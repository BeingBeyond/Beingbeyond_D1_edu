"""Linux device discovery for the D1 examples, independent of the control SDK.

Serial probes only listen or read registers/positions. They never enable torque,
move joints, calibrate, change IDs, or instantiate SDK controllers. USB IDs are
candidate filters, not proof of a device's role. Serial numbers are not required.
"""
import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import struct
import subprocess
import time

import serial
from serial.tools import list_ports


HEAD_ARM_IDS = (11, 12, 21, 22, 23, 24, 25, 26)
SERIAL_USB_IDS = {
    "head_arm": {(0x1A86, 0x55D3)},
    "glove": {(0x1A86, 0x55D3)},
    "exo": {(0x1A86, 0x7523)},
}
SERIAL_BAUDRATES = {"head_arm": 1_000_000, "glove": 115200, "exo": 115200}
CAN_USB_IDS = {(0x0C72, 0x000C)}
CAMERA_USB_IDS = {(0x15AA, 0x1955)}
ANGLE_RE = re.compile(r"=\s*([-+]?\d+(?:\.\d+)?)")


class DeviceDetectionError(RuntimeError):
    pass


def _unique(role, candidates, details=()):
    if len(candidates) != 1:
        reason = "no verified device" if not candidates else "ambiguous devices: " + ", ".join(candidates)
        extra = "\n  " + "\n  ".join(details) if details else ""
        raise DeviceDetectionError(
            "[auto] {}: {}. Connect one matching device, stop programs using it, "
            "or set an explicit device instead of 'auto'.{}".format(role, reason, extra)
        )
    print("[auto] {} -> {}".format(role, candidates[0]))
    return candidates[0]


def _serial_candidates():
    # Never enumerate links as separate devices or depend on enumeration order.
    return {os.path.realpath(p.device): p for p in list_ports.comports()}


def _port_users(device):
    """Best-effort detection of even non-exclusive legacy SDK serial users."""
    target = os.stat(device)
    users = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            for fd in (process / "fd").iterdir():
                try:
                    opened = fd.stat()
                    if stat.S_ISCHR(opened.st_mode) and opened.st_rdev == target.st_rdev:
                        users.append(process.name)
                        break
                except OSError:
                    continue
        except OSError:
            continue
    return users


@contextmanager
def _open_probe(device, baudrate):
    users = _port_users(device)
    if users:
        raise DeviceDetectionError("{} is busy (PID {}); stop its owner before probing".format(device, ", ".join(users)))
    # Set line states before opening; some hardware can still reset on open.
    port = serial.Serial(port=None, baudrate=baudrate, timeout=0.03,
                         write_timeout=0.15, exclusive=True)
    port.dtr = False
    port.rts = False
    port.port = device
    try:
        port.open()
        yield port
    finally:
        port.close()


def _valid_glove_line(line):
    if "ANGLES:" not in line:
        return False
    values = ANGLE_RE.findall(line.split("ANGLES:", 1)[1])
    return len(values) == 6 and all(math.isfinite(float(v)) for v in values)


def _probe_glove(port, timeout=1.2):
    deadline = time.monotonic() + timeout
    valid = 0
    pending = bytearray()
    while time.monotonic() < deadline:
        pending.extend(port.read_until(b"\n", 512))
        while b"\n" in pending:
            raw, _, rest = pending.partition(b"\n")
            pending = bytearray(rest)
            if _valid_glove_line(raw.decode("ascii", errors="replace")):
                valid += 1
                if valid >= 3:
                    return True
            else:
                valid = 0
        if len(pending) > 4096:
            pending.clear()
            valid = 0
    return False


def _head_arm_reply(data, servo_id):
    # ID register read: FF FF <id> 03 <error=0> <id> <checksum>.
    for start in range(max(0, len(data) - 6)):
        frame = data[start:start + 7]
        if (frame[:2] == b"\xff\xff" and frame[2] == servo_id
                and frame[3] == 3 and frame[4] == 0 and frame[5] == servo_id
                and sum(frame[2:]) & 0xFF == 0xFF):
            return True
    return False


def _probe_head_arm(port):
    missing = []
    for servo_id in HEAD_ARM_IDS:
        matched = False
        for _ in range(2):
            port.reset_input_buffer()
            # INST_READ (0x02), ID register (0x05), length 1. No writes.
            packet = bytes((servo_id, 4, 2, 5, 1))
            port.write(b"\xff\xff" + packet + bytes((~sum(packet) & 0xFF,)))
            data = bytearray()
            deadline = time.monotonic() + 0.12
            while time.monotonic() < deadline:
                data.extend(port.read(32))
                if _head_arm_reply(data, servo_id):
                    matched = True
                    break
            if matched:
                break
        if not matched:
            missing.append(servo_id)
    return missing


def _probe_exo(port, servo_ids):
    missing = []
    for servo_id in servo_ids:
        pattern = re.compile(rb"#%03dP(\d{4})(?=[!\r\n\s])" % servo_id)
        matched = False
        for _ in range(2):
            port.reset_input_buffer()
            port.write(("#{:03d}PRAD!".format(servo_id)).encode("ascii"))
            data = bytearray()
            deadline = time.monotonic() + 0.15
            while time.monotonic() < deadline:
                data.extend(port.read(128))
                match = pattern.search(data)
                if match and 500 <= int(match.group(1)) <= 2500:
                    matched = True
                    break
            if matched:
                break
        if not matched:
            missing.append(servo_id)
    return missing


def resolve_serial_ports(requests, baudrates=None, exo_servo_ids=(1, 2, 3, 4, 5, 6)):
    """Resolve requested roles together; explicit paths bypass protocol discovery.

    A busy/unreadable candidate is an error even if another candidate matches:
    the inaccessible device could otherwise hide a second matching controller.
    """
    unknown = set(requests) - set(SERIAL_USB_IDS)
    if unknown:
        raise ValueError("Unknown serial roles: {}".format(sorted(unknown)))
    rates = dict(SERIAL_BAUDRATES, **(baudrates or {}))
    resolved = {role: path for role, path in requests.items() if path != "auto"}
    automatic = set(requests) - set(resolved)
    occupied_paths = {os.path.realpath(path) for path in resolved.values()}
    matches = {role: [] for role in automatic}
    details = {role: [] for role in automatic}
    blocked = {role: False for role in automatic}
    for device, info in sorted(_serial_candidates().items()):
        roles = {role for role in automatic if (info.vid, info.pid) in SERIAL_USB_IDS[role]}
        if not roles or device in occupied_paths:
            continue
        try:
            if roles & {"head_arm", "glove"}:
                # Listen first, even when only the head is requested. Never send
                # binary head queries to an already identified glove.
                with _open_probe(device, rates["glove"]) as port:
                    is_glove = _probe_glove(port)
                    if is_glove:
                        if "glove" in roles:
                            matches["glove"].append(device)
                    elif "head_arm" in roles:
                        port.baudrate = rates["head_arm"]
                        missing = _probe_head_arm(port)
                        if not missing:
                            matches["head_arm"].append(device)
                        else:
                            details["head_arm"].append("{}: head/arm IDs missing: {}".format(device, missing))
                    if not is_glove and "glove" in roles:
                        details["glove"].append("{}: no consecutive valid ANGLES frames".format(device))
            if "exo" in roles:
                with _open_probe(device, rates["exo"]) as port:
                    missing = _probe_exo(port, exo_servo_ids)
                    if not missing:
                        matches["exo"].append(device)
                    else:
                        details["exo"].append("{}: exo IDs missing/invalid: {}".format(device, missing))
        except (OSError, serial.SerialException, DeviceDetectionError) as exc:
            for role in roles:
                blocked[role] = True
                details[role].append("{}: {}".format(device, exc))
    errors = []
    for role in sorted(automatic):
        try:
            if blocked[role]:
                raise DeviceDetectionError("[auto] {}: a candidate is busy or inaccessible.\n  {}".format(role, "\n  ".join(details[role])))
            resolved[role] = _unique(role, matches[role], details[role])
        except DeviceDetectionError as exc:
            errors.append(str(exc))
    if errors:
        raise DeviceDetectionError("\n".join(errors))
    paths = [os.path.realpath(path) for path in resolved.values()]
    if len(set(paths)) != len(paths):
        raise DeviceDetectionError("The same serial port was assigned to multiple roles")
    return resolved


def resolve_head_arm(port="auto", baudrate=1_000_000):
    return resolve_serial_ports({"head_arm": port}, {"head_arm": baudrate})["head_arm"]


def _usb_id(sys_device):
    path = sys_device.resolve()
    for parent in (path,) + tuple(path.parents):
        try:
            return (int((parent / "idVendor").read_text().strip(), 16),
                    int((parent / "idProduct").read_text().strip(), 16))
        except (OSError, ValueError):
            continue
    return None


def resolve_can_iface(iface="auto"):
    if iface != "auto":
        return iface
    candidates = []
    for path in sorted(Path("/sys/class/net").glob("*")):
        try:
            if (path / "type").read_text().strip() == "280" and _usb_id(path / "device") in CAN_USB_IDS:
                candidates.append(path.name)
        except OSError:
            continue
    return _unique("hand CAN", candidates)


def resolve_camera_device(device="auto"):
    if device != "auto":
        return device
    links = {}
    for path in sorted(Path("/dev/v4l/by-id").glob("*video-index0")):
        links[os.path.realpath(str(path))] = str(path)
    candidates = []
    for path in sorted(Path("/sys/class/video4linux").glob("video*")):
        try:
            if (path / "index").read_text().strip() == "0" and _usb_id(path / "device") in CAMERA_USB_IDS:
                node = "/dev/" + path.name
                candidates.append(links.get(os.path.realpath(node), node))
        except OSError:
            continue
    return _unique("stereo camera", candidates)


def resolve_devices(head_arm=None, exo=None, glove=None, can_iface=None, camera=None,
                    head_arm_baud=1_000_000, exo_baud=115200, glove_baud=115200,
                    exo_servo_ids=(1, 2, 3, 4, 5, 6)):
    """Resolve every requested device before any motion-producing SDK setup."""
    result = {}
    # Resolve passive devices first so missing cameras/CAN never trigger probes.
    if can_iface is not None:
        result["can_iface"] = resolve_can_iface(can_iface)
    if camera is not None:
        result["camera"] = resolve_camera_device(camera)
    requests = {role: value for role, value in
                (("head_arm", head_arm), ("exo", exo), ("glove", glove)) if value is not None}
    if requests:
        result.update(resolve_serial_ports(requests,
                      {"head_arm": head_arm_baud, "exo": exo_baud, "glove": glove_baud}, exo_servo_ids))
    return result


def check_can(iface, hand_type="right", bitrate=1_000_000):
    """Read fresh hand joint status on an already configured bus; never set it up."""
    if hand_type not in ("left", "right"):
        raise ValueError("hand_type must be left or right")
    status = subprocess.run(["ip", "-json", "-details", "link", "show", "dev", iface],
                            check=True, stdout=subprocess.PIPE, text=True, timeout=2)
    link = json.loads(status.stdout)[0]
    bus = link.get("linkinfo", {}).get("info_data", {})
    if "UP" not in link.get("flags", []) or bus.get("bittiming", {}).get("bitrate") != bitrate:
        raise DeviceDetectionError("{} must already be UP at {} bit/s for this read-only check".format(iface, bitrate))
    hand_id = 0x27 if hand_type == "right" else 0x28
    with socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW) as bus_socket:
        bus_socket.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER,
                              struct.pack("=II", hand_id, 0xC00007FF))
        bus_socket.bind((iface,))
        bus_socket.settimeout(0.3)
        for _ in range(2):
            # 0x01 with no payload requests status. Commands contain 6 values.
            bus_socket.send(struct.pack("=IB3x8s", hand_id, 1, b"\x01"))
            deadline = time.monotonic() + 0.3
            while time.monotonic() < deadline:
                try:
                    raw = bus_socket.recv(16)
                except socket.timeout:
                    break
                can_id, length, payload = struct.unpack("=IB3x8s", raw)
                if can_id == hand_id and length == 7 and payload[0] == 1:
                    print("[check] {}: {} hand status received (ID 0x{:02x})".format(iface, hand_type, hand_id))
                    return
    raise DeviceDetectionError("{}: no fresh {} hand status response".format(iface, hand_type))


def check_camera(device):
    """Capture one stereo-sized frame, without any SDK or robot initialization."""
    import cv2

    users = _port_users(device)
    if users:
        raise DeviceDetectionError("{} is busy (PID {})".format(device, ", ".join(users)))
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    try:
        if not cap.isOpened():
            raise DeviceDetectionError("Cannot open camera: " + device)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ok, frame = cap.read()
        if not ok or frame is None or frame.shape[0] < 480 or frame.shape[1] < 1280:
            raise DeviceDetectionError("Camera did not return the expected stereo-sized frame: " + device)
        print("[check] {}: captured {}x{}".format(device, frame.shape[1], frame.shape[0]))
    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List serial USB identities only; do not open devices")
    parser.add_argument("--check", action="store_true", help="Also read CAN status and capture a camera frame; no robot motion")
    parser.add_argument("--roles", nargs="+", default=["head_arm", "exo", "glove", "can", "camera"],
                        choices=["head_arm", "exo", "glove", "can", "camera"])
    args = parser.parse_args()
    try:
        if args.list:
            for device, port in sorted(_serial_candidates().items()):
                print("{}: {} VID:PID={} serial={}".format(device, port.description, port.hwid, port.serial_number))
            return 0
        roles = set(args.roles)
        requests = {role: "auto" for role in roles if role in SERIAL_USB_IDS}
        if "can" in roles:
            requests["can_iface"] = "auto"
        if "camera" in roles:
            requests["camera"] = "auto"
        devices = resolve_devices(**requests)
        if args.check:
            if "can_iface" in devices:
                check_can(devices["can_iface"])
            if "camera" in devices:
                check_camera(devices["camera"])
        print("[check] Requested devices resolved successfully; no robot controller was created.")
        return 0
    except (DeviceDetectionError, OSError, subprocess.SubprocessError) as exc:
        print(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
