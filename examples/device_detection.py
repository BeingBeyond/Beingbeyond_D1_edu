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
import sys
import stat
import struct
import subprocess
import time

from diagnostics import bilingual, format_error, role_help, install_error_hook, BilingualArgumentParser, report_error

if __name__ == "__main__":
    install_error_hook()

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


@contextmanager
def device_operation(role, device, stage):
    """Keep the role, selected port and original cause on runtime failures."""
    try:
        yield
    except Exception as exc:
        raise DeviceDetectionError(format_error(role, device, stage, exc)) from exc


def _unique(role, candidates, details=()):
    if len(candidates) != 1:
        reason = "未找到通过验证的设备 / no verified device" if not candidates else "存在多个匹配设备，请显式指定目标端口 / ambiguous devices: " + ", ".join(candidates)
        label, advice = role_help(role)
        extra = "\n  " + "\n  ".join(details) if details else ""
        raise DeviceDetectionError(
            "[auto] {}: {}. 自动识别失败：没有唯一通过验证的设备。 / Discovery failed. Connect one matching device, stop programs using it, "
            "or set an explicit device instead of 'auto'.{}\n{}\n排查建议： / Suggested checks: {}".format(role, reason, extra, label, advice)
        )
    print("[auto] 已识别 / Resolved {} -> {}".format(role, candidates[0]))
    return candidates[0]


def _serial_candidates():
    # Never enumerate links as separate devices or depend on enumeration order.
    return {os.path.realpath(p.device): p for p in list_ports.comports()}


def _port_users(device):
    """Best-effort detection of even non-exclusive legacy SDK serial users."""
    target = os.stat(device)
    users = []
    try:
        processes = list(Path("/proc").iterdir())
    except OSError:
        return users
    for process in processes:
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
        raise DeviceDetectionError("串口被占用，请先确认 PID 对应程序并正常退出 / {} is busy (PID {}); stop its owner before probing".format(device, ", ".join(users)))
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


def _head_arm_error(data, servo_id):
    """Accept a servo error only from a complete, checksum-valid status packet."""
    for start in range(max(0, len(data) - 5)):
        if data[start:start + 3] != b"\xff\xff" + bytes((servo_id,)):
            continue
        length = data[start + 3]
        frame = data[start:start + length + 4]
        if (length in (2, 3) and len(frame) == length + 4
                and sum(frame[2:]) & 0xFF == 0xFF and frame[4]):
            return frame[4]
    return None


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
                error = _head_arm_error(data, servo_id)
                if error is not None:
                    raise DeviceDetectionError(
                        "头臂舵机已回复故障码，请按 ID 核对供电、负载及接线，不能视为舵机缺失 / head/arm servo ID {} replied with error status 0x{:02x}; "
                        "the servo is present but reports a fault".format(servo_id, error)
                    )
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


def resolve_serial_ports(requests, baudrates=None, exo_servo_ids=(1, 2, 3, 4, 5, 6),
                         verify_explicit=False):
    """Resolve roles together; optionally verify explicit paths without USB filters.

    A busy/unreadable candidate is an error even if another candidate matches:
    the inaccessible device could otherwise hide a second matching controller.
    """
    unknown = set(requests) - set(SERIAL_USB_IDS)
    if unknown:
        raise ValueError("未知串口角色，请检查 requests 配置 / Unknown serial roles: {}".format(sorted(unknown)))
    rates = dict(SERIAL_BAUDRATES, **(baudrates or {}))
    resolved = {role: path for role, path in requests.items() if path != "auto"}
    automatic = set(requests) - set(resolved)
    occupied_paths = {os.path.realpath(path) for path in resolved.values()}
    assignments = {}
    for role, path in resolved.items():
        assignments.setdefault(os.path.realpath(path), []).append(role)
    duplicates = ["{} -> {}".format(path, ", ".join(roles))
                  for path, roles in assignments.items() if len(roles) > 1]
    if duplicates:
        raise DeviceDetectionError("同一串口不能分配给多个角色，请分别核对头臂、遥操臂和手套 USB 线及端口配置 / The same serial port was assigned to multiple roles: " + "; ".join(duplicates))
    matches = {role: [] for role in automatic}
    details = {role: [] for role in automatic}
    blocked = {role: False for role in automatic}
    candidates = _serial_candidates() if automatic else {}
    for device, info in sorted(candidates.items()):
        roles = {role for role in automatic if (info.vid, info.pid) in SERIAL_USB_IDS[role]}
        if not roles or device in occupied_paths:
            for role in automatic:
                identity = "{:04x}:{:04x}".format(info.vid, info.pid) if info.vid is not None and info.pid is not None else "unknown"
                reason = "已被显式指定给其他角色 / reserved by an explicit role" if device in occupied_paths else "USB 型号与此角色不符 / USB identity does not match this role"
                details[role].append("候选设备已排除 / {}: USB {} excluded: {}".format(device, identity, reason))
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
                        if "head_arm" in roles:
                            details["head_arm"].append("此端口实际为手套，请检查头臂 USB 线并修正端口配置 / {}: identified as glove; not a head/arm controller".format(device))
                    elif "head_arm" in roles:
                        port.baudrate = rates["head_arm"]
                        missing = _probe_head_arm(port)
                        if not missing:
                            matches["head_arm"].append(device)
                        else:
                            details["head_arm"].append("以下头臂舵机 ID 未收到有效应答；请核对头臂 USB、供电、波特率，断电后检查对应 ID 及上游串联线；无应答不能证明舵机损坏 / {}: no valid head/arm reply for IDs {}; check power, baudrate, wiring or corrupt replies".format(device, missing))
                    if not is_glove and "glove" in roles:
                        details["glove"].append("手套未连续输出有效的 6 路角度数据，检查手套 USB 线、供电、串口及 115200 波特率 / {}: no consecutive valid ANGLES frames".format(device))
            if "exo" in roles:
                with _open_probe(device, rates["exo"]) as port:
                    missing = _probe_exo(port, exo_servo_ids)
                    if not missing:
                        matches["exo"].append(device)
                    else:
                        details["exo"].append("以下遥操臂舵机 ID 应答缺失或无效，请检查遥操臂 USB 线、供电、波特率，断电后检查对应 ID 的舵机线及上游接头 / {}: exo IDs missing/invalid: {}".format(device, missing))
        except (OSError, serial.SerialException, DeviceDetectionError) as exc:
            for role in roles:
                blocked[role] = True
                details[role].append("{} at {} baud: {}".format(device, rates[role], exc))
    errors = []
    if verify_explicit:
        for role, device in resolved.items():
            try:
                with device_operation(role, device, "explicit serial verification"):
                    with _open_probe(device, rates[role]) as port:
                        if role == "glove":
                            if not _probe_glove(port):
                                raise DeviceDetectionError("手套缺少连续有效 ANGLES 数据，请核对手套连线及波特率 / no consecutive valid ANGLES frames")
                        elif role == "head_arm":
                            port.baudrate = rates["glove"]
                            if _probe_glove(port):
                                raise DeviceDetectionError("端口实际连接手套，请重新核对头臂 USB 线和端口配置 / identified as glove; not a head/arm controller")
                            port.baudrate = rates[role]
                            missing = _probe_head_arm(port)
                            if missing:
                                raise DeviceDetectionError("以下头臂舵机 ID 未有效应答；核对供电及波特率，断电后检查对应舵机和上游串联线 / no valid head/arm reply for IDs {}".format(missing))
                        else:
                            missing = _probe_exo(port, exo_servo_ids)
                            if missing:
                                raise DeviceDetectionError("以下遥操臂舵机 ID 未有效应答；核对供电及波特率，断电后检查对应 ID 和上游串联线 / exo IDs missing/invalid: {}".format(missing))
            except DeviceDetectionError as exc:
                errors.append(str(exc))
    for role in sorted(automatic):
        try:
            if blocked[role]:
                label, advice = role_help(role)
                raise DeviceDetectionError("[auto] {} stage=serial verification: 候选串口无法验证，请先处理下方原因 / a candidate could not be verified.\n  {}\n{}\n排查建议： / Suggested checks: {}".format(role, "\n  ".join(details[role]), label, advice))
            if not candidates:
                details[role].append("未枚举到串口，请检查对应设备 USB 线、供电和驱动 / serial inventory is empty; check USB connection and driver enumeration")
            resolved[role] = _unique(role, matches[role], details[role])
        except DeviceDetectionError as exc:
            errors.append(str(exc))
    if errors:
        raise DeviceDetectionError("\n".join(errors))
    paths = [os.path.realpath(path) for path in resolved.values()]
    if len(set(paths)) != len(paths):
        raise DeviceDetectionError("同一串口不能分配给多个设备角色，请分别连接头臂、遥操臂和手套 USB 线并核对配置 / The same serial port was assigned to multiple roles")
    return resolved


def resolve_head_arm(port="auto", baudrate=1_000_000):
    with device_operation("head_arm", port, "serial verification"):
        return resolve_serial_ports({"head_arm": port}, {"head_arm": baudrate},
                                    verify_explicit=True)["head_arm"]


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
    details = []
    for path in sorted(Path("/sys/class/net").glob("*")):
        try:
            if (path / "type").read_text().strip() == "280":
                identity = _usb_id(path / "device")
                if identity in CAN_USB_IDS:
                    candidates.append(path.name)
                else:
                    details.append("已排除非 PCAN 接口，不要仅凭 can0 编号选择设备 / {}: excluded non-PCAN USB identity {} (possibly a built-in CAN controller)".format(path.name, identity))
        except OSError as exc:
            details.append("无法读取接口信息，请核对驱动及系统设备状态 / {}: cannot inspect interface: {}".format(path.name, exc))
    if not candidates:
        details.append("未找到目标 PCAN：检查 PCAN USB 线，运行 lsusb 并核对驱动是否加载 / expected USB 0c72:000c SocketCAN adapter; check lsusb and PCAN driver loading")
    return _unique("hand CAN", candidates, details)


def resolve_camera_device(device="auto"):
    if device != "auto":
        return device
    links = {}
    for path in sorted(Path("/dev/v4l/by-id").glob("*video-index0")):
        links[os.path.realpath(str(path))] = str(path)
    candidates = []
    details = []
    for path in sorted(Path("/sys/class/video4linux").glob("video*")):
        try:
            index = (path / "index").read_text().strip()
            identity = _usb_id(path / "device")
            node = "/dev/" + path.name
            if index == "0" and identity in CAMERA_USB_IDS:
                candidates.append(links.get(os.path.realpath(node), node))
            else:
                details.append("相机候选 USB 型号或视频索引不符，已排除 / {}: excluded USB identity {}, video index {}".format(node, identity, index))
        except OSError as exc:
            details.append("无法读取视频设备信息，请检查相机 USB 和设备节点 / {}: cannot inspect video device: {}".format(path.name, exc))
    if not candidates:
        details.append("未找到目标双目相机：检查相机自己的 USB 线，并核对 lsusb 和 /dev/v4l/by-id/ / expected USB 15aa:1955 video index 0; check lsusb and /dev/v4l/by-id/")
    return _unique("stereo camera", candidates, details)


def resolve_devices(head_arm=None, exo=None, glove=None, can_iface=None, camera=None,
                    head_arm_baud=1_000_000, exo_baud=115200, glove_baud=115200,
                    exo_servo_ids=(1, 2, 3, 4, 5, 6), check_ready=False,
                    hand_type="right", hand_baud=1_000_000):
    """Resolve devices and optionally verify communication before controller setup.

    Readiness checks never configure CAN: it must already be UP at hand_baud.
    Passive failures are collected together; serial probes are then skipped.
    """
    result = {}
    errors = []
    for role, device, resolver in (("can_iface", can_iface, resolve_can_iface),
                                   ("camera", camera, resolve_camera_device)):
        if device is None:
            continue
        try:
            with device_operation(role, device, "discovery"):
                result[role] = resolver(device)
        except DeviceDetectionError as exc:
            errors.append(str(exc))
    if check_ready:
        errors.extend(_check_peripherals(result, hand_type, hand_baud))
    requests = {role: value for role, value in
                (("head_arm", head_arm), ("exo", exo), ("glove", glove)) if value is not None}
    if errors:
        if requests:
            errors.append("[check] CAN/相机预检查失败，以下串口角色尚未检查（不表示已通过） / serial verification skipped because peripheral checks failed: " +
                          ", ".join("{}={}".format(role, path) for role, path in requests.items()))
        raise DeviceDetectionError("\n".join(errors))
    if requests:
        with device_operation("serial roles", str(requests), "discovery/verification"):
            result.update(resolve_serial_ports(requests,
                          {"head_arm": head_arm_baud, "exo": exo_baud, "glove": glove_baud},
                          exo_servo_ids, verify_explicit=check_ready))
    return result


def _check_peripherals(devices, hand_type, hand_baud):
    errors = []
    for role, device in devices.items():
        try:
            if role == "can_iface":
                check_can(device, hand_type, hand_baud)
            elif role == "camera":
                check_camera(device)
        except Exception as exc:
            errors.append("[device] {} port={} stage=communication check: {}".format(role, device, exc))
    return errors


def check_can(iface, hand_type="right", bitrate=1_000_000):
    with device_operation("hand CAN", iface, "status check"):
        _check_can(iface, hand_type, bitrate)


def _check_can(iface, hand_type, bitrate):
    """Read fresh hand joint status on an already configured bus; never set it up."""
    if hand_type not in ("left", "right"):
        raise ValueError("hand_type 仅支持 left（左手）或 right（右手），请修改配置 / hand_type must be left or right")
    status = subprocess.run(["ip", "-json", "-details", "link", "show", "dev", iface],
                            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2)
    link = json.loads(status.stdout)[0]
    bus = link.get("linkinfo", {}).get("info_data", {})
    if "UP" not in link.get("flags", []) or bus.get("bittiming", {}).get("bitrate") != bitrate:
        raise DeviceDetectionError("CAN 未就绪：请先核对 PCAN 接口、供电和接线，再按 README 配置为 UP 及指定波特率；本检查不会自动配置总线 / {} must already be UP at {} bit/s for this read-only check; "
                                   "configure the confirmed PCAN interface first (see lib/pcan/README.md)".format(iface, bitrate))
    if bus.get("state") in ("BUS-OFF", "STOPPED", "SLEEPING"):
        raise DeviceDetectionError("CAN 状态异常：检查手部供电、CAN_H/CAN_L 和终端电阻，确认后按 README 恢复 / {} CAN state is {}; check wiring and recover the bus".format(iface, bus["state"]))
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
                    print("[check] 已收到手部状态 / {}: {} hand status received (ID 0x{:02x})".format(iface, hand_type, hand_id))
                    return
    raise DeviceDetectionError("未收到新的手部应答：核对左右手配置、手部电源、PCAN USB 和 CAN 线；接口 UP 不等于手部有应答 / {}: no fresh {} hand status response".format(iface, hand_type))


def check_camera(device):
    with device_operation("stereo camera", device, "capture check"):
        _check_camera(device)


def _check_camera(device):
    """Capture one stereo-sized frame, without any SDK or robot initialization."""
    import cv2

    users = _port_users(device)
    if users:
        raise DeviceDetectionError("设备被占用，请先确认 PID 对应程序并正常退出 / {} is busy (PID {})".format(device, ", ".join(users)))
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    try:
        if not cap.isOpened():
            raise DeviceDetectionError("无法打开相机，请检查相机 USB 线、video 权限和占用 / Cannot open camera: " + device)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ok, frame = cap.read()
        if not ok or frame is None or frame.shape[0] < 480 or frame.shape[1] < 1280:
            raise DeviceDetectionError("相机没有返回有效的 1280×480 双目图像，请核对 MJPG 格式、USB 线和带宽 / Camera did not return the expected stereo-sized frame: " + device)
        print("[check] 已采集图像 / {}: captured {}x{}".format(device, frame.shape[1], frame.shape[0]))
    finally:
        cap.release()


def main():
    parser = BilingualArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="仅列出串口 USB 信息，不打开设备 / List serial USB identities only; do not open devices")
    parser.add_argument("--check", action="store_true", help="额外读取 CAN 状态和相机图像，不发送运动命令 / Also read CAN status and capture a camera frame; no robot motion")
    parser.add_argument("--roles", nargs="+", default=["head_arm", "exo", "glove", "can", "camera"],
                        choices=["head_arm", "exo", "glove", "can", "camera"])
    parser.add_argument("--hand-type", choices=["left", "right"], default="right")
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
        resolve_devices(**requests, check_ready=args.check, hand_type=args.hand_type)
        print("[check] 指定设备已识别；是否检查 CAN/相机通信取决于 --check，未创建机器人控制器 / Requested devices resolved successfully; no robot controller was created.")
        return 0
    except (DeviceDetectionError, OSError, subprocess.SubprocessError) as exc:
        report_error(exc, file=sys.stdout)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
