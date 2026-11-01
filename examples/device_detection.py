"""Linux device discovery for the D1 examples, independent of the control SDK.

Serial probes only listen or read registers/positions. They never enable torque,
move joints, calibrate, change IDs, or instantiate SDK controllers. USB IDs are
candidate filters, not proof of a device's role. Serial numbers are not required.
"""

from diagnostics import configuration_error, failure

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

from diagnostics import DiagnosticError, DiagnosticGroup, as_diagnostic, format_error, joint_label, joint_read_error, install_error_hook, BilingualArgumentParser, report_error, LABELS
from diagnostics import validate_exo_ids

if __name__ == "__main__":
    if "--debug" in sys.argv:
        os.environ["D1_DEBUG"] = "1"
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


DeviceDetectionError = DiagnosticError


@contextmanager
def device_operation(role, device, stage):
    """Keep the role, selected port and original cause on runtime failures."""
    try:
        yield
    except Exception as exc:
        if as_diagnostic(exc) is not None:
            raise
        raise format_error(role, device, stage, exc) from exc


def _unique(role, candidates, details=()):
    if len(candidates) == 1:
        return candidates[0]
    zh, en = LABELS[role]
    if candidates:
        zh = "PCAN 接口" if role == "hand CAN" else zh
        plural = "PCAN interfaces" if role == "hand CAN" else en + " devices"
        raise failure("S04", "找到多个{}：{}，无法自动选择。".format(zh, "、".join(candidates)),
                      "Multiple matching {} were found: {}.".format(plural, ", ".join(candidates)),
                      "请拔下多余设备，或显式填写要使用的设备路径。", "Disconnect the extra devices or explicitly select a device path.",
                      role=role, candidates=candidates, details=details)
    if role == "hand CAN":
        raise failure("C01", "未识别到 PCAN 接口。", "No matching PCAN interface was found.",
                      "请插好 PCAN 的 USB 线；若已插好，再检查 PCAN 驱动。", "Check the PCAN USB connection, then the PCAN driver if it is already connected.", details=details)
    if role == "stereo camera":
        raise failure("V01", "未识别到双目相机。", "No matching stereo camera was found.",
                      "请检查相机自己的 USB 线。", "Check the camera's own USB cable.", details=details)
    raise failure("S01", "未识别到{}的 USB 串口。".format(zh), "No matching USB serial port was found for the {}.".format(en),
                  "请重新插好{}的 USB 线，再试一次。".format(zh), "Reconnect the {} USB cable and try again.".format(en), role=role)


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
        raise failure("S07", "串口 {} 正被进程 {} 占用。".format(device, ", ".join(users)),
                      "Port {} is in use by process {}.".format(device, ", ".join(users)),
                      "请确认并退出对应程序，再运行本示例。", "Identify and close that program, then run this example again.", device=device, pids=users)
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
                    zh, en = joint_label("head_arm", [servo_id])
                    raise failure("H04", "{}返回故障码 0x{:02x}。".format(zh, error),
                                  "{} reported fault 0x{:02x}.".format((en[:1].upper() + en[1:]), error),
                                  "请停止运行，记录关节编号和故障码，按设备故障说明处理。",
                                  "Stop operation, record the joint number and code, and follow the device fault instructions.",
                                  joint_id=servo_id, fault=error)
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


def _glove_error(device, baudrate):
    return failure("G01", "串口 {} 已打开，但未收到有效手套数据。".format(device),
                   "Port {} opened, but no valid glove data was received.".format(device),
                   "请检查手套 USB 线，并核对手套端口和波特率 {}。".format(baudrate),
                   "Check the glove USB cable, configured port and baudrate {}.".format(baudrate), device=device)


def _wrong_glove(device):
    return failure("H01", "头臂串口 {} 实际连接的是手套。".format(device),
                   "The configured head/arm port {} is connected to the glove.".format(device),
                   "请将头臂串口配置改为头臂控制器对应的端口。", "Set the head/arm port to the robot controller's port.", device=device)


def resolve_serial_ports(requests, baudrates=None, exo_servo_ids=(1, 2, 3, 4, 5, 6),
                         verify_explicit=False):
    if "exo" in requests:
        validate_exo_ids(exo_servo_ids)
    unknown = set(requests) - set(SERIAL_USB_IDS)
    if unknown:
        raise configuration_error("serial roles", sorted(unknown), "head_arm / exo / glove")
    rates = dict(SERIAL_BAUDRATES, **(baudrates or {}))
    resolved = {role: path for role, path in requests.items() if path != "auto"}
    automatic = set(requests) - set(resolved)
    assignments = {}
    for role, path in resolved.items():
        assignments.setdefault(os.path.realpath(path), []).append(role)
    for path, roles in assignments.items():
        if len(roles) > 1:
            zh = "和".join(LABELS[role][0] for role in roles)
            en = " and ".join(LABELS[role][1] for role in roles)
            raise failure("S09", "{}使用了同一串口：{}。".format(zh, path),
                          "{} use the same serial port: {}.".format((en[:1].upper() + en[1:]), path),
                          "请核对两根 USB 线对应的端口，并修改配置。", "Check which port belongs to each USB connection and correct the settings.", device=path, roles=roles)
    matches = {role: [] for role in automatic}
    details = {role: [] for role in automatic}
    blocked = set()
    # Physical ttyS ports are irrelevant to USB discovery, including its failure output.
    candidates = {dev: info for dev, info in (_serial_candidates() if automatic else {}).items()
                  if info.vid is not None or dev.startswith(("/dev/ttyUSB", "/dev/ttyACM"))}
    for device, info in sorted(candidates.items()):
        roles = {role for role in automatic if (info.vid, info.pid) in SERIAL_USB_IDS[role]}
        if device in assignments or not roles:
            continue
        try:
            if roles & {"head_arm", "glove"}:
                # Listen first. Never send head queries to an identified glove.
                with _open_probe(device, rates["glove"]) as port:
                    is_glove = _probe_glove(port)
                    if is_glove:
                        if "glove" in roles:
                            matches["glove"].append(device)
                        if "head_arm" in roles:
                            details["head_arm"].append(_wrong_glove(device))
                    else:
                        if "head_arm" in roles:
                            port.baudrate = rates["head_arm"]
                            missing = _probe_head_arm(port)
                            if missing:
                                details["head_arm"].append(joint_read_error("head_arm", device, missing, HEAD_ARM_IDS, rates["head_arm"]))
                            else:
                                matches["head_arm"].append(device)
                        if "glove" in roles:
                            details["glove"].append(_glove_error(device, rates["glove"]))
            if "exo" in roles:
                with _open_probe(device, rates["exo"]) as port:
                    missing = _probe_exo(port, exo_servo_ids)
                    if missing:
                        details["exo"].append(joint_read_error("exo", device, missing, exo_servo_ids, rates["exo"]))
                    else:
                        matches["exo"].append(device)
        except (OSError, serial.SerialException, DeviceDetectionError) as exc:
            for role in roles:
                blocked.add(role)
                details[role].append(format_error(role, device, "serial verification", exc))
    errors = []
    if verify_explicit:
        for role, device in resolved.items():
            try:
                with device_operation(role, device, "explicit serial verification"):
                    with _open_probe(device, rates[role]) as port:
                        if role == "glove":
                            if not _probe_glove(port):
                                raise _glove_error(device, rates[role])
                        elif role == "head_arm":
                            port.baudrate = rates["glove"]
                            if _probe_glove(port):
                                raise _wrong_glove(device)
                            port.baudrate = rates[role]
                            missing = _probe_head_arm(port)
                            if missing:
                                raise joint_read_error(role, device, missing, HEAD_ARM_IDS, rates[role])
                        else:
                            missing = _probe_exo(port, exo_servo_ids)
                            if missing:
                                raise joint_read_error(role, device, missing, exo_servo_ids, rates[role])
            except DeviceDetectionError as exc:
                errors.append(exc)
    for role in sorted(automatic):
        try:
            if role in blocked:
                errors.extend(details[role])
                continue
            if matches[role]:
                resolved[role] = _unique(role, matches[role])
            elif details[role]:
                errors.extend(details[role])
            else:
                available = [dev for dev in candidates if dev not in assignments]
                if available:
                    zh, en = LABELS[role]
                    raise failure("S03", "已发现 USB 串口 {}，但无法自动确认哪个连接{}。".format("、".join(available), zh),
                                  "USB serial ports {} were found, but none could be selected automatically for the {}.".format(", ".join(available), en),
                                  "请对照拔插确认端口，再将它填入当前示例的{}串口配置。".format(zh),
                                  "Identify the port by reconnecting the device, then enter it in this example's {} port setting.".format(en), candidates=available, role=role)
                resolved[role] = _unique(role, [])
        except DeviceDetectionError as exc:
            errors.append(exc)
    if errors:
        raise DiagnosticGroup(errors)
    paths = [os.path.realpath(path) for path in resolved.values()]
    if len(set(paths)) != len(paths):
        raise failure("S09", "同一串口被分配给多个设备。", "Multiple devices were assigned the same serial port.",
                      "请核对端口配置。", "Check the port settings.")
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
    if exo is not None:
        validate_exo_ids(exo_servo_ids)
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
            errors.append(exc)
    if check_ready:
        errors.extend(_check_peripherals(result, hand_type, hand_baud))
    requests = {role: value for role, value in
                (("head_arm", head_arm), ("exo", exo), ("glove", glove)) if value is not None}
    if errors:
        if requests:
            notes = [("尚未检查：{}。请处理以上问题后重试。".format("、".join(LABELS[r][0] for r in requests)),
                      "Not checked: {}. Resolve the errors above and retry.".format(", ".join(LABELS[r][1] for r in requests)))]
        else:
            notes = []
        raise DiagnosticGroup(errors, notes)
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
            errors.append(format_error(role, device, "communication check", exc))
    return errors


def check_can(iface, hand_type="right", bitrate=1_000_000):
    with device_operation("hand CAN", iface, "status check"):
        _check_can(iface, hand_type, bitrate)


def _check_can(iface, hand_type, bitrate):
    """Read fresh hand joint status on an already configured bus; never set it up."""
    if hand_type not in ("left", "right"):
        raise configuration_error("hand_type", hand_type, "left / right")
    try:
        status = subprocess.run(["ip", "-json", "-details", "link", "show", "dev", iface],
                                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2)
    except subprocess.CalledProcessError as exc:
        if "does not exist" in (exc.stderr or "") or "Cannot find device" in (exc.stderr or ""):
            raise failure("C02", "CAN 接口 {} 不存在。".format(iface), "CAN interface {} does not exist.".format(iface),
                          "请用 ip link show 查看实际接口名，并修改 CAN 配置。", "Check the interface name with ip link show and update the CAN setting.") from exc
        raise
    links = json.loads(status.stdout)
    if not links:
        raise failure("C02", "CAN 接口 {} 不存在。".format(iface), "CAN interface {} does not exist.".format(iface),
                      "请用 ip link show 查看实际接口名，并修改 CAN 配置。", "Check the interface name with ip link show and update the CAN setting.")
    link = links[0]
    bus = link.get("linkinfo", {}).get("info_data", {})
    if bus.get("state") == "BUS-OFF":
        raise failure("C05", "CAN 接口 {} 已进入 BUS-OFF 状态。".format(iface), "CAN interface {} is BUS-OFF.".format(iface),
                      "请先检查手部供电和 CAN 接线，再按 README 恢复接口。", "Check hand power and CAN wiring, then recover the interface using README.")
    if "UP" not in link.get("flags", []):
        raise failure("C03", "CAN 接口 {} 尚未启动。".format(iface), "CAN interface {} is down.".format(iface),
                      "请按 README 启动这个接口，再运行示例。", "Bring up this interface using README, then retry.")
    actual = bus.get("bittiming", {}).get("bitrate")
    if bus.get("state") in ("STOPPED", "SLEEPING") or actual is None:
        raise failure("C07", "CAN 接口 {} 未就绪，状态：{}。".format(iface, bus.get("state", "无法读取")),
                      "CAN interface {} is not ready. State: {}.".format(iface, bus.get("state", "unavailable")),
                      "请按 README 检查接口配置。", "Check its configuration using README.")
    if actual != bitrate:
        raise failure("C04", "CAN 波特率不符：当前 {}，需要 {} bit/s。接口：{}。".format(actual, bitrate, iface),
                      "CAN bitrate mismatch on {}: {}, expected {} bit/s.".format(iface, actual, bitrate),
                      "请按 README 修改该接口的波特率。", "Set the interface bitrate using README.")
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
    raise failure("C06", "{}手未返回状态。CAN 接口：{}。".format("左" if hand_type == "left" else "右", iface),
                  "No status reply from the {} hand on {}.".format(hand_type, iface),
                  "请检查这只手的电源和 CAN 线，并确认左右手配置。", "Check its power and CAN cable, and verify the selected hand side.")


def check_camera(device):
    with device_operation("stereo camera", device, "capture check"):
        _check_camera(device)


def _check_camera(device):
    """Capture one stereo-sized frame, without any SDK or robot initialization."""
    import cv2

    users = _port_users(device)
    if users:
        raise failure("V05", "相机 {} 正被进程 {} 占用。".format(device, ", ".join(users)),
                      "Camera {} is in use by process {}.".format(device, ", ".join(users)),
                      "请退出对应的相机程序后重试。", "Close that camera application and retry.")
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    try:
        if not cap.isOpened():
            raise failure("V06", "无法打开相机：{}。".format(device), "Could not open camera {}.".format(device),
                          "请先退出其他使用相机的程序，再重新插好相机 USB 线。", "Close other camera applications, then reconnect its USB cable.")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise format_error("stereo camera", device, "capture check", TimeoutError("No frame"))
        if frame.shape[0] < 480 or frame.shape[1] < 1280:
            raise failure("V08", "相机图像尺寸不符：实际 {}×{}，需要至少 1280×480。".format(frame.shape[1], frame.shape[0]),
                          "Camera frame size mismatch: got {}x{}, need at least 1280x480.".format(frame.shape[1], frame.shape[0]),
                          "请核对相机设备和采集分辨率。", "Check the selected camera and capture resolution.")
        print("[check] 已采集图像 / {}: captured {}x{}".format(device, frame.shape[1], frame.shape[0]))
    finally:
        cap.release()


def main():
    parser = BilingualArgumentParser(description=__doc__)
    parser.add_argument("--debug", action="store_true", help="显示完整异常 / Show full error details")
    parser.add_argument("--list", action="store_true", help="仅列出串口 USB 信息，不打开设备 / List serial USB identities only; do not open devices")
    parser.add_argument("--check", action="store_true", help="额外读取 CAN 状态和相机图像，不发送运动命令 / Also read CAN status and capture a camera frame; no robot motion")
    parser.add_argument("--roles", nargs="+", default=["head_arm", "exo", "glove", "can", "camera"],
                        choices=["head_arm", "exo", "glove", "can", "camera"])
    parser.add_argument("--hand-type", choices=["left", "right"], default="right")
    args = parser.parse_args()
    if args.debug:
        os.environ["D1_DEBUG"] = "1"
    try:
        if args.list:
            found = False
            for device, port in sorted(_serial_candidates().items()):
                if port.vid is None and not device.startswith(("/dev/ttyUSB", "/dev/ttyACM")):
                    continue
                print("{}: {} VID:PID={} serial={}".format(device, port.description, port.hwid, port.serial_number))
                found = True
            if not found:
                print("未发现可识别的 USB 串口。\nNo recognizable USB serial ports were found.")
            return 0
        roles = set(args.roles)
        requests = {role: "auto" for role in roles if role in SERIAL_USB_IDS}
        if "can" in roles:
            requests["can_iface"] = "auto"
        if "camera" in roles:
            requests["camera"] = "auto"
        devices = resolve_devices(**requests, check_ready=args.check, hand_type=args.hand_type)
        if args.check:
            print("设备检查通过。\nDevice checks passed.")
        else:
            print("设备识别完成：{}。\nDevices detected: {}.".format(
                "、".join("{} {}".format(LABELS[role][0], path) for role, path in devices.items()),
                ", ".join("{} {}".format(LABELS[role][1], path) for role, path in devices.items())))
            pending = [LABELS[role] for role in ("can_iface", "camera") if role in devices]
            if pending:
                print("尚未检查通信：{}（使用 --check 检查）。\nCommunication not checked: {} (use --check).".format(
                    "、".join(zh for zh, en in pending), ", ".join(en for zh, en in pending)))
        return 0
    except (DeviceDetectionError, OSError, subprocess.SubprocessError) as exc:
        report_error(exc, file=sys.stdout)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
