"""Short bilingual diagnostics for examples. No SDK imports or hardware access."""
import argparse
import ast
import errno
import os
import re
import subprocess
import sys
import threading
import traceback


def bilingual(chinese, english):
    return chinese + "\n" + english


LABELS = {
    "head_arm": ("头臂控制器", "head/arm controller"),
    "exo": ("遥操臂", "exoskeleton"), "glove": ("手套", "glove"),
    "hand CAN": ("灵巧手", "hand"), "can_iface": ("灵巧手", "hand"),
    "camera": ("相机", "camera"), "stereo camera": ("相机", "camera"),
    "display": ("图形窗口", "window"),
}

_REPORT_LOCK = threading.Lock()


class DiagnosticError(RuntimeError):
    """User text and technical context are kept separate."""
    def __init__(self, chinese, english=None, action_zh="", action_en="", *, code="unknown", **context):
        self.chinese = chinese
        self.english = english or short_reason(chinese)
        self.action_zh, self.action_en = action_zh, action_en
        self.code, self.context = code, context
        self.reported = False
        self.original = None
        super().__init__(chinese)

    def render(self, severity="error"):
        zh, en = {"error": ("错误", "Error"), "warning": ("警告", "Warning"),
                  "info": ("提示", "Info")}[severity]
        lines = ["[{}] {}".format(zh, self.chinese)]
        if self.action_zh:
            lines.append(self.action_zh)
        lines.append("[{}] {}".format(en, self.english))
        if self.action_en:
            lines.append(self.action_en)
        return "\n".join(lines)

    def __str__(self):
        return self.render()


class DiagnosticGroup(DiagnosticError):
    def __init__(self, errors, notes=()):
        self.errors = []
        self.notes = list(notes)
        seen = set()
        for exc in errors:
            error = as_diagnostic(exc) or format_error(None, None, "runtime", exc)
            children = error.errors if isinstance(error, DiagnosticGroup) else [error]
            if isinstance(error, DiagnosticGroup):
                self.notes.extend(error.notes)
            for child in children:
                key = (child.code, child.chinese, child.english)
                if key not in seen:
                    seen.add(key)
                    self.errors.append(child)
        super().__init__("设备检查未通过。", "Device checks failed.", code="group")

    def render(self, severity="error"):
        blocks = [error.render(severity) for error in self.errors]
        blocks.extend("{}\n{}".format(zh, en) for zh, en in dict.fromkeys(self.notes))
        return "\n".join(blocks)


def as_diagnostic(exc):
    if isinstance(exc, DiagnosticError):
        return exc
    cached = getattr(exc, "_d1_diagnostic", None)
    if cached is not None:
        return cached
    if len(getattr(exc, "args", ())) == 1 and isinstance(exc.args[0], DiagnosticError):
        return exc.args[0]
    return None


def short_reason(exc):
    # Original wording stays in the exception chain / D1_DEBUG output.
    text = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
    text = text.replace("舵机", "关节")
    return re.sub(r"\bservos?\b", "joint", text, flags=re.IGNORECASE)[:180]


def failure(code, zh, en, action_zh, action_en, **context):
    return DiagnosticError(zh, en, action_zh, action_en, code=code, **context)


def configuration_error(name, actual, expected, expected_en=None):
    return failure("P08", "{} 设置错误：{}；要求 {}。".format(name, actual, expected),
                   "Invalid {}: {}; expected {}.".format(name, actual, expected if expected_en is None else expected_en),
                   "请修改 {} 后重试。".format(name), "Correct {} and retry.".format(name))


def validate_exo_ids(ids):
    """Use the SDK protocol's 0..999 range; never rewrite configured IDs."""
    if not isinstance(ids, (list, tuple, range)):
        raise configuration_error("arm_exo_servo_ids", ids, "关节 ID 列表", "a list of joint IDs")
    if not ids or any(type(sid) is not int or not 0 <= sid <= 999 for sid in ids):
        raise configuration_error("arm_exo_servo_ids", ids, "0～999 范围内的整数", "integers in the range 0..999")
    if len(set(ids)) != len(ids):
        raise configuration_error("arm_exo_servo_ids", ids, "每个关节的 ID 不重复", "distinct joint IDs")


def model_file_error(path):
    return failure("P06", "找不到 URDF 文件：{}。".format(path), "URDF file not found: {}.".format(path),
                   "请检查模型路径配置。", "Check the configured model path.")


def missing_joints_error(names):
    return failure("P07", "模型缺少需要的关节或手指：{}。".format(names),
                   "The model is missing required joints or fingers: {}.".format(names),
                   "请核对模型文件和关节配置。", "Check the model file and joint configuration.")


def target_warning(zh="头臂关节", en="Head/arm joints"):
    return failure("H05", "{}未到达目标（超时或无进展）。".format(zh), "{} did not reach the target (timeout or no progress).".format(en),
                   "请检查关节是否受阻，以及目标角度是否超出限位。", "Check for obstruction and verify the target against joint limits.")


def _remember(exc, diagnostic):
    exc._d1_diagnostic = diagnostic
    diagnostic.original = exc
    return diagnostic


def format_error(role, device, stage, exc):
    """Choose one next step; nested boundaries must not add more advice."""
    existing = as_diagnostic(exc)
    if existing is not None:
        return existing
    # SDK initialization reports a full failed-ID list; this is enough evidence
    # to use the same chain analysis as the read-only startup check.
    match = re.fullmatch(r"Servo ids not responding: (\[[\d, ]*\])", str(exc)) if role == "head_arm" else None
    if match:
        missing = ast.literal_eval(match.group(1))
        expected = (11, 12, 21, 22, 23, 24, 25, 26)
        if missing and set(missing).issubset(expected):
            result = joint_read_error(role, device, missing, expected)
            return _remember(exc, result)
    match = re.fullmatch(r"SyncRead returned servo error id=(\d+) err=(\d+)", str(exc)) if role == "head_arm" else None
    if match and int(match.group(2)):
        sid, status = map(int, match.groups())
        zh, en = joint_label(role, [sid])
        result = failure("H04", "{}返回故障码 0x{:02x}。串口：{}。".format(zh, status, device),
                         "{} reported fault 0x{:02x} on {}.".format((en[:1].upper() + en[1:]), status, device),
                         "请停止运行，记录关节编号和故障码，按设备故障说明处理。",
                         "Stop operation, record the joint number and code, and follow the device fault instructions.",
                         role=role, device=device, joint_id=sid, fault=status, raw=str(exc))
        return _remember(exc, result)
    match = re.fullmatch(r"Failed to enable torque for servo ids: (\[[\d, ]*\])", str(exc)) if role == "head_arm" else None
    if match:
        ids = ast.literal_eval(match.group(1))
        if ids:
            zh, en = joint_label(role, ids)
            result = failure("H06", "{}使能失败。串口：{}。".format(zh, device),
                             "Could not enable {} on {}.".format(en, device),
                             "请检查这些关节的供电和接头。", "Check power and connectors for the listed joints.",
                             role=role, device=device, joint_ids=ids, raw=str(exc))
            return _remember(exc, result)
    match = re.match(r"(?:read1 failed|write1 failed|ReadPosSpeed failed|SyncRead returned servo error) id=(\d+)", str(exc)) if role == "head_arm" else None
    if match:
        sid = int(match.group(1))
        zh, en = joint_label(role, [sid])
        result = failure("H06", "{}通信失败。串口：{}。".format(zh, device),
                         "Communication with {} failed on {}.".format(en, device),
                         "请断电检查该关节的接头和上一段连接线。", "Power off and check its connectors and upstream cable.",
                         role=role, device=device, joint_id=sid, raw=str(exc))
        return _remember(exc, result)
    zh, en = LABELS.get(role, ("当前操作", "operation"))
    location = "（{}）".format(device) if device and device != "auto" else ""
    at = " on {}".format(device) if device and device != "auto" else ""
    reason = short_reason(exc)
    context = dict(role=role, device=device, stage=stage, raw=str(exc), stderr=getattr(exc, "stderr", None))
    errnum = getattr(exc, "errno", None)
    failed_path = getattr(exc, "filename", None)
    cause = exc.__cause__
    # SDK wrappers may turn an OS error into RuntimeError. Use only an actual
    # errno from the preserved cause, never words such as 'permission' alone.
    visited = set()
    while cause is not None and id(cause) not in visited:
        visited.add(id(cause))
        if errnum is None:
            errnum = getattr(cause, "errno", None)
        if failed_path is None:
            failed_path = getattr(cause, "filename", None)
        cause = cause.__cause__
    if isinstance(exc, ModuleNotFoundError):
        module = exc.name or reason
        if module == "beingbeyond_d1_edu_sdk":
            result = failure("P02", "当前 Python 环境没有安装 D1 SDK。", "The D1 SDK is not installed in this Python environment.",
                             "请按本仓库 README 安装 SDK，再运行示例。", "Install the SDK using this repository's README, then retry.", **context)
        else:
            result = failure("P01", "当前 Python 环境缺少 {}。".format(module), "Module {} is missing from this Python environment.".format(module),
                             "请在运行本示例的环境中，按 README 安装对应依赖。", "Install the required dependency in this environment using README.", **context)
    elif isinstance(exc, ImportError):
        result = failure("P03", "无法加载模块：{}。".format(reason), "Could not load the module: {}.".format(reason),
                         "请核对当前 Python 环境和模块安装版本。", "Check the Python environment and installed module version.", **context)
    elif isinstance(exc, FileNotFoundError) and stage == "status check" and role in ("hand CAN", "can_iface"):
        result = failure("P10", "缺少系统命令 ip，无法检查 CAN 接口。", "Command ip is missing; the CAN interface could not be checked.",
                         "请安装 iproute2 后重试。", "Install iproute2 and retry.", **context)
    elif isinstance(exc, PermissionError) or errnum in (errno.EACCES, errno.EPERM):
        if failed_path is not None and failed_path != device:
            result = failure("P06", "无法访问文件或命令：{}。".format(failed_path), "Access to file or command {} was denied.".format(failed_path),
                             "请检查该路径的访问权限。", "Check access permissions for that path.", **context)
            return _remember(exc, result)
        if role in ("camera", "stereo camera", "head_arm", "exo", "glove"):
            group = "video" if role in ("camera", "stereo camera") else "dialout"
            action_zh = "请按 README 配置 {} 组权限，然后注销并重新登录。".format(group)
            action_en = "Configure {} group access using README, then log out and back in.".format(group)
        else:
            action_zh, action_en = "请核对当前用户执行这项操作的权限。", "Check the current user's permissions for this operation."
        result = failure("V04" if role in ("camera", "stereo camera") else "S06", "当前用户没有{}{}的访问权限。".format(zh, location), "Access to the {}{} was denied.".format(en, at),
                         action_zh, action_en, **context)
    elif (isinstance(exc, FileNotFoundError) or errnum == errno.ENOENT) and device and device != "auto" and (
        failed_path == device or
        failed_path is None and role in ("head_arm", "exo", "glove") and stage in ("open", "serial verification", "explicit serial verification")
    ):
        result = failure("S05", "{}设备不存在：{}。".format(zh, device), "The configured {} device does not exist: {}.".format(en, device),
                         "请重新连接该设备的 USB 线，再核对配置路径。", "Reconnect its USB cable and check the configured path.", **context)
    elif isinstance(exc, FileNotFoundError) or errnum == errno.ENOENT:
        result = failure("P06", "找不到文件或命令：{}。".format(failed_path or reason), "File or command not found: {}.".format(failed_path or reason),
                         "请核对报错中的路径或命令。", "Check the reported path or command.", **context)
    elif isinstance(exc, (ValueError, TypeError)):
        result = failure("P08", "参数或数据格式错误：{}。".format(reason), "Invalid parameter or data format: {}.".format(reason),
                         "请核对报错中的参数、数值数量和取值范围。", "Check the named parameter, value count and range.", **context)
    elif role == "display":
        result = failure("V11", "图形窗口操作失败：{}。".format(reason), "Window operation failed: {}.".format(reason),
                         "请先在本机桌面终端重试。", "Try again from the local desktop terminal.", **context)
    elif role in ("camera", "stereo camera") and stage == "configure capture":
        result = failure("P08", "相机{}采集设置失败：{}。".format(location, reason), "Could not configure camera capture{}: {}.".format(at, reason),
                         "请核对相机格式和采集分辨率。", "Check the camera format and capture resolution.", **context)
    elif role in ("camera", "stereo camera") and stage in ("read frame", "capture check"):
        result = failure("V07", "相机{}未返回图像。".format(location), "Camera{} did not return an image.".format(at),
                         "请重新插好相机 USB 线，再试一次。", "Reconnect the camera's USB cable and retry.", **context)
    elif stage in ("close", "close_can"):
        result = failure("X05", "关闭{}{}失败：{}。".format(zh, location, reason), "Could not close the {}{}: {}.".format(en, at, reason), "", "", **context)
    elif stage in ("home_and_close", "open_hand"):
        result = failure("X04", "{}{}{}失败。".format(zh, location, "回零" if stage == "home_and_close" else "张手"),
                         "Could not {} the {}{}.".format("home" if stage == "home_and_close" else "open", en, at),
                         "请确认当前姿态，并检查该设备的连接。", "Check its current pose and connection.", **context)
    else:
        operations = {
            "set positions": ("发送关节位置", "sending joint positions"),
            "set_positions": ("发送关节位置", "sending joint positions"),
            "set_joint_pos": ("发送关节位置", "sending joint positions"),
            "read positions": ("读取关节位置", "reading joint positions"),
            "read_joint_pos": ("读取关节位置", "reading joint positions"),
            "get_positions": ("读取关节位置", "reading joint positions"),
            "get_positions_and_velocities": ("读取位置和速度", "reading positions and velocities"),
            "read_frame": ("读取关节角度", "reading joint angles"),
            "get_norm_values": ("读取数据", "reading data"),
            "tactile read": ("读取触觉", "reading tactile data"),
            "set_speed": ("设置速度", "setting speed"),
            "set_torque": ("设置扭矩", "setting torque"),
            "controller initialization": ("初始化", "initializing the controller"),
        }
        verb_zh, verb_en = operations.get(stage, ("", ""))
        actions = {
            "head_arm": ("请检查头臂控制器的 USB 线和机器人供电。", "Check the controller's USB cable and robot power."),
            "exo": ("请检查遥操臂的 USB 线和供电。", "Check the exoskeleton's USB cable and power."),
            "glove": ("请检查手套 USB 线是否松动。", "Check the glove USB connection."),
            "hand CAN": ("请检查手部供电、PCAN USB 线和手部 CAN 线。", "Check hand power, PCAN USB and the hand's CAN cable."),
            "can_iface": ("请检查 PCAN 连接和接口配置。", "Check the PCAN connection and interface configuration."),
        }
        action = actions.get(role, ("请保留完整错误信息并联系支持。", "Save the full error details and contact support."))
        code = {"head_arm": "H06", "glove": "G02", "hand CAN": "C08"}.get(role, "X06") if verb_zh else "X06"
        result = failure(code, "{}{}{}失败：{}。".format(zh, location, verb_zh, reason),
                         "The {}{} failed{}: {}.".format(en, at, " while " + verb_en if verb_en else "", reason), *action, **context)
    return _remember(exc, result)


def report_error(exc, *, role=None, device=None, stage="runtime", file=None, severity="error"):
    file = sys.stderr if file is None else file
    with _REPORT_LOCK:
        diagnostic = format_error(role, device, stage, exc)
        leaves = diagnostic.errors if isinstance(diagnostic, DiagnosticGroup) else [diagnostic]
        printed = [error for error in leaves if not error.reported]
        for error in printed:
            error.reported = True
    for error in printed:
        print(error.render(severity), file=file)
    if printed and isinstance(diagnostic, DiagnosticGroup):
        for zh, en in dict.fromkeys(diagnostic.notes):
            print(zh + "\n" + en, file=file)
    if printed and os.environ.get("D1_DEBUG") == "1":
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=file)
        shown = set()
        current = exc
        while current is not None and id(current) not in shown:
            shown.add(id(current))
            current = current.__cause__ or current.__context__
        records = [diagnostic] + printed if isinstance(diagnostic, DiagnosticGroup) else printed
        for error in records:
            if error.context:
                print("[debug] {} {}".format(error.code, error.context), file=file)
            original = error.original
            if original is not None and id(original) not in shown:
                traceback.print_exception(type(original), original, original.__traceback__, file=file)
                shown.add(id(original))


class BilingualArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        error = failure("P09", "命令行参数错误：{}。".format(short_reason(message)), "Invalid command-line arguments: {}.".format(message),
                        "请运行 {} --help 查看用法。".format(self.prog), "Run {} --help for usage.".format(self.prog))
        self.exit(2, str(error) + "\n")


def install_error_hook():
    def explain(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            return
        report_error(exc)
    sys.excepthook = explain


def require_display():
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise failure("V10", "当前终端没有图形显示会话，无法打开窗口。", "No graphical display session is available for this terminal.",
                      "请在本机桌面终端运行。", "Run from the local desktop terminal.")


def joint_label(role, ids, expected=None):
    """SDK mapping gives physical head/arm numbers; exo IDs are configurable."""
    ids = tuple(ids)
    if role == "head_arm" and all(sid in (11, 12, 21, 22, 23, 24, 25, 26) for sid in ids):
        labels = []
        for chain, base, zh, en in (((11, 12), 10, "头部", "head"), ((21, 22, 23, 24, 25, 26), 20, "臂部", "arm")):
            numbers = [str(sid - base) for sid in chain if sid in ids]
            if numbers:
                labels.append(("{} {} 号关节".format(zh, "、".join(numbers)),
                               "{} joint{} {}".format(en, "s" if len(numbers) > 1 else "", ", ".join(numbers))))
        return "、".join(zh for zh, en in labels), ", ".join(en for zh, en in labels)
    if role == "exo":
        return ("遥操臂关节（通信 ID {}）".format("、".join(map(str, ids))),
                "exoskeleton joint{} (ID{} {})".format("s" if len(ids) > 1 else "", "s" if len(ids) > 1 else "", ", ".join(map(str, ids))))
    names_zh, names_en = [], []
    for sid in ids:
        if role == "head_arm" and sid in (11, 12):
            zh, en = "头部 {} 号关节".format(sid - 10), "head joint {}".format(sid - 10)
        elif role == "head_arm" and 21 <= sid <= 26:
            zh, en = "臂部 {} 号关节".format(sid - 20), "arm joint {}".format(sid - 20)
        else:
            zh, en = "关节（通信 ID {}）".format(sid), "joint (ID {})".format(sid)
        names_zh.append(zh)
        names_en.append(en)
    return "、".join(names_zh), ", ".join(names_en)


def joint_read_error(role, device, missing, expected, baudrate=None, *, invalid=False):
    missing = tuple(sid for sid in expected if sid in set(missing))
    zh, en = joint_label(role, missing, expected)
    all_missing = set(missing) == set(expected)
    if all_missing:
        name_zh, name_en = LABELS[role]
        return failure("H02" if role == "head_arm" else "E01", "串口 {} 已打开，但{}关节均无有效应答。".format(device, "头臂" if role == "head_arm" else "遥操臂"),
                       "Port {} opened, but none of the {} joints returned valid replies.".format(device, "head/arm" if role == "head_arm" else "exoskeleton"),
                       "请检查{}供电，再核对端口和波特率 {}。".format(name_zh, baudrate or "（见配置）"),
                       "Check {} power, then verify the port and baudrate {}.".format(name_en, baudrate or "in the configuration"),
                       role=role, device=device, missing=missing)
    if role == "head_arm":
        head_missing = [sid for sid in missing if sid in (11, 12)]
        arm_missing = [sid for sid in missing if 21 <= sid <= 26]
        if head_missing and arm_missing:
            return DiagnosticGroup([joint_read_error(role, device, part, expected, baudrate, invalid=invalid)
                                    for part in (head_missing, arm_missing)])
    # A link hint is justified only by a good prefix followed by an entirely bad suffix.
    chains = [(11, 12), (21, 22, 23, 24, 25, 26)] if role == "head_arm" else []
    boundaries = []
    for chain in chains:
        if not set(chain).issubset(expected):
            continue
        flags = [sid not in missing for sid in chain]
        first_bad = next((i for i, good in enumerate(flags) if not good), len(chain))
        if 0 < first_bad < len(chain) and not any(flags[first_bad:]):
            left_zh, left_en = joint_label(role, [chain[first_bad - 1]], expected)
            right_zh, right_en = joint_label(role, [chain[first_bad]], expected)
            boundaries.append((left_zh, left_en, right_zh, right_en))
    if boundaries:
        azh = "请断电，优先检查" + "；".join("{}与{}之间的连线".format(a, c) for a, b, c, d in boundaries) + "。"
        aen = "Power off and first check " + "; ".join("the cable between {} and {}".format(b, d) for a, b, c, d in boundaries) + "."
    else:
        azh, aen = ("请断电检查该关节的接头和上一段连接线。", "Power off and check its connectors and upstream cable.") if len(missing) == 1 else (
            "请断电检查这些关节的接头和上一段连接线。", "Power off and check their connectors and upstream cable.")
    return failure("H03" if role == "head_arm" else "E02", "{}{}。串口：{}。".format(zh, "读数无效" if invalid else "无应答", device),
                   "{} {} on {}.".format((en[:1].upper() + en[1:]), "returned invalid readings" if invalid else "did not respond", device), azh, aen,
                   role=role, device=device, missing=missing)


_EXO_WARNINGS = {}


def warn_exo_readings(valid, servo_ids, device, min_valid=None):
    missing = tuple(sid for i, sid in enumerate(servo_ids) if i >= len(valid) or not valid[i])
    previous = _EXO_WARNINGS.get(device)
    if not missing:
        if previous is not None:
            print("[提示] 遥操臂读数已恢复。\n[Info] Exoskeleton readings have recovered.", file=sys.stderr)
            _EXO_WARNINGS.pop(device, None)
        return
    if previous == missing:
        return
    _EXO_WARNINGS[device] = missing
    error = joint_read_error("exo", device, missing, servo_ids, invalid=True)
    if min_valid is not None:
        if sum(valid) < min_valid:
            error.chinese += " 本次未更新机械臂目标。"
            error.english += " The arm target was not updated this cycle."
        else:
            error.chinese += " 对应关节本次保持原目标。"
            error.english += " Those joints kept their previous targets this cycle."
    report_error(error, severity="warning")
