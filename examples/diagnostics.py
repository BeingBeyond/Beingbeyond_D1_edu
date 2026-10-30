"""Bilingual example diagnostics. Standard library only; never touches hardware."""
import argparse
import os
import subprocess
import sys
import time


def bilingual(chinese, english):
    return "{} / {}".format(chinese, english)


_ROLES = {
    "head_arm": ("机器人头臂控制器", "检查机器人头臂控制器 USB 线和供电；停止其他串口程序，核对 dialout 权限、端口及波特率（默认 1000000）。若列出舵机 ID，断电后检查这些 ID 对应的舵机接头及上游串联线。",
                 "Check the robot head/arm USB cable and power, stop other serial programs, and verify dialout access, port and baudrate (default 1000000). For listed servo IDs, power off before checking their connectors and upstream daisy-chain wiring."),
    "exo": ("遥操臂（外骨骼）", "检查遥操臂 USB 线、控制板供电和下游舵机串联线；核对 arm_exo_port、arm_exo_baudrate（默认 115200）及 arm_exo_servo_ids。列出 ID 时，断电后检查对应舵机及上游接头。",
            "Check the exoskeleton USB cable, controller power and servo daisy-chain wiring; verify arm_exo_port, arm_exo_baudrate (default 115200) and arm_exo_servo_ids. Power off before inspecting the listed servo IDs and upstream connectors."),
    "glove": ("遥操手套", "检查手套 USB 线和供电、glove_port 和 glove_baudrate（默认 115200）；确认没有误接到头臂控制器，并退出占用串口的程序。手套应连续输出含 6 路有限数值的 ANGLES 数据。",
              "Check the glove USB cable, power, glove_port and glove_baudrate (default 115200); confirm this is not the head/arm controller and stop other serial users. Expect consecutive ANGLES frames with six finite values."),
    "hand CAN": ("灵巧手 CAN 总线", "检查 PCAN 的 USB 线、手部供电及 CAN_H/CAN_L 连线和终端电阻；用 ip -details -statistics link show <接口> 核对实际接口、UP、波特率（默认 1000000）及 BUS-OFF，核对左右手配置。确认接线后按 README 恢复总线。",
                 "Check the PCAN USB cable, hand power, CAN_H/CAN_L wiring and termination. Use ip -details -statistics link show <interface> to verify the selected interface, UP state, bitrate (default 1000000) and BUS-OFF; verify the hand side. Follow README bus recovery instructions after checking wiring."),
    "stereo camera": ("双目 USB 相机", "检查相机自己的 USB 线、接头和供电（头臂 USB 线不负责相机）；用 lsusb、v4l2-ctl --list-devices 及 /dev/v4l/by-id/ 核对设备，检查 video 权限并退出占用相机的程序。取帧异常时核对 MJPG、1280×480 请求格式和 USB 带宽。",
                      "Check the camera's own USB cable, connectors and power (separate from the head/arm USB cable). Verify enumeration with lsusb, v4l2-ctl --list-devices and /dev/v4l/by-id/, video permissions and other camera users. For capture failures, check the requested MJPG 1280x480 format and USB bandwidth."),
    "display": ("图形显示", "在本地图形桌面终端运行并核对 DISPLAY/WAYLAND_DISPLAY；普通 SSH 不提供显示会话。支持无窗口模式的示例可关闭显示。检查 OpenCV/Qt 图形依赖；显示错误不代表相机线有问题。",
                "Run in a local desktop terminal and verify DISPLAY/WAYLAND_DISPLAY; plain SSH has no display session. Disable visualization where supported. Check OpenCV/Qt GUI dependencies; a display failure does not establish a camera wiring fault."),
    "serial roles": ("串口设备组", "按下方每个角色核对 USB 线、供电、端口和波特率；头臂、遥操臂和手套必须使用各自独立的串口，不能把同一端口分配给多个角色。",
                     "Check each listed role's USB cable, power, port and baudrate. Head/arm, exoskeleton and glove require separate serial ports; do not assign one port to multiple roles."),
}
_ROLES["can_iface"] = _ROLES["hand CAN"]
_ROLES["camera"] = _ROLES["stereo camera"]

_STAGES = {
    "controller initialization": "初始化控制器", "serial verification": "串口通信验证",
    "explicit serial verification": "验证指定串口", "discovery/verification": "识别并验证设备",
    "discovery": "识别设备", "communication check": "通信预检查", "status check": "手部状态预检查",
    "capture check": "相机取帧预检查", "open": "打开设备", "configure capture": "设置采集格式",
    "read frame": "读取图像", "set positions": "发送关节位置", "read positions": "读取关节位置",
    "motion/readback": "运动控制及状态回读", "read_frame": "读取遥操臂舵机角度",
    "get_norm_values": "读取手套数据", "release_torque": "释放遥操臂扭矩",
    "calibration": "零位标定", "close": "关闭设备", "home_and_close": "回零并关闭",
    "open_hand": "张开手部", "close_can": "关闭 CAN", "tactile read": "读取触觉数据",
    "visualization": "显示图形窗口", "runtime": "运行示例",
    "set_positions": "发送头臂关节位置", "set_joint_pos": "发送手部关节位置",
    "get_positions": "读取头臂位置", "get_positions_and_velocities": "读取头臂位置和速度",
    "read_joint_pos": "读取手部位置", "wait_until_reached": "等待头臂到位",
    "set_speed": "设置手部速度", "set_torque": "设置手部扭矩", "set_profile": "设置运动参数",
    "gesture_dance": "执行手部动作演示", "read_matrix_touch_4x10": "读取触觉矩阵",
}


def role_help(role):
    label, chinese, english = _ROLES.get(role, (role, "核对配置与完整异常信息。", "Check configuration and the complete exception."))
    return bilingual(label, role), bilingual(chinese, english)


def format_error(role, device, stage, exc):
    """Keep machine-searchable context and the original error, then add advice."""
    label, advice = role_help(role)
    if "[device] {} port={} stage={}:".format(role, device, stage) in str(exc):
        return str(exc)
    cause = "{}: {}".format(type(exc).__name__, exc)
    if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
        cause += ": " + str(exc.stderr).strip()
    if isinstance(exc, PermissionError):
        advice = bilingual("权限不足：核对当前登录会话的 dialout/video 组权限，重新登录使新增组生效。", "Permission denied: check this session's dialout/video groups; log in again after adding a group.") + "\n" + advice
    elif isinstance(exc, FileNotFoundError):
        advice = bilingual("设备节点、文件或命令不存在：核对报错中的具体路径/命令；重新插拔后端口号可能变化。", "Device, file or command is missing: check the exact path/command; port numbers may change after reconnecting.") + "\n" + advice
    elif isinstance(exc, (ValueError, TypeError)):
        advice = bilingual("参数或数据格式不正确：先核对异常中点名的配置项、数值数量、范围及 SDK/模型版本。", "Invalid parameter or data format: first check the named setting, value count, range and SDK/model version.")
    return ("[device] {} port={} stage={}: {}\n".format(role, device, stage, cause)
            + bilingual("设备：{}；阶段：{}。".format(label, _STAGES.get(stage, stage)), "Device: {}; stage: {}.".format(role, stage))
            + "\n" + bilingual("排查建议：", "Suggested checks:") + " " + advice)


def report_error(exc, *, role=None, device="未指定 / unspecified", stage="runtime", file=None):
    """For caught errors and entry points; do not discard the original message."""
    if file is None:
        file = sys.stderr
    if any(word in str(exc).lower() for word in ("could not connect to display", "cvshowimage", "qt.qpa", "function is not implemented", "no display name")):
        message = format_error("display", device, "visualization", exc)
    elif role is not None:
        message = format_error(role, device, stage, exc)
    elif "排查建议：" in str(exc):
        message = str(exc)
    elif isinstance(exc, ImportError):
        message = bilingual(
            "依赖或 SDK 导入失败。确认使用安装 SDK 的同一 Python 环境；按本仓库 README 安装对应 Python 版本和 CPU 架构的 wheel 及所需选配依赖。NX 的 ARM64 wheel 不能用于 x86；Isaac Gym 等选配组件需确认平台支持。",
            "Dependency or SDK import failed. Use the same Python environment where the SDK is installed. Follow this repository's README for the matching Python/CPU wheel and optional dependencies. NX ARM64 wheels cannot run on x86; check platform support for optional components such as Isaac Gym.") + "\n" + "{}: {}".format(type(exc).__name__, exc)
    else:
        message = bilingual("示例运行失败。按完整异常中的文件、行号、配置项和设备角色定位问题；设备错误请核对对应 USB/CAN 线、供电和端口。反馈时附运行命令及完整异常。详见 docs/TROUBLESHOOTING.md。", "Example failed. Locate the file, line, setting and device role in the complete traceback. For device errors check the corresponding USB/CAN cable, power and port. Include the command and full traceback when reporting; see docs/TROUBLESHOOTING.md.") + "\n" + "{}: {}".format(type(exc).__name__, exc)
    print("[错误 / ERROR] " + message, file=file)


class BilingualArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        super().error(bilingual("命令行参数错误，请按下方英文原因核对参数名称/取值，并用 --help 查看用法。", "Invalid command-line arguments; check the reason below and use --help for usage.") + "\n" + message)


def install_error_hook():
    """Install only for direct script execution, before third-party imports."""
    previous = sys.excepthook

    def explain(exc_type, exc, tb):
        previous(exc_type, exc, tb)
        if not issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
            report_error(exc)

    sys.excepthook = explain


def require_display():
    """Catch absent Linux desktop sessions before native Qt can abort Python."""
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        _, advice = role_help("display")
        raise RuntimeError(bilingual("没有可用的图形显示会话。", "No graphical display session is available.") + "\n" + advice)


_EXO_WARNINGS = {}


def warn_exo_readings(valid, servo_ids, device):
    """Surface invalid SDK readings without changing teleop fallback behavior."""
    missing = [sid for i, sid in enumerate(servo_ids) if i >= len(valid) or not valid[i]]
    if not missing:
        _EXO_WARNINGS.pop(device, None)
        return
    now = time.monotonic()
    if device not in _EXO_WARNINGS or now - _EXO_WARNINGS[device] >= 5.0:
        _EXO_WARNINGS[device] = now
        print("[警告 / WARNING] " + format_error("exo", device, "read_frame", RuntimeError(
            bilingual("遥操臂以下舵机 ID 读数无效：{}；无效通道沿用原有保持/等待策略。".format(missing),
                      "Invalid exoskeleton servo readings for IDs {}; existing hold/wait behavior is retained.".format(missing)))), file=sys.stderr)
