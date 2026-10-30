# 报错与接线排查 / Errors and connection checks

适用于 `examples/0_*.py`～`10_*.py` 及设备诊断工具。项目维护的错误和异常警告提供中英文说明，SDK/操作系统的原始异常及 traceback 保留，方便技术支持继续定位。第三方库和二进制 SDK 自己输出的日志可能仍是英文。

Applies to examples 0–10 and the device diagnostic tool. Maintained errors and failure warnings include Chinese and English explanations. Original SDK/OS errors and tracebacks remain available for support. Messages printed directly by third-party libraries and binary SDKs may still be English.

## 如何读报错 / Reading an error

```text
[device] exo port=/dev/ttyUSB8 stage=read_frame: ...
设备：遥操臂（外骨骼） / exo；阶段：读取遥操臂舵机角度。
排查建议： / Suggested checks: ...
```

- `head_arm`：机器人上的头臂控制器；`exo`：操作者佩戴/操作的遥操臂（外骨骼）；`glove`：手套。三者需要各自独立的 USB 串口。
- `hand CAN` / `can_iface`：灵巧手及 PCAN 链路；`stereo camera` / `camera`：双目 USB 相机；`display`：图形桌面/窗口。
- `port=` 是本次实际选择/尝试的设备或接口，`stage=` 是失败阶段。`auto` 表示识别尚未完成；不能把这个值当作实际端口。
- `ID` 是设备协议中的舵机编号，不是列表下标。按设备接线标识/对应配置定位，不能凭 ID 数字猜关节位置。

`head_arm` is the robot controller; `exo` is the operator's exoskeleton; `glove` is the glove. Each needs its own USB serial port. `hand CAN`/`can_iface` identifies the hand bus, `stereo camera`/`camera` the camera, and `display` the desktop/window. `port=` records the selected/attempted device; `stage=` records the failed operation. `auto` means discovery has not resolved a port. Servo IDs are protocol addresses, not list indices; use the wiring labels and configuration to locate them.

## 优先检查哪条线 / Which connection to check first

检查舵机接头或重新接线前，停止控制程序并断电；检查机器人运动异常时使用实体急停。

Stop the controller and power off before inspecting servo connectors or rewiring. Use the physical emergency stop if robot motion is unsafe.

| 报错角色或现象 / Role or symptom | 检查与处理 / Checks and action |
| --- | --- |
| 头臂串口打不开或断开 / Head/arm port cannot open or disconnects | 检查机器人头臂控制器 USB 线、控制板供电、实际串口、占用及 `dialout` 权限。默认波特率 1000000。 / Check the robot controller USB cable, power, actual port, other port users and `dialout` access. Default baudrate: 1000000. |
| 头臂 `no valid head/arm reply for IDs [...]` | 列出的 ID 没有有效应答。先核对头臂端口、供电和波特率，再断电检查这些 ID 的舵机接头及上游串联线；可能是共同上游断线，不能直接认定舵机坏了。 / Listed IDs did not return valid replies. Verify the controller port, power and baudrate, then power off and inspect their connectors and upstream daisy-chain wiring. A shared upstream disconnect is possible; this does not prove servo damage. |
| `error status 0x...` | 舵机已经应答，但报告故障。保留 ID 和十六进制故障码，核对供电、负载和接线，按设备故障说明或联系支持处理。 / The servo replied but reports a fault. Keep its ID and hexadecimal status; check power, load and wiring and consult device fault documentation or support. |
| 遥操臂 `exo IDs missing/invalid` 或无效读数警告 / Exoskeleton invalid IDs/readings | 检查遥操臂自己的 USB 线、供电、`arm_exo_port`、`arm_exo_baudrate`（默认 115200）和 `arm_exo_servo_ids`（默认 1～6）。断电检查报错 ID 对应舵机线及上游接头。运行时约每 5 秒提示一次，保留原有通道保持/等待策略。 / Check the exoskeleton USB cable, power, port, baudrate (default 115200) and IDs (default 1–6). Power off to inspect the listed servo IDs and upstream wiring. Runtime warnings repeat at most about once per five seconds per port, retaining existing hold/wait behavior. |
| 遥操臂零位标定超时 / Exoskeleton zero-calibration timeout | 若有无效 ID，先处理数据/接线问题；若读数有效，将遥操臂放在零位并保持静止，核对稳定时长和总超时时长。 / Resolve invalid readings/IDs first. With valid data, hold still at zero and check the stability and overall timeout settings. |
| 手套无连续 `ANGLES` / No consecutive glove ANGLES frames | 检查手套 USB 线、供电、端口及默认 115200 波特率；确认该串口实际连接手套，应连续输出 6 路有限角度数值。 / Check glove USB, power, port and default 115200 baudrate. Confirm the actual device is the glove and supplies consecutive six-channel finite angle frames. |
| `identified as glove` | 头臂候选实际连接手套。检查头臂 USB 线和串口分配，不要改舵机 ID 来解决设备选错。 / The head/arm candidate is a glove. Check the controller USB connection and port assignment; changing servo IDs does not correct a wrong device selection. |
| CAN 未 UP、波特率不符、BUS-OFF 或手部无状态 / CAN down, wrong bitrate, BUS-OFF or no hand status | 检查 PCAN USB、手部电源、CAN_H/CAN_L、终端电阻、实际 CAN 接口及左右手配置。按本仓库 README 配置已确认的接口（默认 1000000 bit/s）。接口 UP 不代表手部已应答；不要因名称为 can0 就选用原生 CAN。 / Check PCAN USB, hand power, CAN wiring/termination, actual interface and hand side. Follow this repository's README to configure the confirmed interface (default 1000000 bit/s). UP does not prove a hand response; can0 may be a built-in controller. |
| 相机打不开或运行中取帧失败 / Camera open/capture failure | 检查相机自己的 USB 线及接头。机器人头臂 USB 与相机 USB 是独立链路。核对枚举、`video` 权限、占用、MJPG/1280×480 格式及 USB 带宽。不要盲目改成 `/dev/video0`。 / Check the camera's own USB cable and connectors; it is separate from the head/arm link. Verify enumeration, `video` permissions, other users, MJPG/1280x480 format and USB bandwidth. Do not blindly choose `/dev/video0`. |
| 图像尺寸不足 / Frame too small | 查看实际宽高与期望宽高，核对相机型号、设备路径和采集格式。默认双目并排总宽 1280、高 480，每眼 640×480。 / Compare actual and expected dimensions; check the camera model, path and capture format. Default stereo dimensions are 1280x480 total, 640x480 per eye. |
| Qt、OpenCV、DISPLAY 或 viewer 错误 / Display/viewer error | 在本地图形桌面运行并检查显示变量、图形驱动和 OpenCV/Qt 依赖。普通 SSH 可能没有图形会话。示例 4（NX）支持 `--headless`；`D1Robot.start_vision_thread(display=False)` 仅采集。示例 8 可设置 `enable_visualization=False`。 / Use a local desktop and check display variables, graphics drivers and OpenCV/Qt dependencies. Plain SSH may lack a GUI session. NX example 4 supports `--headless`; `D1Robot.start_vision_thread(display=False)` captures without display; example 8 supports `enable_visualization=False`. |
| 找不到模块、SDK、URDF 或关节名 / Missing module, SDK, URDF or joint | 按本仓库 README 核对解释器、CPU 架构、wheel ABI、依赖与模型文件。NX 的 ARM64 wheel 不能用于 x86；仿真与重定向是额外依赖，不能推定所有 NX 版本支持。 / Check the interpreter, CPU architecture, wheel ABI, dependencies and model files against this repository README. ARM64 wheels do not work on x86; simulation/retargeting have additional dependencies and platform constraints. |

## 无运动诊断命令 / Diagnostics without motion commands

在已经安装本仓库 SDK 和依赖的 Python 环境下，从仓库根目录执行。`--list` 不打开设备；`--check` 读取串口/手部状态和相机图像，不使能运动，但打开串口仍可能导致部分控制板复位。

Run from the repository root in the environment where this repository's SDK and dependencies are installed. `--list` does not open devices. `--check` reads serial/hand status and camera frames without enabling motion; opening a serial port can still reset some controllers.

```bash
python examples/device_detection.py --list
python examples/device_detection.py --roles head_arm --check
python examples/device_detection.py --roles exo glove --check
python examples/device_detection.py --roles can camera --check
python examples/device_detection.py --roles can --hand-type left --check
lsusb
ls -l /dev/serial/by-id/ /dev/v4l/by-id/
v4l2-ctl --list-devices
id -nG
ip -details -statistics link show can1
```

`python` 应替换为当前 SDK 环境的解释器；`can1` 应替换为实际接口。设备缺失时某些命令提示路径不存在，这也是定位线缆/枚举问题的信息。若已安装 `fuser`，可执行 `fuser -v /dev/ttyUSB8` 查看占用，再确认并正常退出对应程序，不要盲目杀进程。

Replace `python` with the SDK environment's interpreter and `can1` with the actual interface. Missing-device errors help identify enumeration/connection issues. If `fuser` is installed, `fuser -v /dev/ttyUSB8` can show port owners; identify and exit the relevant program normally.

权限不足时，参考 README 配置 `dialout`/`video` 组后重新登录；不需要 `chmod 777` 或使用 sudo 启动图形示例。多个匹配设备时显式指定端口，自动识别不会盲选第一个。CAN/相机预检查失败后，提示为 skipped 的串口角色尚未检查。

For permissions, follow README group setup for `dialout`/`video` and log in again; `chmod 777` or sudo GUI execution is unnecessary. Specify a port when several devices match; discovery does not pick the first blindly. Serial roles marked skipped after CAN/camera failures have not yet been checked.

## 反馈时提供 / Include in a support report

提供仓库名称/版本、Python 版本和平台、运行命令、修改过的端口配置、完整终端异常（包括最末的双语说明和之前的 traceback）、设备枚举结果及已检查的线缆/舵机 ID。启动检查通过不保证运行期间设备不会断开。

Include the repository/version, Python/platform, command, changed port settings, full terminal error including the bilingual explanation and preceding traceback, device enumeration, and cables/servo IDs already checked. Successful startup checks do not guarantee connections remain intact during runtime.

原生 Qt/驱动可能在 Python 捕获异常前终止进程，因此无法保证每条原生日志都被翻译。示例会提前检查缺失的显示会话；若 DISPLAY 已设置但无效，仍需按上表核对桌面会话。此次修改经无硬件回归验证，真机接线、热拔插、运动和 GUI 需要在对应平台复验。

Native Qt/drivers can terminate the process before Python catches an exception, so not every native log can be translated. Examples check for missing display sessions early; a set but invalid DISPLAY still needs desktop checks. This update is validated with hardware-free regressions; wiring, hot-unplugging, motion and GUI require validation on the target platform.
