# 报错与自检 / Errors and checks

终端默认只显示当前问题和下一步检查动作。中文、英文分行；同一个错误不重复显示。无关的 `/dev/ttyS*` 不参与 USB 设备报错。

The terminal shows the problem and the next check, with Chinese and English on separate lines. The same error is shown once. Unrelated `/dev/ttyS*` ports are omitted from USB discovery errors.

## 关节与连线 / Joints and cables

机器人头部与臂部使用各自的关节编号。通信 ID 与关节号的对应关系如下；提示按关节号显示。

Head and arm joints are numbered separately. Messages use joint numbers, mapped from communication IDs as follows.

| 位置 / Location | 关节号 / Joint number | 通信 ID / ID |
| --- | --- | --- |
| 头部 / Head | 1、2 | 11、12 |
| 臂部 / Arm | 1～6 / 1–6 | 21～26 / 21–26 |

例如，臂部 1～3 号有应答，4～6 号均无应答时：

For example, if arm joints 1–3 respond and joints 4–6 do not:

```text
[错误] 臂部 4、5、6 号关节无应答。串口：/dev/ttyACM0。
请断电，优先检查臂部 3 号关节与臂部 4 号关节之间的连线。
[Error] Arm joints 4, 5, 6 did not respond on /dev/ttyACM0.
Power off and first check the cable between arm joint 3 and arm joint 4.
```

这表示应优先检查的位置，不代表程序已经确认那段线断了。若只有 4 号无应答、5 和 6 号仍正常，则提示检查 4 号关节的接头。头部与臂部的异常分别显示，不推断头部与臂部之间存在串联线。

This identifies where to check first, without claiming the cable is broken. If only joint 4 fails while joints 5 and 6 respond, check joint 4's connectors. Head and arm failures are reported separately; no shared cable between these chains is inferred.

遥操臂按 `arm_exo_servo_ids` 中的通信 ID 报告异常。配置列表的顺序不一定是实际接线顺序，因此不据此推断哪两个关节之间断了线。

Exoskeleton failures list the communication IDs from `arm_exo_servo_ids`. The list order may differ from the physical wiring, so it is not used to infer a broken cable between adjacent joints.

检查关节接头前先停止程序并断电。机器人运动异常时使用实体急停。

Stop the program and power off before inspecting joint connectors. Use the physical emergency stop if robot motion is unsafe.

## 按报错检查 / Check the reported problem

| 提示 / Message | 下一步 / Next check |
| --- | --- |
| 未识别到头臂控制器、遥操臂或手套 USB 串口。 / No matching USB serial port was found. | 重新插好该设备的 USB 线。若仍没有，用 `lsusb` 对照拔插前后；没有新增设备时，换一根确认能传数据的 USB 线重试。 / Reconnect that device's USB cable. If still absent, compare `lsusb` before and after reconnecting; try a known working USB data cable if no device appears. |
| 已有 USB 串口，但无法自动确认设备。 / USB serial ports exist, but the device cannot be selected automatically. | 对照拔插确认端口，再修改对应配置。只查 `/dev/ttyUSB*` 可能漏掉 `/dev/ttyACM*`。 / Identify the port by reconnecting, then update its setting. Check both `/dev/ttyUSB*` and `/dev/ttyACM*`. |
| 端口不存在。 / The configured port does not exist. | 重插后核对设备路径，端口名称可能改变。 / Verify the device path after reconnecting; the port name may have changed. |
| 没有访问权限。 / Access was denied. | 串口按 README 配置 `dialout`，相机配置 `video`，然后注销并重新登录。 / Follow README group setup: `dialout` for serial, `video` for cameras; log out and back in. |
| 端口正被进程占用。 / The port is in use by a process. | 按提示的 PID 确认程序并正常退出。 / Identify the reported PID and close that application normally. |
| 找到多个设备，无法自动选择。 / Multiple matching devices were found. | 拔下多余设备，或显式填写目标端口。 / Disconnect extra devices or select the intended port explicitly. |
| 头臂串口实际连接手套。 / The head/arm port is connected to the glove. | 将头臂配置改为机器人控制器的端口。 / Set the head/arm port to the robot controller's port. |
| 串口已打开，关节均无有效应答。 / The port opened, but no joints returned valid replies. | 先查该设备供电，再核对端口与提示的波特率。 / Check device power first, then its port and reported baudrate. |
| 部分关节无应答或无有效读数。 / Some joints did not respond or returned invalid readings. | 按提示的关节编号或相邻关节连线位置检查。 / Check the reported joint connectors or cable between adjacent joints. |
| 关节返回故障码。 / A joint reported a fault code. | 记录关节号和故障码，按设备故障说明处理。不能把它当成关节缺失。 / Record the joint number and code and follow the device fault instructions; the joint did respond. |
| 遥操臂标定失败，关节没有有效读数。 / Exoskeleton calibration failed because readings are missing. | 先检查提示的关节连接，再重新标定。 / Check the reported connections before calibrating again. |
| 遥操臂标定超时，未达到静止时长。 / Exoskeleton calibration timed out without the required stable period. | 放到零位并保持静止后重试。 / Place the exoskeleton at zero and hold still before retrying. |
| PCAN 未找到或配置的 CAN 接口不存在。 / PCAN or the configured CAN interface was not found. | 检查 PCAN USB；用 `ip link show` 核对接口名称，再查驱动。NX 驱动说明见 `lib/pcan/README.md`。 / Check PCAN USB, verify the name with `ip link show`, then check the driver. NX driver instructions are in `lib/pcan/README.md`. |
| CAN 未启动、波特率不符或 BUS-OFF。 / CAN is down, has the wrong bitrate, or is BUS-OFF. | 按本仓库 README 处理对应状态。BUS-OFF 时先查手部供电和 CAN 线，再恢复接口。 / Follow this repository's README for that state. For BUS-OFF, check hand power and CAN wiring before recovering the interface. |
| 左/右手未返回状态。 / The selected hand did not reply. | 检查这只手的供电、CAN 线和左右手配置。 / Check that hand's power, CAN cable and selected side. |
| 未识别到相机。 / No matching camera was found. | 检查相机自己的 USB 线；头臂 USB 与相机 USB 是两条独立连接。 / Check the camera's own USB cable; it is separate from the head/arm connection. |
| 无法打开相机。 / The camera could not be opened. | 先退出其他相机程序，再重新插好相机 USB。若已明确查到权限或占用问题，优先按该提示处理。 / Close other camera applications, then reconnect the camera USB cable. Follow a specific permission or occupancy message first if one is available. |
| 相机没有图像或取帧中断。 / No image was returned, or capture was interrupted. | 检查相机 USB 线和接头。仍失败时用 `v4l2-ctl --list-devices` 核对设备。 / Check the USB cable and connectors. If it still fails, verify the camera with `v4l2-ctl --list-devices`. |
| 图像尺寸不符。 / The frame size does not match. | 比较提示中的实际尺寸和要求尺寸，核对相机与分辨率。 / Compare the actual and required sizes and check the selected camera and resolution. |
| 没有显示会话或窗口操作失败。 / No display session is available, or a window operation failed. | 在本机桌面终端运行。NX 示例 4 可用 `--headless`；示例 8 可设 `enable_visualization=False`。 / Run from the local desktop. NX example 4 supports `--headless`; example 8 supports `enable_visualization=False`. |
| 遥操臂关节 ID 配置错误或重复。 / Exoskeleton joint IDs are invalid or duplicated. | 修改 `arm_exo_servo_ids`：每项为 0～999 的整数，不能重复。遥操作示例需要 6 项。 / Set `arm_exo_servo_ids` to distinct integers in the range 0..999. The teleoperation examples require six entries. |
| 参数设置错误、模型文件或关节缺失。 / A setting, model file or joint is invalid or missing. | 按报错中的参数名、实际值、要求值或路径修正。 / Correct the named setting, actual/expected value or file path. |
| 缺少模块或 SDK。 / A module or the SDK is missing. | 在运行示例的同一 Python 环境中按 README 安装。SDK 内部缺依赖时，提示应指向那个依赖。 / Install it in the same Python environment using README. A missing SDK dependency is reported by its module name. |

## 运行状态 / Runtime status

- “本次未更新机械臂目标”只说明这一轮遥操臂读数不足；不代表整机已经停止。
- “对应关节保持原目标”表示其余有效关节仍可能更新。
- “画面显示已关闭，遥操作仍在运行”表示示例 8 只关闭了可视化。
- “尚未检查”不等于检查通过；处理前面的故障后需要重新运行检查。

“Arm target was not updated this cycle” does not mean the robot stopped. “Those joints kept their previous targets” allows other valid joints to update. In example 8, disabling visualization does not stop teleoperation. “Not checked” means the check must be run again after fixing earlier errors.

## 查看完整异常 / Full error details

从仓库根目录运行设备检查，不发送运动命令：

Run a device check from the repository root without sending motion commands:

```bash
python examples/device_detection.py --list
python examples/device_detection.py --roles head_arm --check
python examples/device_detection.py --roles exo glove --check
python examples/device_detection.py --roles can camera --check
```

`--list` 不打开设备，列表为空时也会给出说明；`--check` 会打开串口、请求手部状态或读取图像。打开串口可能导致部分控制板复位。将 `python` 换成已安装 SDK 的解释器。

`--list` does not open devices and reports when no recognizable USB serial ports are found. `--check` opens serial ports, requests hand status or captures images. Opening a serial port may reset some controllers. Replace `python` with the interpreter where the SDK is installed.

需要完整调用栈时：

To show the full traceback:

```bash
python examples/device_detection.py --roles head_arm --check --debug
D1_DEBUG=1 python examples/3_show_vision.py
```

第二条会运行相机示例。`D1_DEBUG=1` 也可用于其他示例，但那些示例原有的硬件动作仍会执行。

The second command runs the camera example. `D1_DEBUG=1` also works with other examples; their normal hardware actions still execute.

反馈问题时提供运行命令、设备路径和完整异常。SDK 或原生驱动直接输出的原始日志可能仍是英文。原生 Qt 直接结束进程时，Python 无法补充说明。手套 SDK 没有提供可靠的收帧时间，程序不会仅凭手指角度不变就报断线。

Include the command, device paths and full error details in a support report. Logs printed directly by the SDK or native drivers may remain in English. Python cannot add a message if native Qt terminates the process. The glove SDK does not provide a reliable last-frame timestamp, so unchanged finger angles are not treated as a disconnection.
