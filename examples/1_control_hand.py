"""
Example 1: Basic control of the DexHand.

This script:
    - Connects to the DexHand over CAN
    - Optionally runs a short "gesture dance" for visual inspection
    - Commands several hand poses using normalized joint values
    - Sets different joint speed and torque limits
    - Prints the current normalized joint positions read from the hardware
    - Shows a live 5-finger 4x10 tactile matrix preview

Use this example to verify that:
    - The hand is wired correctly
    - The confirmed PCAN interface is configured at 1 Mbps before startup
    - The hand responds correctly to high-level normalized joint commands.
    - The hand motion behavior under different speed and torque settings.
    - The tactile matrix frames are received and decoded.
"""
from diagnostics import install_error_hook, report_error, require_display

if __name__ == "__main__":
    install_error_hook()

import time

import cv2
import numpy as np

from beingbeyond_d1_edu_sdk.hand import DexHand
from device_detection import resolve_devices, device_operation


FINGER_NAMES = ["thumb", "index", "middle", "ring", "little"]


def overlay_text(img, lines, x=10, y=25, dy=25):
    out = img.copy()
    for i, line in enumerate(lines):
        yy = y + i * dy
        cv2.putText(
            out,
            line,
            (x, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            line,
            (x, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def tactile_mat_to_bgr(mat, label, cell_size=36, vmin=0, vmax=255):
    arr = np.asarray(mat, dtype=np.float32)
    valid = arr >= 0

    norm = np.zeros(arr.shape, dtype=np.uint8)
    denom = max(float(vmax - vmin), 1.0)
    if np.any(valid):
        clipped = np.clip(arr[valid], float(vmin), float(vmax))
        norm[valid] = np.round((clipped - float(vmin)) * 255.0 / denom).astype(np.uint8)

    color = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    color[~valid] = (32, 32, 32)
    color = cv2.resize(
        color,
        (arr.shape[1] * cell_size, arr.shape[0] * cell_size),
        interpolation=cv2.INTER_NEAREST,
    )
    color = cv2.copyMakeBorder(color, 40, 28, 16, 16, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return overlay_text(color, [label], x=8, y=24, dy=18)


def build_tactile_panel(hand, can_iface, wait_complete=True, timeout=0.08):
    with device_operation("hand CAN", can_iface, "tactile read"):
        thumb, index, middle, ring, little, ts = hand.read_matrix_touch_4x10(
            request=True,
            wait_complete=wait_complete,
            timeout=timeout,
            poll_interval=0.001,
        )
    mats = {
        "thumb": thumb,
        "index": index,
        "middle": middle,
        "ring": ring,
        "little": little,
    }
    body = np.hstack([tactile_mat_to_bgr(mats[name], name) for name in FINGER_NAMES])

    age_ms = -1.0 if ts <= 0 else (time.time() - float(ts)) * 1000.0
    header = np.zeros((58, body.shape[1], 3), dtype=np.uint8)
    header = overlay_text(
        header,
        [
            "RIGHT tactile 4x10",
            f"ts={ts:.3f} age={age_ms:.0f}ms | q=quit",
        ],
        x=8,
        y=20,
        dy=24,
    )
    return np.vstack([header, body])


def show_tactile_preview(hand, can_iface, preview_scale=0.7):
    if not hasattr(hand, "read_matrix_touch_4x10"):
        raise RuntimeError(
            "当前 SDK 缺少触觉读取接口，请按本仓库 README 安装匹配平台和 Python 的 SDK wheel / "
            "The installed SDK does not expose DexHand.read_matrix_touch_4x10. "
            "Install the matching SDK wheel as described in this repository README."
        )

    print("Starting tactile preview. Press q in the window to quit.")
    while True:
        panel = build_tactile_panel(hand, can_iface, wait_complete=True, timeout=0.08)
        if preview_scale != 1.0:
            panel = cv2.resize(
                panel,
                (int(panel.shape[1] * preview_scale), int(panel.shape[0] * preview_scale)),
                interpolation=cv2.INTER_AREA,
            )
        with device_operation("display", "OpenCV", "visualization"):
            cv2.imshow("DexHand tactile", panel)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break


def main():
    require_display()
    print("\033[91m警告：请始终保持实体急停按钮可触及 / WARNING: Always keep the physical emergency stop button within reach,\033[0m")
    print("\033[91m         运动异常时立即按下急停 / and press it immediately if the robot motion looks unsafe.\033[0m\n")

    can_iface = resolve_devices(can_iface="auto", check_ready=True)["can_iface"]
    with device_operation("hand CAN", can_iface, "controller initialization"):
        right_hand = DexHand(hand_type="right", can_iface=can_iface)

    try:
        with device_operation('hand CAN', can_iface, 'gesture_dance'):
            right_hand.gesture_dance(duration_s=3, m=0.1, interval_s=0.1)

        with device_operation('hand CAN', can_iface, 'open_hand'):
            right_hand.open_hand()

        with device_operation('hand CAN', can_iface, 'set_speed'):
            right_hand.set_speed(speed=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        with device_operation('hand CAN', can_iface, 'set_torque'):
            right_hand.set_torque(torque=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        with device_operation('hand CAN', can_iface, 'set_joint_pos'):
            right_hand.set_joint_pos([0.55, 0.8, 0.42, 0.45, 0, 0])
        time.sleep(1)

        with device_operation('hand CAN', can_iface, 'set_speed'):
            right_hand.set_speed(speed=[1, 1, 1, 1, 1, 1])
        with device_operation('hand CAN', can_iface, 'set_torque'):
            right_hand.set_torque(torque=[1, 1, 1, 1, 1, 1])
        with device_operation('hand CAN', can_iface, 'open_hand'):
            right_hand.open_hand()
        time.sleep(1)

        with device_operation('hand CAN', can_iface, 'read_joint_pos'):
            print(right_hand.read_joint_pos())
        time.sleep(1)
        with device_operation('hand CAN', can_iface, 'read_joint_pos'):
            print(right_hand.read_joint_pos())

        show_tactile_preview(right_hand, can_iface)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        report_error(e, role="hand CAN", device=can_iface)

    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            safe_open = [0.0] * 6
            with device_operation('hand CAN', can_iface, 'set_joint_pos'):
                right_hand.set_joint_pos(safe_open)
            time.sleep(0.1)
        except Exception as e:
            report_error(e, role="hand CAN", device=can_iface, stage="open_hand")
        try:
            with device_operation('hand CAN', can_iface, 'close_can'):
                right_hand.close_can()
        except Exception as e:
            report_error(e, role="hand CAN", device=can_iface, stage="close_can")

if __name__ == '__main__':
    main()
