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
    - The CAN interface is configured (e.g. 'can0' at 1 Mbps)
    - The hand responds correctly to high-level normalized joint commands.
    - The hand motion behavior under different speed and torque settings.
    - The tactile matrix frames are received and decoded.
"""
import time

import cv2
import numpy as np

from beingbeyond_d1_edu_sdk.hand import DexHand


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


def build_tactile_panel(hand, wait_complete=True, timeout=0.08):
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


def show_tactile_preview(hand, preview_scale=0.7):
    if not hasattr(hand, "read_matrix_touch_4x10"):
        raise RuntimeError(
            "The installed SDK does not expose DexHand.read_matrix_touch_4x10. "
            "Rebuild/reinstall the SDK from the updated package source."
        )

    print("Starting tactile preview. Press q in the window to quit.")
    while True:
        panel = build_tactile_panel(hand, wait_complete=True, timeout=0.08)
        if preview_scale != 1.0:
            panel = cv2.resize(
                panel,
                (int(panel.shape[1] * preview_scale), int(panel.shape[0] * preview_scale)),
                interpolation=cv2.INTER_AREA,
            )
        cv2.imshow("DexHand tactile", panel)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break


def main():
    print("\033[91mWARNING: Always keep the physical emergency stop button within reach,\033[0m")
    print("\033[91m         and press it immediately if the robot motion looks unsafe.\033[0m\n")

    right_hand = DexHand(hand_type="right", can_iface="can0")

    try:
        right_hand.gesture_dance(duration_s=3, m=0.1, interval_s=0.1)

        right_hand.open_hand()

        right_hand.set_speed(speed=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        right_hand.set_torque(torque=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        right_hand.set_joint_pos([0.55, 0.8, 0.42, 0.45, 0, 0])
        time.sleep(1)

        right_hand.set_speed(speed=[1, 1, 1, 1, 1, 1])
        right_hand.set_torque(torque=[1, 1, 1, 1, 1, 1])
        right_hand.open_hand()
        time.sleep(1)

        print(right_hand.read_joint_pos())
        time.sleep(1)
        print(right_hand.read_joint_pos())

        show_tactile_preview(right_hand)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")

    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            safe_open = [0.0] * 6
            right_hand.set_joint_pos(safe_open)
            time.sleep(0.1)
        except Exception as e:
            print(f"Failed to send safe-open: {e}")
        try:
            right_hand.close_can()
        except Exception as e:
            print(f"Failed to close: {e}")

if __name__ == '__main__':
    main()
