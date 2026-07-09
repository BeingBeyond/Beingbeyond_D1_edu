"""
Example 3: USB stereo RGB viewer.

This script:
  - Opens a side-by-side USB stereo camera
  - Splits the frame into left and right RGB images
  - Displays RGB in one OpenCV window

Press 'q' in the OpenCV window to exit.
"""
import time

import cv2

from vision import StereoRGBCamera


def main():
    device = "/dev/v4l/by-id/usb-SunplusIT_Inc_SPCA2100_PC_Camera-video-index0"
    single_width = 640
    single_height = 480
    fps_request = 30
    fourcc = "MJPG"
    swap_lr = False
    view = "both"  # "both", "left", or "right"

    window_name = "D1 RGB"
    last_time = time.time()
    fps = 0.0

    with StereoRGBCamera(
        device=device,
        single_width=single_width,
        single_height=single_height,
        fps=fps_request,
        fourcc=fourcc,
        swap_lr=swap_lr,
        view=view,
    ) as camera:
        while True:
            now = time.time()
            dt = now - last_time
            last_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt

            try:
                rgb_view = camera.read_view()
            except RuntimeError as exc:
                print("[ERROR]", exc)
                break

            cv2.imshow(window_name, camera.draw_info(rgb_view, fps))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
