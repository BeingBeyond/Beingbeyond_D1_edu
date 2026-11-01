"""
USB stereo RGB camera helper for D1 examples.

This module defines StereoRGBCamera, which:
  - Opens a side-by-side USB stereo camera
  - Splits each frame into left and right color images
  - Builds a single RGB display view from both eyes or either eye
"""

from diagnostics import configuration_error, failure

import cv2
from diagnostics import format_error
from device_detection import resolve_camera_device, device_operation


DEFAULT_DEVICE = "auto"


class StereoRGBCamera:
    def __init__(
        self,
        device=DEFAULT_DEVICE,
        single_width=640,
        single_height=480,
        fps=30,
        fourcc="MJPG",
        swap_lr=False,
        view="both",
    ):
        if view not in ("both", "left", "right"):
            raise configuration_error("view", view, "both / left / right")

        self.device = resolve_camera_device(device)
        self.single_width = single_width
        self.single_height = single_height
        self.fps = fps
        self.fourcc = fourcc
        self.swap_lr = swap_lr
        self.view = view
        self.cap = None
        self._has_frame = False

    def start(self):
        if self.cap is not None:
            return

        capture_width = self.single_width * 2
        capture_height = self.single_height

        with device_operation("stereo camera", self.device, "open"):
            cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            raise failure("V06", "无法打开相机：{}。".format(self.device), "Could not open camera {}.".format(self.device),
                          "请先退出其他使用相机的程序，再重新插好相机 USB 线。", "Close other camera applications, then reconnect its USB cable.")

        try:
            with device_operation("stereo camera", self.device, "configure capture"):
                if self.fourcc.upper() != "NONE":
                    if len(self.fourcc) != 4:
                        raise configuration_error("fourcc", self.fourcc, "4 个字符", "4 characters")
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)
                cap.set(cv2.CAP_PROP_FPS, self.fps)
        except BaseException:
            cap.release()
            raise
        self.cap = cap
        self._has_frame = False

        print("相机已打开：{}。\nCamera opened: {}.".format(self.device, self.device))

    def _actual_fourcc(self):
        value = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        return "".join(chr((value >> (8 * i)) & 0xFF) for i in range(4))

    def read_pair(self):
        if self.cap is None:
            self.start()

        with device_operation("stereo camera", self.device, "read frame"):
            ret, frame = self.cap.read()
        if not ret or frame is None:
            if self._has_frame:
                raise failure("V09", "相机 {} 取帧中断。".format(self.device), "Capture from camera {} was interrupted.".format(self.device),
                              "请检查相机 USB 线和接头是否松动。", "Check its USB cable and connectors.")
            raise format_error("stereo camera", self.device, "read frame", TimeoutError("No frame"))

        frame_h, frame_w = frame.shape[:2]
        expected_w = 2 * self.single_width
        if frame_h < self.single_height or frame_w < expected_w:
            raise failure("V08", "相机图像尺寸不符：实际 {}×{}，需要至少 {}×{}。设备：{}。".format(frame_w, frame_h, expected_w, self.single_height, self.device),
                          "Camera frame size mismatch on {}: got {}x{}, need at least {}x{}.".format(self.device, frame_w, frame_h, expected_w, self.single_height),
                          "请核对相机设备和采集分辨率。", "Check the selected camera and capture resolution.")
        self._has_frame = True

        left_bgr = frame[0:self.single_height, 0:self.single_width]
        right_bgr = frame[
            0:self.single_height,
            self.single_width:2 * self.single_width,
        ]

        if self.swap_lr:
            left_bgr, right_bgr = right_bgr, left_bgr

        return left_bgr, right_bgr

    def read_view(self):
        left_bgr, right_bgr = self.read_pair()
        if self.view == "left":
            return left_bgr
        if self.view == "right":
            return right_bgr

        left_labeled = self.draw_label(left_bgr, "Left eye")
        right_labeled = self.draw_label(right_bgr, "Right eye")
        return cv2.hconcat([left_labeled, right_labeled])

    def draw_info(self, image, fps):
        out = image.copy()
        text = f"FPS: {fps:.1f} | {self.device} | {self.view}"
        cv2.putText(
            out,
            text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return out

    @staticmethod
    def draw_label(image, label):
        out = image.copy()
        cv2.putText(
            out,
            label,
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return out

    def stop(self):
        if self.cap is None:
            return
        try:
            self.cap.release()
        finally:
            self.cap = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
