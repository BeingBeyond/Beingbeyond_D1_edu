"""
USB stereo RGB camera helper for D1 examples.

This module defines StereoRGBCamera, which:
  - Opens a side-by-side USB stereo camera
  - Splits each frame into left and right color images
  - Builds a single RGB display view from both eyes or either eye
"""
import cv2


DEFAULT_DEVICE = "/dev/v4l/by-id/usb-SunplusIT_Inc_SPCA2100_PC_Camera-video-index0"


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
            raise ValueError("view must be 'both', 'left', or 'right'")

        self.device = device
        self.single_width = single_width
        self.single_height = single_height
        self.fps = fps
        self.fourcc = fourcc
        self.swap_lr = swap_lr
        self.view = view
        self.cap = None

    def start(self):
        if self.cap is not None:
            return

        capture_width = self.single_width * 2
        capture_height = self.single_height

        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open camera: {self.device}. "
                "Check /dev/video* and /dev/v4l/by-id/, then set device "
                "in the example script or pass vision_device when creating "
                "D1Robot."
            )

        if self.fourcc.upper() != "NONE":
            if len(self.fourcc) != 4:
                raise ValueError("fourcc must have 4 characters, e.g. MJPG or YUYV.")
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap = cap

        print("============================================================")
        print("Camera opened")
        print("Device:", self.device)
        print("Requested resolution:", capture_width, "x", capture_height)
        print("Actual resolution:", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "x",
              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print("Requested FPS:", self.fps)
        print("Actual FPS:", cap.get(cv2.CAP_PROP_FPS))
        print("Requested FOURCC:", self.fourcc)
        print("Actual FOURCC:", self._actual_fourcc())
        print("View:", self.view)
        print("Controls: q to quit vision window")
        print("============================================================")

    def _actual_fourcc(self):
        value = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        return "".join(chr((value >> (8 * i)) & 0xFF) for i in range(4))

    def read_pair(self):
        if self.cap is None:
            self.start()

        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise RuntimeError("Failed to read frame.")

        frame_h, frame_w = frame.shape[:2]
        expected_w = 2 * self.single_width
        if frame_h < self.single_height or frame_w < expected_w:
            raise RuntimeError(
                f"Frame size is too small. Got {frame_w}x{frame_h}, "
                f"expected at least {expected_w}x{self.single_height}."
            )

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
