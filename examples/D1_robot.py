import os
import sys
import threading
import time
import traceback
from diagnostics import format_error, report_error

from beingbeyond_d1_edu_sdk import HeadArmRobot
from beingbeyond_d1_edu_sdk.hand import DexHand
from device_detection import resolve_devices, device_operation


class D1Robot:
    def __init__(self, urdf_path: str, arm_dev: str, arm_baud: int,
                 hand_type: str, hand_can: str, hand_baud: int,
                 vision_device: str = "auto",
                 enable_vision: bool = True):
        self.head_arm = None
        self.hand = None
        self.vision = None
        self._vision_thread = None
        self._vision_stop_event = None
        self._vision_error = None
        self._closed = False

        devices = resolve_devices(head_arm=arm_dev, can_iface=hand_can,
                                  camera=vision_device if enable_vision else None,
                                  head_arm_baud=arm_baud, hand_type=hand_type,
                                  hand_baud=hand_baud, check_ready=True)
        self.devices = devices
        try:
            if enable_vision:
                from vision import StereoRGBCamera

                self.vision = StereoRGBCamera(device=devices["camera"])
            with device_operation("head_arm", devices["head_arm"], "controller initialization"):
                self.head_arm = HeadArmRobot(
                    urdf_path=urdf_path,
                    dev=devices["head_arm"],
                    baudrate=arm_baud,
                )
            with device_operation("hand CAN", devices["can_iface"], "controller initialization"):
                self.hand = DexHand(
                    hand_type=hand_type,
                    can_iface=devices["can_iface"],
                    baudrate=hand_baud,
                )
        except BaseException:
            self.close(home=False, open_hand=False)
            raise

    def start_vision_thread(
        self,
        window_name: str = "D1 RGB",
        display: bool = True,
        startup_timeout: float = 10.0,
    ):
        """
        Start RGB capture and wait for the first frame before returning.

        Pass display=False for SSH/headless capture without OpenCV windows.
        Press 'q' in the window to stop when display is enabled.
        """
        if self.vision is None:
            raise RuntimeError("当前 D1Robot 未启用视觉；需要相机时请设置 enable_vision=True / Vision is disabled for this D1Robot.")
        if startup_timeout <= 0:
            raise ValueError("startup_timeout 必须大于 0，请修改首帧等待超时配置 / startup_timeout must be positive")
        if self._vision_thread is not None and self._vision_thread.is_alive():
            return  # already running

        if display and sys.platform.startswith("linux") and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        ):
            print("[VisionThread] 未检测到图形桌面会话，仅采集图像而不显示窗口 / No display session detected; capturing without a window.")
            display = False

        self._vision_stop_event = threading.Event()
        self._vision_error = None
        ready = threading.Event()
        stop_event = self._vision_stop_event

        def _vision_loop():
            last_time = time.monotonic()
            fps = 0.0
            window_created = False

            try:
                if display:
                    import cv2

                with self.vision as camera:
                    while not stop_event.is_set():
                        vis = camera.read_view()
                        now = time.monotonic()
                        dt = now - last_time
                        last_time = now
                        if dt > 0:
                            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt

                        if display:
                            with device_operation("display", window_name, "visualization"):
                                cv2.imshow(window_name, camera.draw_info(vis, fps))
                            window_created = True
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                stop_event.set()
                        ready.set()

            except Exception as e:
                self._vision_error = e
                report_error(e)
                traceback.print_exc()
            finally:
                stop_event.set()
                ready.set()
                if display and window_created:
                    try:
                        cv2.destroyWindow(window_name)
                    except Exception:
                        pass

        self._vision_thread = threading.Thread(
            target=_vision_loop,
            daemon=True,
        )
        self._vision_thread.start()
        if not ready.wait(startup_timeout):
            self.stop_vision_thread()
            raise RuntimeError(format_error("stereo camera", self.vision.device, "read frame",
                                            TimeoutError("等待相机首帧超时 / Timed out waiting for the first camera frame.")))
        self._raise_vision_error()

    def _raise_vision_error(self):
        if self._vision_error is not None:
            if "stage=visualization" in str(self._vision_error):
                raise RuntimeError(str(self._vision_error)) from self._vision_error
            raise RuntimeError(format_error("stereo camera", self.vision.device, "read frame",
                                            RuntimeError("D1 相机采集失败 / D1 camera capture failed: {}".format(self._vision_error)))) from self._vision_error

    def stop_vision_thread(self, join_timeout: float = 2.0):
        """
        Request the vision thread to stop and join it.
        """
        if self._vision_stop_event is not None:
            self._vision_stop_event.set()
        if self._vision_thread is not None and self._vision_thread.is_alive():
            self._vision_thread.join(join_timeout)

    def vision_stop_requested(self) -> bool:
        self._raise_vision_error()
        return (
            self._vision_stop_event is not None
            and self._vision_stop_event.is_set()
        )

    def set_q(self, q):
        with device_operation("head_arm", self.devices["head_arm"], "set positions"):
            self.head_arm.set_positions(q[:8])
        with device_operation("hand CAN", self.devices["can_iface"], "set positions"):
            self.hand.set_joint_pos(q[8:])

    def get_q(self):
        with device_operation("head_arm", self.devices["head_arm"], "read positions"):
            q_arm = self.head_arm.get_positions()
        with device_operation("hand CAN", self.devices["can_iface"], "read positions"):
            q_hand = self.hand.read_joint_pos()
        return q_arm + q_hand

    def close(self, *, home: bool = True, open_hand: bool = True):
        """Release resources; intentional normal exits keep the demo's homing behavior."""
        if self._closed:
            return
        self._closed = True
        # Stop vision thread first
        try:
            self.stop_vision_thread()
        except Exception as exc:
            report_error(exc, role="stereo camera", device=self.devices.get("camera"), stage="close")

        # Then release robot hardware
        if self.vision is not None and (
            self._vision_thread is None or not self._vision_thread.is_alive()
        ):
            try:
                self.vision.stop()
            except Exception as exc:
                report_error(exc, role="stereo camera", device=self.devices.get("camera"), stage="close")
        if self.hand is not None:
            if open_hand:
                try:
                    self.hand.open_hand()
                except Exception as exc:
                    report_error(exc, role="hand CAN", device=self.devices["can_iface"], stage="open_hand")
            try:
                self.hand.close_can()
            except Exception as exc:
                report_error(exc, role="hand CAN", device=self.devices["can_iface"], stage="close_can")
        if self.head_arm is not None:
            try:
                if home:
                    self.head_arm.home_and_close()
                else:
                    self.head_arm.close()
            except Exception as exc:
                report_error(exc, role="head_arm", device=self.devices["head_arm"], stage="home_and_close" if home else "close")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        normal_exit = exc_type is None
        self.close(home=normal_exit, open_hand=normal_exit)
