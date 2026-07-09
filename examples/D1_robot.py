import threading
import time
import traceback

from beingbeyond_d1_edu_sdk import HeadArmRobot
from beingbeyond_d1_edu_sdk.hand import DexHand

from vision import StereoRGBCamera

class D1Robot:
    def __init__(self, urdf_path: str, arm_dev: str, arm_baud: int,
                 hand_type: str, hand_can: str, hand_baud: int,
                 vision_device: str = "/dev/v4l/by-id/usb-SunplusIT_Inc_SPCA2100_PC_Camera-video-index0"):
        self.head_arm = HeadArmRobot(
            urdf_path=urdf_path,
            dev=arm_dev,
            baudrate=arm_baud,
        )
        self.hand = DexHand(
            hand_type=hand_type,
            can_iface=hand_can,
            baudrate=hand_baud,
        )

        self.vision = StereoRGBCamera(device=vision_device)

        # Vision thread control
        self._vision_thread = None
        self._vision_stop_event = None

    def start_vision_thread(
        self,
        window_name: str = "D1 RGB",
    ):
        """
        Start a background thread that continuously displays RGB.
        Press 'q' in the window to stop the vision thread.
        """
        if self._vision_thread is not None and self._vision_thread.is_alive():
            return  # already running

        self._vision_stop_event = threading.Event()

        def _vision_loop():
            import cv2

            last_time = time.time()
            fps = 0.0

            try:
                with self.vision as camera:
                    while not self._vision_stop_event.is_set():
                        now = time.time()
                        dt = now - last_time
                        last_time = now
                        if dt > 0:
                            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt

                        vis = camera.read_view()
                        cv2.imshow(window_name, camera.draw_info(vis, fps))

                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            self._vision_stop_event.set()
                            break

            except Exception as e:
                print("[VisionThread] Error:")
                print(e)
                traceback.print_exc()
            finally:
                if self._vision_stop_event is not None:
                    self._vision_stop_event.set()
                try:
                    import cv2
                    cv2.destroyWindow(window_name)
                except Exception:
                    pass

        self._vision_thread = threading.Thread(
            target=_vision_loop,
            daemon=True,
        )
        self._vision_thread.start()

    def stop_vision_thread(self, join_timeout: float = 2.0):
        """
        Request the vision thread to stop and join it.
        """
        if self._vision_stop_event is not None:
            self._vision_stop_event.set()
        if self._vision_thread is not None and self._vision_thread.is_alive():
            self._vision_thread.join(join_timeout)

    def vision_stop_requested(self) -> bool:
        return (
            self._vision_stop_event is not None
            and self._vision_stop_event.is_set()
        )

    def set_q(self, q):
        self.head_arm.set_positions(q[:8])
        self.hand.set_joint_pos(q[8:])

    def get_q(self):
        q_arm = self.head_arm.get_positions()
        q_hand = self.hand.read_joint_pos()
        return q_arm + q_hand

    def close(self):
        # Stop vision thread first
        try:
            self.stop_vision_thread()
        except Exception:
            pass

        # Then release robot hardware
        try:
            self.hand.open_hand()
            self.hand.close_can()
        except Exception:
            pass
        try:
            self.head_arm.home_and_close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
