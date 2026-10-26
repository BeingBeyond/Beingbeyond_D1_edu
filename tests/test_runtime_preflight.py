"""Mock regressions for controller boundaries and camera error attribution."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import os
import runpy
import sys
import unittest
from unittest.mock import MagicMock, patch

EXAMPLES = Path(os.environ.get("D1_EXAMPLES_DIR", str(Path(__file__).resolve().parents[1] / "examples"))).resolve()
sys.path.insert(0, str(EXAMPLES))

import device_detection as detection
import D1_robot as runtime
import vision


class RuntimePreflightTests(unittest.TestCase):
    def setUp(self):
        self.devices = {"head_arm": "/dev/ttyACM7", "can_iface": "can7", "camera": "/dev/video8"}
        self.args = dict(urdf_path="unused.urdf", arm_dev="auto", arm_baud=1000000,
                         hand_type="left", hand_can="auto", hand_baud=1000000)

    def test_failed_preflight_prevents_both_controllers(self):
        with patch.object(runtime, "resolve_devices", side_effect=detection.DeviceDetectionError("camera failed")), \
                patch.object(runtime, "HeadArmRobot") as arm, patch.object(runtime, "DexHand") as hand:
            with self.assertRaisesRegex(detection.DeviceDetectionError, "camera failed"):
                runtime.D1Robot(**self.args)
        arm.assert_not_called()
        hand.assert_not_called()

    def test_hand_example_checks_communication_before_creating_controller(self):
        namespace = runpy.run_path(str(EXAMPLES / "1_control_hand.py"), run_name="test_hand_example")
        main = namespace["main"]
        preflight = MagicMock(side_effect=detection.DeviceDetectionError("CAN no response"))
        controller = MagicMock()
        with patch.dict(main.__globals__, resolve_devices=preflight, DexHand=controller), redirect_stdout(StringIO()):
            with self.assertRaisesRegex(detection.DeviceDetectionError, "CAN no response"):
                main()
        preflight.assert_called_once_with(can_iface="auto", check_ready=True)
        controller.assert_not_called()

    def test_head_example_cannot_initialize_after_detection_failure(self):
        namespace = runpy.run_path(str(EXAMPLES / "2_control_head_arm.py"), run_name="test_head_example")
        main = namespace["main"]
        resolve = MagicMock(side_effect=detection.DeviceDetectionError("wrong controller at /dev/ttyACM7"))
        controller = MagicMock()
        with patch.dict(main.__globals__, resolve_head_arm=resolve, HeadArmRobot=controller), redirect_stdout(StringIO()):
            with self.assertRaisesRegex(detection.DeviceDetectionError, "/dev/ttyACM7"):
                main()
        controller.assert_not_called()

    def test_no_vision_omits_camera_but_preserves_readiness_and_hand_side(self):
        devices = {key: value for key, value in self.devices.items() if key != "camera"}
        with patch.object(runtime, "resolve_devices", return_value=devices) as resolve, \
                patch.object(runtime, "HeadArmRobot"), patch.object(runtime, "DexHand"), \
                patch.object(vision, "StereoRGBCamera") as camera:
            robot = runtime.D1Robot(**self.args, enable_vision=False)
            self.assertIsNone(resolve.call_args[1]["camera"])
            self.assertTrue(resolve.call_args[1]["check_ready"])
            self.assertEqual(resolve.call_args[1]["hand_type"], "left")
            robot.close(home=False, open_hand=False)
        camera.assert_not_called()

    def test_hand_init_failure_reports_port_and_closes_arm_without_homing(self):
        with patch.object(runtime, "resolve_devices", return_value=self.devices), \
                patch.object(runtime, "HeadArmRobot") as arm, \
                patch.object(runtime, "DexHand", side_effect=OSError("lost adapter")), \
                patch.object(vision, "StereoRGBCamera") as camera:
            with self.assertRaisesRegex(detection.DeviceDetectionError,
                                        "hand CAN port=can7 stage=controller initialization.*lost adapter"):
                runtime.D1Robot(**self.args)
        arm.return_value.close.assert_called_once()
        arm.return_value.home_and_close.assert_not_called()
        camera.return_value.stop.assert_called_once()

    def test_runtime_arm_error_keeps_selected_port_and_original_cause(self):
        with patch.object(runtime, "resolve_devices", return_value=self.devices), \
                patch.object(runtime, "HeadArmRobot") as arm, patch.object(runtime, "DexHand") as hand, \
                patch.object(vision, "StereoRGBCamera"):
            robot = runtime.D1Robot(**self.args)
            arm.return_value.set_positions.side_effect = OSError("USB disconnected")
            with self.assertRaisesRegex(detection.DeviceDetectionError,
                                        "head_arm port=/dev/ttyACM7 stage=set positions.*USB disconnected") as caught:
                robot.set_q([0.0] * 14)
            self.assertIsInstance(caught.exception.__cause__, OSError)
            hand.return_value.set_joint_pos.assert_not_called()
            robot.close(home=False, open_hand=False)

    def test_camera_read_failure_includes_device(self):
        camera = vision.StereoRGBCamera(device="/dev/video8")
        camera.cap = MagicMock()
        camera.cap.read.return_value = (False, None)
        with self.assertRaisesRegex(RuntimeError, "port=/dev/video8 stage=read frame.*Failed to read frame"):
            camera.read_pair()

    def test_camera_configuration_failure_releases_capture(self):
        camera = vision.StereoRGBCamera(device="/dev/video8")
        with patch.object(vision.cv2, "VideoCapture") as capture:
            capture.return_value.set.side_effect = OSError("configuration failed")
            with self.assertRaisesRegex(detection.DeviceDetectionError,
                                        "stereo camera port=/dev/video8 stage=configure capture.*configuration failed"):
                camera.start()
        capture.return_value.release.assert_called_once()
        self.assertIsNone(camera.cap)

    def test_camera_preflight_failure_releases_capture(self):
        with patch.object(detection, "_port_users", return_value=[]), \
                patch.object(vision.cv2, "VideoCapture") as capture:
            capture.return_value.isOpened.return_value = False
            with self.assertRaisesRegex(detection.DeviceDetectionError,
                                        "stereo camera port=/dev/video8 stage=capture check"):
                detection.check_camera("/dev/video8")
        capture.return_value.release.assert_called_once()

    def test_cli_reports_both_peripheral_failures_and_returns_failure(self):
        with patch.object(sys, "argv", ["device_detection.py", "--roles", "can", "camera", "--check"]), \
                patch.object(detection, "resolve_can_iface", side_effect=OSError("CAN absent")), \
                patch.object(detection, "resolve_camera_device", side_effect=OSError("camera absent")), \
                redirect_stdout(StringIO()) as output:
            self.assertEqual(detection.main(), 1)
        self.assertIn("CAN absent", output.getvalue())
        self.assertIn("camera absent", output.getvalue())

    def test_list_does_not_open_or_probe_devices(self):
        with patch.object(sys, "argv", ["device_detection.py", "--list"]), \
                patch.object(detection, "_serial_candidates", return_value={}), \
                patch.object(detection, "_open_probe") as probe, \
                patch.object(detection, "check_can") as can_check, \
                patch.object(detection, "check_camera") as camera_check:
            self.assertEqual(detection.main(), 0)
        probe.assert_not_called()
        can_check.assert_not_called()
        camera_check.assert_not_called()


if __name__ == "__main__":
    unittest.main()
