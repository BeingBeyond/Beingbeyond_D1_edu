"""Exercise example 8's preflight boundary without opening hardware."""
from dataclasses import replace
from pathlib import Path
import runpy
import sys
import unittest
from unittest.mock import patch

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))


class ExampleIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Importing the example creates no controllers and does not run main().
        cls.namespace = runpy.run_path(str(EXAMPLES / "8_exo_teleop_real.py"), run_name="test_example_8")

    def setUp(self):
        self.cfg = self.namespace["TeleopRealCfg"]()
        self.robot = self.namespace["TeleopReal"](self.cfg)

    def test_batch_resolution_preserves_config_and_forwards_custom_parameters(self):
        from exo_glove_teleop_common import resolve_teleop_devices
        cfg = replace(self.cfg, arm_exo_servo_ids=[2, 3, 4, 5, 6, 7])
        devices = {"exo": "/dev/ttyUSB8", "glove": "/dev/ttyACM9", "head_arm": "/dev/ttyACM7",
                   "can_iface": "can3", "camera": "/dev/video4"}
        with patch("exo_glove_teleop_common.resolve_devices", return_value=devices) as resolve:
            result = resolve_teleop_devices(cfg)
        self.assertEqual(result.arm_exo_port, "/dev/ttyUSB8")
        self.assertEqual(result.glove_port, "/dev/ttyACM9")
        self.assertEqual(result.robot_arm_dev, "/dev/ttyACM7")
        self.assertEqual(result.robot_hand_can, "can3")
        self.assertEqual(result.robot_vision_device, "/dev/video4")
        self.assertEqual(cfg.robot_arm_dev, "auto")
        self.assertEqual(cfg.glove_port, "auto")
        self.assertEqual(resolve.call_args[1]["exo_servo_ids"], [2, 3, 4, 5, 6, 7])

    def test_discovery_failure_prevents_all_controller_initialization(self):
        run_globals = self.robot.run.__globals__
        with patch.dict(run_globals, resolve_teleop_devices=lambda cfg: (_ for _ in ()).throw(RuntimeError("missing device"))), \
                patch.object(self.robot, "_open_inputs") as inputs, \
                patch.object(self.robot, "_open_robot") as robot, \
                patch.object(self.robot, "_move_to_initial_pose") as motion:
            with self.assertRaisesRegex(RuntimeError, "missing device"):
                self.robot.run()
            inputs.assert_not_called()
            robot.assert_not_called()
            motion.assert_not_called()

    def test_resolved_config_is_installed_before_inputs_open(self):
        resolved = replace(self.cfg, arm_exo_port="/dev/exo", glove_port="/dev/glove", robot_arm_dev="/dev/arm")
        def stop_before_open():
            self.assertIs(self.robot.cfg, resolved)
            raise RuntimeError("stopped before any control")
        with patch.dict(self.robot.run.__globals__, resolve_teleop_devices=lambda cfg: resolved), \
                patch.object(self.robot, "_open_inputs", side_effect=stop_before_open), \
                patch.object(self.robot, "_open_robot") as robot:
            with self.assertRaisesRegex(RuntimeError, "stopped before any control"):
                self.robot.run()
            robot.assert_not_called()


if __name__ == "__main__":
    unittest.main()
