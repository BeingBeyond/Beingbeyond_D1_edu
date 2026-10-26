"""Exercise example 8's preflight boundary without opening hardware."""
from dataclasses import replace
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import os
import runpy
import sys
import unittest
from unittest.mock import Mock, patch

EXAMPLES = Path(os.environ.get("D1_EXAMPLES_DIR", str(Path(__file__).resolve().parents[1] / "examples"))).resolve()
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
        self.assertTrue(resolve.call_args[1]["check_ready"])

    def test_disabled_camera_is_omitted_from_teleop_preflight(self):
        from exo_glove_teleop_common import resolve_teleop_devices
        cfg = replace(self.cfg, enable_camera=False)
        devices = {"exo": "/dev/exo", "glove": "/dev/glove", "head_arm": "/dev/head", "can_iface": "can7"}
        with patch("exo_glove_teleop_common.resolve_devices", return_value=devices) as resolve:
            resolved = resolve_teleop_devices(cfg)
        self.assertIsNone(resolve.call_args[1]["camera"])
        self.assertFalse(resolved.enable_camera)

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

    def test_partial_input_startup_failure_closes_exo_and_prevents_robot_setup(self):
        exo = Mock()
        def fail_after_exo_open():
            self.robot.arm_exo = exo
            raise RuntimeError("glove initialization failed")
        with patch.dict(self.robot.run.__globals__, resolve_teleop_devices=lambda cfg: cfg), \
                patch.object(self.robot, "_open_inputs", side_effect=fail_after_exo_open), \
                patch.object(self.robot, "_open_robot") as robot:
            with self.assertRaisesRegex(RuntimeError, "glove initialization failed"):
                self.robot.run()
        exo.close.assert_called_once()
        robot.assert_not_called()

    def test_initial_pose_failure_closes_without_homing_or_opening_hand(self):
        controller = Mock()
        def assign_controller():
            self.robot.robot = controller
        with patch.dict(self.robot.run.__globals__, resolve_teleop_devices=lambda cfg: cfg), \
                patch.object(self.robot, "_open_inputs"), \
                patch.object(self.robot, "_open_robot", side_effect=assign_controller), \
                patch.object(self.robot, "_move_to_initial_pose", side_effect=RuntimeError("lost serial")):
            with self.assertRaisesRegex(RuntimeError, "lost serial"):
                self.robot.run()
        controller.close.assert_called_once_with(home=False, open_hand=False)

    def test_runtime_failure_propagates_and_closes_without_extra_motion(self):
        self.robot.cfg = replace(self.cfg, enable_camera=False)
        controller = Mock()
        def assign_controller():
            self.robot.robot = controller
        with patch.dict(self.robot.run.__globals__, resolve_teleop_devices=lambda cfg: cfg), \
                patch.object(self.robot, "_open_inputs"), \
                patch.object(self.robot, "_open_robot", side_effect=assign_controller), \
                patch.object(self.robot, "_move_to_initial_pose"), \
                patch.object(self.robot, "_build_teleop", side_effect=RuntimeError("serial disconnected")), \
                redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            with self.assertRaisesRegex(RuntimeError, "serial disconnected"):
                self.robot.run()
        controller.close.assert_called_once_with(home=False, open_hand=False)

    def test_simulator_startup_failure_releases_external_input(self):
        with patch.dict(sys.modules, {"pybullet": Mock(), "pybullet_data": Mock()}):
            namespace = runpy.run_path(str(EXAMPLES / "7_exo_teleop_sim.py"), run_name="test_example_7")
            simulator = namespace["TeleopSimulator"](namespace["TeleopSimCfg"]())
            exo = Mock()
            def fail_after_exo_open():
                simulator.arm_exo = exo
                raise RuntimeError("glove initialization failed")
            with patch.dict(simulator.run.__globals__, validate_cfg=lambda cfg: None,
                            resolve_teleop_devices=lambda cfg: cfg), \
                    patch.object(simulator, "_open_inputs", side_effect=fail_after_exo_open), \
                    patch.object(simulator, "_open_sim") as simulation, redirect_stdout(StringIO()):
                with self.assertRaisesRegex(RuntimeError, "glove initialization failed"):
                    simulator.run()
            exo.close.assert_called_once()
            simulation.assert_not_called()


if __name__ == "__main__":
    unittest.main()
