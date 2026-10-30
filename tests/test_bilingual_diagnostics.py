"""Hardware-free regressions for actionable bilingual example failures."""
import ast
from contextlib import redirect_stderr
from io import StringIO
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

EXAMPLES = Path(os.environ.get("D1_EXAMPLES_DIR", str(Path(__file__).resolve().parents[1] / "examples"))).resolve()
sys.path.insert(0, str(EXAMPLES))
import diagnostics


class BilingualDiagnosticsTests(unittest.TestCase):
    def test_each_device_has_specific_bilingual_wiring_advice(self):
        for role, chinese, english in (
            ("head_arm", "头臂", "head/arm USB"),
            ("exo", "遥操臂", "exoskeleton USB"),
            ("glove", "手套", "glove USB"),
            ("hand CAN", "CAN_H/CAN_L", "PCAN USB"),
            ("stereo camera", "相机自己的 USB 线", "camera's own USB"),
        ):
            with self.subTest(role=role):
                message = diagnostics.format_error(role, "/dev/test7", "open", OSError("unplugged"))
                for expected in (chinese, english, "/dev/test7", "stage=open", "unplugged", "排查建议", "Suggested checks"):
                    self.assertIn(expected, message)

    def test_permission_and_missing_node_have_distinct_actions(self):
        denied = diagnostics.format_error("exo", "/dev/ttyUSB8", "open", PermissionError("denied"))
        missing = diagnostics.format_error("exo", "/dev/ttyUSB8", "open", FileNotFoundError("gone"))
        self.assertIn("重新登录", denied)
        self.assertIn("dialout", denied)
        self.assertIn("端口号可能变化", missing)
        self.assertIn("gone", missing)

    def test_configuration_errors_do_not_require_rewiring(self):
        message = diagnostics.format_error("head_arm", "/dev/ttyACM7", "set_positions", ValueError("need 8 values"))
        self.assertIn("参数或数据格式", message)
        self.assertIn("need 8 values", message)
        self.assertNotIn("断电后检查", message)

    def test_subprocess_stderr_is_preserved(self):
        error = subprocess.CalledProcessError(1, ["ip"], stderr="Cannot find device can7")
        message = diagnostics.format_error("hand CAN", "can7", "status check", error)
        self.assertIn("Cannot find device can7", message)
        self.assertIn("灵巧手", message)

    def test_native_gui_error_has_display_advice(self):
        out = StringIO()
        diagnostics.report_error(RuntimeError("cvShowImage: function is not implemented"), file=out)
        self.assertIn("图形显示", out.getvalue())
        self.assertIn("DISPLAY", out.getvalue())
        self.assertIn("不代表相机线有问题", out.getvalue())

    def test_no_display_is_explained_before_opening_windows(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(sys, "platform", "linux"):
            with self.assertRaisesRegex(RuntimeError, "没有可用的图形显示会话.*No graphical display"):
                diagnostics.require_display()

    def test_exo_invalid_ids_are_throttled_and_reset_after_recovery(self):
        diagnostics._EXO_WARNINGS.clear()
        with patch.object(diagnostics.time, "monotonic", side_effect=[10, 11, 16, 17]), redirect_stderr(StringIO()) as output:
            diagnostics.warn_exo_readings([True, False], [3, 8, 9], "/dev/ttyUSB8")
            diagnostics.warn_exo_readings([True, False], [3, 8, 9], "/dev/ttyUSB8")
            diagnostics.warn_exo_readings([True, False], [3, 8, 9], "/dev/ttyUSB8")
            diagnostics.warn_exo_readings([True, True, True], [3, 8, 9], "/dev/ttyUSB8")
            diagnostics.warn_exo_readings([True, False], [3, 8, 9], "/dev/ttyUSB8")
        text = output.getvalue()
        self.assertEqual(text.count("[警告 / WARNING]"), 3)
        self.assertIn("[8, 9]", text)
        self.assertIn("/dev/ttyUSB8", text)
        self.assertIn("上游接头", text)

    def test_entry_points_explain_missing_dependencies_before_hardware_access(self):
        # Fail every hardware/optional import, even when installed on this host.
        code = '''
import builtins, runpy, sys
sys.path.insert(0, sys.argv[1])
original = builtins.__import__
blocked = {"numpy", "cv2", "scipy", "serial", "beingbeyond_d1_edu_sdk", "isaacgym", "pybullet", "pinocchio", "tkinter"}
def fail_import(name, *args, **kwargs):
    if name.split(".")[0] in blocked:
        raise ModuleNotFoundError("injected missing dependency: " + name)
    return original(name, *args, **kwargs)
builtins.__import__ = fail_import
example = sys.argv[2]
sys.argv = [example]
runpy.run_path(example, run_name="__main__")
'''
        for path in sorted(EXAMPLES.glob("[0-9]*.py")) + [EXAMPLES / "device_detection.py"]:
            with self.subTest(example=path.name):
                result = subprocess.run([sys.executable, "-c", code, str(EXAMPLES), str(path)],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
                self.assertNotEqual(result.returncode, 0)
                for expected in ("依赖或 SDK 导入失败", "Dependency or SDK import failed", "injected missing dependency", "Traceback"):
                    self.assertIn(expected, result.stderr)

    def test_argument_errors_are_bilingual_and_keep_exit_code(self):
        parser = diagnostics.BilingualArgumentParser(prog="test")
        with redirect_stderr(StringIO()) as output, self.assertRaises(SystemExit) as caught:
            parser.parse_args(["--wrong"])
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("命令行参数错误", output.getvalue())
        self.assertIn("unrecognized arguments", output.getvalue())

    def test_all_example_files_parse_with_python38_syntax(self):
        for path in EXAMPLES.glob("*.py"):
            with self.subTest(path=path.name):
                tree = ast.parse(path.read_text(), filename=str(path), feature_version=8)
                compile(tree, str(path), "exec")

    def test_servo_fault_and_missing_reply_preserve_role_id_and_cause(self):
        import device_detection as detection
        fault = RuntimeError("servo ID 23 replied with error status 0x04; present")
        with self.assertRaises(detection.DeviceDetectionError) as caught:
            with detection.device_operation("head_arm", "/dev/ttyACM7", "serial verification"):
                raise fault
        self.assertIs(caught.exception.__cause__, fault)
        for value in ("头臂", "ID 23", "0x04", "present", "/dev/ttyACM7", "舵机接头"):
            self.assertIn(value, str(caught.exception))
        with patch.object(detection, "_serial_candidates", return_value={}):
            with self.assertRaisesRegex(detection.DeviceDetectionError, "遥操臂"):
                detection.resolve_serial_ports({"exo": "auto"})

    def test_duplicate_serial_roles_explain_independent_connections(self):
        import device_detection as detection
        with self.assertRaises(detection.DeviceDetectionError) as caught:
            detection.resolve_serial_ports({"head_arm": "/dev/same", "glove": "/dev/same"})
        for value in ("同一串口", "USB 线", "multiple roles", "head_arm", "glove", "/dev/same"):
            self.assertIn(value, str(caught.exception))

    def test_failed_camera_frame_is_bilingual_and_keeps_selected_path(self):
        import vision
        camera = vision.StereoRGBCamera(device="/dev/video8")
        camera.cap = Mock()
        camera.cap.read.return_value = (False, None)
        with self.assertRaises(RuntimeError) as caught:
            camera.read_pair()
        for value in ("相机没有返回图像", "Failed to read frame", "/dev/video8", "相机自己的 USB 线"):
            self.assertIn(value, str(caught.exception))

    def test_calibration_timeout_names_exo_port_and_configured_servo_ids(self):
        from exo_glove_teleop_common import calibrate_arm_exo_zero
        cfg = Mock(zero_calibration_timeout_s=0, arm_exo_port="/dev/ttyUSB8", arm_exo_servo_ids=[4, 5, 6, 7, 8, 9])
        exo = Mock()
        with self.assertRaises(RuntimeError) as caught:
            calibrate_arm_exo_zero(exo, cfg)
        exo.read_frame.assert_not_called()
        for value in ("零位标定超时", "Timed out", "/dev/ttyUSB8", "[4, 5, 6, 7, 8, 9]", "保持零位静止"):
            self.assertIn(value, str(caught.exception))

    def test_game_posture_failures_name_the_exact_bus_and_stop_later_commands(self):
        import device_detection as detection
        # Isolate the actual helper from optional MediaPipe/retargeting imports.
        tree = ast.parse((EXAMPLES / "10_rock_paper_scissors_real.py").read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "apply_posture")
        function.returns = None
        for arg in function.args.args:
            arg.annotation = None
        namespace = {"device_operation": detection.device_operation, "deg_list_to_rad": list}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "game_helper", "exec"), namespace)
        arm, hand = Mock(), Mock()
        devices = {"head_arm": "/dev/ttyACM7", "can_iface": "can9"}
        cause = OSError("link lost")
        arm.set_positions.side_effect = cause
        with self.assertRaises(detection.DeviceDetectionError) as caught:
            namespace["apply_posture"](arm, hand, (0, 0), (0,) * 6, (0,) * 6, devices=devices)
        self.assertIs(caught.exception.__cause__, cause)
        self.assertIn("/dev/ttyACM7", str(caught.exception))
        hand.set_joint_pos.assert_not_called()
        arm.set_positions.side_effect = None
        hand.set_joint_pos.side_effect = cause
        with self.assertRaisesRegex(detection.DeviceDetectionError, "hand CAN port=can9"):
            namespace["apply_posture"](arm, hand, (0, 0), (0,) * 6, (0,) * 6, devices=devices)


if __name__ == "__main__":
    unittest.main()
