"""Hardware-free tests for concise output and joint-level connection advice."""
import ast
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import os
from pathlib import Path
import subprocess
import sys
import threading
import runpy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

EXAMPLES = Path(os.environ.get("D1_EXAMPLES_DIR", str(Path(__file__).resolve().parents[1] / "examples"))).resolve()
sys.path.insert(0, str(EXAMPLES))
import diagnostics as diag
import device_detection as detection


class BilingualDiagnosticsTests(unittest.TestCase):
    def test_known_error_is_four_lines_without_internal_metadata(self):
        error = diag.format_error("exo", "/dev/ttyUSB8", "open", PermissionError("denied"))
        text = str(error)
        self.assertEqual(len(text.splitlines()), 4)
        self.assertIn("dialout", text)
        self.assertNotIn("stage=", text)
        self.assertNotIn("servo", text.lower())
        self.assertNotIn("舵机", text)
        self.assertNotIn("关节连线", text)
        self.assertEqual(error.context["stage"], "open")
        self.assertEqual(error.context["raw"], "denied")

    def test_nested_contexts_keep_the_cause_and_print_once(self):
        original = PermissionError("denied")
        with self.assertRaises(diag.DiagnosticError) as caught:
            with detection.device_operation("serial roles", "auto", "discovery"):
                with detection.device_operation("exo", "/dev/ttyUSB8", "open"):
                    raise original
        self.assertIs(caught.exception.__cause__, original)
        with redirect_stderr(StringIO()) as out:
            diag.report_error(caught.exception)
            diag.report_error(caught.exception)
        self.assertEqual(len(out.getvalue().splitlines()), 4)
        self.assertNotIn("serial roles", out.getvalue())
        self.assertNotIn("Traceback", out.getvalue())

    def test_group_deduplicates_and_marks_unchecked_roles_once(self):
        error = diag.failure("V01", "未识别到双目相机。", "No stereo camera found.", "请检查相机 USB 线。", "Check the camera USB cable.")
        group = diag.DiagnosticGroup([error, diag.DiagnosticGroup([error])], [("尚未检查：头臂。", "Not checked: head/arm.")])
        out = StringIO()
        diag.report_error(group, file=out)
        diag.report_error(group, file=out)
        self.assertEqual(out.getvalue().count("[错误]"), 1)
        self.assertEqual(out.getvalue().count("尚未检查"), 1)

    def test_no_usb_candidates_ignore_all_ttyS_ports(self):
        ports = {"/dev/ttyS{}".format(i): SimpleNamespace(vid=None, pid=None) for i in range(64)}
        with patch.object(detection, "_serial_candidates", return_value=ports), patch.object(detection, "_open_probe") as probe:
            with self.assertRaises(diag.DiagnosticGroup) as caught:
                detection.resolve_serial_ports({"head_arm": "auto"})
        self.assertIn("未识别到头臂控制器", str(caught.exception))
        self.assertNotIn("ttyS", str(caught.exception))
        self.assertEqual(len(str(caught.exception).splitlines()), 4)
        probe.assert_not_called()

    def test_invalid_exo_ids_fail_before_any_device_access(self):
        from exo_glove_teleop_common import validate_common_cfg
        for ids in ([1, 2, 3, 3, 5, 6], [], [1, -1], [1000], [True], [1.5], ["3"], None, 1):
            with self.subTest(ids=ids), patch.object(detection, "resolve_can_iface") as can, patch.object(detection, "_serial_candidates") as ports:
                with self.assertRaises(diag.DiagnosticError) as caught:
                    detection.resolve_devices(exo="auto", exo_servo_ids=ids, can_iface="auto", check_ready=True)
                self.assertEqual(caught.exception.code, "P08")
                self.assertIn("arm_exo_servo_ids", str(caught.exception))
                can.assert_not_called()
                ports.assert_not_called()
                with self.assertRaises(diag.DiagnosticError) as cfg_error:
                    validate_common_cfg(SimpleNamespace(arm_exo_servo_ids=ids))
                self.assertIn("arm_exo_servo_ids", str(cfg_error.exception))

    def test_serial_entry_also_rejects_duplicate_exo_ids_before_opening(self):
        with patch.object(detection, "_open_probe") as opened, self.assertRaises(diag.DiagnosticError):
            detection.resolve_serial_ports({"exo": "/dev/exo"}, exo_servo_ids=[1, 1], verify_explicit=True)
        opened.assert_not_called()

    def test_valid_exo_ids_follow_sdk_range_without_being_renumbered(self):
        for ids in ([0, 42, 999], (1, 3, 6), range(1, 7)):
            original = list(ids)
            diag.validate_exo_ids(ids)
            self.assertEqual(list(ids), original)

    def test_empty_serial_inventory_prints_a_result(self):
        with patch.object(sys, "argv", ["device_detection.py", "--list"]), patch.object(detection, "_serial_candidates", return_value={}), redirect_stdout(StringIO()) as out:
            self.assertEqual(detection.main(), 0)
        self.assertIn("未发现可识别的 USB 串口", out.getvalue())
        self.assertIn("No recognizable USB serial ports", out.getvalue())

    def test_two_threads_report_the_same_error_once(self):
        error = diag.failure("V11", "窗口失败。", "Window failed.", "请在桌面运行。", "Run from the desktop.")
        entered, release = threading.Event(), threading.Event()
        render = error.render
        failures = []
        out = StringIO()
        def slow_render(severity):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test timed out")
            return render(severity)
        def report():
            try:
                diag.report_error(error, file=out)
            except Exception as exc:
                failures.append(exc)
        with patch.object(error, "render", side_effect=slow_render):
            first = threading.Thread(target=report)
            second = threading.Thread(target=report)
            first.start()
            try:
                self.assertTrue(entered.wait(5))
                second.start()
                second.join(2)
                self.assertFalse(second.is_alive())
            finally:
                release.set()
                first.join(5)
                if second.ident is not None:
                    second.join(5)
        self.assertFalse(failures)
        self.assertEqual(out.getvalue().count("[错误]"), 1)
        self.assertEqual(len(out.getvalue().splitlines()), 4)

    def test_tactile_key_poll_failure_is_a_display_error(self):
        namespace = runpy.run_path(str(EXAMPLES / "1_control_hand.py"), run_name="test_tactile_keys")
        show = namespace["show_tactile_preview"]
        import vision
        with patch.dict(show.__globals__, build_tactile_panel=Mock(return_value=object())), patch.object(vision.cv2, "imshow"), patch.object(vision.cv2, "waitKey", side_effect=RuntimeError("event polling failed")):
            with self.assertRaises(diag.DiagnosticError) as caught:
                show(Mock(), "can7", preview_scale=1)
        self.assertEqual(caught.exception.code, "V11")
        self.assertNotIn("CAN", str(caught.exception))

    def test_background_key_poll_failure_is_not_a_camera_error(self):
        import D1_robot as runtime
        import vision
        devices = {"head_arm": "/dev/arm", "can_iface": "can7", "camera": "/dev/video8"}
        with patch.object(runtime, "resolve_devices", return_value=devices), patch.object(runtime, "HeadArmRobot"), patch.object(runtime, "DexHand"), patch.object(vision, "StereoRGBCamera"), patch.dict(os.environ, DISPLAY=":test"), patch.object(vision.cv2, "imshow"), patch.object(vision.cv2, "destroyWindow"), patch.object(vision.cv2, "waitKey", side_effect=RuntimeError("event polling failed")), redirect_stderr(StringIO()) as out:
            robot = runtime.D1Robot("unused.urdf", "auto", 1000000, "right", "auto", 1000000)
            try:
                with self.assertRaises(diag.DiagnosticError) as caught:
                    robot.start_vision_thread()
                self.assertEqual(caught.exception.code, "V11")
                diag.report_error(caught.exception)
                self.assertEqual(out.getvalue().count("[错误]"), 1)
                self.assertNotIn("相机", out.getvalue())
            finally:
                robot.close(home=False, open_hand=False)

    def test_head_and_arm_numbering_are_separate(self):
        zh, en = diag.joint_label("head_arm", [11, 12, 21, 26])
        self.assertEqual(zh, "头部 1、2 号关节、臂部 1、6 号关节")
        self.assertIn("head joints 1, 2", en)
        self.assertIn("arm joints 1, 6", en)

    def test_arm_good_prefix_bad_suffix_points_to_three_four_link(self):
        error = diag.joint_read_error("head_arm", "/dev/ttyACM0", [24, 25, 26], detection.HEAD_ARM_IDS, 1000000)
        self.assertIn("臂部 3 号关节与臂部 4 号关节之间的连线", error.action_zh)
        self.assertIn("between arm joint 3 and arm joint 4", error.action_en)
        self.assertNotIn("头部", error.chinese)
        self.assertNotIn("舵机", str(error))
        self.assertNotIn("servo", str(error).lower())

    def test_isolated_missing_joint_does_not_claim_chain_disconnect(self):
        error = diag.joint_read_error("head_arm", "/dev/ttyACM0", [24], detection.HEAD_ARM_IDS)
        self.assertIn("臂部 4 号关节", error.chinese)
        self.assertNotIn("臂部 5", error.chinese)
        self.assertNotIn("之间", error.action_zh)

    def test_head_and_arm_failures_have_separate_connection_advice(self):
        error = diag.joint_read_error("head_arm", "port", [12, 24, 25, 26], detection.HEAD_ARM_IDS)
        self.assertIsInstance(error, diag.DiagnosticGroup)
        self.assertEqual(len(error.errors), 2)
        self.assertIn("head joint 1 and head joint 2", str(error.errors[0]))
        self.assertIn("arm joint 3 and arm joint 4", str(error.errors[1]))

    def test_missing_head_chain_does_not_invent_head_arm_link(self):
        error = diag.joint_read_error("head_arm", "port", [11, 12], detection.HEAD_ARM_IDS)
        self.assertNotIn("臂部", str(error))
        self.assertNotIn("之间", error.action_zh)

    def test_entire_arm_missing_does_not_invent_good_joint(self):
        error = diag.joint_read_error("head_arm", "port", [21, 22, 23, 24, 25, 26], detection.HEAD_ARM_IDS)
        self.assertNotIn("之间", error.action_zh)
        self.assertNotIn("头部", error.chinese)

    def test_all_joints_missing_prioritize_power_not_individual_links(self):
        error = diag.joint_read_error("head_arm", "port", detection.HEAD_ARM_IDS, detection.HEAD_ARM_IDS, 1000000)
        self.assertEqual(error.code, "H02")
        self.assertIn("供电", error.action_zh)
        self.assertIn("1000000", error.action_zh)
        self.assertNotIn("之间", error.action_zh)

    def test_custom_exo_ids_do_not_claim_physical_order_or_adjacent_cable(self):
        error = diag.joint_read_error("exo", "/dev/ttyUSB8", [42, 7, 9], [8, 3, 14, 42, 7, 9], 115200)
        self.assertIn("通信 ID 42、7、9", error.chinese)
        self.assertNotIn("4 号关节", str(error))
        self.assertNotIn("之间", error.action_zh)

    def test_sdk_missing_id_error_uses_verified_joint_mapping(self):
        error = diag.format_error("head_arm", "port", "controller initialization", RuntimeError("Servo ids not responding: [24, 25, 26]"))
        self.assertEqual(error.code, "H03")
        self.assertIn("between arm joint 3 and arm joint 4", str(error))

    def test_warning_only_on_change_and_recovery_and_keeps_fallback_description(self):
        diag._EXO_WARNINGS.clear()
        ids = [1, 2, 3, 4, 5, 6]
        with redirect_stderr(StringIO()) as out:
            diag.warn_exo_readings([True, True, True, False, False, False], ids, "port", 4)
            for _ in range(50):
                diag.warn_exo_readings([True, True, True, False, False, False], ids, "port", 4)
            diag.warn_exo_readings([True, True, True, True, False, False], ids, "port", 4)
            diag.warn_exo_readings([True] * 6, ids, "port", 4)
            diag.warn_exo_readings([True] * 6, ids, "port", 4)
        text = out.getvalue()
        self.assertEqual(text.count("[警告]"), 2)
        self.assertEqual(text.count("读数已恢复"), 1)
        self.assertIn("未更新机械臂目标", text)
        self.assertIn("保持原目标", text)
        self.assertNotIn("机器人已停止", text)

    def test_dependency_errors_name_the_actual_missing_module(self):
        for module, code in (("beingbeyond_d1_edu_sdk", "P02"), ("numpy", "P01")):
            error = diag.format_error(None, None, "runtime", ModuleNotFoundError("missing", name=module))
            self.assertEqual(error.code, code)
            self.assertEqual(len(str(error).splitlines()), 4)
        error = diag.format_error(None, None, "runtime", ImportError("unresolved symbol"))
        self.assertEqual(error.code, "P03")
        self.assertNotIn("架构", str(error))

    def test_entrypoints_catch_missing_imports_before_hardware(self):
        code = '''
import builtins, runpy, sys
sys.path.insert(0, sys.argv[1])
original = builtins.__import__
blocked = {"numpy", "cv2", "scipy", "serial", "beingbeyond_d1_edu_sdk", "isaacgym", "pybullet", "pinocchio", "tkinter"}
def fail_import(name, *args, **kwargs):
    if name.split(".")[0] in blocked:
        raise ModuleNotFoundError("missing " + name, name=name)
    return original(name, *args, **kwargs)
builtins.__import__ = fail_import
example = sys.argv[2]
sys.argv = [example]
runpy.run_path(example, run_name="__main__")
'''
        for path in sorted(EXAMPLES.glob("[0-9]*.py")) + [EXAMPLES / "device_detection.py"]:
            with self.subTest(path=path.name):
                env = dict(os.environ, D1_DEBUG="0")
                result = subprocess.run([sys.executable, "-c", code, str(EXAMPLES), str(path)], env=env, capture_output=True, text=True, timeout=10)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(len(result.stderr.splitlines()), 4)
                self.assertNotIn("Traceback", result.stderr)
                self.assertIn("[Error]", result.stderr)

    def test_debug_preserves_original_traceback_and_command_stderr(self):
        original = subprocess.CalledProcessError(1, ["ip"], stderr="original detail")
        try:
            with detection.device_operation("hand CAN", "can9", "status check"):
                raise original
        except diag.DiagnosticError as error:
            with patch.dict(os.environ, D1_DEBUG="1"), redirect_stderr(StringIO()) as out:
                diag.report_error(error)
        self.assertIn("Traceback", out.getvalue())
        self.assertIn("original detail", out.getvalue())

    def test_group_debug_keeps_each_unraised_original_error(self):
        errors = []
        for role, port in (("exo", "/dev/exo"), ("glove", "/dev/glove")):
            try:
                raise PermissionError(13, "original-" + role, port)
            except PermissionError as error:
                errors.append(diag.format_error(role, port, "serial verification", error))
        with patch.dict(os.environ, D1_DEBUG="1"), redirect_stderr(StringIO()) as out:
            diag.report_error(diag.DiagnosticGroup(errors))
        for text in ("original-exo", "original-glove", "/dev/exo", "/dev/glove", "serial verification"):
            self.assertIn(text, out.getvalue())

    def test_diagnostic_debug_flag_also_covers_import_failures(self):
        code = '''
import builtins, runpy, sys
sys.path.insert(0, sys.argv[1])
original = builtins.__import__
def fail_import(name, *args, **kwargs):
    if name == "serial":
        raise ModuleNotFoundError("missing serial", name=name)
    return original(name, *args, **kwargs)
builtins.__import__ = fail_import
example = sys.argv[2]
sys.argv = [example, "--debug"]
runpy.run_path(example, run_name="__main__")
'''
        result = subprocess.run([sys.executable, "-c", code, str(EXAMPLES), str(EXAMPLES / "device_detection.py")], env=dict(os.environ, D1_DEBUG="0"), capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Traceback", result.stderr)
        self.assertIn("missing serial", result.stderr)

    def test_group_debug_keeps_sdk_missing_id_message(self):
        original = RuntimeError("Servo ids not responding: [12, 24, 25, 26]")
        error = diag.format_error("head_arm", "/dev/arm", "controller initialization", original)
        self.assertIsInstance(error, diag.DiagnosticGroup)
        with patch.dict(os.environ, D1_DEBUG="1"), redirect_stderr(StringIO()) as out:
            diag.report_error(error)
        self.assertIn(str(original), out.getvalue())

    def test_can_permission_does_not_suggest_serial_or_video_groups(self):
        error = diag.format_error("hand CAN", "can7", "status check", PermissionError(1, "denied"))
        self.assertNotIn("dialout", str(error))
        self.assertNotIn("video", str(error))
        self.assertIn("can7", str(error))

    def test_missing_model_is_not_reported_as_missing_usb(self):
        for original in (FileNotFoundError("missing model"), FileNotFoundError(2, "missing", "/models/d1.urdf")):
            error = diag.format_error("head_arm", "/dev/ttyACM0", "controller initialization", original)
            self.assertEqual(error.code, "P06")
            self.assertNotIn("USB", str(error))

    def test_model_permission_is_not_reported_as_serial_permission(self):
        error = diag.format_error("head_arm", "/dev/ttyACM0", "controller initialization", PermissionError(13, "denied", "/models/d1.urdf"))
        self.assertEqual(error.code, "P06")
        self.assertIn("/models/d1.urdf", str(error))
        self.assertNotIn("dialout", str(error))

    def test_sdk_reported_joint_fault_is_not_treated_as_a_disconnected_joint(self):
        error = diag.format_error("head_arm", "/dev/ttyACM0", "read positions", RuntimeError("SyncRead returned servo error id=23 err=4"))
        self.assertEqual(error.code, "H04")
        self.assertIn("臂部 3 号关节", str(error))
        self.assertIn("0x04", str(error))
        self.assertNotIn("连线", str(error))

    def test_sdk_enable_failure_does_not_infer_a_broken_link(self):
        error = diag.format_error("head_arm", "/dev/ttyACM0", "controller initialization", RuntimeError("Failed to enable torque for servo ids: [24, 25, 26]"))
        self.assertIn("臂部 4、5、6 号关节使能失败", str(error))
        self.assertNotIn("之间", str(error))

    def test_multiple_pcan_candidates_are_called_interfaces_not_hands(self):
        with self.assertRaises(diag.DiagnosticError) as caught:
            detection._unique("hand CAN", ["can7", "can9"])
        self.assertIn("PCAN 接口", str(caught.exception))
        self.assertNotIn("多个灵巧手", str(caught.exception))

    def test_serial_only_cli_does_not_mention_unrequested_camera_or_can(self):
        with patch.object(sys, "argv", ["device_detection.py", "--roles", "exo"]), patch.object(detection, "resolve_devices", return_value={"exo": "/dev/exo"}), redirect_stdout(StringIO()) as out:
            self.assertEqual(detection.main(), 0)
        self.assertIn("/dev/exo", out.getvalue())
        self.assertNotIn("CAN", out.getvalue())
        self.assertNotIn("相机", out.getvalue())

    def test_sdk_wrapped_permission_error_uses_actual_errno(self):
        try:
            try:
                raise PermissionError(13, "denied")
            except PermissionError as cause:
                raise RuntimeError("SDK open failed") from cause
        except RuntimeError as error:
            diagnostic = diag.format_error("glove", "port", "open", error)
        self.assertEqual(diagnostic.code, "S06")

    def test_unknown_error_does_not_turn_permission_words_into_a_fact(self):
        error = diag.format_error("exo", "port", "open", RuntimeError("maybe permission denied"))
        self.assertEqual(error.code, "X06")

    def test_serial_missing_path_uses_actual_errno(self):
        import serial
        error = diag.format_error("exo", "/dev/missing", "open", serial.SerialException(2, "missing port"))
        self.assertEqual(error.code, "S05")
        self.assertIn("/dev/missing", str(error))

    def test_can_down_wrong_bitrate_and_missing_bitrate_are_distinct(self):
        import json
        for flags, data, code in (([], {}, "C03"), (["UP"], {"bittiming": {"bitrate": 500000}}, "C04"), (["UP"], {}, "C07")):
            status = [{"flags": flags, "linkinfo": {"info_data": data}}]
            with patch.object(detection.subprocess, "run", return_value=SimpleNamespace(stdout=json.dumps(status))), patch.object(detection.socket, "socket") as sock:
                with self.assertRaises(diag.DiagnosticError) as caught:
                    detection.check_can("can7")
            self.assertEqual(caught.exception.code, code)
            sock.assert_not_called()

    def test_argument_error_is_four_lines_and_exit_two(self):
        parser = diag.BilingualArgumentParser(prog="demo.py")
        with redirect_stderr(StringIO()) as out, self.assertRaises(SystemExit) as caught:
            parser.parse_args(["--wrong"])
        self.assertEqual(caught.exception.code, 2)
        self.assertEqual(len(out.getvalue().splitlines()), 4)
        self.assertIn("demo.py --help", out.getvalue())

    def test_no_display_explains_local_desktop(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(sys, "platform", "linux"):
            with self.assertRaises(diag.DiagnosticError) as caught:
                diag.require_display()
        self.assertEqual(caught.exception.code, "V10")
        self.assertIn("本机桌面终端", str(caught.exception))

    def test_camera_false_open_does_not_claim_permission_or_occupancy(self):
        import vision
        camera = vision.StereoRGBCamera(device="/dev/video8")
        with patch.object(vision.cv2, "VideoCapture") as cap:
            cap.return_value.isOpened.return_value = False
            with self.assertRaises(diag.DiagnosticError) as caught:
                camera.start()
        self.assertEqual(caught.exception.code, "V06")
        self.assertNotIn("权限", str(caught.exception))
        self.assertNotIn("正被", str(caught.exception))
        cap.return_value.release.assert_called_once()

    def test_camera_empty_frame_and_wrong_size_are_distinct(self):
        import vision
        camera = vision.StereoRGBCamera(device="/dev/video8")
        camera.cap = Mock()
        camera.cap.read.return_value = (False, None)
        with self.assertRaises(diag.DiagnosticError) as caught:
            camera.read_pair()
        self.assertEqual(caught.exception.code, "V07")
        camera.cap.read.return_value = (True, SimpleNamespace(shape=(240, 640, 3)))
        with self.assertRaises(diag.DiagnosticError) as caught:
            camera.read_pair()
        self.assertEqual(caught.exception.code, "V08")
        self.assertIn("640×240", str(caught.exception))
        self.assertNotIn("带宽", str(caught.exception))
        camera._has_frame = True
        camera.cap.read.return_value = (False, None)
        with self.assertRaises(diag.DiagnosticError) as caught:
            camera.read_pair()
        self.assertEqual(caught.exception.code, "V09")

    def test_calibration_timeout_distinguishes_invalid_data_from_movement(self):
        from exo_glove_teleop_common import calibrate_arm_exo_zero
        cfg = SimpleNamespace(zero_calibration_timeout_s=0.8, zero_calibration_poll_s=0,
                              zero_calibration_stable_s=3, zero_calibration_max_delta_deg=3,
                              arm_exo_port="port", arm_exo_servo_ids=[1, 2, 3, 4, 5, 6], arm_exo_baudrate=115200)
        for angles, code in (([1, 2, 3, None, None, None], "E03"), ([0] * 6, "E04")):
            exo = Mock()
            exo.read_frame.return_value.angles_deg = angles
            ticks = iter([0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.9, 1.0, 1.1])
            with patch("exo_glove_teleop_common.time.monotonic", side_effect=lambda: next(ticks)), redirect_stdout(StringIO()) as out:
                with self.assertRaises(diag.DiagnosticError) as caught:
                    calibrate_arm_exo_zero(exo, cfg)
            self.assertEqual(caught.exception.code, code)
            self.assertEqual(len(out.getvalue().splitlines()), 2)

    def test_game_posture_failure_stops_later_commands_and_keeps_exact_bus(self):
        tree = ast.parse((EXAMPLES / "10_rock_paper_scissors_real.py").read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "apply_posture")
        function.returns = None
        for arg in function.args.args:
            arg.annotation = None
        namespace = {"device_operation": detection.device_operation, "deg_list_to_rad": list}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "game_helper", "exec"), namespace)
        arm, hand = Mock(), Mock()
        cause = OSError("link lost")
        arm.set_positions.side_effect = cause
        with self.assertRaises(diag.DiagnosticError) as caught:
            namespace["apply_posture"](arm, hand, (0, 0), (0,) * 6, (0,) * 6, devices={"head_arm": "/dev/ttyACM7", "can_iface": "can9"})
        self.assertIs(caught.exception.__cause__, cause)
        self.assertEqual(caught.exception.context["device"], "/dev/ttyACM7")
        hand.set_joint_pos.assert_not_called()

    def test_python38_compilation(self):
        for path in EXAMPLES.glob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path), feature_version=8)
            compile(tree, str(path), "exec")


if __name__ == "__main__":
    unittest.main()
