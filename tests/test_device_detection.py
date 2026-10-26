"""Protocol and discovery regressions; these tests never access real devices."""
from contextlib import contextmanager
import itertools
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

EXAMPLES = Path(os.environ.get("D1_EXAMPLES_DIR", str(Path(__file__).resolve().parents[1] / "examples"))).resolve()
sys.path.insert(0, str(EXAMPLES))
import device_detection as detection


def response(sid, error=0):
    body = bytes((sid, 3, error, sid))
    return b"\xff\xff" + body + bytes((~sum(body) & 255,))


class SerialStub:
    def __init__(self, corrupt=False):
        self.commands = []
        self.pending = b""
        self.corrupt = corrupt

    def reset_input_buffer(self):
        self.pending = b""

    def write(self, command):
        self.commands.append(command)
        if command.startswith(b"\xff\xff"):
            self.pending = response(command[2])
            if self.corrupt:
                self.pending = self.pending[:-1] + bytes((self.pending[-1] ^ 1,))
        else:
            self.pending = command[:4] + b"P1500!"

    def read(self, size):
        # Force replies to span reads, as USB serial packets may be fragmented.
        chunk, self.pending = self.pending[:2], self.pending[2:]
        return chunk


class ProtocolTests(unittest.TestCase):
    def test_head_fault_is_not_reported_as_missing(self):
        port = SerialStub()
        def fault(command):
            body = bytes((command[2], 2, 4))
            port.pending = b"\xff\xff" + body + bytes((~sum(body) & 255,))
        port.write = fault
        with self.assertRaisesRegex(detection.DeviceDetectionError, "ID 11.*0x04.*present"):
            detection._probe_head_arm(port)

    def test_head_error_parser_rejects_query_echo_and_bad_checksum(self):
        query = bytes((11, 4, 2, 5, 1))
        echo = b"\xff\xff" + query + bytes((~sum(query) & 255,))
        self.assertIsNone(detection._head_arm_error(echo, 11))
        bad = response(11, 4)[:-1] + b"\x00"
        self.assertIsNone(detection._head_arm_error(bad, 11))

    def test_head_reply_rejects_echo_wrong_id_error_and_bad_checksum(self):
        frame = response(11)
        self.assertTrue(detection._head_arm_reply(b"noise" + frame, 11))
        self.assertFalse(detection._head_arm_reply(frame[:-1], 11))
        self.assertFalse(detection._head_arm_reply(frame, 12))
        self.assertFalse(detection._head_arm_reply(response(11, 1), 11))
        self.assertFalse(detection._head_arm_reply(frame[:-1] + b"\x00", 11))

    def test_head_probe_only_sends_id_register_reads(self):
        port = SerialStub()
        self.assertEqual(detection._probe_head_arm(port), [])
        self.assertEqual([cmd[2] for cmd in port.commands], list(detection.HEAD_ARM_IDS))
        for command in port.commands:
            self.assertEqual(command[3:7], bytes((4, 2, 5, 1)))
            self.assertEqual(sum(command[2:]) & 255, 255)

    def test_corrupted_head_responses_are_not_accepted(self):
        ticks = itertools.count(step=0.025)
        with patch.object(detection.time, "monotonic", side_effect=lambda: next(ticks)):
            self.assertEqual(detection._probe_head_arm(SerialStub(corrupt=True)), list(detection.HEAD_ARM_IDS))

    def test_glove_requires_six_finite_channels(self):
        self.assertTrue(detection._valid_glove_line("ANGLES: a=1 b=-2 c=3.5 d=4 e=5 f=6"))
        self.assertFalse(detection._valid_glove_line("OTHER: a=1 b=2 c=3 d=4 e=5 f=6"))
        self.assertFalse(detection._valid_glove_line("ANGLES: a=1 b=2"))
        self.assertFalse(detection._valid_glove_line("ANGLES: " + " a=1" * 7))

    def test_glove_handles_partial_lines_and_requires_three_frames(self):
        line = b"ANGLES: a=1 b=2 c=3 d=4 e=5 f=6\n"
        chunks = iter([line[:8], line[8:], b"noise\n", line, line, line])
        port = SimpleNamespace(read_until=lambda *args: next(chunks))
        self.assertTrue(detection._probe_glove(port))

    def test_exo_probe_uses_only_requested_position_reads(self):
        port = SerialStub()
        self.assertEqual(detection._probe_exo(port, [1, 3, 6]), [])
        self.assertEqual(port.commands, [b"#001PRAD!", b"#003PRAD!", b"#006PRAD!"])

    def test_exo_rejects_wrong_id_or_out_of_range_positions(self):
        for reply in (b"#999P1500!", b"#001P9999!", b"#001P15000!"):
            port = SerialStub()
            port.write = lambda command: setattr(port, "pending", reply)
            ticks = itertools.count(step=0.025)
            with patch.object(detection.time, "monotonic", side_effect=lambda: next(ticks)):
                self.assertEqual(detection._probe_exo(port, [1]), [1])


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.probes = []

    @contextmanager
    def open_port(self, device, baud):
        self.probes.append((device, baud))
        yield SimpleNamespace(device=device, baudrate=baud)

    def discover(self, candidates, roles=None):
        with patch.object(detection, "_serial_candidates", return_value=candidates), \
                patch.object(detection, "_open_probe", side_effect=self.open_port), \
                patch.object(detection, "_probe_glove", side_effect=lambda port: port.device == "/dev/ttyACM7"), \
                patch.object(detection, "_probe_head_arm", return_value=[]), \
                patch.object(detection, "_probe_exo", return_value=[]):
            return detection.resolve_serial_ports(roles or {"head_arm": "auto", "glove": "auto", "exo": "auto"})

    def test_reordered_ports_and_changed_serial_numbers(self):
        candidates = {
            "/dev/ttyUSB9": SimpleNamespace(vid=0x1a86, pid=0x7523, serial_number=None),
            "/dev/ttyACM7": SimpleNamespace(vid=0x1a86, pid=0x55d3, serial_number="NEW-GLOVE"),
            "/dev/ttyACM4": SimpleNamespace(vid=0x1a86, pid=0x55d3, serial_number="NEW-HEAD"),
            "/dev/ttyUSB0": SimpleNamespace(vid=0x1234, pid=0x5678),
        }
        devices = self.discover(candidates)
        self.assertEqual(devices, {"head_arm": "/dev/ttyACM4", "glove": "/dev/ttyACM7", "exo": "/dev/ttyUSB9"})
        self.assertNotIn("/dev/ttyUSB0", [device for device, _ in self.probes])

    def test_detected_glove_is_never_sent_head_queries(self):
        with patch.object(detection, "_probe_head_arm") as head_probe:
            with patch.object(detection, "_serial_candidates", return_value={
                "/dev/ttyACM7": SimpleNamespace(vid=0x1a86, pid=0x55d3)}), \
                    patch.object(detection, "_open_probe", side_effect=self.open_port), \
                    patch.object(detection, "_probe_glove", return_value=True):
                with self.assertRaises(detection.DeviceDetectionError):
                    detection.resolve_serial_ports({"head_arm": "auto"})
            head_probe.assert_not_called()

    def test_wrong_role_error_includes_actual_candidate_port(self):
        with patch.object(detection, "_serial_candidates", return_value={
                "/dev/ttyACM7": SimpleNamespace(vid=0x1a86, pid=0x55d3)}), \
                patch.object(detection, "_open_probe", side_effect=self.open_port), \
                patch.object(detection, "_probe_glove", return_value=True):
            with self.assertRaisesRegex(detection.DeviceDetectionError, "/dev/ttyACM7: identified as glove"):
                detection.resolve_serial_ports({"head_arm": "auto"})

    def test_unsupported_serial_is_listed_without_opening_it(self):
        with self.assertRaisesRegex(detection.DeviceDetectionError, "/dev/ttyUSB8: USB 1234:5678 excluded"):
            self.discover({"/dev/ttyUSB8": SimpleNamespace(vid=0x1234, pid=0x5678)}, {"exo": "auto"})
        self.assertEqual(self.probes, [])

    def test_explicit_override_is_verified_without_usb_filter(self):
        with patch.object(detection, "_serial_candidates") as inventory, \
                patch.object(detection, "_open_probe", side_effect=self.open_port), \
                patch.object(detection, "_probe_exo", return_value=[]):
            self.assertEqual(detection.resolve_serial_ports({"exo": "/dev/custom"}, verify_explicit=True),
                             {"exo": "/dev/custom"})
        inventory.assert_not_called()
        self.assertEqual(self.probes, [("/dev/custom", 115200)])

    def test_explicit_failure_keeps_role_port_stage_and_permission(self):
        with patch.object(detection, "_open_probe", side_effect=PermissionError("permission denied")):
            with self.assertRaisesRegex(detection.DeviceDetectionError, "exo port=/dev/custom stage=explicit serial verification.*permission denied"):
                detection.resolve_serial_ports({"exo": "/dev/custom"}, verify_explicit=True)

    def test_explicit_alias_conflict_fails_before_any_probe(self):
        with patch.object(detection.os.path, "realpath", return_value="/dev/ttyUSB4"), \
                patch.object(detection, "_serial_candidates") as inventory:
            with self.assertRaisesRegex(detection.DeviceDetectionError, "/dev/ttyUSB4 -> exo, glove"):
                detection.resolve_serial_ports({"exo": "/alias", "glove": "/dev/ttyUSB4"})
        inventory.assert_not_called()

    def test_passive_errors_are_aggregated_and_serial_skip_is_explicit(self):
        with patch.object(detection, "resolve_can_iface", side_effect=detection.DeviceDetectionError("missing CAN")), \
                patch.object(detection, "resolve_camera_device", side_effect=detection.DeviceDetectionError("missing camera")), \
                patch.object(detection, "resolve_serial_ports") as probe:
            with self.assertRaises(detection.DeviceDetectionError) as caught:
                detection.resolve_devices(head_arm="auto", can_iface="auto", camera="auto", check_ready=True)
        message = str(caught.exception)
        self.assertIn("missing CAN", message)
        self.assertIn("missing camera", message)
        self.assertIn("serial verification skipped", message)
        self.assertIn("head_arm=auto", message)
        probe.assert_not_called()

    def test_communication_failures_are_aggregated_before_serial_probes(self):
        with patch.object(detection, "check_can", side_effect=OSError("bus unavailable")), \
                patch.object(detection, "check_camera", side_effect=RuntimeError("capture failed")), \
                patch.object(detection, "resolve_serial_ports") as probe:
            with self.assertRaises(detection.DeviceDetectionError) as caught:
                detection.resolve_devices(head_arm="auto", can_iface="can7", camera="/dev/video8", check_ready=True)
        self.assertIn("can_iface port=can7 stage=communication check: bus unavailable", str(caught.exception))
        self.assertIn("camera port=/dev/video8 stage=communication check: capture failed", str(caught.exception))
        probe.assert_not_called()

    def test_readiness_forwards_hand_side_bitrate_and_explicit_verification(self):
        with patch.object(detection, "check_can") as can_check, \
                patch.object(detection, "resolve_serial_ports", return_value={"head_arm": "/dev/head"}) as serial_probe:
            devices = detection.resolve_devices(head_arm="/dev/head", can_iface="can7", hand_type="left",
                                                hand_baud=500000, check_ready=True)
        can_check.assert_called_once_with("can7", "left", 500000)
        self.assertTrue(serial_probe.call_args[1]["verify_explicit"])
        self.assertEqual(devices["head_arm"], "/dev/head")

    def test_two_matching_exos_are_ambiguous(self):
        info = SimpleNamespace(vid=0x1a86, pid=0x7523)
        with self.assertRaisesRegex(detection.DeviceDetectionError, "ambiguous"):
            self.discover({"/dev/ttyUSB1": info, "/dev/ttyUSB2": info}, {"exo": "auto"})

    def test_busy_candidate_cannot_hide_ambiguity(self):
        info = SimpleNamespace(vid=0x1a86, pid=0x7523)
        @contextmanager
        def busy(device, baud):
            if device.endswith("1"):
                raise detection.DeviceDetectionError("busy PID 123")
            yield SimpleNamespace(device=device)
        with patch.object(detection, "_serial_candidates", return_value={"/dev/ttyUSB1": info, "/dev/ttyUSB2": info}), \
                patch.object(detection, "_open_probe", side_effect=busy), \
                patch.object(detection, "_probe_exo", return_value=[]):
            with self.assertRaisesRegex(detection.DeviceDetectionError, "busy"):
                detection.resolve_serial_ports({"exo": "auto"})

    def test_missing_device_and_partial_head_fail(self):
        with self.assertRaisesRegex(detection.DeviceDetectionError, "no verified device"):
            self.discover({}, {"exo": "auto"})
        with patch.object(detection, "_serial_candidates", return_value={"/dev/ttyACM0": SimpleNamespace(vid=0x1a86, pid=0x55d3)}), \
                patch.object(detection, "_open_probe", side_effect=self.open_port), \
                patch.object(detection, "_probe_glove", return_value=False), \
                patch.object(detection, "_probe_head_arm", return_value=[26]):
            with self.assertRaisesRegex(detection.DeviceDetectionError, "26"):
                detection.resolve_serial_ports({"head_arm": "auto"})

    def test_duplicate_explicit_paths_are_rejected(self):
        with self.assertRaisesRegex(detection.DeviceDetectionError, "multiple roles"):
            detection.resolve_serial_ports({"head_arm": "/same", "glove": "/same"})

    def test_serial_aliases_are_deduplicated(self):
        infos = [SimpleNamespace(device="/dev/ttyUSB0"), SimpleNamespace(device="/alias")]
        with patch.object(detection.list_ports, "comports", return_value=infos), \
                patch.object(detection.os.path, "realpath", return_value="/dev/ttyUSB0"):
            self.assertEqual(len(detection._serial_candidates()), 1)

    def test_open_probe_refuses_existing_legacy_process(self):
        with patch.object(detection, "_port_users", return_value=["123"]), \
                patch.object(detection.serial, "Serial") as serial_open:
            with self.assertRaisesRegex(detection.DeviceDetectionError, "123"):
                with detection._open_probe("/dev/ttyUSB0", 115200):
                    pass
            serial_open.assert_not_called()

    def test_proc_scan_finds_a_nonexclusive_open_terminal(self):
        master, slave = os.openpty()
        try:
            self.assertIn(str(os.getpid()), detection._port_users(os.ttyname(slave)))
        finally:
            os.close(master)
            os.close(slave)

    def test_no_devices_are_opened_when_passive_preflight_fails(self):
        with patch.object(detection, "resolve_can_iface", side_effect=detection.DeviceDetectionError("missing CAN")), \
                patch.object(detection, "resolve_serial_ports") as serial_probe:
            with self.assertRaises(detection.DeviceDetectionError):
                detection.resolve_devices(head_arm="auto", can_iface="auto")
            serial_probe.assert_not_called()


class PeripheralTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def path(self, name):
        return self.root / name.lstrip("/")

    def usb_node(self, name, vid, pid, **attrs):
        path = self.path(name)
        (path / "device").mkdir(parents=True)
        (path / "device/idVendor").write_text(vid)
        (path / "device/idProduct").write_text(pid)
        for attr, value in attrs.items():
            (path / attr).write_text(value)
        return path

    def test_can_discovers_renumbered_adapter_and_rejects_duplicates(self):
        self.usb_node("/sys/class/net/can7", "0c72", "000c", type="280")
        self.usb_node("/sys/class/net/can0", "1234", "5678", type="280")
        with patch.object(detection, "Path", side_effect=self.path):
            self.assertEqual(detection.resolve_can_iface(), "can7")
        self.usb_node("/sys/class/net/can9", "0c72", "000c", type="280")
        with patch.object(detection, "Path", side_effect=self.path):
            with self.assertRaisesRegex(detection.DeviceDetectionError, "ambiguous"):
                detection.resolve_can_iface()

    def test_camera_filters_builtin_and_secondary_video_nodes(self):
        self.usb_node("/sys/class/video4linux/video8", "15aa", "1955", index="0")
        self.usb_node("/sys/class/video4linux/video9", "15aa", "1955", index="1")
        self.usb_node("/sys/class/video4linux/video0", "04f2", "b7b8", index="0")
        with patch.object(detection, "Path", side_effect=self.path):
            self.assertEqual(detection.resolve_camera_device(), "/dev/video8")
        self.usb_node("/sys/class/video4linux/video12", "15aa", "1955", index="0")
        with patch.object(detection, "Path", side_effect=self.path):
            with self.assertRaisesRegex(detection.DeviceDetectionError, "ambiguous"):
                detection.resolve_camera_device()

    def test_can_check_sends_only_status_request(self):
        import json
        import struct
        from unittest.mock import MagicMock
        channel = MagicMock()
        channel.recv.return_value = struct.pack("=IB3x8s", 0x27, 7, b"\x01\x10\x20\x30\x40\x50\x60")
        channel.__enter__.return_value = channel
        status = [{"flags": ["UP"], "linkinfo": {"info_data": {"bittiming": {"bitrate": 1000000}}}}]
        with patch.object(detection.socket, "socket", return_value=channel), \
                patch.object(detection.subprocess, "run", return_value=SimpleNamespace(stdout=json.dumps(status))):
            detection.check_can("can7")
        channel.send.assert_called_once_with(struct.pack("=IB3x8s", 0x27, 1, b"\x01"))

    def test_can_check_does_not_reconfigure_a_down_link(self):
        with patch.object(detection.subprocess, "run", return_value=SimpleNamespace(stdout='[{"flags": []}]')) as command, \
                patch.object(detection.socket, "socket") as connect:
            with self.assertRaisesRegex(detection.DeviceDetectionError, "already be UP"):
                detection.check_can("can7")
            connect.assert_not_called()
            self.assertEqual(command.call_args[0][0], ["ip", "-json", "-details", "link", "show", "dev", "can7"])

    def test_can_bus_off_is_rejected_before_sending(self):
        status = '[{"flags":["UP"],"linkinfo":{"info_data":{"state":"BUS-OFF","bittiming":{"bitrate":1000000}}}}]'
        with patch.object(detection.subprocess, "run", return_value=SimpleNamespace(stdout=status)), \
                patch.object(detection.socket, "socket") as connect:
            with self.assertRaisesRegex(detection.DeviceDetectionError, "hand CAN port=can7.*BUS-OFF"):
                detection.check_can("can7")
        connect.assert_not_called()

    def test_missing_ip_command_keeps_interface_and_stage(self):
        with patch.object(detection.subprocess, "run", side_effect=FileNotFoundError("ip not installed")):
            with self.assertRaisesRegex(detection.DeviceDetectionError, "hand CAN port=can7 stage=status check.*ip not installed"):
                detection.check_can("can7")


if __name__ == "__main__":
    unittest.main()
