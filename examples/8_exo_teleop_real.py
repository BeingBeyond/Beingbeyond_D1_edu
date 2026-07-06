from dataclasses import dataclass, field
import time
import traceback
from typing import List, Optional

import numpy as np

from beingbeyond_d1_edu_sdk.exo import ExoDriver, format_servo_ids
from beingbeyond_d1_edu_sdk.glove_driver import GloveReader
from beingbeyond_d1_edu_sdk.urdf_path import get_default_urdf_path

from D1_robot import D1Robot
from exo_glove_teleop_common import (
    build_arm_exo_cfg,
    build_glove_cfg,
    calibrate_arm_exo_zero,
    glove_norm_to_real_hand,
    transform_arm_exo_to_arm_deg,
    validate_common_cfg,
)
from utils import deg_list_to_rad


@dataclass
class TeleopRealCfg:
    arm_exo_port: str = "/dev/ttyUSB0"
    arm_exo_baudrate: int = 115200
    arm_exo_servo_ids: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    arm_exo_read_hz: float = 10.0
    arm_exo_command_delay_s: float = 0.008
    release_arm_exo_torque_on_start: bool = True

    glove_port: str = "/dev/ttyACM1"
    glove_baudrate: int = 115200
    glove_timeout_s: float = 0.01
    glove_auto_calib: bool = True

    robot_arm_dev: str = "/dev/ttyACM0"
    robot_arm_baud: int = 1_000_000
    robot_hand_type: str = "right"
    robot_hand_can: str = "can0"
    robot_hand_baud: int = 1_000_000

    arm_min_valid: int = 4

    head_deg: List[float] = field(default_factory=lambda: [-15.0, -60.0])
    arm_init_deg: List[float] = field(default_factory=lambda: [0.0, -90.0, 90.0, 0.0, 0.0, 0.0])
    arm_sign: List[int] = field(default_factory=lambda: [1, 1, 1, 1, 1, 1])
    arm_limit_low_deg: List[float] = field(default_factory=lambda: [-150.0, -90.0, -90.0, -150.0, -100.0, -150.0])
    arm_limit_high_deg: List[float] = field(default_factory=lambda: [150.0, 90.0, 90.0, 150.0, 85.0, 150.0])

    hand_init_norm: List[float] = field(default_factory=lambda: [0.0] * 6)
    hand_speed: List[float] = field(default_factory=lambda: [1.0] * 6)

    zero_calibration_timeout_s: float = 5.0
    zero_calibration_poll_s: float = 0.05
    zero_calibration_stable_s: float = 3.0
    zero_calibration_max_delta_deg: float = 3.0

    loop_hz: float = 30.0
    dbg: bool = False


def validate_cfg(cfg: TeleopRealCfg) -> None:
    validate_common_cfg(cfg)
    for name, values, n in (
        ("head_deg", cfg.head_deg, 2),
        ("arm_limit_low_deg", cfg.arm_limit_low_deg, 6),
        ("arm_limit_high_deg", cfg.arm_limit_high_deg, 6),
        ("hand_init_norm", cfg.hand_init_norm, 6),
        ("hand_speed", cfg.hand_speed, 6),
    ):
        if len(values) != n:
            raise ValueError(f"{name} must contain {n} values")


class TeleopReal:
    def __init__(self, cfg: TeleopRealCfg = TeleopRealCfg()) -> None:
        self.cfg = cfg
        self.arm_exo: Optional[ExoDriver] = None
        self.glove: Optional[GloveReader] = None
        self.robot: Optional[D1Robot] = None
        self.arm_zero_offsets_deg: Optional[List[float]] = None
        self.teleop = np.zeros(14, dtype=np.float64)
        self.teleop[0:2] = np.deg2rad(np.asarray(cfg.head_deg, dtype=np.float64))
        self.teleop[2:8] = np.deg2rad(np.asarray(cfg.arm_init_deg, dtype=np.float64))
        self.teleop[8:14] = np.asarray(cfg.hand_init_norm, dtype=np.float64)

    def close(self) -> None:
        try:
            if self.robot is not None:
                self.robot.close()
        except Exception:
            pass
        try:
            if self.glove is not None:
                self.glove.close()
        except Exception:
            pass
        try:
            if self.arm_exo is not None:
                self.arm_exo.close()
        except Exception:
            pass

    def _open_inputs(self) -> None:
        self.arm_exo = ExoDriver(build_arm_exo_cfg(self.cfg))
        self.arm_exo.open()
        if self.cfg.release_arm_exo_torque_on_start:
            print("[Info] Releasing arm exo torque.")
            self.arm_exo.release_torque()

        self.glove = GloveReader(build_glove_cfg(self.cfg))
        self.arm_zero_offsets_deg = calibrate_arm_exo_zero(self.arm_exo, self.cfg)

    def _open_robot(self) -> None:
        self.robot = D1Robot(
            urdf_path=get_default_urdf_path(),
            arm_dev=self.cfg.robot_arm_dev,
            arm_baud=self.cfg.robot_arm_baud,
            hand_type=self.cfg.robot_hand_type,
            hand_can=self.cfg.robot_hand_can,
            hand_baud=self.cfg.robot_hand_baud,
        )
        self.robot.__enter__()
        self.robot.hand.set_speed(speed=list(self.cfg.hand_speed))

    def _move_to_initial_pose(self) -> None:
        assert self.robot is not None
        q_headarm = deg_list_to_rad(self.cfg.head_deg + self.cfg.arm_init_deg)
        self.robot.set_q(q_headarm + list(self.cfg.hand_init_norm))
        self.robot.head_arm.wait_until_reached(
            q_headarm,
            active_joint_indices=range(8),
            pos_tol_deg=5.0,
            vel_tol_deg_s=20.0,
            timeout_s=5.0,
        )

    def _read_arm(self) -> np.ndarray:
        assert self.arm_exo is not None
        assert self.arm_zero_offsets_deg is not None

        frame = self.arm_exo.read_frame()
        valid = [read.ok and read.angle_deg is not None for read in frame.reads]
        if sum(valid) < self.cfg.arm_min_valid:
            return self.teleop[2:8].copy()

        angles_deg = [
            float(read.angle_deg) if read.angle_deg is not None else zero
            for read, zero in zip(frame.reads, self.arm_zero_offsets_deg)
        ]
        arm_deg = transform_arm_exo_to_arm_deg(
            angles_deg,
            self.arm_zero_offsets_deg,
            self.cfg.arm_init_deg,
            self.cfg.arm_sign,
            self.cfg.arm_limit_low_deg,
            self.cfg.arm_limit_high_deg,
        )
        prev = self.teleop[2:8].copy()
        arm_rad = np.deg2rad(np.asarray(arm_deg, dtype=np.float64))
        for i, ok in enumerate(valid):
            if ok:
                prev[i] = arm_rad[i]
        return prev

    def _read_hand(self) -> np.ndarray:
        assert self.glove is not None
        return glove_norm_to_real_hand(self.glove.get_norm_values())

    def _build_teleop(self) -> np.ndarray:
        x = self.teleop.copy()
        x[0:2] = np.deg2rad(np.asarray(self.cfg.head_deg, dtype=np.float64))
        x[2:8] = self._read_arm()
        x[8:14] = self._read_hand()
        return x

    def _apply(self, x: np.ndarray) -> None:
        assert self.robot is not None
        self.robot.set_q(x.tolist())

    def run(self) -> None:
        print("=== Exo + Glove Teleop Real ===")
        print("\033[91mWARNING: Always keep the physical emergency stop button within reach.\033[0m")
        print("\033[91m         Press it immediately if the robot motion looks unsafe.\033[0m\n")

        validate_cfg(self.cfg)
        self._open_inputs()
        self._open_robot()
        self._move_to_initial_pose()

        print("[Info] Arm exo ids:", format_servo_ids(self.cfg.arm_exo_servo_ids))
        print("Start real teleop. Ctrl+C to exit.")

        dt = 1.0 / self.cfg.loop_hz
        try:
            while True:
                t0 = time.perf_counter()
                self.teleop = self._build_teleop()
                self._apply(self.teleop)

                if self.cfg.dbg:
                    print(
                        "head=", np.array2string(self.teleop[0:2], precision=3, suppress_small=True),
                        "arm=", np.array2string(self.teleop[2:8], precision=3, suppress_small=True),
                        "hand=", np.array2string(self.teleop[8:14], precision=3, suppress_small=True),
                    )

                sleep_s = dt - (time.perf_counter() - t0)
                if sleep_s > 0:
                    time.sleep(sleep_s)

        except KeyboardInterrupt:
            print("\nInterrupted by user (Ctrl+C).")
        except Exception as exc:
            print(f"Error: {exc}")
            traceback.print_exc()
        finally:
            self.close()


def main() -> None:
    TeleopReal(TeleopRealCfg()).run()


if __name__ == "__main__":
    main()
