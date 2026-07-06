from dataclasses import dataclass, field
import math
from pathlib import Path
import time
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
import pybullet as p
import pybullet_data

from beingbeyond_d1_edu_sdk.exo import ExoDriver, format_servo_ids
from beingbeyond_d1_edu_sdk.glove_driver import GloveReader
from beingbeyond_d1_edu_sdk.urdf_path import get_default_urdf_path

from exo_glove_teleop_common import (
    ARM_NAMES,
    HEAD_NAMES,
    SIM_HAND_NAMES,
    build_arm_exo_cfg,
    build_glove_cfg,
    calibrate_arm_exo_zero,
    clamp,
    glove_norm_to_sim_hand_norm,
    transform_arm_exo_to_arm_deg,
    validate_common_cfg,
    validate_urdf_path,
)


@dataclass
class TeleopSimCfg:
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

    urdf_path: str = field(default_factory=get_default_urdf_path)
    use_gui: bool = True
    use_fixed_base: bool = True
    load_plane: bool = True
    gravity_z: float = -9.81

    arm_min_valid: int = 4
    head_init_deg: List[float] = field(default_factory=lambda: [-45.0, -45.0])
    arm_init_deg: List[float] = field(default_factory=lambda: [0.0, -90.0, 90.0, 0.0, 0.0, 0.0])
    arm_sign: List[int] = field(default_factory=lambda: [1, 1, 1, 1, 1, 1])

    zero_calibration_timeout_s: float = 10.0
    zero_calibration_poll_s: float = 0.05
    zero_calibration_stable_s: float = 3.0
    zero_calibration_max_delta_deg: float = 3.0

    loop_hz: float = 30.0
    dbg: bool = False


def validate_cfg(cfg: TeleopSimCfg) -> None:
    validate_common_cfg(cfg)
    if len(cfg.head_init_deg) != 2:
        raise ValueError("head_init_deg must contain 2 values")
    validate_urdf_path(cfg.urdf_path)


def build_name_index(robot_id: int) -> Tuple[Dict[str, int], Dict[str, Tuple[float, float]]]:
    name_to_idx: Dict[str, int] = {}
    limits: Dict[str, Tuple[float, float]] = {}
    for joint_idx in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, joint_idx)
        name = info[1].decode("utf-8")
        name_to_idx[name] = joint_idx
        limits[name] = (float(info[8]), float(info[9]))
    return name_to_idx, limits


def ensure_joints_exist(name_to_idx: Dict[str, int], names: List[str]) -> None:
    missing = [name for name in names if name not in name_to_idx]
    if missing:
        raise KeyError(f"URDF missing expected joints: {missing}")


def set_joint_position(robot_id: int, joint_idx: int, q: float) -> None:
    p.resetJointState(robot_id, joint_idx, q)
    p.setJointMotorControl2(
        robot_id,
        joint_idx,
        controlMode=p.POSITION_CONTROL,
        targetPosition=q,
    )


def start_pybullet(cfg: TeleopSimCfg) -> int:
    p.connect(p.GUI if cfg.use_gui else p.DIRECT)
    pybullet_data_path = Path(pybullet_data.getDataPath()).resolve()
    p.setAdditionalSearchPath(str(pybullet_data_path))
    p.setAdditionalSearchPath(str(Path(cfg.urdf_path).resolve().parent))
    p.setGravity(0.0, 0.0, cfg.gravity_z)
    if cfg.load_plane:
        p.loadURDF(str(pybullet_data_path / "plane.urdf"))
    return p.loadURDF(str(Path(cfg.urdf_path).resolve()), useFixedBase=cfg.use_fixed_base)


def glove_norm_to_hand_rad(norm_values: List[float], limits: Dict[str, Tuple[float, float]]) -> List[float]:
    joint_norm = glove_norm_to_sim_hand_norm(norm_values)
    q = []
    for name, value in zip(SIM_HAND_NAMES, joint_norm):
        low, high = limits[name]
        if low < high:
            q.append(low + float(value) * (high - low))
        else:
            q.append(0.0)
    return q


class TeleopSimulator:
    def __init__(self, cfg: TeleopSimCfg = TeleopSimCfg()) -> None:
        self.cfg = cfg
        self.arm_exo: Optional[ExoDriver] = None
        self.glove: Optional[GloveReader] = None
        self.robot_id: Optional[int] = None
        self.name_to_idx: Dict[str, int] = {}
        self.limits: Dict[str, Tuple[float, float]] = {}
        self.arm_zero_offsets_deg: Optional[List[float]] = None

    def close(self) -> None:
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
        try:
            if p.isConnected():
                p.disconnect()
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

    def _open_sim(self) -> None:
        self.robot_id = start_pybullet(self.cfg)
        self.name_to_idx, self.limits = build_name_index(self.robot_id)
        ensure_joints_exist(self.name_to_idx, HEAD_NAMES + ARM_NAMES + SIM_HAND_NAMES)
        self._reset_pose()
        p.setRealTimeSimulation(0)
        print("[Info] Loaded URDF:", Path(self.cfg.urdf_path).resolve())

    def _reset_pose(self) -> None:
        assert self.robot_id is not None
        for name, q in zip(HEAD_NAMES, [math.radians(v) for v in self.cfg.head_init_deg]):
            set_joint_position(self.robot_id, self.name_to_idx[name], q)
        for name, q in zip(ARM_NAMES, [math.radians(v) for v in self.cfg.arm_init_deg]):
            set_joint_position(self.robot_id, self.name_to_idx[name], q)
        for name in SIM_HAND_NAMES:
            set_joint_position(self.robot_id, self.name_to_idx[name], 0.0)

    def _apply(self, arm_q: List[float], hand_q: List[float]) -> None:
        assert self.robot_id is not None
        for name, q in zip(HEAD_NAMES, [math.radians(v) for v in self.cfg.head_init_deg]):
            p.setJointMotorControl2(self.robot_id, self.name_to_idx[name], p.POSITION_CONTROL, targetPosition=q)
        for name, q in zip(ARM_NAMES, arm_q):
            p.setJointMotorControl2(self.robot_id, self.name_to_idx[name], p.POSITION_CONTROL, targetPosition=q)
        for name, q in zip(SIM_HAND_NAMES, hand_q):
            p.setJointMotorControl2(self.robot_id, self.name_to_idx[name], p.POSITION_CONTROL, targetPosition=q)

    def run(self) -> None:
        print("=== Exo + Glove Teleop Sim ===")
        validate_cfg(self.cfg)
        self._open_inputs()
        self._open_sim()

        assert self.arm_exo is not None
        assert self.glove is not None
        assert self.arm_zero_offsets_deg is not None

        dt = 1.0 / self.cfg.loop_hz
        print("[Info] Arm exo ids:", format_servo_ids(self.cfg.arm_exo_servo_ids))
        print("Start sim teleop. Ctrl+C to exit.")

        try:
            while True:
                t0 = time.perf_counter()

                frame = self.arm_exo.read_frame()
                arm_q = [p.getJointState(self.robot_id, self.name_to_idx[name])[0] for name in ARM_NAMES]
                valid = [read.ok and read.angle_deg is not None for read in frame.reads]
                if sum(valid) >= self.cfg.arm_min_valid:
                    arm_angles_deg = [
                        float(read.angle_deg) if read.angle_deg is not None else zero
                        for read, zero in zip(frame.reads, self.arm_zero_offsets_deg)
                    ]
                    arm_deg_new = transform_arm_exo_to_arm_deg(
                        arm_angles_deg,
                        self.arm_zero_offsets_deg,
                        self.cfg.arm_init_deg,
                        self.cfg.arm_sign,
                    )
                    arm_q_new = [
                        clamp(math.radians(q), self.limits[name][0], self.limits[name][1])
                        for name, q in zip(ARM_NAMES, arm_deg_new)
                    ]
                    for i, ok in enumerate(valid):
                        if ok:
                            arm_q[i] = arm_q_new[i]

                hand_q = glove_norm_to_hand_rad(self.glove.get_norm_values(), self.limits)
                self._apply(arm_q, hand_q)

                if p.stepSimulation() is False:
                    break

                if self.cfg.dbg:
                    print(
                        "arm=", np.array2string(np.asarray(arm_q), precision=3, suppress_small=True),
                        "hand=", np.array2string(np.asarray(hand_q), precision=3, suppress_small=True),
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
            try:
                self._reset_pose()
                for _ in range(3):
                    p.stepSimulation()
                    time.sleep(0.01)
            except Exception:
                pass
            self.close()


def main() -> None:
    TeleopSimulator(TeleopSimCfg()).run()


if __name__ == "__main__":
    main()
