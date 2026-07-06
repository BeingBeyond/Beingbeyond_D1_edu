import math
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from beingbeyond_d1_edu_sdk.exo import ExoDriver, ExoDriverCfg
from beingbeyond_d1_edu_sdk.glove_driver import GloveConfig


RED = "\033[31m"
RESET = "\033[0m"

HEAD_NAMES = ["joint_7_head_yaw", "joint_8_head_pitch"]
ARM_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
SIM_HAND_NAMES = [
    "thumb_cmc_yaw",
    "thumb_cmc_pitch",
    "thumb_ip",
    "index_mcp_pitch",
    "index_dip",
    "middle_mcp_pitch",
    "middle_dip",
    "ring_mcp_pitch",
    "ring_dip",
    "pinky_mcp_pitch",
    "pinky_dip",
]


def print_red(message: str, end: str = "\n") -> None:
    print(f"{RED}{message}{RESET}", end=end, flush=True)


def require_len(name: str, values: Sequence[object], expected: int) -> None:
    if len(values) != expected:
        raise ValueError(f"{name} must contain {expected} values")


def validate_common_cfg(cfg) -> None:
    require_len("arm_exo_servo_ids", cfg.arm_exo_servo_ids, 6)
    require_len("arm_init_deg", cfg.arm_init_deg, 6)
    require_len("arm_sign", cfg.arm_sign, 6)
    if any(sign == 0 for sign in cfg.arm_sign):
        raise ValueError("arm_sign values must be non-zero")
    if cfg.loop_hz <= 0.0:
        raise ValueError("loop_hz must be positive")


def validate_urdf_path(urdf_path: str) -> None:
    if not Path(urdf_path).is_file():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")


def build_arm_exo_cfg(cfg) -> ExoDriverCfg:
    return ExoDriverCfg(
        port=cfg.arm_exo_port,
        baudrate=cfg.arm_exo_baudrate,
        servo_ids=list(cfg.arm_exo_servo_ids),
        read_hz=cfg.arm_exo_read_hz,
        command_delay_s=cfg.arm_exo_command_delay_s,
    )


def build_glove_cfg(cfg) -> GloveConfig:
    return GloveConfig(
        port=cfg.glove_port,
        baudrate=cfg.glove_baudrate,
        timeout=cfg.glove_timeout_s,
        use_auto_calib=cfg.glove_auto_calib,
        offset_deg=[60.0, -70.0, 120.0, 120.0, 120.0, 120.0]
    )


def all_angles_valid(angles: List[Optional[float]]) -> bool:
    return all(angle is not None for angle in angles)


def to_float_angles(angles: List[Optional[float]]) -> List[float]:
    return [float(angle) for angle in angles if angle is not None]


def max_angle_delta_deg(current: List[float], previous: List[float]) -> float:
    return max(abs(angle - last_angle) for angle, last_angle in zip(current, previous))


def calibrate_arm_exo_zero(exo: ExoDriver, cfg) -> List[float]:
    print_red("ARM EXO ZERO CALIBRATION")
    print_red("Move the arm exo to the zero position and hold still.")

    deadline_s = time.monotonic() + cfg.zero_calibration_timeout_s
    previous_angles: Optional[List[float]] = None
    stable_since_s: Optional[float] = None
    last_countdown_s: Optional[int] = None

    while time.monotonic() < deadline_s:
        angles = exo.read_frame().angles_deg
        if not all_angles_valid(angles):
            previous_angles = None
            stable_since_s = None
            last_countdown_s = None
            print_red("\rWaiting for valid arm exo readings...                    ", end="")
            time.sleep(cfg.zero_calibration_poll_s)
            continue

        current_angles = to_float_angles(angles)
        now_s = time.monotonic()
        if previous_angles is None:
            previous_angles = current_angles
            stable_since_s = now_s
            time.sleep(cfg.zero_calibration_poll_s)
            continue

        delta_deg = max_angle_delta_deg(current_angles, previous_angles)
        if delta_deg <= cfg.zero_calibration_max_delta_deg:
            if stable_since_s is None:
                stable_since_s = now_s
            stable_elapsed_s = now_s - stable_since_s
            remaining_s = max(0.0, cfg.zero_calibration_stable_s - stable_elapsed_s)
            countdown_s = int(math.ceil(remaining_s))
            if countdown_s != last_countdown_s:
                print_red(f"\rHold still... {countdown_s}s remaining.                 ", end="")
                last_countdown_s = countdown_s
            if stable_elapsed_s >= cfg.zero_calibration_stable_s:
                print_red("\rArm exo zero calibration complete.                       ")
                print("arm_zero_offsets_deg:", [round(v, 3) for v in current_angles])
                return current_angles
        else:
            stable_since_s = now_s
            last_countdown_s = None
            print_red(f"\rMovement detected ({delta_deg:.2f} deg). Hold still.     ", end="")

        previous_angles = current_angles
        time.sleep(cfg.zero_calibration_poll_s)

    raise RuntimeError("Timed out while calibrating arm exo zero.")


def clamp(value: float, low: float, high: float) -> float:
    if low < high:
        return max(low, min(high, value))
    return value


def transform_arm_exo_to_arm_deg(
    angles_deg: Sequence[float],
    zero_offsets_deg: Sequence[float],
    arm_init_deg: Sequence[float],
    arm_sign: Sequence[int],
    limit_low_deg: Optional[Sequence[float]] = None,
    limit_high_deg: Optional[Sequence[float]] = None,
) -> List[float]:
    target = []
    for i, (q0, angle, zero, sign) in enumerate(
        zip(arm_init_deg, angles_deg, zero_offsets_deg, arm_sign)
    ):
        q = float(q0) + float(sign) * (float(angle) - float(zero))
        if limit_low_deg is not None and limit_high_deg is not None:
            q = clamp(q, float(limit_low_deg[i]), float(limit_high_deg[i]))
        target.append(q)
    return target


def glove_norm_to_real_hand(norm_values: Sequence[float]) -> np.ndarray:
    return np.clip(np.asarray(norm_values, dtype=np.float64).reshape(6), 0.0, 1.0)


def glove_norm_to_sim_hand_norm(norm_values: Sequence[float]) -> List[float]:
    x = glove_norm_to_real_hand(norm_values)
    # DexHand 6-channel order is thumb pitch, thumb yaw, index, middle, ring, pinky.
    # The URDF hand starts with thumb yaw, then thumb pitch and thumb IP.
    return [
        float(x[1]),
        float(x[0]),
        float(x[0]),
        float(x[2]),
        float(x[2]),
        float(x[3]),
        float(x[3]),
        float(x[4]),
        float(x[4]),
        float(x[5]),
        float(x[5]),
    ]
