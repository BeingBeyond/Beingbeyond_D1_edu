"""
Example 10: Rock-paper-scissors on the real DexHand.

Pipeline:
  USB stereo RGB camera -> MediaPipe right-hand landmarks -> dex-retargeting qpos
  -> classify human rock/paper/scissors from five finger bend thresholds
  -> command the real right DexHand to play the winning gesture.

This example moves the head + arm to an initial pose, then controls the
DexHand during the game. Keep the physical emergency stop button within reach.
Press q in the OpenCV camera window to exit.
"""

import importlib
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from beingbeyond_d1_edu_sdk import HeadArmRobot
from beingbeyond_d1_edu_sdk.hand import DexHand
from beingbeyond_d1_edu_sdk.urdf_path import get_default_urdf_path

from vision import StereoRGBCamera
from utils import deg_list_to_rad


EXAMPLES_DIR = Path(__file__).resolve().parent
D1_EDU_DIR = EXAMPLES_DIR.parent
DEX_RETARGETING_DIR = D1_EDU_DIR / "lib" / "dex-retargeting"
DEX_VECTOR_EXAMPLE_DIR = DEX_RETARGETING_DIR / "example" / "vector_retargeting"

sys.path.append(str(DEX_RETARGETING_DIR))
sys.path.append(str(DEX_VECTOR_EXAMPLE_DIR))

from dex_retargeting.constants import RobotName, RetargetingType, HandType, get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig
from single_hand_detector import SingleHandDetector


FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")
GESTURE_LABELS = {
    "rock": "ROCK",
    "paper": "PAPER",
    "scissors": "SCISSORS",
    "unknown": "UNKNOWN",
}


@dataclass(frozen=True)
class StereoCfg:
    device: str = "/dev/v4l/by-id/usb-SunplusIT_Inc_SPCA2100_PC_Camera-video-index0"
    single_width: int = 640
    single_height: int = 480
    fps: int = 30
    fourcc: str = "MJPG"
    swap_lr: bool = False
    view: str = "both"


@dataclass(frozen=True)
class HeadArmCfg:
    dev: str = "/dev/ttyACM0"
    baudrate: int = 1_000_000
    q_init_deg: Tuple[float, ...] = (
        0.0, -15.0,
         20.0, -60.0, 50.0, 0.0, 40.0, 30.0,
    )
    wait_pos_tol_deg: float = 10.0
    wait_vel_tol_deg_s: float = 20.0
    wait_timeout_s: float = 3.0


@dataclass(frozen=True)
class HandCfg:
    hand_type: str = "right"
    can_iface: str = "can0"
    baudrate: int = 1_000_000
    speed: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    torque: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    open_value: float = 0.0
    closed_value: float = 1.0
    thumb_pitch_value: float = 0.7
    thumb_yaw_value: float = 0.5
    non_thumb_delay_s: float = 0.15
    thumb_yaw_delay_s: float = 0.05
    thumb_pitch_delay_s: float = 0.2


@dataclass(frozen=True)
class RpsArmCfg:
    enabled: bool = True
    wait: bool = False
    wait_pos_tol_deg: float = 8.0
    wait_vel_tol_deg_s: float = 15.0
    wait_timeout_s: float = 2.0


@dataclass(frozen=True)
class RpsCfg:
    thumb_threshold: float = 0.45
    index_threshold: float = 0.45
    middle_threshold: float = 0.45
    ring_threshold: float = 0.35
    pinky_threshold: float = 0.30
    stable_frames: int = 3
    unknown_reset_frames: int = 2

    @property
    def thresholds(self) -> Dict[str, float]:
        return {
            "thumb": self.thumb_threshold,
            "index": self.index_threshold,
            "middle": self.middle_threshold,
            "ring": self.ring_threshold,
            "pinky": self.pinky_threshold,
        }


@dataclass(frozen=True)
class IdleCfg:
    enabled: bool = False
    trigger_after_s: float = 2.0
    dance_interval_s: float = 1.0
    dance_steps: int = 4
    thumbs_up_hold_s: float = 2.0
    wait_pos_tol_deg: float = 5.0
    wait_vel_tol_deg_s: float = 5.0
    wait_timeout_s: float = 3.0


@dataclass
class StableGesture:
    last_raw: str = "unknown"
    last_stable: str = "unknown"
    count: int = 0
    unknown_count: int = 0

    def update(self, raw: str, stable_frames: int, unknown_reset_frames: int) -> str:
        if raw == self.last_raw:
            self.count += 1
        else:
            self.last_raw = raw
            self.count = 1

        if raw == "unknown":
            self.unknown_count += 1
            if self.unknown_count >= unknown_reset_frames:
                self.last_stable = "unknown"
            return self.last_stable

        self.unknown_count = 0
        if self.count >= stable_frames:
            self.last_stable = raw
        return self.last_stable


def init_right_hand_retargeting():
    robot_name = RobotName.d1hand

    dex_pkg = importlib.util.find_spec("dex_retargeting")
    if dex_pkg and dex_pkg.origin:
        robot_dir = Path(dex_pkg.origin).absolute().parent.parent / "assets" / "robots" / "hands"
    else:
        robot_dir = DEX_RETARGETING_DIR / "assets" / "robots" / "hands"
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))

    config_path = get_default_config_path(robot_name, RetargetingType.dexpilot, HandType.right)
    return RetargetingConfig.load_from_file(config_path).build()


def retarget_joint_pos(retargeting, joint_pos: np.ndarray) -> np.ndarray:
    retargeting_type = retargeting.optimizer.retargeting_type
    indices = retargeting.optimizer.target_link_human_indices

    if retargeting_type == "POSITION":
        ref_value = joint_pos[indices, :]
    else:
        origin_indices = indices[0, :]
        task_indices = indices[1, :]
        ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]

    return retargeting.retarget(ref_value)


def infer_finger_indices(joint_names) -> Dict[str, Tuple[int, ...]]:
    aliases = {
        "thumb": ("thumb",),
        "index": ("index",),
        "middle": ("middle",),
        "ring": ("ring",),
        "pinky": ("pinky", "little"),
    }
    indices = {finger: [] for finger in FINGER_ORDER}
    for i, name in enumerate(joint_names):
        lname = name.lower()
        for finger, keys in aliases.items():
            if any(key in lname for key in keys):
                indices[finger].append(i)

    missing = [finger for finger, values in indices.items() if not values]
    if missing:
        raise RuntimeError(
            f"Cannot infer finger qpos indices for {missing}. "
            f"Retargeting joints are: {list(joint_names)}"
        )
    return {finger: tuple(values) for finger, values in indices.items()}


def finger_bend_scores(qpos: np.ndarray, finger_indices: Dict[str, Tuple[int, ...]]) -> Dict[str, float]:
    scores = {}
    for finger, indices in finger_indices.items():
        values = np.abs(qpos[list(indices)])
        scores[finger] = float(np.max(values))
    return scores


def classify_rps(scores: Dict[str, float], thresholds: Dict[str, float]) -> str:
    bent = {finger: scores[finger] >= thresholds[finger] for finger in FINGER_ORDER}
    non_thumb = ("index", "middle", "ring", "pinky")
    bent_non_thumb_count = sum(1 for finger in non_thumb if bent[finger])
    index_open = scores["index"] < thresholds["index"] * 0.55
    middle_open = scores["middle"] < thresholds["middle"] * 0.55

    if (
        not bent["index"]
        and not bent["middle"]
        and not bent["ring"]
        and not bent["pinky"]
    ):
        return "paper"

    if (
        scores["index"] < thresholds["index"] * 0.95
        and scores["middle"] < thresholds["middle"] * 0.95
        and scores["ring"] < thresholds["ring"] * 1.25
        and scores["pinky"] < thresholds["pinky"] * 1.25
    ):
        return "paper"

    if (
        index_open
        and middle_open
        and bent["ring"]
        and bent["pinky"]
    ):
        return "scissors"

    if bent_non_thumb_count >= 3:
        return "rock"

    if bent["index"] and bent["middle"] and (bent["ring"] or bent["pinky"]):
        return "rock"

    if (
        index_open
        and middle_open
        and (bent["ring"] or bent["pinky"])
        and scores["thumb"] >= thresholds["thumb"] * 0.8
    ):
        return "scissors"

    if (
        scores["index"] < thresholds["index"] * 0.35
        and scores["middle"] < thresholds["middle"] * 0.35
        and scores["ring"] < thresholds["ring"] * 0.45
        and scores["pinky"] < thresholds["pinky"] * 0.45
    ):
        return "paper"

    return "unknown"


def winning_gesture(human_gesture: str) -> Optional[str]:
    return {
        "rock": "paper",
        "paper": "scissors",
        "scissors": "rock",
    }.get(human_gesture)


def build_real_hand_gesture_sequences(cfg: HandCfg) -> Dict[str, Tuple[Tuple[np.ndarray, float], ...]]:
    open_value = float(cfg.open_value)
    closed_value = float(cfg.closed_value)
    thumb_pitch = float(cfg.thumb_pitch_value)
    thumb_yaw = float(cfg.thumb_yaw_value)

    return {
        "paper": ((np.full(6, open_value, dtype=np.float64), 0.0),),
        # DexHand order: thumb pitch, thumb yaw, index, middle, ring, pinky.
        "rock": (
            (np.asarray([open_value, open_value, closed_value, closed_value, closed_value, closed_value], dtype=np.float64), cfg.non_thumb_delay_s),
            (np.asarray([open_value, thumb_yaw, closed_value, closed_value, closed_value, closed_value], dtype=np.float64), cfg.thumb_yaw_delay_s),
            (np.asarray([thumb_pitch, thumb_yaw, closed_value, closed_value, closed_value, closed_value], dtype=np.float64), cfg.thumb_pitch_delay_s),
        ),
        "scissors": (
            (np.asarray([open_value, open_value, open_value, open_value, closed_value, closed_value], dtype=np.float64), cfg.non_thumb_delay_s),
            (np.asarray([open_value, thumb_yaw, open_value, open_value, closed_value, closed_value], dtype=np.float64), cfg.thumb_yaw_delay_s),
            (np.asarray([thumb_pitch, thumb_yaw, open_value, open_value, closed_value, closed_value], dtype=np.float64), cfg.thumb_pitch_delay_s),
        ),
    }


def apply_real_hand_gesture_sequence(hand: DexHand, sequence: Tuple[Tuple[np.ndarray, float], ...]) -> None:
    for qpos, delay_s in sequence:
        hand.set_joint_pos(np.clip(qpos, 0.0, 1.0).tolist())
        if delay_s > 0.0:
            time.sleep(delay_s)


def build_rps_head_arm_postures() -> Dict[str, Tuple[Tuple[float, ...], Tuple[float, ...]]]:
    return {
        "paper": (
            (0.0, -10.0),
            (20.0, -60.0, 50.0, 0.0, 40.0, 20.0),
        ),
        "rock": (
            (-5.0, -25.0),
            (25.0, -65.0, 60.0, 30.0, 45.0, -20.0),
        ),
        "scissors": (
            (5.0, -15.0),
            (30.0, -55.0, 45.0, 0.0, -20.0, 0.0),
        ),
    }


def apply_rps_head_arm_posture(
    head_arm: HeadArmRobot,
    gesture: str,
    postures: Dict[str, Tuple[Tuple[float, ...], Tuple[float, ...]]],
    cfg: RpsArmCfg,
) -> None:
    if not cfg.enabled:
        return

    posture = postures.get(gesture)
    if posture is None:
        return

    head_deg, arm_deg = posture
    head_arm_q = deg_list_to_rad(list(head_deg) + list(arm_deg))
    head_arm.set_positions(head_arm_q)

    if cfg.wait:
        head_arm.wait_until_reached(
            head_arm_q,
            pos_tol_deg=cfg.wait_pos_tol_deg,
            vel_tol_deg_s=cfg.wait_vel_tol_deg_s,
            timeout_s=cfg.wait_timeout_s,
        )


def apply_posture(
    head_arm: HeadArmRobot,
    hand: DexHand,
    head_deg: Tuple[float, ...],
    arm_deg: Tuple[float, ...],
    hand_q_norm: Tuple[float, ...],
    wait: bool = False,
    idle_cfg: Optional[IdleCfg] = None,
) -> Optional[float]:
    head_q = deg_list_to_rad(list(head_deg))
    arm_q = deg_list_to_rad(list(arm_deg))
    head_arm_q = head_q + arm_q

    head_arm.set_positions(head_arm_q)
    hand.set_joint_pos(list(hand_q_norm))

    if not wait:
        return None

    wait_cfg = idle_cfg or IdleCfg()
    return head_arm.wait_until_reached(
        head_arm_q,
        pos_tol_deg=wait_cfg.wait_pos_tol_deg,
        vel_tol_deg_s=wait_cfg.wait_vel_tol_deg_s,
        timeout_s=wait_cfg.wait_timeout_s,
    )


def apply_thumbs_up_idle_posture(head_arm: HeadArmRobot, hand: DexHand, idle_cfg: IdleCfg) -> None:
    elapsed = apply_posture(
        head_arm,
        hand,
        head_deg=(0.0, 0.0),
        arm_deg=(20.0, -60.0, 50.0, 90.0, 70.0, 10.0),
        hand_q_norm=(0.0, 0.0, 1.0, 1.0, 1.0, 1.0),
        wait=True,
        idle_cfg=idle_cfg,
    )
    if elapsed is not None:
        print(f"Reached idle thumbs-up posture in {elapsed:.2f} s.")
    else:
        print("Idle thumbs-up posture NOT reached (timeout or no progress).")


class IdleMotion:
    def __init__(
        self,
        cfg: IdleCfg,
        ready_head_arm_deg: Tuple[float, ...],
        ready_hand_q_norm: Tuple[float, ...],
    ):
        self.cfg = cfg
        self.ready_head_arm_deg = ready_head_arm_deg
        self.ready_hand_q_norm = ready_hand_q_norm
        self.mode: Optional[str] = None
        self.next_step_time = 0.0
        self.dance_index = 0
        self.dance_steps_done = 0
        self.completed = False

    def reset(self) -> None:
        self.mode = None
        self.next_step_time = 0.0
        self.dance_index = 0
        self.dance_steps_done = 0
        self.completed = False

    def _move_to_ready_posture(self, head_arm: HeadArmRobot, hand: DexHand) -> None:
        apply_posture(
            head_arm,
            hand,
            head_deg=self.ready_head_arm_deg[:2],
            arm_deg=self.ready_head_arm_deg[2:],
            hand_q_norm=self.ready_hand_q_norm,
            wait=True,
            idle_cfg=self.cfg,
        )
        self.completed = True
        print("Idle action finished. Back to rock-paper-scissors ready posture.")

    def update(self, head_arm: HeadArmRobot, hand: DexHand, now: float) -> Optional[str]:
        if self.completed:
            return None

        if self.mode is None:
            self.mode = random.choice(("dance", "thumbs_up"))
            self.next_step_time = 0.0
            self.dance_index = 0
            self.dance_steps_done = 0
            print(f"No active rock-paper-scissors gesture detected. Idle action: {self.mode}.")

        if self.mode == "thumbs_up":
            if self.next_step_time == 0.0:
                apply_thumbs_up_idle_posture(head_arm, hand, self.cfg)
                self.next_step_time = now + self.cfg.thumbs_up_hold_s
            elif now >= self.next_step_time:
                self._move_to_ready_posture(head_arm, hand)
            return self.mode

        if now < self.next_step_time:
            return self.mode

        if self.dance_steps_done >= self.cfg.dance_steps:
            self._move_to_ready_posture(head_arm, hand)
            return self.mode

        dance_postures = (
            (
                (-25.0, 35.0),
                (30.0, -50.0, 30.0, 0.0, -30.0, 0.0),
                (0.0, 1.0, 0.0, 1.0, 1.0, 0.0),
            ),
            (
                (-25.0, -15.0),
                (30.0, -70.0, 60.0, 0.0, 30.0, 0.0),
                (0.0, 1.0, 0.0, 1.0, 1.0, 0.0),
            ),
        )
        head_deg, arm_deg, hand_q_norm = dance_postures[self.dance_index]
        apply_posture(head_arm, hand, head_deg, arm_deg, hand_q_norm)
        self.dance_index = (self.dance_index + 1) % len(dance_postures)
        self.dance_steps_done += 1
        self.next_step_time = now + self.cfg.dance_interval_s
        return self.mode


def move_head_arm_to_initial_pose(head_arm: HeadArmRobot, cfg: HeadArmCfg) -> None:
    q_init_deg = list(cfg.q_init_deg)
    if len(q_init_deg) != 8:
        raise ValueError("q_init_deg must contain 8 values")

    q_init_rad = deg_list_to_rad(q_init_deg)
    print(f"\n[Step] Move head + arm to initial pose (deg): {q_init_deg}")
    head_arm.set_positions(q_init_rad)
    head_arm.wait_until_reached(
        q_init_rad,
        active_joint_indices=range(8),
        pos_tol_deg=cfg.wait_pos_tol_deg,
        vel_tol_deg_s=cfg.wait_vel_tol_deg_s,
        timeout_s=cfg.wait_timeout_s,
    )
    time.sleep(0.3)


def draw_overlay(
    image: np.ndarray,
    fps: float,
    human: str,
    robot: Optional[str],
    scores: Optional[Dict[str, float]],
    thresholds: Dict[str, float],
) -> np.ndarray:
    out = image.copy()
    lines = [
        f"FPS: {fps:.1f}",
        f"Human: {GESTURE_LABELS.get(human, human)}",
        f"DexHand: {GESTURE_LABELS.get(robot, robot) if robot else '-'}",
    ]
    if scores is not None:
        score_text = " ".join(
            f"{finger[0]}:{scores[finger]:.2f}/{thresholds[finger]:.2f}"
            for finger in FINGER_ORDER
        )
        lines.append(score_text)
    lines.append("q: quit")

    for i, line in enumerate(lines):
        y = 28 + i * 26
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    stereo_cfg = StereoCfg()
    head_arm_cfg = HeadArmCfg()
    hand_cfg = HandCfg()
    rps_arm_cfg = RpsArmCfg()
    rps_cfg = RpsCfg()
    idle_cfg = IdleCfg()

    print("\033[91mWARNING: This example controls real D1 head + arm and DexHand.\033[0m")
    print("\033[91mKeep the physical emergency stop button within reach.\033[0m\n")

    retargeting = init_right_hand_retargeting()
    detector = SingleHandDetector(hand_type="Right", selfie=False)
    finger_indices = infer_finger_indices(retargeting.joint_names)
    robot_gestures = build_real_hand_gesture_sequences(hand_cfg)
    rps_head_arm_postures = build_rps_head_arm_postures()

    print(f"Retargeting output joints: {retargeting.joint_names}")
    print(f"Finger qpos indices: {finger_indices}")
    print(f"Finger thresholds: {rps_cfg.thresholds}")

    head_arm = None
    right_hand = None
    try:
        head_arm = HeadArmRobot(
            urdf_path=get_default_urdf_path(),
            dev=head_arm_cfg.dev,
            baudrate=head_arm_cfg.baudrate,
            return_delay=0,
        )
        move_head_arm_to_initial_pose(head_arm, head_arm_cfg)

        right_hand = DexHand(hand_type=hand_cfg.hand_type, can_iface=hand_cfg.can_iface, baudrate=hand_cfg.baudrate)
        right_hand.set_speed(speed=list(hand_cfg.speed))
        right_hand.set_torque(torque=list(hand_cfg.torque))
        right_hand.open_hand()
        time.sleep(0.2)

        window_name = "D1 Rock-Paper-Scissors Real"
        last_time = time.time()
        fps = 0.0
        frame_idx = 0
        stable = StableGesture()
        idle_motion = IdleMotion(
            idle_cfg,
            ready_head_arm_deg=head_arm_cfg.q_init_deg,
            ready_hand_q_norm=(hand_cfg.open_value,) * 6,
        )
        inactive_since = time.monotonic()
        last_robot_gesture = None
        last_scores = None

        print("Starting real DexHand rock-paper-scissors. Press q in the OpenCV window to exit.")
        with StereoRGBCamera(
            device=stereo_cfg.device,
            single_width=stereo_cfg.single_width,
            single_height=stereo_cfg.single_height,
            fps=stereo_cfg.fps,
            fourcc=stereo_cfg.fourcc,
            swap_lr=stereo_cfg.swap_lr,
            view=stereo_cfg.view,
        ) as camera:
            while True:
                now = time.time()
                dt = now - last_time
                last_time = now
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt

                try:
                    bgr_view = camera.read_view()
                except RuntimeError as exc:
                    print(f"[ERROR] {exc}")
                    break

                rgb = cv2.cvtColor(bgr_view, cv2.COLOR_BGR2RGB)
                _, joint_pos, _, _ = detector.detect(rgb)

                human_raw = "unknown"
                human_stable = stable.last_stable
                robot_gesture = last_robot_gesture
                active_rps = False

                if joint_pos is None:
                    human_stable = stable.update("unknown", rps_cfg.stable_frames, rps_cfg.unknown_reset_frames)
                    robot_gesture = None
                    last_robot_gesture = None
                    if frame_idx % 30 == 0:
                        print("Right hand is not detected.")
                else:
                    human_qpos = retarget_joint_pos(retargeting, joint_pos)
                    last_scores = finger_bend_scores(human_qpos, finger_indices)
                    human_raw = classify_rps(last_scores, rps_cfg.thresholds)
                    human_stable = stable.update(human_raw, rps_cfg.stable_frames, rps_cfg.unknown_reset_frames)
                    robot_gesture = winning_gesture(human_stable)
                    active_rps = robot_gesture is not None

                    if human_stable == "unknown":
                        last_robot_gesture = None

                    if robot_gesture is not None and robot_gesture != last_robot_gesture:
                        idle_motion.reset()
                        apply_rps_head_arm_posture(
                            head_arm,
                            robot_gesture,
                            rps_head_arm_postures,
                            rps_arm_cfg,
                        )
                        apply_real_hand_gesture_sequence(right_hand, robot_gestures[robot_gesture])
                        last_robot_gesture = robot_gesture

                    if frame_idx % 30 == 0:
                        print(
                            f"raw={human_raw} stable={human_stable} "
                            f"robot={robot_gesture} scores={last_scores}"
                        )

                if active_rps:
                    inactive_since = time.monotonic()
                    idle_motion.reset()
                elif idle_cfg.enabled and time.monotonic() - inactive_since >= idle_cfg.trigger_after_s:
                    idle_motion.update(head_arm, right_hand, time.monotonic())

                overlay = draw_overlay(
                    camera.draw_info(bgr_view, fps),
                    fps=fps,
                    human=human_stable,
                    robot=robot_gesture,
                    scores=last_scores,
                    thresholds=rps_cfg.thresholds,
                )
                cv2.imshow(window_name, overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                frame_idx += 1

    except KeyboardInterrupt:
        pass
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            if right_hand is not None:
                right_hand.open_hand()
                time.sleep(0.2)
        except Exception as exc:
            print(f"Failed to send safe-open: {exc}")
        try:
            if right_hand is not None:
                right_hand.close_can()
        except Exception as exc:
            print(f"Failed to close CAN: {exc}")
        try:
            if head_arm is not None:
                head_arm.home_and_close()
        except Exception as exc:
            print(f"Failed to close head + arm: {exc}")


if __name__ == "__main__":
    main()
