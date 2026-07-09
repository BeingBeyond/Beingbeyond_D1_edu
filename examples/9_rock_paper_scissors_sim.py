"""
Example 9: Rock-paper-scissors in simulation.

Pipeline:
  USB stereo RGB camera -> MediaPipe right-hand landmarks -> dex-retargeting qpos
  -> classify human rock/paper/scissors from five finger bend thresholds
  -> command the Isaac Gym D1 hand to play the winning gesture.

Press q in the OpenCV camera window, or close the Isaac Gym viewer, to exit.
"""

import importlib
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pinocchio

from isaacgym import gymapi

from beingbeyond_d1_edu_sdk.urdf_path import get_default_urdf_path


EXAMPLES_DIR = Path(__file__).resolve().parent
D1_EDU_DIR = EXAMPLES_DIR.parent
DEX_RETARGETING_DIR = D1_EDU_DIR / "lib" / "dex-retargeting"
DEX_VECTOR_EXAMPLE_DIR = DEX_RETARGETING_DIR / "example" / "vector_retargeting"

sys.path.append(str(DEX_RETARGETING_DIR))
sys.path.append(str(DEX_VECTOR_EXAMPLE_DIR))

from dex_retargeting.constants import RobotName, RetargetingType, HandType, get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig
from single_hand_detector import SingleHandDetector
from vision import StereoRGBCamera


FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")
HEAD_ARM_JOINT_NAMES = (
    "joint_7_head_yaw",
    "joint_8_head_pitch",
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
)
GESTURE_LABELS = {
    "rock": "ROCK",
    "paper": "PAPER",
    "scissors": "SCISSORS",
    "unknown": "UNKNOWN",
}


@dataclass(frozen=True)
class AssetCfg:
    asset_path: str = get_default_urdf_path()
    fix_base_link: bool = True
    flip_visual_attachments: bool = False
    armature: float = 0.01


@dataclass(frozen=True)
class SimCfg:
    dt: float = 1.0 / 60.0
    substeps: int = 2
    up_axis: int = gymapi.UP_AXIS_Z
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.8)
    use_gpu_pipeline: bool = False
    compute_device_id: int = 0
    graphics_device_id: int = 0
    physics_engine: int = gymapi.SIM_PHYSX
    num_threads: int = 0
    use_gpu: bool = False


@dataclass(frozen=True)
class DofDriveCfg:
    stiffness: float = 400.0
    damping: float = 40.0
    drive_mode: int = gymapi.DOF_MODE_POS


@dataclass(frozen=True)
class CameraCfg:
    cam_pos: Tuple[float, float, float] = (1.5, 1.5, 1.2)
    cam_target: Tuple[float, float, float] = (0.0, 0.0, 0.45)


@dataclass(frozen=True)
class HeadArmCfg:
    q_init_deg: Tuple[float, ...] = (
        0.0,
        0.0,
        0.0,
        -60.0,
        60.0,
        0.0,
        0.0,
        0.0,
    )


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
class RpsCfg:
    thumb_threshold: float = 0.45
    index_threshold: float = 0.45
    middle_threshold: float = 0.45
    ring_threshold: float = 0.40
    pinky_threshold: float = 0.40
    open_value: float = 0.0
    closed_value: float = 1.0
    thumb_pitch_value: float = 0.7
    thumb_yaw_value: float = 0.5
    thumb_yaw_delay_s: float = 0.1
    thumb_pitch_delay_s: float = 1.0
    non_thumb_delay_s: float = 0.5
    stable_frames: int = 5
    unknown_reset_frames: int = 3

    @property
    def thresholds(self) -> Dict[str, float]:
        return {
            "thumb": self.thumb_threshold,
            "index": self.index_threshold,
            "middle": self.middle_threshold,
            "ring": self.ring_threshold,
            "pinky": self.pinky_threshold,
        }


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


def create_sim(gym, cfg: SimCfg) -> gymapi.Sim:
    sim_params = gymapi.SimParams()
    sim_params.dt = cfg.dt
    sim_params.substeps = cfg.substeps
    sim_params.up_axis = cfg.up_axis
    sim_params.gravity = gymapi.Vec3(*cfg.gravity)

    if cfg.physics_engine == gymapi.SIM_FLEX:
        sim_params.flex.solver_type = 5
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 15
        sim_params.flex.relaxation = 0.75
        sim_params.flex.warm_start = 0.8
    elif cfg.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = cfg.num_threads
        sim_params.physx.use_gpu = cfg.use_gpu

    sim_params.use_gpu_pipeline = cfg.use_gpu_pipeline
    if cfg.use_gpu_pipeline:
        print("WARNING: Forcing CPU pipeline.")

    sim = gym.create_sim(cfg.compute_device_id, cfg.graphics_device_id, cfg.physics_engine, sim_params)
    if sim is None:
        raise RuntimeError("Failed to create sim")
    return sim


def create_viewer(gym, sim) -> gymapi.Viewer:
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise RuntimeError("Failed to create viewer")
    return viewer


def add_ground(gym, sim) -> None:
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)


def load_asset(gym, sim, cfg: AssetCfg):
    asset_root = os.path.dirname(cfg.asset_path)
    asset_file = os.path.basename(cfg.asset_path)

    opt = gymapi.AssetOptions()
    opt.fix_base_link = cfg.fix_base_link
    opt.flip_visual_attachments = cfg.flip_visual_attachments
    opt.armature = cfg.armature

    asset = gym.load_asset(sim, asset_root, asset_file, opt)
    return asset, os.path.join(asset_root, asset_file)


def create_env_and_actor(gym, sim, asset):
    spacing = 2.0
    env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)
    env = gym.create_env(sim, env_lower, env_upper, 1)

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
    pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    actor = gym.create_actor(env, asset, pose, "d1", 0, 1)
    return env, actor


def configure_dofs(gym, env, actor, cfg: DofDriveCfg):
    dof_props = gym.get_asset_dof_properties(gym.get_actor_asset(env, actor))
    for i in range(len(dof_props)):
        dof_props["driveMode"][i] = cfg.drive_mode
        dof_props["stiffness"][i] = cfg.stiffness
        dof_props["damping"][i] = cfg.damping
    gym.set_actor_dof_properties(env, actor, dof_props)

    num_dofs = gym.get_actor_dof_count(env, actor)
    dof_states = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
    targets = np.zeros(num_dofs, dtype=np.float32)
    gym.set_actor_dof_position_targets(env, actor, targets)
    return num_dofs, dof_states, targets


def set_viewer_camera(gym, viewer, cfg: CameraCfg) -> None:
    cam_pos = gymapi.Vec3(*cfg.cam_pos)
    cam_target = gymapi.Vec3(*cfg.cam_target)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)


def apply_initial_head_arm_pose(
    gym,
    env,
    actor,
    joint_names,
    targets: np.ndarray,
    dof_states: np.ndarray,
    cfg: HeadArmCfg,
) -> None:
    if len(cfg.q_init_deg) != len(HEAD_ARM_JOINT_NAMES):
        raise ValueError(f"q_init_deg must contain {len(HEAD_ARM_JOINT_NAMES)} values")

    q_init_rad = np.deg2rad(np.asarray(cfg.q_init_deg, dtype=np.float64))
    for name, q in zip(HEAD_ARM_JOINT_NAMES, q_init_rad):
        try:
            idx = joint_names.index(name)
        except ValueError:
            print(f"WARNING: Initial joint {name} is missing in Isaac asset and will be skipped.")
            continue
        targets[idx] = float(q)
        dof_states["pos"][idx] = float(q)
        dof_states["vel"][idx] = 0.0

    gym.set_actor_dof_position_targets(env, actor, targets)
    gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_ALL)


def build_qpos_map(gym_joint_names, retarget_joint_names) -> np.ndarray:
    mapping = []
    for name in retarget_joint_names:
        try:
            mapping.append(gym_joint_names.index(name))
        except ValueError:
            mapping.append(-1)
    return np.array(mapping, dtype=np.int32)


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

    if all(bent.values()):
        return "rock"

    if not any(bent.values()):
        return "paper"

    if (
        bent["thumb"]
        and not bent["index"]
        and not bent["middle"]
        and bent["ring"]
        and bent["pinky"]
    ):
        return "scissors"

    return "unknown"


def winning_gesture(human_gesture: str) -> Optional[str]:
    return {
        "rock": "paper",
        "paper": "scissors",
        "scissors": "rock",
    }.get(human_gesture)


def set_thumb_values(qpos: np.ndarray, joint_names, thumb_indices: Tuple[int, ...], pitch: float, yaw: float) -> None:
    for idx in thumb_indices:
        lname = joint_names[idx].lower()
        if "yaw" in lname:
            qpos[idx] = yaw
        else:
            qpos[idx] = pitch


def build_robot_gesture_sequences(
    joint_names,
    finger_indices: Dict[str, Tuple[int, ...]],
    cfg: RpsCfg,
) -> Dict[str, Tuple[Tuple[np.ndarray, float], ...]]:
    n = len(joint_names)

    def make_qpos(bent_fingers, thumb_pitch=None, thumb_yaw=None) -> np.ndarray:
        qpos = np.full(n, cfg.open_value, dtype=np.float32)
        for finger in bent_fingers:
            if finger == "thumb":
                set_thumb_values(
                    qpos,
                    joint_names,
                    finger_indices["thumb"],
                    cfg.thumb_pitch_value if thumb_pitch is None else thumb_pitch,
                    cfg.thumb_yaw_value if thumb_yaw is None else thumb_yaw,
                )
            else:
                for idx in finger_indices[finger]:
                    qpos[idx] = cfg.closed_value
        if "thumb" not in bent_fingers and (thumb_pitch is not None or thumb_yaw is not None):
            set_thumb_values(
                qpos,
                joint_names,
                finger_indices["thumb"],
                cfg.open_value if thumb_pitch is None else thumb_pitch,
                cfg.open_value if thumb_yaw is None else thumb_yaw,
            )
        return qpos

    return {
        "paper": ((make_qpos(()), 0.0),),
        "rock": (
            (make_qpos(("index", "middle", "ring", "pinky")), cfg.non_thumb_delay_s),
            (make_qpos(("index", "middle", "ring", "pinky"), thumb_pitch=0.0, thumb_yaw=cfg.thumb_yaw_value), cfg.thumb_yaw_delay_s),
            (make_qpos(FINGER_ORDER), cfg.thumb_pitch_delay_s),
        ),
        "scissors": (
            (make_qpos(("ring", "pinky")), cfg.non_thumb_delay_s),
            (make_qpos(("ring", "pinky"), thumb_pitch=0.0, thumb_yaw=cfg.thumb_yaw_value), cfg.thumb_yaw_delay_s),
            (make_qpos(("thumb", "ring", "pinky")), cfg.thumb_pitch_delay_s),
        ),
    }


def apply_robot_qpos(
    gym,
    env,
    actor,
    targets: np.ndarray,
    qpos_map: np.ndarray,
    qpos: np.ndarray,
) -> None:
    for retarget_idx, gym_idx in enumerate(qpos_map):
        if gym_idx == -1:
            continue
        targets[gym_idx] = qpos[retarget_idx]

    gym.set_actor_dof_position_targets(env, actor, targets)


def apply_robot_gesture_sequence(
    gym,
    env,
    actor,
    targets: np.ndarray,
    qpos_map: np.ndarray,
    sequence: Tuple[Tuple[np.ndarray, float], ...],
) -> None:
    for qpos, delay_s in sequence:
        apply_robot_qpos(gym, env, actor, targets, qpos_map, qpos)
        if delay_s > 0.0:
            time.sleep(delay_s)


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
        f"D1: {GESTURE_LABELS.get(robot, robot) if robot else '-'}",
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
    asset_cfg = AssetCfg()
    sim_cfg = SimCfg()
    dof_cfg = DofDriveCfg()
    cam_cfg = CameraCfg()
    head_arm_cfg = HeadArmCfg()

    gym = gymapi.acquire_gym()
    stereo_cfg = StereoCfg()
    rps_cfg = RpsCfg()

    sim = create_sim(gym, sim_cfg)
    viewer = create_viewer(gym, sim)
    add_ground(gym, sim)

    asset, urdf_path = load_asset(gym, sim, asset_cfg)
    print(f"Loaded Isaac asset: {urdf_path}")
    env, actor = create_env_and_actor(gym, sim, asset)
    num_dofs, dof_states, current_targets = configure_dofs(gym, env, actor, dof_cfg)
    set_viewer_camera(gym, viewer, cam_cfg)

    retargeting = init_right_hand_retargeting()
    detector = SingleHandDetector(hand_type="Right", selfie=False)

    gym_joint_names = gym.get_actor_dof_names(env, actor)
    apply_initial_head_arm_pose(
        gym=gym,
        env=env,
        actor=actor,
        joint_names=gym_joint_names,
        targets=current_targets,
        dof_states=dof_states,
        cfg=head_arm_cfg,
    )

    retarget_joint_names = retargeting.joint_names
    qpos_map = build_qpos_map(gym_joint_names, retarget_joint_names)
    finger_indices = infer_finger_indices(retarget_joint_names)
    robot_gestures = build_robot_gesture_sequences(retarget_joint_names, finger_indices, rps_cfg)

    print(f"Isaac DOF count: {num_dofs}")
    print(f"Retargeting output joints: {retarget_joint_names}")
    print(f"Finger qpos indices: {finger_indices}")
    print(f"Finger thresholds: {rps_cfg.thresholds}")
    print(f"Isaac qpos map: {qpos_map}")
    missing = [name for name, idx in zip(retarget_joint_names, qpos_map) if idx == -1]
    if missing:
        print(f"WARNING: Retargeted joints missing in Isaac asset and will be skipped: {missing}")

    window_name = "D1 RPS RGB"
    last_time = time.time()
    fps = 0.0
    frame_idx = 0
    stable = StableGesture()
    last_robot_gesture = None
    last_scores = None

    print("Starting D1 rock-paper-scissors. Press q in the OpenCV window to exit.")
    try:
        with StereoRGBCamera(
            device=stereo_cfg.device,
            single_width=stereo_cfg.single_width,
            single_height=stereo_cfg.single_height,
            fps=stereo_cfg.fps,
            fourcc=stereo_cfg.fourcc,
            swap_lr=stereo_cfg.swap_lr,
            view=stereo_cfg.view,
        ) as camera:
            while not gym.query_viewer_has_closed(viewer):
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

                if joint_pos is None:
                    if frame_idx % 30 == 0:
                        print("Right hand is not detected.")
                else:
                    human_qpos = retarget_joint_pos(retargeting, joint_pos)
                    last_scores = finger_bend_scores(human_qpos, finger_indices)
                    human_raw = classify_rps(last_scores, rps_cfg.thresholds)
                    human_stable = stable.update(human_raw, rps_cfg.stable_frames, rps_cfg.unknown_reset_frames)
                    robot_gesture = winning_gesture(human_stable)

                    if human_stable == "unknown":
                        last_robot_gesture = None

                    if robot_gesture is not None and robot_gesture != last_robot_gesture:
                        apply_robot_gesture_sequence(
                            gym=gym,
                            env=env,
                            actor=actor,
                            targets=current_targets,
                            qpos_map=qpos_map,
                            sequence=robot_gestures[robot_gesture],
                        )
                        last_robot_gesture = robot_gesture

                    if frame_idx % 30 == 0:
                        print(
                            f"raw={human_raw} stable={human_stable} "
                            f"robot={robot_gesture} scores={last_scores}"
                        )

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

                gym.simulate(sim)
                gym.fetch_results(sim, True)
                gym.step_graphics(sim)
                gym.draw_viewer(viewer, sim, True)
                gym.sync_frame_time(sim)

                frame_idx += 1
    finally:
        cv2.destroyAllWindows()
        gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
