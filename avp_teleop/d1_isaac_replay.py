"""
D1 Isaac Gym Replay (NPZ)

Replays recorded VisionPro (or operator) tracking data in Isaac Gym:
- Optional hand retargeting (fingers -> robot hand joints)
- Arm/head IK control (wrist pose -> robot end-effector pose)
- Visualizes the target frame axes in the viewer

Input data (NPZ) expected keys:
- timestamps: (T,) float
- right_wrist_poses: (T,4,4) float  (SE(3) transform, row-major)
- right_pinch: (T,) float
- optional right_fingers: (T,25,4,4) float  (25 joint/link transforms)

Key logic:
- Calibrate origin implicitly at t=0:
  - initial recorded wrist pose as reference
  - initial robot EE pose from q_init as reference
- Position mapping:
  pos_delta = correction_R @ (pos(t) - pos(0)) * pos_scale
  ik_target_pos = initial_ee_pos + pos_delta
  clamp z >= min_z
- Orientation mapping:
  R_delta_data = R(t) * R(0)^-1
  R_delta = correction * R_delta_data * correction^-1
  R_target = R_delta * R_robot_init
- IK:
  iterative outer loop; each iteration calls kin.ik_ee_quatpose_with_arm_only()
  stops when position/orientation error below tolerances

Controls / behavior:
- Runs continuously; when reach end of frames, resets to initial pose and restarts replay.
- Viewer: closes to exit.

Dependencies:
- Isaac Gym Preview 4
- beingbeyond_d1_sdk
- dex_retargeting
"""


import os
import math
import time
import importlib
import pinocchio
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from isaacgym import gymapi
from isaacgym import gymutil

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from beingbeyond_d1_edu_sdk.pin_kinematics import D1Kinematics, D1KinematicsConfig

from beingbeyond_d1_edu_sdk.urdf_path import get_default_urdf_path

HEAD_JOINT_NAMES = ("joint_7_head_yaw", "joint_8_head_pitch")
ARM_JOINT_NAMES = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6")


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


@dataclass(frozen=True)
class DofDriveCfg:
    stiffness: float = 400.0
    damping: float = 40.0
    drive_mode: int = gymapi.DOF_MODE_POS


@dataclass(frozen=True)
class CameraCfg:
    cam_pos: Tuple[float, float, float] = (1.5, 1.5, 1.5)
    cam_target: Tuple[float, float, float] = (0.0, 0.0, 0.5)


@dataclass(frozen=True)
class ReplayCfg:
    tracking_data_path: str = "demo_data/demo_data.npz"
    q_init_deg: Tuple[float, ...] = (
        0.0, 0.0,          # head_yaw, head_pitch
        0.0, -60.0, 60.0,
        0.0, 0.0, 0.0,
    )
    correction_euler_deg: Tuple[float, float, float] = (0.0, 0.0, -90.0)
    pos_scale: float = 1.0
    min_z: float = 0.05


@dataclass(frozen=True)
class IKCfg:
    max_outer_iters: int = 5
    pos_tol: float = 0.005
    rot_tol: float = 0.5
    print_every: int = 10


OPERATOR2AVP_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ],
    dtype=np.float64,
)

def load_npz_tracking(filepath: str) -> Dict[str, np.ndarray]:
    data = np.load(filepath)
    result = {
        "timestamps": data["timestamps"],
        "right_wrist_poses": data["right_wrist_poses"],
        "right_pinch": data["right_pinch"],
    }
    if "right_fingers" in data:
        result["right_fingers"] = data["right_fingers"]
    return result


def two_mat_batch_mul(batch_T: np.ndarray, left_rot: np.ndarray) -> np.ndarray:
    out = np.tile(np.eye(4, dtype=batch_T.dtype), (batch_T.shape[0], 1, 1))
    out[:, :3, :3] = np.matmul(left_rot[None, ...], batch_T[:, :3, :3])
    out[:, :3, 3] = batch_T[:, :3, 3] @ left_rot.T
    return out


def joint_avp2hand(finger_T: np.ndarray) -> np.ndarray:
    finger_index = np.array(
        [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24],
        dtype=np.int64,
    )
    return finger_T[finger_index]


def build_qpos_map(gym_joint_names, retarget_joint_names) -> np.ndarray:
    m = []
    for name in retarget_joint_names:
        try:
            m.append(gym_joint_names.index(name))
        except ValueError:
            m.append(-1)
    return np.array(m, dtype=np.int32)


def rotation_angle_error(R_target: np.ndarray, R_curr: np.ndarray) -> float:
    R_diff = R_target @ R_curr.T
    tr = np.trace(R_diff)
    cos_theta = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cos_theta))



def create_sim(gym, args, cfg: SimCfg) -> gymapi.Sim:
    sim_params = gymapi.SimParams()
    sim_params.dt = cfg.dt
    sim_params.substeps = cfg.substeps
    sim_params.up_axis = cfg.up_axis
    sim_params.gravity = gymapi.Vec3(*cfg.gravity)

    if args.physics_engine == gymapi.SIM_FLEX:
        sim_params.flex.solver_type = 5
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 15
        sim_params.flex.relaxation = 0.75
        sim_params.flex.warm_start = 0.8
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = args.num_threads
        sim_params.physx.use_gpu = args.use_gpu

    sim_params.use_gpu_pipeline = cfg.use_gpu_pipeline
    if args.use_gpu_pipeline:
        print("WARNING: Forcing CPU pipeline.")

    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
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


def create_env_and_actor(gym, sim, asset) -> Tuple[gymapi.Env, int]:
    spacing = 2.0
    env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)

    env = gym.create_env(sim, env_lower, env_upper, 1)

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
    pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    actor = gym.create_actor(env, asset, pose, "d1", 0, 1)
    return env, actor


def configure_dofs(gym, env, actor, cfg: DofDriveCfg) -> Tuple[int, np.ndarray]:
    dof_props = gym.get_asset_dof_properties(gym.get_actor_asset(env, actor))
    for i in range(len(dof_props)):
        dof_props["driveMode"][i] = cfg.drive_mode
        dof_props["stiffness"][i] = cfg.stiffness
        dof_props["damping"][i] = cfg.damping
    gym.set_actor_dof_properties(env, actor, dof_props)

    num_dofs = gym.get_actor_dof_count(env, actor)
    dof_states = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
    return num_dofs, dof_states


def set_viewer_camera(gym, viewer, cfg: CameraCfg) -> None:
    cam_pos = gymapi.Vec3(*cfg.cam_pos)
    cam_target = gymapi.Vec3(*cfg.cam_target)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)


def build_named_dof_map(gym_joint_names, target_joint_names) -> np.ndarray:
    missing = [name for name in target_joint_names if name not in gym_joint_names]
    if missing:
        raise ValueError(f"Missing joints in Isaac asset: {missing}")
    return np.array([gym_joint_names.index(name) for name in target_joint_names], dtype=np.int32)


def apply_named_targets(targets: np.ndarray, dof_indices: np.ndarray, values: np.ndarray) -> None:
    if len(dof_indices) != len(values):
        raise ValueError(f"Target length mismatch: {len(dof_indices)} indices, {len(values)} values")
    targets[dof_indices] = values


def set_actor_joint_targets(
    gym,
    env,
    actor,
    targets: np.ndarray,
    dof_states: np.ndarray,
    dof_indices: np.ndarray,
    values: np.ndarray,
) -> None:
    apply_named_targets(targets, dof_indices, values)
    dof_states["pos"][dof_indices] = values
    dof_states["vel"][dof_indices] = 0.0
    gym.set_actor_dof_position_targets(env, actor, targets)
    gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_ALL)


def set_initial_targets(
    gym,
    env,
    actor,
    num_dofs: int,
    dof_states: np.ndarray,
    head_arm_dof_indices: np.ndarray,
    q_init_rad: np.ndarray,
) -> np.ndarray:
    targets = np.zeros(num_dofs, dtype=np.float32)
    set_actor_joint_targets(gym, env, actor, targets, dof_states, head_arm_dof_indices, q_init_rad)
    return targets



def init_kinematics(urdf_path: str) -> D1Kinematics:
    print(f"Initializing D1 Kinematics with URDF: {urdf_path}")
    kin_cfg = D1KinematicsConfig(
        urdf_path=urdf_path,
        base_link="link0",
        ee_link="ee",
        camera_link="camera",
    )
    return D1Kinematics(kin_cfg)


def init_right_hand_retargeting() -> object:
    print("Initializing Dex Retargeting...")
    robot_name = RobotName.d1hand

    dex_pkg = importlib.util.find_spec("dex_retargeting")
    if dex_pkg and dex_pkg.origin:
        robot_dir = Path(dex_pkg.origin).absolute().parent.parent / "assets" / "robots" / "hands"
        RetargetingConfig.set_default_urdf_dir(str(robot_dir))

    override = dict(add_dummy_free_joint=True)
    right_config_path = get_default_config_path(robot_name, RetargetingType.dexpilot, HandType.right)
    right_cfg = RetargetingConfig.load_from_file(right_config_path, override=override)
    right_retargeting = right_cfg.build()
    return right_retargeting



def compute_target_pose(
    replay_cfg: ReplayCfg,
    initial_ee_pos: np.ndarray,
    initial_ee_R: np.ndarray,
    initial_rec_pos: np.ndarray,
    initial_rec_R: np.ndarray,
    curr_rec_pos: np.ndarray,
    curr_rec_R: np.ndarray,
) -> Tuple[np.ndarray, R]:
    r_correction = R.from_euler("xyz", replay_cfg.correction_euler_deg, degrees=True)
    mat_corr = r_correction.as_matrix()

    pos_delta = curr_rec_pos - initial_rec_pos
    pos_delta = mat_corr @ pos_delta
    pos_delta = pos_delta * replay_cfg.pos_scale

    ik_target_pos = initial_ee_pos + pos_delta
    if ik_target_pos[2] < replay_cfg.min_z:
        ik_target_pos[2] = replay_cfg.min_z

    r_rec_init = R.from_matrix(initial_rec_R)
    r_rec_curr = R.from_matrix(curr_rec_R)

    r_delta_data = r_rec_curr * r_rec_init.inv()
    r_delta = r_correction * r_delta_data * r_correction.inv()

    r_robot_init = R.from_matrix(initial_ee_R)
    r_target = r_delta * r_robot_init

    return ik_target_pos, r_target


def solve_ik_iterative(
    kin: D1Kinematics,
    ik_cfg: IKCfg,
    q_head: np.ndarray,
    q_arm: np.ndarray,
    ik_target_pos: np.ndarray,
    r_target: R,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    q_head_iter = q_head.copy()
    q_arm_iter = q_arm.copy()

    q_target_xyzw = r_target.as_quat()  # xyzw
    target_quatpose = np.zeros(7, dtype=float)
    target_quatpose[0:3] = ik_target_pos
    target_quatpose[3:7] = q_target_xyzw

    pos_err = 1e9
    rot_err = 1e9

    for _ in range(ik_cfg.max_outer_iters):
        T_curr = kin.ee_in_base(q_head_iter, q_arm_iter)
        p_curr = T_curr[:3, 3]
        R_curr = T_curr[:3, :3]

        pos_err = float(np.linalg.norm(ik_target_pos - p_curr))
        rot_err = rotation_angle_error(r_target.as_matrix(), R_curr)

        if pos_err < ik_cfg.pos_tol and rot_err < ik_cfg.rot_tol:
            break

        q_head_sol, q_arm_sol, cost, inner_iters = kin.ik_ee_quatpose_with_arm_only(
            target_ee_quatpose=target_quatpose,
            q_head=q_head_iter,
            q_arm_init=q_arm_iter,
        )
        q_head_iter = q_head_sol
        q_arm_iter = q_arm_sol

    return q_head_iter, q_arm_iter, pos_err, rot_err


def apply_hand_retargeting_if_available(
    recorded_fingers_frame: np.ndarray,
    right_retargeting: object,
    qpos_map: np.ndarray,
    current_targets: np.ndarray,
) -> None:
    if recorded_fingers_frame is None:
        return
    if np.isnan(recorded_fingers_frame).all():
        return

    joint_pose = two_mat_batch_mul(recorded_fingers_frame, OPERATOR2AVP_RIGHT.T)
    joint_pos = joint_avp2hand(joint_pose)[:, :3, 3]

    indices = right_retargeting.optimizer.target_link_human_indices
    origin_indices = indices[0, :]
    task_indices = indices[1, :]
    ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]

    qpos = right_retargeting.retarget(ref_value)
    for i, gym_idx in enumerate(qpos_map):
        if gym_idx != -1:
            current_targets[gym_idx] = qpos[i]


def draw_target_frame(gym, viewer, env, pos: np.ndarray, rot: np.ndarray, axis_len: float = 0.1) -> None:
    gym.clear_lines(viewer)

    vertices = np.zeros((3, 2, 3), dtype=np.float32)
    colors = np.array(
        [
            [1, 0, 0],  # X
            [0, 1, 0],  # Y
            [0, 0, 1],  # Z
        ],
        dtype=np.float32,
    )

    vpos = pos.astype(np.float32)
    Rm = rot.astype(np.float32)

    vertices[0, 0] = vpos
    vertices[0, 1] = vpos + Rm @ np.array([axis_len, 0, 0], dtype=np.float32)

    vertices[1, 0] = vpos
    vertices[1, 1] = vpos + Rm @ np.array([0, axis_len, 0], dtype=np.float32)

    vertices[2, 0] = vpos
    vertices[2, 1] = vpos + Rm @ np.array([0, 0, axis_len], dtype=np.float32)

    gym.add_lines(viewer, env, 3, vertices, colors)



def main() -> None:
    asset_cfg = AssetCfg()
    sim_cfg = SimCfg()
    dof_cfg = DofDriveCfg()
    cam_cfg = CameraCfg()
    replay_cfg = ReplayCfg()
    ik_cfg = IKCfg()

    # Gym + args
    gym = gymapi.acquire_gym()
    args = gymutil.parse_arguments(description="D1 Replay Example (Refactored)")

    # Sim / Viewer / Ground
    sim = create_sim(gym, args, sim_cfg)
    viewer = create_viewer(gym, sim)
    add_ground(gym, sim)

    # Asset / Env / Actor
    script_dir = os.path.dirname(os.path.abspath(__file__))
    asset, urdf_path = load_asset(gym, sim, asset_cfg)
    env, actor = create_env_and_actor(gym, sim, asset)

    # DOFs
    num_dofs, dof_states = configure_dofs(gym, env, actor, dof_cfg)

    # Kinematics / Retargeting
    kin = init_kinematics(urdf_path)
    right_retargeting = init_right_hand_retargeting()

    retarget_joint_names = right_retargeting.joint_names
    print(f"Retargeting Output Joints: {retarget_joint_names}")

    gym_joint_names = gym.get_actor_dof_names(env, actor)
    print(f"Gym Actor Joints: {gym_joint_names}")

    qpos_map = build_qpos_map(gym_joint_names, retarget_joint_names)
    print(f"Qpos Map: {qpos_map}")
    head_arm_joint_names = HEAD_JOINT_NAMES + ARM_JOINT_NAMES
    head_arm_dof_indices = build_named_dof_map(gym_joint_names, head_arm_joint_names)
    print(f"Head/arm DOF map: {dict(zip(head_arm_joint_names, head_arm_dof_indices.tolist()))}")

    # Initial pose
    q_init_rad = np.array([math.radians(v) for v in replay_cfg.q_init_deg], dtype=np.float64)
    if len(q_init_rad) != len(head_arm_joint_names):
        raise ValueError(
            f"q_init_deg length ({len(q_init_rad)}) does not match head/arm joint count "
            f"({len(head_arm_joint_names)})"
        )
    current_targets = set_initial_targets(
        gym,
        env,
        actor,
        num_dofs,
        dof_states,
        head_arm_dof_indices,
        q_init_rad,
    )

    set_viewer_camera(gym, viewer, cam_cfg)

    # IK state (head 2, arm 6)
    q_head = q_init_rad[:2].copy()
    q_arm = q_init_rad[2:8].copy()

    # Load data
    data_file = replay_cfg.tracking_data_path
    print(f"Loading data from {data_file}")
    recorded = load_npz_tracking(data_file)
    timestamps = recorded["timestamps"]
    recorded_poses = recorded["right_wrist_poses"]
    recorded_fingers = recorded.get("right_fingers", None)

    # Initial transforms
    T_base_ee_init = kin.ee_in_base(q_head, q_arm)
    initial_ee_pos = T_base_ee_init[:3, 3].copy()
    initial_ee_R = T_base_ee_init[:3, :3].copy()
    initial_ee_quat = R.from_matrix(initial_ee_R).as_quat()  # xyzw

    print("Initial EE Pos:", initial_ee_pos)
    print("Initial EE Quat (xyzw):", initial_ee_quat)

    initial_hand_T = recorded_poses[0]
    initial_rec_pos = initial_hand_T[:3, 3].copy()
    initial_rec_R = initial_hand_T[:3, :3].copy()
    print("Initial Recorded Pos:", initial_rec_pos)

    print("Starting simulation loop...")

    frame_idx = 0
    num_frames = len(timestamps)

    while not gym.query_viewer_has_closed(viewer):
        if frame_idx >= num_frames:
            print("Replay finished, restarting...")
            frame_idx = 0

            # Reset robot to initial state
            q_head = q_init_rad[:2].copy()
            q_arm = q_init_rad[2:8].copy()
            current_targets[:] = 0.0
            set_actor_joint_targets(
                gym,
                env,
                actor,
                current_targets,
                dof_states,
                head_arm_dof_indices,
                q_init_rad,
            )

        # Current recorded pose
        hand_T = recorded_poses[frame_idx]
        curr_rec_pos = hand_T[:3, 3]
        curr_rec_R = hand_T[:3, :3]

        # Hand retargeting
        if recorded_fingers is not None:
            apply_hand_retargeting_if_available(
                recorded_fingers_frame=recorded_fingers[frame_idx],
                right_retargeting=right_retargeting,
                qpos_map=qpos_map,
                current_targets=current_targets,
            )

        # Compute IK target pose
        ik_target_pos, r_target = compute_target_pose(
            replay_cfg=replay_cfg,
            initial_ee_pos=initial_ee_pos,
            initial_ee_R=initial_ee_R,
            initial_rec_pos=initial_rec_pos,
            initial_rec_R=initial_rec_R,
            curr_rec_pos=curr_rec_pos,
            curr_rec_R=curr_rec_R,
        )

        # IK solve + apply
        try:
            q_head_new, q_arm_new, pos_err, rot_err = solve_ik_iterative(
                kin=kin,
                ik_cfg=ik_cfg,
                q_head=q_head,
                q_arm=q_arm,
                ik_target_pos=ik_target_pos,
                r_target=r_target,
            )

            if frame_idx % ik_cfg.print_every == 0:
                print(
                    f"Frame {frame_idx}: Final err pos={pos_err * 1000:.1f}mm, rot={np.rad2deg(rot_err):.1f}deg"
                )

            q_head = q_head_new
            q_arm = q_arm_new

            # Apply to robot targets
            apply_named_targets(current_targets, head_arm_dof_indices, np.concatenate([q_head, q_arm]))
            gym.set_actor_dof_position_targets(env, actor, current_targets)

        except Exception as e:
            print(f"IK Failed frame {frame_idx}: {e}")

        # Step physics + render
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)

        # Visualize target frame
        draw_target_frame(gym, viewer, env, ik_target_pos, r_target.as_matrix(), axis_len=0.1)

        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

        frame_idx += 1

    print("Done")
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
