"""
D1 Real-time Control with Vision Pro Stream in Isaac Gym

- Combines Vision Pro hand stream (shared memory) + Isaac Gym control loop.
- Keeps original behavior:
  - Multiprocessing streamer process writes hand pose + finger transforms to SHM
  - Main process reads SHM, performs retargeting + IK, drives Isaac Gym robot
  - Press 'S' in viewer to calibrate (set origin) and start teleoperation
  - Optional video streaming: Isaac Gym camera -> RGB SHM -> streamer update_frame()
"""

import os
import math
import time
import importlib
import multiprocessing
from multiprocessing import shared_memory
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

import pinocchio

from isaacgym import gymapi
from isaacgym import gymutil

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig

from avp_stream import VisionProStreamer
from beingbeyond_d1_edu_sdk.pin_kinematics import D1Kinematics, D1KinematicsConfig
from beingbeyond_d1_edu_sdk.urdf_path import get_default_urdf_path


@dataclass(frozen=True)
class SharedMemCfg:
    """
    Control SHM layout (float64):
      [0]   timestamp
      [1]   has_right_hand (1.0/0.0)
      [2]   pinch_distance
      [3:19]  right_wrist_matrix (16 floats, 4x4 row-major)
      [19]  frame_counter (monotonic in streamer process)
      [20:420] right_fingers (400 floats, 25x4x4 row-major)
    """
    name: str = "d1_vision_pro_shm"
    n_f64: int = 420
    n_bytes: int = 420 * 8


@dataclass(frozen=True)
class RgbShmCfg:
    """
    RGB SHM layout:
      offset 0:   frame_counter (float64)
      offset 8:   RGB uint8 data (H*W*C)
    """
    name: str = "d1_vision_pro_rgb_shm"
    width: int = 1280
    height: int = 720
    channels: int = 3

    @property
    def rgb_size(self) -> int:
        return self.width * self.height * self.channels

    @property
    def n_bytes(self) -> int:
        return 8 + self.rgb_size


@dataclass(frozen=True)
class RuntimeCfg:
    visualize_video: bool = True


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
    use_gpu_pipeline: bool = False  # keep forced CPU pipeline


@dataclass(frozen=True)
class DofDriveCfg:
    stiffness: float = 400.0
    damping: float = 40.0
    drive_mode: int = gymapi.DOF_MODE_POS


@dataclass(frozen=True)
class ViewerCfg:
    cam_pos: Tuple[float, float, float] = (0.8, 0.8, 0.8)
    cam_target: Tuple[float, float, float] = (0.0, 0.0, 0.4)


@dataclass(frozen=True)
class ControlCfg:
    # Initial Joint Config: head(2) + arm(6) = 8
    q_init_deg: Tuple[float, ...] = (
        0.0, 0.0,          # head_yaw, head_pitch
        0.0, -60.0, 60.0,  # shoulder / elbow
        0.0, 0.0, 0.0,     # wrist joints
    )
    correction_euler_deg: Tuple[float, float, float] = (0.0, 0.0, -90.0)
    pos_scale: float = 1.0
    min_z: float = 0.05

    max_outer_iters: int = 5
    pos_tol: float = 0.01
    rot_tol: float = 0.1


@dataclass
class TeleopState:
    calibrated: bool = False
    request_calibration: bool = False
    initial_hand_pos: Optional[np.ndarray] = None
    initial_hand_rot: Optional[np.ndarray] = None
    last_processed_frame_counter: float = -1.0


@dataclass
class FpsMeter:
    last_log_time: float = 0.0
    frames_since_log: int = 0

    def reset(self) -> None:
        self.last_log_time = time.time()
        self.frames_since_log = 0

    def tick_and_maybe_log(self, fid: float) -> None:
        self.frames_since_log += 1
        now = time.time()
        if now - self.last_log_time > 1.0:
            dt = now - self.last_log_time
            fps = self.frames_since_log / max(dt, 1e-6)
            print(f"FPS: {fps:.1f} | FID: {fid:.0f}")
            self.last_log_time = now
            self.frames_since_log = 0


OPERATOR2AVP_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ],
    dtype=np.float64,
)


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


def try_unlink_shm(name: str) -> None:
    try:
        shm = shared_memory.SharedMemory(name=name)
        shm.close()
        shm.unlink()
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"WARNING: Failed to unlink SHM '{name}': {e}")


def open_control_shm(cfg: SharedMemCfg) -> Tuple[shared_memory.SharedMemory, np.ndarray]:
    shm = shared_memory.SharedMemory(name=cfg.name)
    buf = np.ndarray((cfg.n_f64,), dtype=np.float64, buffer=shm.buf)
    return shm, buf


def open_rgb_shm(cfg: RgbShmCfg) -> Tuple[shared_memory.SharedMemory, np.ndarray, np.ndarray]:
    shm = shared_memory.SharedMemory(name=cfg.name)
    counter = np.ndarray((1,), dtype=np.float64, buffer=shm.buf, offset=0)
    rgb = np.ndarray((cfg.height, cfg.width, cfg.channels), dtype=np.uint8, buffer=shm.buf, offset=8)
    return shm, counter, rgb


def create_control_shm(cfg: SharedMemCfg) -> Tuple[shared_memory.SharedMemory, np.ndarray]:
    try:
        shm = shared_memory.SharedMemory(name=cfg.name, create=True, size=cfg.n_bytes)
    except FileExistsError:
        shm = shared_memory.SharedMemory(name=cfg.name)
    buf = np.ndarray((cfg.n_f64,), dtype=np.float64, buffer=shm.buf)
    buf[:] = 0.0
    return shm, buf


def create_rgb_shm(cfg: RgbShmCfg) -> Tuple[shared_memory.SharedMemory, np.ndarray, np.ndarray]:
    try:
        shm = shared_memory.SharedMemory(name=cfg.name, create=True, size=cfg.n_bytes)
    except FileExistsError:
        shm = shared_memory.SharedMemory(name=cfg.name)
    counter = np.ndarray((1,), dtype=np.float64, buffer=shm.buf, offset=0)
    rgb = np.ndarray((cfg.height, cfg.width, cfg.channels), dtype=np.uint8, buffer=shm.buf, offset=8)
    counter[0] = 0.0
    return shm, counter, rgb


def streamer_worker(ip: str, stop_event, shm_cfg: SharedMemCfg, rgb_cfg: RgbShmCfg, runtime_cfg: RuntimeCfg) -> None:
    """
    Worker process:
      - Connects to VisionProStreamer
      - Writes latest right hand data into Control SHM
      - If visualize_video enabled: reads RGB SHM and feeds streamer.update_frame()
    """
    from avp_stream import VisionProStreamer
    import cv2  # noqa: F401

    print(f"[StreamerProcess] Starting streamer process for {ip}...")

    shm, shared_buffer = create_control_shm(shm_cfg)

    rgb_shm = None
    rgb_counter = None
    rgb_data = None
    if runtime_cfg.visualize_video:
        rgb_shm, rgb_counter, rgb_data = create_rgb_shm(rgb_cfg)

    try:
        streamer = VisionProStreamer(ip=ip, record=False)
        if runtime_cfg.visualize_video:
            streamer.configure_video(format="rgb", fps=60, size=f"{rgb_cfg.width}x{rgb_cfg.height}")
        streamer.start_webrtc()
        print("[StreamerProcess] Streamer initialized. Waiting for data...")
    except Exception as e:
        print(f"[StreamerProcess] CRITICAL ERROR: Failed to initialize streamer: {e}")
        try:
            shm.close()
            shm.unlink()
        except Exception:
            pass
        if runtime_cfg.visualize_video and rgb_shm is not None:
            try:
                rgb_shm.close()
                rgb_shm.unlink()
            except Exception:
                pass
        return

    frame_count = 0
    last_data_id = -1
    last_log_time = time.time()
    last_rgb_frame_idx = -1.0

    while not stop_event.is_set():
        try:
            latest = streamer.get_latest()
            current_raw = latest.raw if latest else None

            if current_raw is not None and id(current_raw) != last_data_id:
                last_data_id = id(current_raw)
                if latest.right is not None:
                    shared_buffer[0] = time.time()

                    wrist = latest.right.wrist
                    if wrist is not None and not np.isnan(wrist).any():
                        shared_buffer[1] = 1.0
                        shared_buffer[2] = float(latest.right.pinch_distance)
                        shared_buffer[3:19] = wrist.flatten()

                        frame_count += 1
                        shared_buffer[19] = float(frame_count)

                        if "right_fingers" in current_raw:
                            shared_buffer[20:420] = current_raw["right_fingers"].flatten()
                    else:
                        shared_buffer[1] = 0.0
                else:
                    shared_buffer[1] = 0.0

            if runtime_cfg.visualize_video and rgb_counter is not None:
                current_rgb_idx = float(rgb_counter[0])
                if current_rgb_idx > last_rgb_frame_idx:
                    last_rgb_frame_idx = current_rgb_idx
                    streamer.update_frame(rgb_data)

            if time.time() - last_log_time > 1.0:
                val_readback = float(shared_buffer[6])
                video_idx_log = f"{last_rgb_frame_idx:.0f}" if runtime_cfg.visualize_video else "Off"
                print(
                    f"[StreamerProcess] Frames: {frame_count} | "
                    f"SHM Wrist[0,3] (Tx): {val_readback:.4f} | "
                    f"Video Idx: {video_idx_log}"
                )
                last_log_time = time.time()

            time.sleep(0.01)

        except Exception as e:
            print(f"[StreamerProcess] Error in loop: {e}")
            time.sleep(0.1)

    print("[StreamerProcess] Stopping...")
    try:
        shm.close()
        shm.unlink()
    except Exception:
        pass

    if runtime_cfg.visualize_video and rgb_shm is not None:
        try:
            rgb_shm.close()
            rgb_shm.unlink()
        except Exception:
            pass


# -----------------------------
# Isaac Gym helpers
# -----------------------------

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


def load_asset(gym, sim, cfg: AssetCfg) -> gymapi.Asset:
    if not os.path.isfile(cfg.asset_path):
        raise FileNotFoundError(f"URDF not found: {cfg.asset_path}")

    asset_root = os.path.dirname(cfg.asset_path)
    asset_file = os.path.basename(cfg.asset_path)

    opt = gymapi.AssetOptions()
    opt.fix_base_link = cfg.fix_base_link
    opt.flip_visual_attachments = cfg.flip_visual_attachments
    opt.armature = cfg.armature

    print(f"Loading asset '{asset_file}' from '{asset_root}'")
    asset = gym.load_asset(sim, asset_root, asset_file, opt)
    return asset


def create_env_and_actor(gym, sim, asset) -> Tuple[gymapi.Env, int]:
    env_lower = gymapi.Vec3(-2.0, 0.0, -2.0)
    env_upper = gymapi.Vec3(2.0, 2.0, 2.0)
    env = gym.create_env(sim, env_lower, env_upper, 1)

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
    pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    actor = gym.create_actor(env, asset, pose, "d1", 0, 1)
    return env, actor


def configure_dofs(gym, env, actor, asset, cfg: DofDriveCfg) -> Tuple[int, np.ndarray]:
    dof_props = gym.get_asset_dof_properties(asset)
    for i in range(len(dof_props)):
        dof_props["driveMode"][i] = cfg.drive_mode
        dof_props["stiffness"][i] = cfg.stiffness
        dof_props["damping"][i] = cfg.damping
    gym.set_actor_dof_properties(env, actor, dof_props)

    num_dofs = gym.get_asset_dof_count(asset)
    dof_states = gym.get_actor_dof_states(env, actor, gymapi.STATE_NONE)
    return num_dofs, dof_states


def set_initial_targets(gym, env, actor, num_dofs: int, dof_states: np.ndarray, q_init_rad: np.ndarray) -> np.ndarray:
    targets = np.zeros(num_dofs, dtype=np.float32)
    n = min(len(q_init_rad), num_dofs)
    targets[:n] = q_init_rad[:n]
    gym.set_actor_dof_position_targets(env, actor, targets)
    gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_POS)
    return targets


def set_viewer_camera(gym, viewer, cfg: ViewerCfg) -> None:
    cam_pos = gymapi.Vec3(*cfg.cam_pos)
    cam_target = gymapi.Vec3(*cfg.cam_target)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)


def create_rgb_camera_sensor(gym, env, cam_pos, cam_target, rgb_cfg: RgbShmCfg) -> int:
    camera_props = gymapi.CameraProperties()
    camera_props.width = rgb_cfg.width
    camera_props.height = rgb_cfg.height
    camera_props.enable_tensors = False
    cam = gym.create_camera_sensor(env, camera_props)
    gym.set_camera_location(cam, env, cam_pos, cam_target)
    return cam


# -----------------------------
# Kinematics / retargeting
# -----------------------------

def init_kinematics(urdf_path: str) -> D1Kinematics:
    kin_cfg = D1KinematicsConfig(urdf_path=urdf_path)
    return D1Kinematics(kin_cfg)


def init_right_hand_retargeting(robot_name: RobotName) -> object:
    print("Initializing Dex Retargeting...")

    dex_pkg = importlib.util.find_spec("dex_retargeting")
    if dex_pkg and dex_pkg.origin:
        robot_dir = Path(dex_pkg.origin).absolute().parent.parent / "assets" / "robots" / "hands"
        RetargetingConfig.set_default_urdf_dir(str(robot_dir))

    override = dict(add_dummy_free_joint=True)
    right_config_path = get_default_config_path(robot_name, RetargetingType.dexpilot, HandType.right)
    right_cfg = RetargetingConfig.load_from_file(right_config_path, override=override)
    return right_cfg.build()


# -----------------------------
# Control computation
# -----------------------------

def compute_target_pose(
    ctrl_cfg: ControlCfg,
    initial_ee_pos: np.ndarray,
    initial_ee_R: np.ndarray,
    initial_hand_pos: np.ndarray,
    initial_hand_rot: np.ndarray,
    curr_hand_pos: np.ndarray,
    curr_hand_R: np.ndarray,
) -> Tuple[np.ndarray, R]:
    r_correction = R.from_euler("xyz", ctrl_cfg.correction_euler_deg, degrees=True)
    mat_corr = r_correction.as_matrix()

    pos_delta = curr_hand_pos - initial_hand_pos
    pos_delta = mat_corr @ pos_delta
    pos_delta = pos_delta * ctrl_cfg.pos_scale

    ik_target_pos = initial_ee_pos + pos_delta
    if ik_target_pos[2] < ctrl_cfg.min_z:
        ik_target_pos[2] = ctrl_cfg.min_z

    r_rec_init = R.from_matrix(initial_hand_rot)
    r_rec_curr = R.from_matrix(curr_hand_R)
    r_delta_data = r_rec_curr * r_rec_init.inv()
    r_delta = r_correction * r_delta_data * r_correction.inv()

    r_robot_init = R.from_matrix(initial_ee_R)
    r_target = r_delta * r_robot_init
    return ik_target_pos, r_target


def solve_ik_iterative(
    kin: D1Kinematics,
    ctrl_cfg: ControlCfg,
    q_head: np.ndarray,
    q_arm: np.ndarray,
    ik_target_pos: np.ndarray,
    r_target: R,
) -> Tuple[np.ndarray, np.ndarray]:
    q_head_iter = q_head.copy()
    q_arm_iter = q_arm.copy()

    q_target_xyzw = r_target.as_quat()
    target_quatpose = np.zeros(7, dtype=float)
    target_quatpose[0:3] = ik_target_pos
    target_quatpose[3:7] = q_target_xyzw

    for _ in range(ctrl_cfg.max_outer_iters):
        T_curr = kin.ee_in_base(q_head_iter, q_arm_iter)
        p_curr = T_curr[:3, 3]
        R_curr = T_curr[:3, :3]

        pos_err = float(np.linalg.norm(ik_target_pos - p_curr))
        rot_err = rotation_angle_error(r_target.as_matrix(), R_curr)

        if pos_err < ctrl_cfg.pos_tol and rot_err < ctrl_cfg.rot_tol:
            break

        q_head_sol, q_arm_sol, cost, inner_iters = kin.ik_ee_quatpose_with_arm_only(
            target_ee_quatpose=target_quatpose,
            q_head=q_head_iter,
            q_arm_init=q_arm_iter,
        )
        q_head_iter = q_head_sol
        q_arm_iter = q_arm_sol

    return q_head_iter, q_arm_iter


def apply_hand_retargeting(
    fingers_mat_25: np.ndarray,
    right_retargeting: object,
    qpos_map: np.ndarray,
    current_targets: np.ndarray,
) -> None:
    if fingers_mat_25 is None:
        return
    if np.isnan(fingers_mat_25).all():
        return

    joint_pose = two_mat_batch_mul(fingers_mat_25, OPERATOR2AVP_RIGHT.T)
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
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

    vpos = pos.astype(np.float32)
    Rm = rot.astype(np.float32)

    vertices[0, 0] = vpos
    vertices[0, 1] = vpos + Rm @ np.array([axis_len, 0, 0], dtype=np.float32)

    vertices[1, 0] = vpos
    vertices[1, 1] = vpos + Rm @ np.array([0, axis_len, 0], dtype=np.float32)

    vertices[2, 0] = vpos
    vertices[2, 1] = vpos + Rm @ np.array([0, 0, axis_len], dtype=np.float32)

    gym.add_lines(viewer, env, 3, vertices, colors)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    shm_cfg = SharedMemCfg()
    rgb_cfg = RgbShmCfg()
    runtime_cfg = RuntimeCfg()
    asset_cfg = AssetCfg()
    sim_cfg = SimCfg()
    dof_cfg = DofDriveCfg()
    viewer_cfg = ViewerCfg()
    ctrl_cfg = ControlCfg()

    custom_parameters = [{"name": "--ip", "type": str, "help": "Vision Pro IP address"}]
    args = gymutil.parse_arguments(description="D1 Real-time Control", custom_parameters=custom_parameters)

    stop_event = None
    streamer_process = None

    ctrl_shm = None
    ctrl_buf = None

    rgb_shm = None
    rgb_counter = None
    rgb_data = None

    if args.ip:
        print(f"Connecting to Vision Pro at {args.ip}...")

        try_unlink_shm(shm_cfg.name)
        if runtime_cfg.visualize_video:
            try_unlink_shm(rgb_cfg.name)

        stop_event = multiprocessing.Event()
        streamer_process = multiprocessing.Process(
            target=streamer_worker,
            args=(args.ip, stop_event, shm_cfg, rgb_cfg, runtime_cfg),
            daemon=True,
        )
        streamer_process.start()
        time.sleep(1.0)

        try:
            ctrl_shm, ctrl_buf = open_control_shm(shm_cfg)
            if runtime_cfg.visualize_video:
                rgb_shm, rgb_counter, rgb_data = open_rgb_shm(rgb_cfg)
            print("Shared Memory (Control & Video) connected successfully.")
        except FileNotFoundError:
            print("CRITICAL: Shared Memory not found! Streamer process might have failed.")
            return
    else:
        print("WARNING: No IP provided. Running in simulation-only mode (no input).")

    gym = gymapi.acquire_gym()
    sim = create_sim(gym, args, sim_cfg)
    viewer = create_viewer(gym, sim)

    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_S, "calib_start")
    add_ground(gym, sim)

    asset = load_asset(gym, sim, asset_cfg)
    env, actor = create_env_and_actor(gym, sim, asset)
    num_dofs, dof_states = configure_dofs(gym, env, actor, asset, dof_cfg)

    q_init_rad = np.array([math.radians(v) for v in ctrl_cfg.q_init_deg], dtype=np.float64)
    current_targets = set_initial_targets(gym, env, actor, num_dofs, dof_states, q_init_rad)

    set_viewer_camera(gym, viewer, viewer_cfg)

    camera_handle = None
    cam_pos_vec = gymapi.Vec3(*viewer_cfg.cam_pos)
    cam_target_vec = gymapi.Vec3(*viewer_cfg.cam_target)
    if runtime_cfg.visualize_video:
        camera_handle = create_rgb_camera_sensor(gym, env, cam_pos_vec, cam_target_vec, rgb_cfg)

    print(f"Initializing D1 Kinematics with URDF: {asset_cfg.asset_path}")
    kin = init_kinematics(asset_cfg.asset_path)

    robot_name = RobotName.linkero6
    right_retargeting = init_right_hand_retargeting(robot_name)

    retarget_joint_names = right_retargeting.joint_names
    print(f"Retargeting Output Joints: {retarget_joint_names}")

    gym_joint_names = gym.get_actor_dof_names(env, actor)
    print(f"Gym Actor Joints: {gym_joint_names}")

    qpos_map = build_qpos_map(gym_joint_names, retarget_joint_names)
    print(f"Qpos Map: {qpos_map}")

    q_head = q_init_rad[:2].copy()
    q_arm = q_init_rad[2:8].copy()

    T_base_ee_init = kin.ee_in_base(q_head, q_arm)
    initial_ee_pos = T_base_ee_init[:3, 3].copy()
    initial_ee_R = T_base_ee_init[:3, :3].copy()

    state = TeleopState()
    fps = FpsMeter()
    fps.reset()

    print("Starting real-time control loop...")
    print("Please put on Vision Pro and ensure stream is active.")
    print("Press 'S' to calibrate and start teleoperation.")

    sim_frame_idx = 0

    while not gym.query_viewer_has_closed(viewer):
        for evt in gym.query_viewer_action_events(viewer):
            if evt.action == "calib_start" and evt.value > 0:
                print("Key 'S' pressed. Requesting calibration...")
                state.request_calibration = True

        has_hand_data = False
        curr_pos = None
        curr_R = None
        fingers_mat = None
        fid = -1.0

        if ctrl_buf is not None:
            fid = float(ctrl_buf[19])
            if fid > state.last_processed_frame_counter:
                state.last_processed_frame_counter = fid

                if float(ctrl_buf[1]) > 0.5:
                    has_hand_data = True

                    wrist_flat = ctrl_buf[3:19].copy()
                    wrist_mat = wrist_flat.reshape(4, 4)
                    curr_pos = wrist_mat[:3, 3].copy()
                    curr_R = wrist_mat[:3, :3].copy()

                    fingers_flat = ctrl_buf[20:420].copy()
                    fingers_mat = fingers_flat.reshape(25, 4, 4)

        if has_hand_data:
            if fingers_mat is not None:
                apply_hand_retargeting(fingers_mat, right_retargeting, qpos_map, current_targets)

            if not state.calibrated and state.request_calibration:
                print("Calibrating origin...")
                state.initial_hand_pos = curr_pos.copy()
                state.initial_hand_rot = curr_R.copy()
                state.calibrated = True
                state.request_calibration = False
                print(f"Calibration Done. Initial Hand Pos: {state.initial_hand_pos}")

            if state.calibrated and state.initial_hand_pos is not None and state.initial_hand_rot is not None:
                ik_target_pos, r_target = compute_target_pose(
                    ctrl_cfg=ctrl_cfg,
                    initial_ee_pos=initial_ee_pos,
                    initial_ee_R=initial_ee_R,
                    initial_hand_pos=state.initial_hand_pos,
                    initial_hand_rot=state.initial_hand_rot,
                    curr_hand_pos=curr_pos,
                    curr_hand_R=curr_R,
                )

                try:
                    q_head_new, q_arm_new = solve_ik_iterative(
                        kin=kin,
                        ctrl_cfg=ctrl_cfg,
                        q_head=q_head,
                        q_arm=q_arm,
                        ik_target_pos=ik_target_pos,
                        r_target=r_target,
                    )
                    q_head = q_head_new
                    q_arm = q_arm_new

                    current_targets[0:2] = q_head
                    current_targets[2:8] = q_arm
                    gym.set_actor_dof_position_targets(env, actor, current_targets)

                except Exception as e:
                    print(f"IK Failed: {e}")

                draw_target_frame(gym, viewer, env, ik_target_pos, r_target.as_matrix(), axis_len=0.1)
                fps.tick_and_maybe_log(fid=fid)

        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)

        if (
            runtime_cfg.visualize_video
            and rgb_shm is not None
            and camera_handle is not None
            and rgb_counter is not None
            and rgb_data is not None
        ):
            gym.render_all_camera_sensors(sim)
            image = gym.get_camera_image(sim, env, camera_handle, gymapi.IMAGE_COLOR)
            image = image.reshape(rgb_cfg.height, rgb_cfg.width, 4)
            rgb_data[:] = image[:, :, :3]
            rgb_counter[0] = float(sim_frame_idx)

        gym.sync_frame_time(sim)
        sim_frame_idx += 1

    print("Done")

    if stop_event is not None:
        stop_event.set()
    if streamer_process is not None:
        streamer_process.join(timeout=1.0)
        if streamer_process.is_alive():
            streamer_process.terminate()

    try:
        if ctrl_shm is not None:
            ctrl_shm.close()
    except Exception:
        pass

    try:
        if rgb_shm is not None:
            rgb_shm.close()
    except Exception:
        pass

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()