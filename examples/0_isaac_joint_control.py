"""
Example 0: Isaac Gym joint control.

This script loads the D1 EDU URDF in Isaac Gym and opens a separate Tkinter
panel with one slider per DOF. Moving a slider updates the corresponding
Isaac Gym DOF position target in real time.

Use this simulation example before running the real robot to learn the D1 joint
names, order, ranges, and positive motion directions.

Keyboard in the Isaac Gym viewer:
  C: center all joints
  R: randomize all joints

Close the Isaac Gym viewer or the slider panel to exit.
"""

import math
import os
import random
import time
import tkinter as tk
from dataclasses import dataclass
from multiprocessing import Array, Event, Process
from typing import List, Sequence, Tuple

import numpy as np

from isaacgym import gymapi

from beingbeyond_d1_edu_sdk.urdf_path import get_default_urdf_path


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
class ViewerCfg:
    cam_pos: Tuple[float, float, float] = (1.5, 1.5, 1.2)
    cam_target: Tuple[float, float, float] = (0.0, 0.0, 0.45)


@dataclass(frozen=True)
class SliderCfg:
    width: int = 520
    height: int = 760
    slider_length: int = 280
    fallback_limit: float = math.pi
    value_resolution: float = 0.001


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
    return asset, os.path.join(asset_root, asset_file)


def create_env_and_actor(gym, sim, asset):
    env_lower = gymapi.Vec3(-2.0, 0.0, -2.0)
    env_upper = gymapi.Vec3(2.0, 2.0, 2.0)
    env = gym.create_env(sim, env_lower, env_upper, 1)

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
    pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    actor = gym.create_actor(env, asset, pose, "d1", 0, 1)
    return env, actor


def configure_dofs(gym, env, actor, asset, cfg: DofDriveCfg):
    dof_props = gym.get_asset_dof_properties(asset)
    for i in range(len(dof_props)):
        dof_props["driveMode"][i] = cfg.drive_mode
        dof_props["stiffness"][i] = cfg.stiffness
        dof_props["damping"][i] = cfg.damping
    gym.set_actor_dof_properties(env, actor, dof_props)

    num_dofs = gym.get_asset_dof_count(asset)
    dof_states = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
    targets = np.zeros(num_dofs, dtype=np.float32)
    gym.set_actor_dof_position_targets(env, actor, targets)
    gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_ALL)
    return dof_props, dof_states, targets


def set_viewer_camera(gym, viewer, cfg: ViewerCfg) -> None:
    cam_pos = gymapi.Vec3(*cfg.cam_pos)
    cam_target = gymapi.Vec3(*cfg.cam_target)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)


def sanitize_limits(
    lower: Sequence[float],
    upper: Sequence[float],
    cfg: SliderCfg,
) -> Tuple[np.ndarray, np.ndarray]:
    lo = np.asarray(lower, dtype=np.float64).copy()
    hi = np.asarray(upper, dtype=np.float64).copy()

    for i in range(len(lo)):
        bad = (
            not np.isfinite(lo[i])
            or not np.isfinite(hi[i])
            or hi[i] <= lo[i]
            or abs(hi[i] - lo[i]) > 100.0
        )
        if bad:
            lo[i] = -cfg.fallback_limit
            hi[i] = cfg.fallback_limit
    return lo, hi


def center_values(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    values = np.zeros_like(lower, dtype=np.float64)
    for i, (lo, hi) in enumerate(zip(lower, upper)):
        if lo <= 0.0 <= hi:
            values[i] = 0.0
        else:
            values[i] = 0.5 * (lo + hi)
    return values


def random_values(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.asarray([random.uniform(float(lo), float(hi)) for lo, hi in zip(lower, upper)], dtype=np.float64)


def write_shared(shared_targets, values: np.ndarray) -> None:
    with shared_targets.get_lock():
        for i, value in enumerate(values):
            shared_targets[i] = float(value)


def read_shared(shared_targets, out: np.ndarray) -> None:
    with shared_targets.get_lock():
        for i in range(len(out)):
            out[i] = shared_targets[i]


def slider_panel_worker(
    joint_names: List[str],
    lower_list: List[float],
    upper_list: List[float],
    initial_list: List[float],
    shared_targets,
    stop_event,
    randomize_event,
    center_event,
    cfg: SliderCfg,
) -> None:
    lower = np.asarray(lower_list, dtype=np.float64)
    upper = np.asarray(upper_list, dtype=np.float64)
    current = np.asarray(initial_list, dtype=np.float64)

    root = tk.Tk()
    root.title("D1 Isaac Gym Joint Control")
    root.geometry(f"{cfg.width}x{cfg.height}")

    top = tk.Frame(root)
    top.pack(fill=tk.X, padx=8, pady=6)

    def set_all(values: np.ndarray) -> None:
        nonlocal current
        current = np.clip(values.astype(np.float64), lower, upper)
        for i, var in enumerate(slider_vars):
            var.set(float(current[i]))
            value_vars[i].set(f"{current[i]: .3f}")
        write_shared(shared_targets, current)

    def on_randomize() -> None:
        set_all(random_values(lower, upper))

    def on_center() -> None:
        set_all(center_values(lower, upper))

    tk.Button(top, text="Randomize", command=on_randomize).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=3)
    tk.Button(top, text="Center", command=on_center).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=3)

    canvas = tk.Canvas(root)
    scrollbar = tk.Scrollbar(root, orient=tk.VERTICAL, command=canvas.yview)
    scroll_frame = tk.Frame(canvas)

    scroll_frame.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    slider_vars = []
    value_vars = []

    def on_slider(idx: int, value: str) -> None:
        current[idx] = float(value)
        value_vars[idx].set(f"{current[idx]: .3f}")
        write_shared(shared_targets, current)

    for idx, name in enumerate(joint_names):
        row = tk.Frame(scroll_frame)
        row.pack(fill=tk.X, padx=8, pady=4)

        label = tk.Label(row, text=name, width=28, anchor="w")
        label.pack(side=tk.LEFT)

        var = tk.DoubleVar(value=float(current[idx]))
        value_var = tk.StringVar(value=f"{current[idx]: .3f}")
        slider_vars.append(var)
        value_vars.append(value_var)

        slider = tk.Scale(
            row,
            from_=float(lower[idx]),
            to=float(upper[idx]),
            resolution=cfg.value_resolution,
            orient=tk.HORIZONTAL,
            length=cfg.slider_length,
            variable=var,
            showvalue=False,
            command=lambda value, i=idx: on_slider(i, value),
        )
        slider.pack(side=tk.LEFT, padx=5)

        value_label = tk.Label(row, textvariable=value_var, width=8, anchor="e")
        value_label.pack(side=tk.LEFT)

    def poll_events() -> None:
        if stop_event.is_set():
            root.destroy()
            return
        if randomize_event.is_set():
            randomize_event.clear()
            on_randomize()
        if center_event.is_set():
            center_event.clear()
            on_center()
        root.after(50, poll_events)

    def on_close() -> None:
        stop_event.set()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    write_shared(shared_targets, current)
    root.after(50, poll_events)
    root.mainloop()


def main() -> None:
    asset_cfg = AssetCfg()
    sim_cfg = SimCfg()
    dof_cfg = DofDriveCfg()
    viewer_cfg = ViewerCfg()
    slider_cfg = SliderCfg()

    gym = gymapi.acquire_gym()
    sim = create_sim(gym, sim_cfg)
    viewer = create_viewer(gym, sim)
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_C, "center")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "randomize")
    add_ground(gym, sim)

    asset, urdf_path = load_asset(gym, sim, asset_cfg)
    print(f"Loaded Isaac asset: {urdf_path}")

    env, actor = create_env_and_actor(gym, sim, asset)
    dof_props, dof_states, targets = configure_dofs(gym, env, actor, asset, dof_cfg)
    set_viewer_camera(gym, viewer, viewer_cfg)

    joint_names = list(gym.get_actor_dof_names(env, actor))
    lower, upper = sanitize_limits(dof_props["lower"], dof_props["upper"], slider_cfg)
    initial = center_values(lower, upper)

    targets[:] = initial.astype(np.float32)
    dof_states["pos"][:] = targets
    dof_states["vel"][:] = 0.0
    gym.set_actor_dof_position_targets(env, actor, targets)
    gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_ALL)

    print(f"DOF count: {len(joint_names)}")
    for i, name in enumerate(joint_names):
        print(f"{i:02d}: {name:32s} [{lower[i]: .3f}, {upper[i]: .3f}] init={initial[i]: .3f}")

    shared_targets = Array("d", initial.tolist(), lock=True)
    stop_event = Event()
    randomize_event = Event()
    center_event = Event()
    panel = Process(
        target=slider_panel_worker,
        args=(
            joint_names,
            lower.tolist(),
            upper.tolist(),
            initial.tolist(),
            shared_targets,
            stop_event,
            randomize_event,
            center_event,
            slider_cfg,
        ),
        daemon=True,
    )
    panel.start()

    last_targets = targets.astype(np.float64)
    tmp_targets = np.zeros_like(last_targets)

    print("Joint control is running.")
    print("Use the slider panel to move joints. Press C to center, R to randomize.")

    try:
        while not gym.query_viewer_has_closed(viewer) and not stop_event.is_set():
            for evt in gym.query_viewer_action_events(viewer):
                if evt.value <= 0:
                    continue
                if evt.action == "center":
                    center = center_values(lower, upper)
                    write_shared(shared_targets, center)
                    center_event.set()
                elif evt.action == "randomize":
                    rand = random_values(lower, upper)
                    write_shared(shared_targets, rand)
                    randomize_event.set()

            read_shared(shared_targets, tmp_targets)
            if not np.allclose(tmp_targets, last_targets, atol=1e-6):
                targets[:] = tmp_targets.astype(np.float32)
                gym.set_actor_dof_position_targets(env, actor, targets)
                last_targets[:] = tmp_targets

            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)

    finally:
        stop_event.set()
        time.sleep(0.1)
        if panel.is_alive():
            panel.terminate()
            panel.join(timeout=1.0)
        gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
