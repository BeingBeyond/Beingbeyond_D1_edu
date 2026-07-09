import multiprocessing
import time
from pathlib import Path
from typing import Optional, Tuple, List
from queue import Empty
from typing import Optional

import cv2
import subprocess
import numpy as np
import sapien
import tyro
from loguru import logger
#from sapien.asset import create_dome_envmap
#from sapien.utils import Viewer

from dex_retargeting.constants import RobotName, RetargetingType, HandType, get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig
from single_hand_wrist_detector import SingleHandWristDetector, get_aligned_images, MujocoTrans
import pyrealsense2 as rs
from dex_retargeting.constants import RobotName, HandType
from dex_retargeting.retargeting_config import RetargetingConfig
from mujoco_tasks.table_cube_inspire import TableCubeInspireEnv
from tqdm import trange

def start_retargeting(queue: multiprocessing.Queue, robot: RobotName, tau=0.9):
    wrist_offset = np.array([-0.3,0,1])
    env = TableCubeInspireEnv(render_mode="human")
    config_path = get_default_config_path(robot, RetargetingType.dexpilot, HandType.right)
    override = dict(add_dummy_free_joint=True)
    config = RetargetingConfig.load_from_file(config_path, override=override)
    retargeting = config.build()
    #print(retargeting.joint_names, env.actuator_names)
    retarget2sim = np.array([retargeting.joint_names.index(n[2:]) for n in env.actuator_names]).astype(int)
    print(retarget2sim) # actuator indices mapping from retargeting results to mujoco sim
    env.reset()
    env.render()
    subprocess.run(['wmctrl', '-a', 'demo'])

    last_act, last_wrist_xyz, last_wrist_rpy = np.zeros(12), wrist_offset, np.zeros(3)
    while True:
        #if last_act is not None:
        env.step(last_act, last_wrist_xyz, last_wrist_rpy)
        
        try:
            joint_pos, wrist_xyz, wrist_rpy = queue.get(timeout=0) #(timeout=1)
        except Empty:
            #print("Fail to fetch joint_pos.")
            #return
            continue
        if joint_pos is None:
            print("hand is not detected.")
            continue

        # hand vector retargeting
        indices = retargeting.optimizer.target_link_human_indices
        origin_indices = indices[0, :]
        task_indices = indices[1, :]
        ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
        act_qpos = retargeting.retarget(ref_value)[retarget2sim]
        last_act = (1-tau)*act_qpos + tau*last_act
        last_wrist_xyz = (1-tau)*(wrist_xyz+wrist_offset) + tau*last_wrist_xyz
        last_wrist_rpy = (1-tau)*wrist_rpy + tau*last_wrist_rpy
        


def produce_frame(queue: multiprocessing.Queue):
    hand_type = "Right"
    detector = SingleHandWristDetector(hand_type=hand_type, selfie=False)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)

    cv2.namedWindow("demo", cv2.WINDOW_NORMAL)
    cv2.moveWindow("demo", 1500, 0)

    while True:
        color_intrin, depth_intrin, img_color, img_depth, aligned_depth_frame = get_aligned_images(pipeline, align)

        frame = img_color
        image = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
        joint_pos, wrist_xyz, wrist_rpy, keypoint_2d = detector.detect_hand_wrist(
            image, aligned_depth_frame, depth_intrin, MujocoTrans)
        queue.put([joint_pos, wrist_xyz, wrist_rpy])

        detector.draw_skeleton_on_image(frame, keypoint_2d)
        cv2.imshow("demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def main(robot: RobotName):
    robot_dir = Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    RetargetingConfig.set_default_urdf_dir(robot_dir)
    print(robot)

    queue = multiprocessing.Queue(maxsize=1000)
    producer_process = multiprocessing.Process(target=produce_frame, args=(queue,))
    consumer_process = multiprocessing.Process(target=start_retargeting, args=(queue, robot))

    producer_process.start()
    consumer_process.start()

    producer_process.join()
    consumer_process.join()
    time.sleep(1)
    print("done")


if __name__ == "__main__":
    tyro.cli(main)
