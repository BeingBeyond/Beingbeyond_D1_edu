import multiprocessing
import time
from pathlib import Path
from typing import Optional, Tuple, List
from queue import Empty
from typing import Optional

import cv2
import numpy as np
import sapien
import tyro
from loguru import logger
#from sapien.asset import create_dome_envmap
#from sapien.utils import Viewer

from dex_retargeting.constants import RobotName, RetargetingType, HandType, get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig
from single_hand_wrist_detector import SingleHandWristDetector, get_aligned_images

import pyrealsense2 as rs

from dataset import DexYCBVideoDataset
from dex_retargeting.constants import RobotName, HandType
from dex_retargeting.retargeting_config import RetargetingConfig
from hand_robot_viewer import RobotHandDatasetSAPIENViewer
from hand_viewer import HandDatasetSAPIENViewer


def start_retargeting(queue: multiprocessing.Queue, robots: Optional[Tuple[RobotName]], 
    data_root: Path, data_id: int):
    dataset = DexYCBVideoDataset(data_root, hand_type="right")
    viewer = RobotHandDatasetSAPIENViewer(list(robots), HandType.right, headless=False)
    sampled_data = dataset[data_id]
    viewer.load_object_hand(sampled_data)
    viewer.render_realtime_data(queue, sampled_data)
    


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
            image, aligned_depth_frame, depth_intrin)
        queue.put([joint_pos, wrist_xyz, wrist_rpy])

        detector.draw_skeleton_on_image(frame, keypoint_2d)
        cv2.imshow("demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def main(
    robots: Optional[List[RobotName]], dexycb_dir: str, data_id: int
):
    data_root = Path(dexycb_dir).absolute()
    robot_dir = Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    RetargetingConfig.set_default_urdf_dir(robot_dir)
    if not data_root.exists():
        raise ValueError(f"Path to DexYCB dir: {data_root} does not exist.")
    else:
        print(f"Using DexYCB dir: {data_root}")

    queue = multiprocessing.Queue(maxsize=1000)
    producer_process = multiprocessing.Process(target=produce_frame, args=(queue,))
    consumer_process = multiprocessing.Process(target=start_retargeting, 
        args=(queue, robots, data_root, data_id))

    producer_process.start()
    consumer_process.start()

    producer_process.join()
    consumer_process.join()
    time.sleep(1)

    print("done")


if __name__ == "__main__":
    tyro.cli(main)
