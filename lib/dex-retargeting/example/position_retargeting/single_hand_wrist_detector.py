'''
Detect single hand with wrist position and 
'''

import numpy as np
from single_hand_detector import SingleHandDetector
from scipy.spatial.transform import Rotation as R
import pyrealsense2 as rs
import time


class SapienTrans:
    # d435 coordinate to simulator coordinate
    Camera2RobotXYZ = np.array([0,0,0])
    Camera2RobotRPY = np.array([-np.pi/2, 0, -np.pi/2])
    Camera2RobotRot = R.from_euler('xyz', Camera2RobotRPY, degrees=False).as_matrix()
    # MediaPipe wrist coordinate to simulator coordinate
    #MPMirror = np.array([1,1,1])
    MP2RobotRPY = np.array([0, np.pi/2, 0])
    MP2RobotRot = R.from_euler('xyz', MP2RobotRPY, degrees=False).as_matrix()
    MP2RobotRPY_init_diff = np.array([-np.pi/2,0,-np.pi/2]) # initial state difference

class MujocoTrans:
    # d435 coordinate to simulator coordinate
    Camera2RobotXYZ = np.array([0,0,0])
    Camera2RobotRPY = np.array([-np.pi/2, 0, -np.pi/2])
    Camera2RobotRot = R.from_euler('xyz', Camera2RobotRPY, degrees=False).as_matrix()
    # MediaPipe wrist coordinate to simulator coordinate
    #MPMirror = np.array([-1,1,1])
    MP2RobotRPY = np.array([-np.pi/2, 0, -np.pi])
    MP2RobotRot = R.from_euler('xyz', MP2RobotRPY, degrees=False).as_matrix()
    MP2RobotRPY_init_diff = np.array([np.pi/2,0,np.pi/2]) # initial state difference


''' 
获取对齐图像帧与相机参数
'''
def get_aligned_images(pipeline, align):
    
    frames = pipeline.wait_for_frames()     # 等待获取图像帧，获取颜色和深度的框架集     
    aligned_frames = align.process(frames)      # 获取对齐帧，将深度框与颜色框对齐  

    aligned_depth_frame = aligned_frames.get_depth_frame()      # 获取对齐帧中的的depth帧 
    aligned_color_frame = aligned_frames.get_color_frame()      # 获取对齐帧中的的color帧

    #### 获取相机参数 ####
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics     # 获取深度参数（像素坐标系转相机坐标系会用到）
    color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics     # 获取相机内参


    #### 将images转为numpy arrays ####  
    img_color = np.asanyarray(aligned_color_frame.get_data())       # RGB图  
    img_depth = np.asanyarray(aligned_depth_frame.get_data())       # 深度图（默认16位）

    return color_intrin, depth_intrin, img_color, img_depth, aligned_depth_frame

''' 
获取指定像素三维坐标
'''
def get_3d_camera_coordinate(depth_pixel, aligned_depth_frame, depth_intrin):
    x = depth_pixel[0]
    y = depth_pixel[1]
    dis = aligned_depth_frame.get_distance(x, y)        # 获取该像素点对应的深度
    # print ('depth: ',dis)       # 深度单位是m
    camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dis)
    # print ('camera_coordinate: ',camera_coordinate)
    return dis, camera_coordinate

''' 
获取对齐图像帧与相机参数
'''
def get_aligned_images(pipeline, align):
    
    frames = pipeline.wait_for_frames()     # 等待获取图像帧，获取颜色和深度的框架集     
    aligned_frames = align.process(frames)      # 获取对齐帧，将深度框与颜色框对齐  

    aligned_depth_frame = aligned_frames.get_depth_frame()      # 获取对齐帧中的的depth帧 
    aligned_color_frame = aligned_frames.get_color_frame()      # 获取对齐帧中的的color帧

    #### 获取相机参数 ####
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics     # 获取深度参数（像素坐标系转相机坐标系会用到）
    color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics     # 获取相机内参


    #### 将images转为numpy arrays ####  
    img_color = np.asanyarray(aligned_color_frame.get_data())       # RGB图  
    img_depth = np.asanyarray(aligned_depth_frame.get_data())       # 深度图（默认16位）

    return color_intrin, depth_intrin, img_color, img_depth, aligned_depth_frame



class SingleHandWristDetector(SingleHandDetector):
    
    def detect_hand_wrist(self, rgb, aligned_depth_frame, depth_intrin, trans=SapienTrans):
        num_box, joint_pos, keypoint_2d, mediapipe_wrist_rot = self.detect(rgb)
        if num_box==0:
            return None, None, None, None

        wrist_rpy = R.from_matrix(mediapipe_wrist_rot).as_euler('xyz', degrees=False)
        #print('mp: ', wrist_rpy)
        wrist_rpy = np.dot(trans.MP2RobotRot, wrist_rpy) + trans.MP2RobotRPY_init_diff
        #print('robot: ', wrist_rpy)
        #print(rgb.shape)
        wrist_xy = self.parse_keypoint_2d(keypoint_2d, rgb.shape[0:2])[0]
        wrist_xy = [np.clip(int(wrist_xy[0]),0,639), np.clip(int(wrist_xy[1]),0,479)]
        #print(wrist_xy)
        dis, wrist_xyz = get_3d_camera_coordinate(wrist_xy, aligned_depth_frame, depth_intrin)
        #print(camera_coordinate)

        wrist_xyz = np.dot(trans.Camera2RobotRot, wrist_xyz)+trans.Camera2RobotXYZ
        return joint_pos, wrist_xyz, wrist_rpy, keypoint_2d
   
