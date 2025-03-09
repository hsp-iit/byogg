import open3d as o3d
import numpy as np
import camtools
import cv2
import torch

# TODO: Substitute SE3 from lietorch with SE3 from Spatial Math Toolbox for Python
from odometry.dpvo.lietorch import SE3
import pytransform3d.rotations as pyR
import pytransform3d.transformations as pyT

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random

# PCD Utilities

def draw_pcd_with_camera_frame(color, depth, K):
    K = o3d.camera.PinholeCameraIntrinsic(width=640, 
                                            height=480, 
                                            fx=K[0, 0], 
                                            fy=K[1, 1], 
                                            cx=K[0, 2], 
                                            cy=K[1, 2])
    geometries = []
    pcd, _ = make_point_cloud(color, depth, K, convert_rgb_to_intensity=False)
    geometries.append(pcd)
    frame_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
    geometries.append(frame_axes)
    camera_frame_rs = camtools.camera._create_camera_frame(K=K.intrinsic_matrix, 
                                                        T=np.eye(4), 
                                                        image_wh=(640, 480), 
                                                        size=K.get_focal_length()[0] / (20 * 1000),
                                                        color=[0, 1, 0],
                                                        up_triangle=False,
                                                        center_ray=False)
    geometries.append(camera_frame_rs)
    o3d.visualization.draw_geometries(geometries)

def make_point_cloud(color_bgr, depth, o3d_intrinsic, convert_rgb_to_intensity=True, bgr=True):
    if bgr:
        color_o3d_img = o3d.geometry.Image(cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB))
    else:
        color_o3d_img = o3d.geometry.Image(color_bgr)

    depth_o3d_img = o3d.geometry.Image(depth)
    rgbd_o3d_img = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d_img, depth_o3d_img,
        convert_rgb_to_intensity=convert_rgb_to_intensity
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_o3d_img, o3d_intrinsic
    )
    return pcd, rgbd_o3d_img

def get_pcd_geometry_with_camera_frame(color, depth, K):
    K = o3d.camera.PinholeCameraIntrinsic(width=640, 
                                            height=480, 
                                            fx=K[0, 0], 
                                            fy=K[1, 1], 
                                            cx=K[0, 2], 
                                            cy=K[1, 2])
    geometries = []
    pcd, _ = make_point_cloud(color, depth, K, convert_rgb_to_intensity=False)
    geometries.append(pcd)
    frame_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
    geometries.append(frame_axes)
    camera_frame_rs = camtools.camera._create_camera_frame(K=K.intrinsic_matrix, 
                                                        T=np.eye(4), 
                                                        image_wh=(640, 480), 
                                                        size=K.get_focal_length()[0] / (20 * 1000),
                                                        color=[0, 1, 0],
                                                        up_triangle=False,
                                                        center_ray=False)
    geometries.append(camera_frame_rs)
    return geometries

def get_pcd_geometry_with_camera_poses(color, depth, K, camera_poses, scale_factor):
    K = o3d.camera.PinholeCameraIntrinsic(width=640, 
                                            height=480, 
                                            fx=K[0, 0], 
                                            fy=K[1, 1], 
                                            cx=K[0, 2], 
                                            cy=K[1, 2])
    geometries = []
    pcd, _ = make_point_cloud(color, depth, K, convert_rgb_to_intensity=False)
    geometries.append(pcd)
    frame_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
    geometries.append(frame_axes)

    for pose in camera_poses:   
        T = SE3(torch.tensor(pose)).inv().matrix().numpy()
        T[:3, 3] *= scale_factor

        camera_frame = camtools.camera._create_camera_frame(K=K.intrinsic_matrix, 
                                                            T=T, 
                                                            image_wh=(640, 480), 
                                                            size=K.get_focal_length()[0] / (20 * 1000),
                                                            color=[0, 1, 0],
                                                            up_triangle=False,
                                                            center_ray=False)
        geometries.append(camera_frame)

    return geometries


