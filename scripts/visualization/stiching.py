import numpy as np
import open3d as o3d
from utils.camera import load_intrinsics
import copy
import torch
from torchvision import transforms
from odometry.dpvo.lietorch import SE3

# Part of the code is taken from here: https://github.com/smartroboticslab/deep_prob_feature_track/blob/main/code/tools/rgbd_odometry.py
# rgb 0 and 28

def load_rgbd_image(rgbd_path):
    pcd_data = np.load(rgbd_path, allow_pickle=True).item()
    color, depth, K = pcd_data['rgb'], pcd_data['depth'],  pcd_data['K']
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=o3d.geometry.Image(color),
        depth=o3d.geometry.Image(depth),
        depth_scale=1,
        depth_trunc=2.0,
        convert_rgb_to_intensity=False
    )
    return rgbd_image, depth

def load_camera_poses(cameras_path, depth):
    cameras = np.load(cameras_path, allow_pickle=True)
    sparse_depth_path = cameras_path.replace('poses', 'depths')
    x, y, invdepths = np.load(sparse_depth_path, allow_pickle=True)

    valid_mask = np.bitwise_and((1/invdepths) >= 0, (1/invdepths) <=2)
    h, w = depth.shape

    rescale = transforms.Compose([transforms.Resize((int(h/4), int(w/4)))])
    depth = rescale(torch.tensor(depth).reshape(1, 1, h, w)).squeeze()
    scale_factor = np.median(depth[y, x][valid_mask] * invdepths[valid_mask])
    cameras[:, :3] *= scale_factor

    se3cameras = []
    for camera in cameras:
        se3camera = SE3(torch.tensor(camera)).inv().matrix().numpy()
        se3cameras.append(se3camera)

    return np.stack(se3cameras, axis=0)

def draw_registration_result(source, target, transformation, name='Open3D'):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp], window_name=name)

if __name__ == '__main__':
    intrinsic = load_intrinsics(calib_txt='src/cameras/calib/d435.txt')
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width=640, 
                                        height=480, 
                                        fx=intrinsic[0, 0], 
                                        fy=intrinsic[1, 1], 
                                        cx=intrinsic[0, 2], 
                                        cy=intrinsic[1, 2])

    rgbd_0, depth_0 = load_rgbd_image('/docker_volume/pcd_data/pred_pcd/pred_mustard0.npy')
    rgbd_1, depth_1 = load_rgbd_image('/docker_volume/pcd_data/pred_pcd/pred_mustard28.npy')
    
    
    # option =  o3d.pipelines.odometry.OdometryOption(depth_min=0.01, depth_diff_max=1.0)
    # odo_opt = o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm()

    odo_init = np.identity(4)

    # [is_success, T_10, info] = o3d.pipelines.odometry.compute_rgbd_odometry(
    #     rgbd_0, rgbd_1, intrinsic,
    #     odo_init, odo_opt, option)
    
    # print(T_10)

    pcd_0 = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_0, intrinsic)
    pcd_1 = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_1, intrinsic)

    # print(pcd_0)
    # print(pcd_1)

    cameras = load_camera_poses('results/dpvo/poses/mustard0.npy', depth_0)
    T_10 = cameras[28]

    draw_registration_result(pcd_0, pcd_1, odo_init, 'init')
    draw_registration_result(pcd_0, pcd_1, T_10, 'aligned')