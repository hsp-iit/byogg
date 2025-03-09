import pyransac3d as pyrsc
import numpy as np
import open3d as o3d
from utils.camera import load_intrinsics
import time 

def load_rgbd_image(rgbd_path):
    pcd_data = np.load(rgbd_path, allow_pickle=True).item()
    color, depth, K = pcd_data['rgb'], pcd_data['depth'],  pcd_data['K']
    color = np.array(color[:, :, ::-1])

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=o3d.geometry.Image(color),
        depth=o3d.geometry.Image(depth),
        depth_scale=1,
        depth_trunc=2.0,
        convert_rgb_to_intensity=False
    )
    return rgbd_image, depth

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtracting max for numerical stability
    return exp_x / np.sum(exp_x)

if __name__ == "__main__":
    # Code partially from https://github.com/leomariga/pyRANSAC-3D/blob/Animations/test_plane.py
    
    intrinsic = load_intrinsics(calib_txt='src/cameras/calib/d435.txt')
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width=640, 
                                        height=480, 
                                        fx=intrinsic[0, 0], 
                                        fy=intrinsic[1, 1], 
                                        cx=intrinsic[0, 2], 
                                        cy=intrinsic[1, 2])

    rgbd_0, depth_0 = load_rgbd_image('/docker_volume/pcd_data/pred_pcd/pred_pitcher3_ft_encoder.npy')
    # rgbd_0, depth_0 = load_rgbd_image('/docker_volume/pcd_data/pred_pcd/pred_box_003_cracker_box_power.npy')

    pcd_0 = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_0, intrinsic)

    points = np.asarray(pcd_0.points)
    plane1 = pyrsc.Plane()

    start = time.time()
    best_eq, best_inliers1 = plane1.fit(points, 0.02)
    end = time.time()
    plane1 = pcd_0.select_by_index(best_inliers1).paint_uniform_color([1, 0, 0])
    outliers = pcd_0.select_by_index(best_inliers1, invert=True)

    print(f"[RANSAC plane-fitting] end - start: {end - start}")

    obb = plane1.get_oriented_bounding_box()
    # obb2 = plane1.get_axis_aligned_bounding_box()
    obb.color = [0, 0, 1]
    # obb2.color = [0, 1, 0]
    
    # Fit another plane to the remaining points of the pcd
    out_points = np.asarray(outliers.points)
    out_colors = np.asarray(outliers.colors)

    start = time.time()
    invdepths = 1 / out_points[:, 2]
    probs = softmax(invdepths)
    center_indexes = np.random.choice(range(out_points.shape[0]), size=20000, replace=False, p=probs)
    # center_indexes = np.random.choice(range(out_points.shape[0]), size=20000, replace=False)
    end = time.time()

    print(f"[sampling] end - start: {end - start}")

    samples = out_points[center_indexes, :]
    colors = out_colors[center_indexes, :]

    samples_pcd = o3d.geometry.PointCloud()
    samples_pcd.points = o3d.utility.Vector3dVector(samples)
    samples_pcd.colors = o3d.utility.Vector3dVector(colors)

    # o3d.visualization.draw_geometries([obb, samples_pcd])