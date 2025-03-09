import open3d as o3d
import numpy as np
import cv2

from src.grasping.contact_graspnet.utils.data_utils import load_available_input_data, depth2pc

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    o3d.visualization.draw_geometries([inlier_cloud])

pcd_path = '/docker_volume/pcd_data/pred_pcd/pred_crackers_box.npy'

x_range = [-0.5, 0.5]
y_range = [-0.5, 0.5]
z_range = [0.2, 1.0]

pcd_data = np.load(pcd_path, allow_pickle=True).item()
color, depth, K = pcd_data['rgb'], pcd_data['depth'], pcd_data['K']
mm_depth = (depth * 1000).astype(np.uint16)

o3dK = o3d.camera.PinholeCameraIntrinsic(width=640, 
                                            height=480, 
                                            fx=K[0, 0], 
                                            fy=K[1, 1], 
                                            cx=K[0, 2], 
                                            cy=K[1, 2])

color_o3d_img = o3d.geometry.Image(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
depth_o3d_img = o3d.geometry.Image(depth)
rgbd_o3d_img = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_o3d_img, depth_o3d_img,
    convert_rgb_to_intensity=False
)

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_o3d_img, o3dK
)

# print("Load a ply point cloud, print it, and render it")
# o3d.visualization.draw_geometries([pcd])

print("Downsample the point cloud with a voxel of 0.02")
voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.02/1000)
# o3d.visualization.draw_geometries([voxel_down_pcd])

print("Statistical oulier removal")
cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=20,
                                                    std_ratio=0.5)
display_inlier_outlier(voxel_down_pcd, ind)
