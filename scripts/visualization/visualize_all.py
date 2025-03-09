import open3d as o3d
import numpy as np
import torch
import os
from utils.camera import load_intrinsics
from pcd import make_point_cloud
import cv2
import pickle as pkl

from torchvision import transforms
from utils.ops import compute_scale_factor
import torch.nn.functional as F
from odometry.dpvo.lietorch import SE3
import camtools
import grasping.contact_graspnet.utils.mesh_utils as mesh_utils
import roma
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--rgb', type=str, required=False)
parser.add_argument('--depth', type=str, required=False)
parser.add_argument('--cameras', type=str, required=False)
parser.add_argument('--invDepths', type=str, required=False)
parser.add_argument('--coords', type=str, required=False)
parser.add_argument('--grasps', type=str, required=False)
parser.add_argument('--viz_cameras', type=int, required=False, default=1)

VIZ_DEBUG = 0
VIZ_CAMERAS = 1
DIST_THRESH = 0.80
KNN = 1

'''
python scripts/visualization/visualize_all.py --pcd pred_mustard0.npy \
                                            --cameras mustard0.npy \
                                            --grasps predictions_pred_mustard0.npz

python scripts/visualization/visualize_all.py --pcd pred_pitcher_pinch_1006.npy \
                                    --cameras pitcher_pinch_1006.npy \
                                    --grasps predictions_pred_pitcher_pinch_1006.npz   

python scripts/visualization/visualize_all.py --pcd pred_mustard_pinch_8.npy \
                                    --cameras mustard_pinch_8.npy \
                                    --grasps predictions_pred_mustard_pinch_8.npz

python scripts/visualization/visualize_all.py --pcd pred_crackers_box.npy \
                                            --cameras crackers_box.npy \
                                            --grasps predictions_pred_crackers_box.npz                                                                                                            
'''

def draw_pcd(rgb_path, depth_path, geometries, K):

    color = cv2.imread(rgb_path)
    depth = np.load(depth_path)
    mm_depth = (depth * 1000).astype(np.uint16)    
    pcd, _ = make_point_cloud(color, mm_depth, K, convert_rgb_to_intensity=False)
    geometries.append(pcd)
    return geometries, depth

def draw_cameras(cameras_path, inv_depth_path, coords_path, K, depth, geometries, scaling_method='pooling'):
    assert scaling_method in ['pooling', 'rescaling']
    assert os.path.exists(cameras_path)
    with open(cameras_path, 'rb') as f:
        camera = pkl.load(f)
    with open(inv_depth_path, 'rb') as f:
        inv_depths = pkl.load(f)
    with open(coords_path, 'rb') as f:
        coords = pkl.load(f)

    camera = SE3(torch.tensor(camera)).inv().matrix().numpy()
    x, y = coords

    scale_factor = compute_scale_factor(x.astype(int), 
                                        y.astype(int), 
                                        dense_depth=depth, 
                                        inv_sparse_depths=inv_depths, 
                                        h=480, 
                                        w=640,
                                        scaling_method=scaling_method)
    camera[:, :3] *= scale_factor
    # camera = SE3(torch.tensor(camera)).inv().matrix().numpy() 
    frame = camtools.camera._create_camera_frame(K=K.intrinsic_matrix, 
                                                T=camera, 
                                                image_wh=(640, 480), 
                                                size=K.get_focal_length()[0] / (20 * 1000),
                                                color=[0, 0, 1],
                                                up_triangle=False,
                                                center_ray=False)
    geometries.append(frame)
    return geometries, camera

def draw_grasps(grasps_path, geometries, color=[1.,0.,0], cam_pose= np.eye(4), tube_radius=0.0008, end_trajectory=None):
    assert os.path.exists(grasps_path)
    with open(grasps_path, 'rb') as f:
        grasp_data = pkl.load(f)

    # gposes = grasp_data['pred_grasps_cam'].item()[-1]
    # gscores = grasp_data['scores'].item()[-1]
    # gcontacts = grasp_data['contact_pts'].item()[-1]
    # gopenings = grasp_data['gripper_openings'].item()[-1]        

    gquats = grasp_data[:, :4]
    gtranslations = grasp_data[:, 4:7]
    gopenings = grasp_data[:, 7]
    gscores = grasp_data[:, 8]

    grots = roma.unitquat_to_rotmat(torch.tensor(gquats)).numpy()
    gposes = np.tile(np.eye(4), (gtranslations.shape[0], 1, 1))
    gposes[:, :3, :3] = grots
    gposes[:, :3, 3] = gtranslations

    near_color = color
    distant_color = [1., 0, 0]

    gripper = mesh_utils.create_gripper('panda')
    gripper_control_points = gripper.get_control_point_tensor(1, False, convex_hull=False).squeeze()
    mid_point = 0.5*(gripper_control_points[1, :] + gripper_control_points[2, :])
    grasp_line_plot = np.array([np.zeros((3,)), mid_point, gripper_control_points[1], gripper_control_points[3], 
                                gripper_control_points[1], gripper_control_points[2], gripper_control_points[4]])

    gripper_points = []

    for i,(g,g_opening) in enumerate(zip(gposes, gopenings)):
        gripper_control_points_closed = grasp_line_plot.copy()
        gripper_control_points_closed[2:,0] = np.sign(grasp_line_plot[2:,0]) * g_opening/2
        
        gripper_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
        gripper_frame.transform(g)

        pts = np.matmul(gripper_control_points_closed, g[:3, :3].T)
        pts += np.expand_dims(g[:3, 3], 0)
        pts_homog = np.concatenate((pts, np.ones((7, 1))),axis=1)
        pts = np.dot(pts_homog, cam_pose.T)[:,:3]
        
        if VIZ_DEBUG:
            mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
            mesh_sphere.translate(pts[1])
            geometries.append(mesh_sphere) 
        
        gripper_points.append((i, pts, gripper_frame, g, g_opening))

    M = len(gripper_points)

    if end_trajectory is not None:
        print(end_trajectory)
        # If the last camera pose of the trajectory is given, evidence the nearest KNN grasp poses.
        # To do that, we sort the grasp poses based on the norm-2 distance between the camera image center 
        # (i.e., given by the camera translation vector) and the mid-point of the grasp pose (center of grasp opening).
        end_trajectory = end_trajectory[:3, 3]

        if VIZ_DEBUG:
            mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
            mesh_sphere.translate(end_trajectory)
            geometries.append(mesh_sphere) 

        # Filter grippers based on the distance from the center of the last camera in the trajectory.
        gripper_points = [(idx, pts, frames, gpose, gopen) for idx, pts, frames, gpose, gopen in gripper_points if np.linalg.norm(pts[1] - end_trajectory) < DIST_THRESH]
        gripper_points = sorted(gripper_points, key=lambda x: np.linalg.norm(x[1][1] - end_trajectory))

        # DEBUG: Only plot the first choices
        # gripper_points = gripper_points[:1]

        # Plot the nearest 10 grasp poses in green, all the rest in red.
        gcolors = [near_color for _ in range(KNN)] + [distant_color for _ in range(M-KNN)]
        # gcolors = [near_color for _ in range(1)]
    else:
        gcolors = None

    for k, (idx, pts, frame, gpose, gopen) in enumerate(gripper_points):
        gripper_color = color if gcolors is None else gcolors[k]
        lines = [[i, i+1] for i in range(len(pts)-1)]

        if idx == 59:
            gripper_color = [0.0, 1.0, 0.0]
            candidate_pts = pts

        gripper_mesh = mesh_utils.LineMesh(pts, lines, colors=gripper_color, radius=tube_radius)
        gripper_mesh_geoms = gripper_mesh.cylinder_segments
    
        geometries += [*gripper_mesh_geoms]

    # #### DEBUG: Draw an additional grasp with identity transform as reference. ####
    # gpose = np.eye(4)
    # gopening = np.max(gopenings)
    # gripper_control_points_closed = grasp_line_plot.copy()
    # gripper_control_points_closed[2:,0] = np.sign(grasp_line_plot[2:,0]) * gopening/2
    
    # gripper_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
    # gripper_frame.transform(gpose)
    
    # pts = np.matmul(gripper_control_points_closed, gpose[:3, :3].T)
    # pts += np.expand_dims(gpose[:3, 3], 0)
    # pts_homog = np.concatenate((pts, np.ones((7, 1))),axis=1)
    # pts = np.dot(pts_homog, cam_pose.T)[:,:3]
    # lines = [[i, i+1] for i in range(len(pts)-1)]
    # gripper_mesh = mesh_utils.LineMesh(pts, lines, colors=[0.4, 0.4, 1.], radius=tube_radius)
    # gripper_mesh_geoms = gripper_mesh.cylinder_segments
    # geometries += [*gripper_mesh_geoms]
    # #### #################################################################### ####

    return geometries, gposes, candidate_pts

if __name__ == '__main__':
    args = parser.parse_args()
    geometries = []

    rgb_path = args.rgb
    depth_path = args.depth
    cameras_path = args.cameras
    grasps_path = args.grasps
    inv_depth_path = args.invDepths
    coords_path = args.coords

    VIZ_CAMERAS = int(args.viz_cameras)

    K = load_intrinsics(calib_txt='src/cameras/calib/hannes.txt')
    K = o3d.camera.PinholeCameraIntrinsic(width=640, 
                                        height=480, 
                                        fx=K[0, 0], 
                                        fy=K[1, 1], 
                                        cx=K[0, 2], 
                                        cy=K[1, 2])


    geometries, depth = draw_pcd(rgb_path, depth_path, geometries, K)
    geometries, lastCamera = draw_cameras(cameras_path, inv_depth_path, coords_path, K, depth, geometries, scaling_method='rescaling')
    geometries, grasps, candidate_pts = draw_grasps(grasps_path, geometries, end_trajectory=lastCamera)

    # print(candidate_pts)
    # # Define two points
    # points = np.array([lastCamera[:3, 3], candidate_pts[1]])  # Start at (0, 0, 0) and end at (1, 0, 0)
    # # Define the line between these points (indices in the points array)
    # print(points)
    # lines = np.array([[0, 1]])
    # # Create a LineSet object
    # line_set = o3d.geometry.LineSet()
    # # Assign the points and lines to the LineSet
    # line_set.points = o3d.utility.Vector3dVector(points)
    # line_set.lines = o3d.utility.Vector2iVector(lines)
    # geometries.append(line_set)

    o3d.visualization.draw_geometries(geometries)
