import yarp
import sys
import yaml
import json

import open3d as o3d
import numpy as np

import torch
import torch.multiprocessing as mp

import os
from utils.camera import load_intrinsics
from utils.ops import compute_scale_factor
import grasping.contact_graspnet.utils.mesh_utils as mesh_utils

from pcd import make_point_cloud
import queue
import signal
# from multiprocessing import Lock, Queue, Event, Process

import time
import argparse
import matplotlib.pyplot as plt

# TODO: Substitute SE3 from lietorch with SE3 from Spatial Math Toolbox for Python
from odometry.dpvo.lietorch import SE3
import roma
import camtools
from pytransform3d.rotations import matrix_from_quaternion

DEBUG = False
VERBOSE = False
SHOW_GRASPS = True
KNN = 1
DIST_THRESH = 0.3

yarp.Network.init()

# TODO: Delete from the visualizer the grippers that are filtered out due to the distance thresholding.
# Now this is possible as we associate a unique id to each geometry (so, also to each cylinder mesh of the gripper),
# so we can delete a specific geometry from the visualizer and the rendering process.

def vprint(message):
    print(message if VERBOSE else "", end='\n' if VERBOSE else '')

# RFModule-interface to Open3D non-blocking visualizer
class VisualizerRFModule(yarp.RFModule):

    def __init__(self, geometry_queue, close_window_event):
        super().__init__()
        self.geometry_queue = geometry_queue
        self.close_window_event = close_window_event

    def configure(self, rf):    
        self.module_name = "visualizer"
        with open("src/yarp-app/configs/default.yaml", "r") as f:
            cfg = yaml.safe_load(f)

        self.num_patches = cfg['odometry']['dpvo_config']['PATCHES_PER_FRAME']
        self.cfg = cfg

        # Init gripper (panda) params
        gripper = mesh_utils.create_gripper('panda')
        gripper_control_points = gripper.get_control_point_tensor(1, False, convex_hull=False).squeeze()
        mid_point = 0.5*(gripper_control_points[1, :] + gripper_control_points[2, :])
        self.grasp_line_plot = np.array([np.zeros((3,)), mid_point, gripper_control_points[1], gripper_control_points[3], 
                                    gripper_control_points[1], gripper_control_points[2], gripper_control_points[4]])

        calib = cfg["global"]["calib"]
        self.K = load_intrinsics(calib)

        vprint(f"[VISUALIZER] Loading camera intrinsics: {self.K}")

        self.o3dK = o3d.camera.PinholeCameraIntrinsic(width=640, 
                                height=480, 
                                fx=self.K[0, 0], 
                                fy=self.K[1, 1], 
                                cx=self.K[0, 2], 
                                cy=self.K[1, 2])
        
        self.width = cfg["global"]["img_width"]
        self.height = cfg["global"]["img_height"]
        self.scaling_method = cfg["global"]["scaling_method"]
        self.pcd = None
        self.fps = cfg["yarp"]["fps"]["visualizer"]
        self.show_controller_candidate = cfg['visualization']['show_controller_candidate']
        
        self.nearColor = np.array([0., 1., 0.])
        self.distantColor = np.array([1., 0., 0.])
        
        if self.show_controller_candidate:
            self.nearColor = np.array([0.4, 1., 0.4])
            self.candidateColor = np.array([0., 0.6, 0.2])
            self.last_candidate = None

        ### Input ports
        port_name = "/" + self.module_name + "/receive/rgb/image:i"
        self.input_rgb_port = yarp.BufferedPortImageRgb()
        self.input_rgb_port.open(port_name)
        print(f"{port_name} opened") 

        port_name = "/" + self.module_name + "/receive/depth/image:i"
        self.input_depth_port = yarp.BufferedPortImageFloat()
        self.input_depth_port.open(port_name)
        print(f"{port_name} opened") 

        port_name = "/" + self.module_name + "/receive/cameras/se3:i"
        self.input_cameras_port = yarp.BufferedPortBottle()
        self.input_cameras_port.open(port_name)
        self.input_cameras_port.setStrict()
        print(f"{port_name} opened")

        port_name = "/" + self.module_name + "/receive/patches/coords:i"
        self.input_patch_coords_port = yarp.BufferedPortBottle()
        self.input_patch_coords_port.open(port_name)
        print(f"{port_name} opened")

        port_name = "/" + self.module_name + "/receive/patches/depths:i"
        self.input_patch_depths_port = yarp.BufferedPortBottle()
        self.input_patch_depths_port.open(port_name)
        self.input_patch_depths_port.setStrict()
        print(f"{port_name} opened")

        port_name = "/" + self.module_name + "/receive/grasps/se3:i"
        self.input_grasps_port = yarp.BufferedPortBottle()
        self.input_grasps_port.open(port_name)
        print(f"{port_name} opened")

        # Port to notify the visualizer about the need of clearing all the geometries.
        port_name = "/" + self.module_name + "/receive/clear:i"
        self.input_clear_port = yarp.BufferedPortBottle()
        self.input_clear_port.open(port_name)
        print(f"{port_name} opened")

        port_name = "/" + self.module_name + "/receive/candidate:i"
        self.input_candidate_port = yarp.BufferedPortBottle()
        self.input_candidate_port.open(port_name)
        print(f"{port_name} opened")

        ### Prepare input image and pose buffers
        self.input_rgb_array = np.ones(
            (cfg["global"]["img_height"], cfg["global"]["img_width"], 3), dtype=np.uint8
        )
        self.input_rgb_image = yarp.ImageRgb()
        self.input_rgb_image.resize(
            self.input_rgb_array.shape[1], self.input_rgb_array.shape[0]
        )
        self.input_rgb_image.setExternal(
            self.input_rgb_array, self.input_rgb_array.shape[1], self.input_rgb_array.shape[0]
        )

        self.input_depth_array = np.ones(
            (cfg["global"]["img_height"], cfg["global"]["img_width"]), dtype=np.float32
        )
        self.input_depth_image = yarp.ImageFloat()
        self.input_depth_image.resize(
            self.input_depth_array.shape[1], self.input_depth_array.shape[0]
        )
        self.input_depth_image.setExternal(
            self.input_depth_array, self.input_depth_array.shape[1], self.input_depth_array.shape[0]
        )

        self.patchCoordsX = np.zeros((self.num_patches, ), dtype=np.float32)
        self.patchCoordsY = np.zeros((self.num_patches, ), dtype=np.float32)
        self.invDepths = np.zeros((self.num_patches, ), dtype=np.float32)

        self.hasRGB = False
        self.hasDepth = False
        self.hasRGBD = False
        self.hasGrasps = False
        self.hasCoords = False
        self.cameraPoses = None
        self.grasp_geometries = []

        if DEBUG:
            self.outputFile = open('tests/debug/visualizerOutput.txt', 'w')

        print(f"[VISUALIZER] Running at {self.fps} fps.")
        return True

    def clear(self):
        self.geometry_queue.put(('clear', None))

        self.patchCoordsX = np.zeros((self.num_patches, ), dtype=np.float32)
        self.patchCoordsY = np.zeros((self.num_patches, ), dtype=np.float32)
        self.invDepths = np.zeros((self.num_patches, ), dtype=np.float32)

        self.hasRGB = False
        self.hasDepth = False
        self.hasRGBD = False
        self.hasGrasps = False
        self.hasCoords = False  
        self.cameraPoses = None
        self.grasp_geometries = []

    def respond(self, command, reply):
        return super().respond(command, reply)

    def cleanup(self):
        self.input_rgb_port.close()
        self.input_depth_port.close()
        self.input_cameras_port.close()
        self.input_patch_coords_port.close()
        self.input_patch_depths_port.close()
        if self.input_grasps_port:
            self.input_grasps_port.close()
        if DEBUG:
            self.outputFile.close()
        self.input_clear_port.close()
        self.input_candidate_port.close()

    def interruptModule(self):
        self.close_window_event.set()
        self.input_rgb_port.interrupt()
        self.input_depth_port.interrupt()
        self.input_cameras_port.interrupt()
        self.input_patch_coords_port.interrupt()
        self.input_patch_depths_port.interrupt()
        if self.input_grasps_port:
            self.input_grasps_port.interrupt()
        self.input_clear_port.interrupt()
        self.input_candidate_port.interrupt()
        return True
    
    def getPeriod(self):
        return 1/float(self.fps)

    def draw_grasps(self, grasps, color=[1, 0, 0], cam_pose=np.eye(4), tube_radius=0.0008):
        # Grasps are encoded as 9D vectors in the form:
        # [4D quaternion (unit), 3D trans, opening, score]
        vprint("[VISUALIZER] Drawing new grasps...")

        gquats = grasps[:, :4]
        gtranslations = grasps[:, 4:7]
        gopenings = grasps[:, 7]
        gscores = grasps[:, 8]

        grots = roma.unitquat_to_rotmat(torch.tensor(gquats)).numpy()
        gposes = np.tile(np.eye(4), (gtranslations.shape[0], 1, 1))
        gposes[:, :3, :3] = grots
        gposes[:, :3, 3] = gtranslations

        # TODO: Wrap this code inside a function to import both in visualize_all.py script and here.
        # Avoid code duplication.
        for gripper_idx, (g,g_opening) in enumerate(zip(gposes, gopenings)):
            gripper_control_points_closed = self.grasp_line_plot.copy()
            gripper_control_points_closed[2:,0] = np.sign(self.grasp_line_plot[2:,0]) * g_opening/2
            
            # gripper_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
            # gripper_frame.transform(g)

            pts = np.matmul(gripper_control_points_closed, g[:3, :3].T)
            pts += np.expand_dims(g[:3, 3], 0)
            pts_homog = np.concatenate((pts, np.ones((7, 1))),axis=1)
            pts = np.dot(pts_homog, cam_pose.T)[:,:3]
            
            lines = [[i, i+1] for i in range(len(pts)-1)]
            gripper_mesh = mesh_utils.LineMesh(pts, lines, colors=color, radius=tube_radius)
            gripper_mesh_geoms = gripper_mesh.cylinder_segments[:]

            # Store gripper meshes in order to update them (color them).
            # NOTE: We also use gripper_idx as an id of the generated graps, in order to keep track of them for color updates.
            self.grasp_geometries.append((gripper_idx, pts, gripper_mesh_geoms))
        
        vprint("[VISUALIZER] Adding grasp geometries to visualizer...")

        vertices = {}
        triangles = {}
        vertex_colors = {}        

        for gripper_idx, _, gripper_geoms in self.grasp_geometries:
            for mesh_idx, mesh in enumerate(gripper_geoms):
                mesh_id = f'gripper_{gripper_idx}_{mesh_idx}'
                vertices[mesh_id] = np.asarray(mesh.vertices)
                triangles[mesh_id] = np.asarray(mesh.triangles)
                vertex_colors[mesh_id] = np.asarray(mesh.vertex_colors)

        self.geometry_queue.put(('grippers_register', vertices, triangles, vertex_colors))

    def update_grasps(self, last_camera_pose, near_color=[0, 1, 0], distant_color=[1, 0, 0], candidate_id=None, candidate_color=None):
        # Check if we already have grasp geometries in our visualizer.
        if len(self.grasp_geometries) == 0:
            return
        
        vprint("[VISUALIZER] Updating grasps...")
        M = len(self.grasp_geometries)
        # If the last camera pose of the trajectory is given, evidence the nearest KNN grasp poses.
        # To do that, we sort the grasp poses based on the norm-2 distance between the camera image center 
        # (i.e., given by the camera translation vector) and the mid-point of the grasp pose (center of grasp opening).
        hand_pos = last_camera_pose[:3]

        # Filter grippers based on the distance from the center of the last camera in the trajectory.
        if DEBUG:
            vprint(f"Hand position: {hand_pos}")
            self.geometry_queue.put((f'debug_sphere_hand_pos', hand_pos, [0, 0, 1]))
            grippers = []
            for gripper_id, pts, mesh_geoms in self.grasp_geometries:
                dist_from_hand = np.linalg.norm(pts[1] - hand_pos)
                # vprint(f"Gripper Params - id: {gripper_id}, center: {pts[1]}, distance from hand: {dist_from_hand}") 
                self.geometry_queue.put((f'debug_sphere_gripper_{gripper_id}', pts[1], distant_color))
                if dist_from_hand < DIST_THRESH:
                    vprint("Selected as potential candidate grasp.")
                    grippers.append((gripper_id, pts, mesh_geoms)) 
        else:
            grippers = [(gripper_id, pts, mesh_geoms) for gripper_id, pts, mesh_geoms in self.grasp_geometries if np.linalg.norm(pts[1] - hand_pos) < DIST_THRESH]
        
        # TODO: Perfectly understand which point on the camera frame and gripper mesh we are selecting when computing the distance.
        grippers = sorted(grippers, key=lambda x: np.linalg.norm(x[1][1] - hand_pos))
        
        candidates = grippers[:KNN]
        not_candidates = grippers[KNN:]
        updates = {}

        # We want to define a color "priority" for color updates in gripper geometries.
        # If the gripper should be colored as red, this update should always be done, as the gripper is no more a candidate.
        # If the gripper should be colored as dark green, this update should always be done, as the gripper is now the candidate selected by the controller.
        # If the gripper should be colored as light green, this update should be done only if the previous color is not dark green. This scenario can happen 
        # when the controller and the visualizer converges to the same candidate selection (it should happen indeed!), the visualizer and the controller are not synchronized 
        # and the visualizer is slower than the controller. This is basically the most common scenario. If this update is done anyway, we lose track of the gripper selected by the controller.
        # This control is done by also specifying when the update should happen: 'always', or a color to check. If the current color is equal to the given one, we don't do the update.

        for (gripper_id, pts, mesh_geoms) in candidates:
            if candidate_id is not None and gripper_id == candidate_id:
                assert candidate_color is not None
                color = (candidate_color, 'always')
            else:
                if candidate_color is not None:
                    color = (near_color, candidate_color)
                else:
                    color = (near_color, 'always')

            for mesh_idx, _ in enumerate(mesh_geoms):
                mesh_id = f'gripper_{gripper_id}_{mesh_idx}'
                updates[mesh_id] = color

        for (gripper_id, pts, mesh_geoms) in not_candidates:
            color = (distant_color, 'always')

            for mesh_idx, _ in enumerate(mesh_geoms):
                mesh_id = f'gripper_{gripper_id}_{mesh_idx}'
                updates[mesh_id] = color

        self.geometry_queue.put(('grippers_update', updates))

    def update_cameras(self, cameras, color=[0, 0, 1]):
        h, w = self.cfg["global"]["img_height"], self.cfg["global"]["img_width"]
        cameras = SE3(torch.tensor(cameras)).inv().matrix().numpy()
        for idx, camera in enumerate(cameras):
            frame = camtools.camera._create_camera_frame(K=self.o3dK.intrinsic_matrix, 
                                                            T=camera, 
                                                            image_wh=(w, h), 
                                                            size=self.o3dK.get_focal_length()[0] / (20 * 1000),
                                                            color=color,
                                                            up_triangle=False,
                                                            center_ray=False)
            
            points = np.asarray(frame.points)
            lines = np.asarray(frame.lines)
            colors = np.asarray(frame.colors)

            self.geometry_queue.put((f'camera_{idx}', points, lines, colors))

    def draw_camera(self, camera, id, color=[0, 0, 1]):
        h, w = self.cfg["global"]["img_height"], self.cfg["global"]["img_width"]
        camera = SE3(torch.tensor(camera)).inv().matrix().numpy()
        frame = camtools.camera._create_camera_frame(K=self.o3dK.intrinsic_matrix, 
                                                        T=camera, 
                                                        image_wh=(w, h), 
                                                        size=self.o3dK.get_focal_length()[0] / (20 * 1000),
                                                        color=color,
                                                        up_triangle=False,
                                                        center_ray=False)

        points = np.asarray(frame.points)
        lines = np.asarray(frame.lines)
        colors = np.asarray(frame.colors)

        self.geometry_queue.put((id, points, lines, colors))

    def updateModule(self):
        clearBottle = self.input_clear_port.read(shouldWait=False)
        if clearBottle is not None:
            self.clear()

        if not self.hasRGBD:
            vprint("[VISUALIZER] Reading RGB-D image, polling...")
            if not self.hasRGB:
                rgb = self.input_rgb_port.read(shouldWait=False)
                if rgb is not None:
                    self.input_rgb_image.copy(rgb)
                    self.hasRGB = True
            if not self.hasDepth:
                depth = self.input_depth_port.read(shouldWait=False)
                if depth is not None:
                    self.input_depth_image.copy(depth)
                    self.hasDepth = True
            if self.hasRGB and self.hasDepth:
                mm_depth = (self.input_depth_array * 1000).astype(np.uint16)
                self.pcd, _ = make_point_cloud(self.input_rgb_array,
                                            depth=mm_depth,
                                            o3d_intrinsic=self.o3dK,
                                            convert_rgb_to_intensity=False,
                                            bgr=False)
                
                points = np.asarray(self.pcd.points)
                colors = np.asarray(self.pcd.colors)
                self.geometry_queue.put(('pcd', points, colors))

                self.hasRGBD = True 

        if SHOW_GRASPS and not self.hasGrasps:
            vprint("[VISUALIZER] Reading grasps, polling...")
            graspBottle = self.input_grasps_port.read(shouldWait=False)
            if graspBottle is not None:
                numGrasps = len(graspBottle.toString().split(' ')) // 9
                grasps = np.zeros((numGrasps, 9), dtype=np.float32)

                for graspIdx in range(numGrasps):
                    for jIdx in range(9):
                        grasps[graspIdx, jIdx] = graspBottle.get(graspIdx * 9 + jIdx).asFloat32()

                self.draw_grasps(grasps)
                self.hasGrasps = True

        # Polling on candidate selection made by the controller
        if self.show_controller_candidate:
            candidateBottle = self.input_candidate_port.read(shouldWait=False)
            if candidateBottle is not None:
                self.last_candidate = candidateBottle.get(0).asInt32()

        # Reading cameras from the cameras port.
        vprint("[VISUALIZER] Reading new cameras, polling...")
        camerasVisualizerBottle = self.input_cameras_port.read(shouldWait=False)

        if camerasVisualizerBottle is not None:
            camerasBottleData = camerasVisualizerBottle.toString()
            cameraBatches = len(camerasBottleData.split(' ')) // 7
            self.cameraPoses = np.zeros((cameraBatches, 7), dtype=np.float32)

            for frameIdx in range(cameraBatches):
                for jIdx in range(7):
                    self.cameraPoses[frameIdx, jIdx] = camerasVisualizerBottle.get(frameIdx * 7 + jIdx).asFloat32()
            
            vprint(f'[VISUALIZER] Last camera transform received: {self.cameraPoses[cameraBatches-1]}')

            if DEBUG:
                self.outputFile.write(f'[VISUALIZER] Last camera bottle received: {camerasVisualizerBottle.toString()}\n')

        vprint("[VISUALIZER] Reading patches, polling...")
        # NOTE: If we already have patch coords (and dense depth) for sampling on the dense depth map and compute the scale factor,
        # we start reading from inv.depth input port. Otherwise, we first wait to receive them.
        # Note that a camera is not drawn until we start receiving inv. depths. In this way, we can draw a camera
        # after applying a scale factor to its translation vector.

        if not self.hasCoords:
            coordsVisualizerBottle = self.input_patch_coords_port.read(shouldWait=False)
            if coordsVisualizerBottle is not None:
                for patchIdx in range(self.num_patches):
                    self.patchCoordsX[patchIdx] = coordsVisualizerBottle.get(2 * patchIdx).asFloat32()
                    self.patchCoordsY[patchIdx] = coordsVisualizerBottle.get(2 * patchIdx + 1).asFloat32()
                self.hasCoords = True

        elif self.hasCoords and self.hasRGBD:
            invDepthsVisualizerBottle = self.input_patch_depths_port.read(shouldWait=False)
            if invDepthsVisualizerBottle is not None:
                for patchIdx in range(self.num_patches):
                    self.invDepths[patchIdx] = invDepthsVisualizerBottle.get(patchIdx).asFloat32()

                scale_factor = compute_scale_factor(self.patchCoordsX.astype(int), 
                                                    self.patchCoordsY.astype(int), 
                                                    dense_depth=self.input_depth_array, 
                                                    inv_sparse_depths=self.invDepths, 
                                                    h=self.height, 
                                                    w=self.width,
                                                    scaling_method=self.scaling_method)
                #vprint(f'[VISUALIZER] scale_factor = {scale_factor}')
                if self.cameraPoses is not None:
                    # Cameras are stored as 7D vectors (3D trans + unit quaternions)
                    self.cameraPoses[:, :3] *= scale_factor
                    self.update_cameras(self.cameraPoses)
                    # self.draw_camera(self.cameraPoses[-1], id=f'camera_{len(self.cameraPoses)-1}')
                    if SHOW_GRASPS:
                        if self.show_controller_candidate and self.last_candidate is not None:
                            self.update_grasps(last_camera_pose=self.cameraPoses[-1], 
                                                near_color=self.nearColor,
                                                distant_color=self.distantColor,
                                                candidate_id=self.last_candidate,
                                                candidate_color=self.candidateColor)
                        else:
                            self.update_grasps(last_camera_pose=self.cameraPoses[-1],
                                               near_color=self.nearColor,
                                               distant_color=self.distantColor)

        return True


class GraspVO_Visualizer(object):

    def __init__(self, geometry_queue, close_window_event):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name='Hannes-GraspVO - Open3D Visualizer')

        with open("src/yarp-app/configs/default.yaml", "r") as f:
            cfg = yaml.safe_load(f)

        viz_cfg = cfg['visualization']
        self.apply_view = viz_cfg['apply_view']
        ctr = self.vis.get_view_control()
        ctr.set_lookat([0, 0, 0])
        
        if self.apply_view:
            with open(viz_cfg['view']['cfg'], 'r') as view_cfg_file:
                self.view_params = json.load(view_cfg_file)["trajectory"][0]
                print(f"[VISUALIZER - Rendering Loop] Setting view params: {self.view_params}")

        self.flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        self.scale_transform = 2.0
        self.geometry_queue = geometry_queue
        self.vis_geometries = {} # Dictionary to mantain geometry references for updates.
        self.close_window_event = close_window_event
        self.dt = float(viz_cfg['render_delta_t'])

    def run(self):
        print("[VISUALIZER - Rendering Loop] Running the visualizer...")
        updateTime = time.time()
        #TODO: Test if we can use self.vis.poll_events() to understand when the visualization window has been closed.
        while not self.close_window_event.is_set():
            if not self.geometry_queue.empty():
                # NOTE: The Visualizer RFModule is responsible to assign geom_ids to new geometries.
                geom = self.geometry_queue.get(block=False)
                # print('got geom: ', geom)
                if geom is not None:
                    geom_id, geom_data = geom[0], geom[1:]

                    if 'pcd' in geom_id:
                        assert len(geom_data) == 2
                        points, colors = geom_data

                        if geom_id not in self.vis_geometries:
                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(points)
                            pcd.colors = o3d.utility.Vector3dVector(colors)
                            pcd.transform(self.flip_transform)
                            pcd.scale(self.scale_transform, center=(0, 0, 0))

                            self.vis.add_geometry(pcd)
                            self.vis_geometries[geom_id] = pcd
                            
                            print("[VISUALIZER] Adding pcd geometry to visualizer.")
                            if self.apply_view:
                                print(f"[VISUALIZER - Rendering Loop] Setting view params: {self.view_params}")
                                ctr = self.vis.get_view_control()
                                ctr.set_front(self.view_params["front"])
                                ctr.set_lookat(self.view_params["lookat"])
                                ctr.set_up(self.view_params["up"])
                                ctr.set_zoom(self.view_params["zoom"])
                        else:
                            pcd = self.vis_geometries[geom_id]
                            pcd.points = o3d.utility.Vector3dVector(points)
                            pcd.colors = o3d.utility.Vector3dVector(colors)
                            pcd.transform(self.flip_transform)
                            pcd.scale(self.scale_transform, center=(0, 0, 0))
                            self.vis.update_geometry(pcd)

                    elif 'camera' in geom_id:
                        assert len(geom_data) == 3
                        points, lines, colors = geom_data
                        
                        if geom_id not in self.vis_geometries:
                            cameraFrame = o3d.geometry.LineSet()
                            cameraFrame.points = o3d.utility.Vector3dVector(points)
                            cameraFrame.lines = o3d.utility.Vector2iVector(lines)
                            cameraFrame.colors =  o3d.utility.Vector3dVector(colors)
                            cameraFrame.transform(self.flip_transform)
                            cameraFrame.scale(self.scale_transform, center=(0, 0, 0))

                            self.vis.add_geometry(cameraFrame, reset_bounding_box=False)
                            self.vis_geometries[geom_id] = cameraFrame
                        else:
                            cameraFrame = self.vis_geometries[geom_id]
                            cameraFrame.points = o3d.utility.Vector3dVector(points)
                            cameraFrame.lines =  o3d.utility.Vector2iVector(lines)
                            cameraFrame.colors = o3d.utility.Vector3dVector(colors)
                            cameraFrame.transform(self.flip_transform)
                            cameraFrame.scale(self.scale_transform, center=(0, 0, 0))
                            self.vis.update_geometry(cameraFrame)

                    elif 'grippers' in geom_id:
                        if geom_id == 'grippers_register':
                            assert len(geom_data) == 3 
                            vertices, triangles, vertex_colors = geom_data
                            mesh_ids = list(vertices.keys())

                            for mesh_id in mesh_ids:
                                gripperMesh = o3d.geometry.TriangleMesh()
                                gripperMesh.vertices = o3d.utility.Vector3dVector(vertices[mesh_id])
                                gripperMesh.triangles = o3d.utility.Vector3iVector(triangles[mesh_id])
                                gripperMesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors[mesh_id])
                                gripperMesh.transform(self.flip_transform)
                                gripperMesh.scale(self.scale_transform, center=(0, 0, 0))
                                
                                self.vis.add_geometry(gripperMesh, reset_bounding_box=False)
                                self.vis_geometries[mesh_id] = gripperMesh

                        elif geom_id == 'grippers_update':
                            assert len(geom_data) == 1
                            (updates,) = geom_data
                            
                            for mesh_id in updates:
                                (color, priority) = updates[mesh_id]
                                current_color = np.asarray(self.vis_geometries[mesh_id].vertex_colors)[0]
                                
                                if priority != 'always':
                                    assert isinstance(priority, np.ndarray)

                                if priority == 'always' or not (current_color == priority).all():
                                    self.vis_geometries[mesh_id].paint_uniform_color(color)
                                    self.vis.update_geometry(self.vis_geometries[mesh_id])

                        else:
                            pass

                    elif 'debug_sphere' in geom_id:
                        assert len(geom_data) == 2
                        center, color = geom_data
                        print(f"[VISUALIZER - Rendering Loop] Debug Sphere Params - id: {geom_id}, center: {center}, color: {color}")
                        if geom_id in self.vis_geometries:
                            self.vis.remove_geometry(self.vis_geometries[geom_id], reset_bounding_box=False)
                        
                        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                        sphere.translate(center)
                        sphere.transform(self.flip_transform)
                        sphere.scale(self.scale_transform, center=(0, 0, 0))
                        sphere.paint_uniform_color(color)
                        self.vis.add_geometry(sphere, reset_bounding_box=False)
                        self.vis_geometries[geom_id] = sphere

                    elif 'clear' in geom_id:
                        self.clear()
                    else:
                        pass
            
            currentTime = time.time()
            if currentTime - updateTime > self.dt:
                self.vis.poll_events()
                self.vis.update_renderer()
                updateTime = currentTime
            
        print('[VISUALIZER] Stop event has been set!')
        self.vis.destroy_window()

    def clear(self):
        self.vis.clear_geometries()
        self.vis.poll_events()
        self.vis.update_renderer()

        self.vis_geometries = {} # Reset geometry dictionary.
        # Clear all the geometries still in queue.
        try:
            while True:
                self.geometry_queue.get_nowait()
        except queue.Empty:
            pass
        
def handle_sigint(signum, frame, close_event):
    print("SIGINT received. Terminating processes...")
    close_event.set()

def yarpMain(geometry_queue, close_window_event):
    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.setDefaultContext("VisualizerRFModule")
    conffile = rf.find("from").asString()
    if not conffile:
        print("Using default conf file")
        rf.setDefaultConfigFile("conf.ini")
    else:
        rf.setDefaultConfigFile(rf.find("from").asString())
    
    rf.configure(sys.argv)
    # Run module
    manager = VisualizerRFModule(geometry_queue, close_window_event)
    try:
        manager.runModule(rf)
    finally:
        print('[VISUALIZER] Closing VisualizerRFModule due to an error...')
        manager.cleanup() 

if __name__ == '__main__':
    # NOTE: See https://github.com/isl-org/Open3D/issues/389#issuecomment-396858138
    # on why the non-blocking visualizer loop should be run on the main thread.
    mp.set_start_method("spawn")

    geometry_queue = mp.Queue()
    close_window_event = mp.Event()

    # Set up the signal handler for SIGINT
    signal.signal(signal.SIGINT, lambda signum, frame: handle_sigint(signum, frame, close_window_event))

    # Run YARP worker on a separate process.
    yarpWorker = mp.Process(target=yarpMain, args=(geometry_queue, close_window_event,))
    yarpWorker.start()

    # Run visualizer on the main thread.
    visualizer = GraspVO_Visualizer(geometry_queue, close_window_event)
    visualizer.run()
    yarpWorker.join()
