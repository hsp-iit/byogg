import yarp
import sys
import yaml
import os
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("[GRASPER] Available GPU devices: ", physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import torch
import roma
import open3d as o3d

from utils.camera import load_intrinsics
from pcd import build_pcd_from_rgbd, make_point_cloud
import argparse

import grasping.contact_graspnet.utils.config_utils as config_utils
from grasping.contact_graspnet.model import GraspEstimator
import grasping.contact_graspnet.utils.mesh_utils as mesh_utils

yarp.Network.init()

# Grasp generation module to take in input multiple point clouds instead of a single one.
# NOTE: This module does only communicate with the depther and reader/camera module.
# It's made for wrapping multiple inference calls to Contact-GraspNet and testing this module alone.

class MultiGrasperRFModule(yarp.RFModule):

    def configure(self, rf):
        self.module_name = "multigrasper"
        with open("src/yarp-app/configs/default.yaml", "r") as f:
            cfg = yaml.safe_load(f)

        self.fps = cfg['yarp']['fps']['multigrasper']
        calib = cfg['global']['calib']
        self.K = load_intrinsics(calib)

        self.width = cfg["global"]["img_width"]
        self.height = cfg["global"]["img_height"]

        self.o3dK = o3d.camera.PinholeCameraIntrinsic(width=self.width, 
                            height=self.height, 
                            fx=self.K[0, 0], 
                            fy=self.K[1, 1], 
                            cx=self.K[0, 2], 
                            cy=self.K[1, 2])

        self.min_depth = cfg["mde"]["min_depth"]
        self.max_depth = cfg["mde"]["max_depth"]
        self.flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        self.scale_transform = 4.0

        gripper = mesh_utils.create_gripper('panda')
        gripper_control_points = gripper.get_control_point_tensor(1, False, convex_hull=False).squeeze()
        mid_point = 0.5*(gripper_control_points[1, :] + gripper_control_points[2, :])
        self.grasp_line_plot = np.array([np.zeros((3,)), mid_point, gripper_control_points[1], gripper_control_points[3], 
                                    gripper_control_points[1], gripper_control_points[2], gripper_control_points[4]])

        self.ckpt_dir = os.path.join(cfg['grasping']['model_ckpts_path'], 
                                     cfg['grasping']['model_name'])
        
        self.cfg = cfg['grasping']['contact_grasp_cfg']

        ### Output ports
        port_name = "/" + self.module_name + "/forward/grasps/se3:o"
        self.output_grasps_port = yarp.BufferedPortImageFloat()
        self.output_grasps_port.open(port_name)
        self.output_grasps_port.writeStrict()
        print(f"{port_name} opened")            
        ### Input ports
        port_name = "/" + self.module_name + "/receive/rgb/image:i"
        self.input_rgb_port = yarp.BufferedPortImageRgb()
        self.input_rgb_port.open(port_name)
        print(f"{port_name} opened")

        port_name = "/" + self.module_name + "/receive/depth/image:i"
        self.input_depth_port = yarp.BufferedPortImageFloat()
        self.input_depth_port.open(port_name)
        print(f"{port_name} opened")

        ### Init input rgb and depth buffers
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

        # Create a TF session and init grasper model        
        _cfg = config_utils.load_config(
                    checkpoint_dir=self.ckpt_dir,
                    batch_size=self.cfg['forward_passes'],
                    arg_configs=self.cfg['arg_configs'])
        
        self.grasper = GraspEstimator(_cfg)
        self.grasper.build_network()

        saver = tf.train.Saver(save_relative_paths=True)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        # config.log_device_placement = True
        
        self.sess = tf.Session(config=config)

        self.grasper.load_weights(self.sess, saver, self.ckpt_dir, mode='test')
        self.flag = False

        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

        print(f'[multiGRASPER] Running at {self.fps} fps.')
        
        return True

    def respond(self, command, reply):
        return super().respond(command, reply)

    def cleanup(self):
        self.input_rgb_port.close()
        self.input_depth_port.close()

    def interruptModule(self):
        self.vis.close()
        self.vis.destroy_window()
        self.input_rgb_port.interrupt()
        self.input_depth_port.interrupt()
        return True
    
    def getPeriod(self):
        return 1/float(self.fps)

    def draw_grasps(self, grasps, color=[1, 0, 0], cam_pose=np.eye(4), tube_radius=0.0008):
        # Grasps are encoded as 9D vectors in the form:
        # [4D quaternion (unit), 3D trans, opening, score]
        print("[multiGRASPER] Drawing new grasps...")
        gquats = grasps[:, :4]
        gtranslations = grasps[:, 4:7]
        gopenings = grasps[:, 7]
        gscores = grasps[:, 8]

        grots = roma.unitquat_to_rotmat(torch.tensor(gquats)).numpy()

        gposes = np.tile(np.eye(4), (gtranslations.shape[0], 1, 1))
        gposes[:, :3, :3] = grots
        gposes[:, :3, 3] = gtranslations
        ggeometries = []

        for i,(g,g_opening) in enumerate(zip(gposes, gopenings)):
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
            ggeometries += [*gripper_mesh_geoms]

        print("[multiGRASPER] Adding grasp geometries to visualizer.")

        for geom in ggeometries:
            geom.transform(self.flip_transform)
            geom.scale(self.scale_transform, center=(0, 0, 0))
            self.vis.add_geometry(geom)

    def updateModule(self):
        print('[multiGRASPER] Waiting to read depth and rgb...')

        # TODO: As ports are not bufferized on the TX side, add reading from a single
        #       RGB-D port for ensuring RGB+D synchronization.

        rgb = self.input_rgb_port.read()
        self.input_rgb_image.copy(rgb)
        assert(
            self.input_rgb_array.__array_interface__['data'][0] == \
            self.input_rgb_image.getRawImage().__int__()
        )

        depth = self.input_depth_port.read()
        self.input_depth_image.copy(depth)
        assert(
            self.input_depth_array.__array_interface__['data'][0] == \
            self.input_depth_image.getRawImage().__int__()
        )

        # Transform the depth in the required format (e.g. convert from meters to mm).
        mm_depth = (self.input_depth_array * 1000).astype(np.uint16)
        pcd, _ = make_point_cloud(color_bgr=self.input_rgb_array, 
                        depth=mm_depth, 
                        o3d_intrinsic=self.o3dK, 
                        convert_rgb_to_intensity=False,
                        bgr=False)

        pcd.transform(self.flip_transform)
        pcd.scale(self.scale_transform, center=(0, 0, 0))
 
        print('[multiGRASPER] Generating Grasps...')
        # TODO: Read correct z_range from config file.
        pc_full, pc_segments, pc_colors = self.grasper.extract_point_clouds(self.input_depth_array, self.K, segmap=None, rgb=self.input_rgb_array,
                                                                            skip_border_objects=False, z_range=[0.0, 2.0], x_range=[-3.0, 3.0], y_range=[-3.0, 3.0]) 

        # Draw point cloud here.
        
        pred_grasps_cam, scores, contact_pts, gripper_openings = self.grasper.predict_scene_grasps(self.sess, pc_full, pc_segments={}, 
                                                                                        local_regions=False, filter_grasps=False, forward_passes=1)  
        
        self.n_grasps = pred_grasps_cam[-1].shape[0]
        print(f'[multiGRASPER] Generated {self.n_grasps} grasps.')
        # Encoding grasps to send them via Yarp.
        self.encoded_grasps = np.full(shape=(self.n_grasps, 9), fill_value=np.nan, dtype=np.float32)

        for idx in range(self.n_grasps):
            gpose = pred_grasps_cam[-1][idx]
            rot, trans = torch.from_numpy(gpose[:3, :3]), gpose[:3, 3]
            uquat = roma.rotmat_to_unitquat(rot).numpy()
            gopening = gripper_openings[-1][idx]
            gscore = scores[-1][idx]
            self.encoded_grasps[idx] = np.hstack([uquat, trans, np.float32(gopening),  np.float32(gscore)])

        self.encoded_grasps = np.vstack(self.encoded_grasps)
        # TODO: Change the implementation of this instance of draw_grasps function to avoid encoding grasps.
        self.vis.clear_geometries()
        self.vis.add_geometry(pcd) 
        self.draw_grasps(self.encoded_grasps)
        self.vis.poll_events()
        self.vis.update_renderer()

        return True
    
if __name__ == '__main__':
    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.setDefaultContext("MultiGrasperRFModule")

    conffile = rf.find("from").asString()
    if not conffile:
        print("Using default conf file")
        rf.setDefaultConfigFile("conf.ini")
    else:
        rf.setDefaultConfigFile(rf.find("from").asString())

    rf.configure(sys.argv)

    # Run module
    manager = MultiGrasperRFModule()
    try:
        manager.runModule(rf)
    finally:
        print('Closing MultiGrasperRFModule due to an error...')
        manager.cleanup() 
