import yarp
import sys
import yaml
import os
import time
import numpy as np
import pyransac3d as ransac

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("[GRASPER] Available GPU devices: ", physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import torch
import roma

from utils.camera import load_intrinsics
import grasping.contact_graspnet.utils.config_utils as config_utils
from grasping.contact_graspnet.model import GraspEstimator

yarp.Network.init()

class GraspingRFModule(yarp.RFModule):

    def configure(self, rf):
        self.module_name = "grasper"
        with open("src/yarp-app/configs/default.yaml", "r") as f:
            cfg = yaml.safe_load(f)

        self.fps = cfg['yarp']['fps']['grasper']
        calib = cfg['global']['calib']
        self.K = load_intrinsics(calib)

        self.min_depth = cfg["mde"]["min_depth"]
        self.max_depth = cfg["mde"]["max_depth"]

        self.ckpt_dir = os.path.join(cfg['grasping']['model_ckpts_path'], 
                                     cfg['grasping']['model_name'])
        
        self.cfg = cfg['grasping']['contact_grasp_cfg']

        self.img_height = cfg['global']['img_height']
        self.img_width = cfg['global']['img_width']
        self.remove_table = cfg['pcd']['remove-table']
        self.sampling_strategy = cfg['pcd']['sampling-strategy']

        ### Command/RPC port
        self.cmd_port = yarp.Port()
        self.cmd_port.open('/' + self.module_name + '/command:i')
        print('{:s} opened'.format('/' + self.module_name + '/command:i'))
        self.attach(self.cmd_port)

        # Input ports
        port_name = "/" + self.module_name + "/receive/rgb/image:i"
        self.input_rgb_port = yarp.BufferedPortImageRgb()
        self.input_rgb_port.open(port_name)
        print(f"{port_name} opened") 

        port_name = "/" + self.module_name + "/receive/depth/image:i"
        self.input_depth_port = yarp.BufferedPortImageFloat()
        self.input_depth_port.open(port_name)
        print(f"{port_name} opened") 

        ### Output ports
        port_name = "/" + self.module_name + "/forward/grasps/se3:o"
        self.output_grasps_port = yarp.Port()
        self.output_grasps_port.open(port_name)
        print(f"{port_name} opened")

        ### Prepare input image and depth buffers
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

        # Perform dummy inference on random depth to load all the needed TF libraries.
        depth = np.random.random((cfg["global"]["img_height"], cfg["global"]["img_width"])).astype(np.float32)
        rgb = np.zeros((cfg["global"]["img_height"], cfg["global"]["img_width"], 3), dtype=np.uint8)

        pc_full, _, _= self.grasper.extract_point_clouds(depth, self.K, segmap=None, rgb=rgb,
                                                                            skip_border_objects=False, z_range=[0.0, 2.0])
        self.grasper.predict_scene_grasps(self.sess, pc_full, pc_segments={}, 
                                        local_regions=False, filter_grasps=False, forward_passes=1)  
        
        self.attachTimestamp = cfg['global']['dump']['timestamps']
        self.timestamp = None

        print(f'[GRASPER] Running at {self.fps} fps.')
        return True

    def respond(self, command, reply):
        return super().respond(command, reply)

    def cleanup(self):
        self.input_rgb_port.close()
        self.input_depth_port.close()
        self.output_grasps_port.close()

    def interruptModule(self):
        self.input_rgb_port.interrupt()
        self.input_depth_port.interrupt()
        self.output_grasps_port.interrupt()
        return True
    
    def getPeriod(self):
        return 1/float(self.fps)

    def updateModule(self):
        print('[GRASPER] Waiting to read rgb-d image...')
        rgb = self.input_rgb_port.read()
        self.input_rgb_image.copy(rgb)

        depth = self.input_depth_port.read()
        self.input_depth_image.copy(depth)        

        print('[GRASPER] Generating Grasps...')
        # TODO: Read correct z_range from config file.
        pc_full, pc_segments, pc_colors = self.grasper.extract_point_clouds(self.input_depth_array, self.K, 
                                                                            segmap=None, rgb=self.input_rgb_array,
                                                                            skip_border_objects=False, z_range=[0.0, 2.0])

        if self.remove_table:
            plane = ransac.Plane()
            _, inliers = plane.fit(pc_full, thresh=0.02)
            outliers = np.ones(pc_full.shape[0], dtype=bool)
            outliers[inliers] = False
            pc_full = pc_full[outliers]

        pred_grasps_cam, scores, contact_pts, gripper_openings = self.grasper.predict_scene_grasps(self.sess, pc_full, pc_segments={}, 
                                                                                        local_regions=False, filter_grasps=False, forward_passes=1,
                                                                                         sampling_strategy=self.sampling_strategy)  
        if self.attachTimestamp:
            self.timestamp = time.time()

        self.n_grasps = pred_grasps_cam[-1].shape[0]
        print(f'[GRASPER] Generated {self.n_grasps} grasps.')

        graspBottle = yarp.Bottle()
        for graspIdx in range(self.n_grasps):
            gpose = pred_grasps_cam[-1][graspIdx]
            rot, trans = torch.from_numpy(gpose[:3, :3]), gpose[:3, 3]
            uquat = roma.rotmat_to_unitquat(rot).numpy()
            gopening = gripper_openings[-1][graspIdx]
            gscore = scores[-1][graspIdx] 
            embeddedGrasp = np.hstack([uquat, trans, np.float32(gopening),  np.float32(gscore)])
            for value in embeddedGrasp:
                graspBottle.addFloat32(value.item())

        if self.attachTimestamp:
            envelopeBottle = yarp.Bottle()
            envelopeBottle.addFloat64(self.timestamp)
            self.output_grasps_port.setEnvelope(envelopeBottle)
            
        self.output_grasps_port.write(graspBottle)

        return True
    
if __name__ == '__main__':
    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.setDefaultContext("GraspingRFModule")

    conffile = rf.find("from").asString()
    if not conffile:
        print("Using default conf file")
        rf.setDefaultConfigFile("conf.ini")
    else:
        rf.setDefaultConfigFile(rf.find("from").asString())

    rf.configure(sys.argv)

    # Run module
    manager = GraspingRFModule()
    try:
        manager.runModule(rf)
    finally:
        print('Closing GraspingRFModule due to an error...')
        manager.cleanup() 
