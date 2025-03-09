import yarp
import sys
import yaml
import os
import numpy as np
import torch
import ctypes
import time
import argparse
import cv2

from multiprocessing import Lock
from odometry.dpvo.dpvo import DPVO
from odometry.dpvo.config import cfg as _cfg
from utils.camera import load_intrinsics

DEBUG = False
VIZ = False

yarp.Network.init()

class VisualOdometryRFModule(yarp.RFModule):

    def configure(self, rf):
        self.module_name = "odometer"
        with open("src/yarp-app/configs/default.yaml", "r") as f:
            cfg = yaml.safe_load(f)

        self._cfg = _cfg
        # Build yacs config node from yaml cfg
        for key, value in cfg['odometry']['dpvo_config'].items():
            if key in self._cfg:
                self._cfg.merge_from_list([key, value])    

        self.network = os.path.join(cfg['odometry']['model_ckpts_path'], f"{cfg['odometry']['model_name']}.pth")
        self.fps = cfg['yarp']['fps']['odometer']
        calib = cfg['global']['calib']
        
        self.num_patches = cfg['odometry']['dpvo_config']['PATCHES_PER_FRAME']
        self.device = torch.device('cuda:0')

        K = load_intrinsics(calib)
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        K = np.array([fx, fy, cx, cy])
        self.K = torch.from_numpy(K).to(self.device)

        self.image_height = cfg['global']['img_height']
        self.image_width = cfg['global']['img_width']
        self.visualization = cfg['visualization']['active']

        self.slam = DPVO(self._cfg, 
                         self.network, 
                         self.device, 
                         ht=self.image_height,
                         wd=self.image_width,
                         viz=VIZ)

        self.t = 0              # Number of inference steps for DPVO.
        self.counter = 0        # Number of received RGB frames.

        ### Input ports
        port_name = "/" + self.module_name + "/receive/rgb/image:i"
        self.input_rgb_port = yarp.BufferedPortImageRgb()
        self.input_rgb_port.open(port_name)
        self.input_rgb_port.setStrict()
        print(f"{port_name} opened")

        ### Input port for receiving ctrigger (i.e., to reset the pose-graph optimization)
        port_name = "/" + self.module_name + "/receive/ctrigger:i"
        self.ctrigger_port = yarp.BufferedPortBottle()
        self.ctrigger_port.open(port_name)
        print(f"{port_name} opened")

        ### Command/RPC port
        self.cmd_port = yarp.Port()
        self.cmd_port.open('/' + self.module_name + '/command:i')
        print('{:s} opened'.format('/' + self.module_name + '/command:i'))
        self.attach(self.cmd_port)

        ### Output ports
        # NOTE: Note that connections with visualizer and controller are doubled because
        # communication is bufferized with visualizer (we want to get all the information)
        # but not with controller (we just want the newest updates from modules).

        # Port through which updates to camera poses are forwarded
        port_name = "/" + self.module_name + "/forward/to/visualizer/cameras/se3:o"
        self.output_cameras_to_visualizer_port = yarp.BufferedPortBottle()
        self.output_cameras_to_visualizer_port.open(port_name)
        self.output_cameras_to_visualizer_port.writeStrict()
        print(f"{port_name} opened")

        port_name = "/" + self.module_name + "/forward/to/controller/camera/se3:o"
        self.output_camera_to_controller_port = yarp.Port()
        self.output_camera_to_controller_port.open(port_name)
        print(f"{port_name} opened")

        port_name = "/" + self.module_name + "/forward/to/visualizer/patches/coords:o"
        self.output_patch_coords_to_visualizer_port = yarp.Port()
        self.output_patch_coords_to_visualizer_port.open(port_name)
        print(f"{port_name} opened")

        port_name = "/" + self.module_name + "/forward/to/visualizer/patches/depths:o"
        self.output_patch_depths_to_visualizer_port = yarp.BufferedPortBottle()
        self.output_patch_depths_to_visualizer_port.open(port_name)
        self.output_patch_depths_to_visualizer_port.writeStrict()
        print(f"{port_name} opened")

        port_name = "/" + self.module_name + "/forward/to/controller/patches/coords:o"
        self.output_patch_coords_to_controller_port = yarp.Port()
        self.output_patch_coords_to_controller_port.open(port_name)
        print(f"{port_name} opened")

        port_name = "/" + self.module_name + "/forward/to/controller/patches/depths:o"
        self.output_patch_depths_to_controller_port = yarp.Port()
        self.output_patch_depths_to_controller_port.open(port_name)
        print(f"{port_name} opened")

        ### Prepare input image buffers
        self.input_rgb_array = np.ones(
            (cfg["global"]["img_height"], cfg["global"]["img_width"], 3), np.uint8
        )
        self.input_rgb_image = yarp.ImageRgb()
        self.input_rgb_image.resize(
            self.input_rgb_array.shape[1], self.input_rgb_array.shape[0]
        )
        self.input_rgb_image.setExternal(
            self.input_rgb_array, self.input_rgb_array.shape[1], self.input_rgb_array.shape[0]
        )

        self.pendingReset = False
        self.coordsSent = False

        if DEBUG:
            self.outputFile = open('tests/debug/odometerOutput.txt', 'w')

        self.attachTimestamp = cfg['global']['dump']['timestamps']
        self.timestamp = None

        print(f"[ODOMETER] Running at {self.fps} fps.")
        return True

    def respond(self, command, reply):
        return super().respond(command, reply)

    def cleanup(self):
        self.input_rgb_port.close()
        self.cmd_port.close()
        self.ctrigger_port.close()
        self.output_cameras_to_visualizer_port.close()
        self.output_camera_to_controller_port.close()
        self.output_patch_coords_to_visualizer_port.close()
        self.output_patch_depths_to_visualizer_port.close()
        self.output_patch_coords_to_controller_port.close()
        self.output_patch_depths_to_controller_port.close()
        if DEBUG:
            self.outputFile.close()
    
    def interruptModule(self):
        self.input_rgb_port.interrupt()
        self.cmd_port.interrupt()
        self.ctrigger_port.interrupt()
        self.output_cameras_to_visualizer_port.interrupt()
        self.output_camera_to_controller_port.interrupt()
        self.output_patch_coords_to_visualizer_port.interrupt()
        self.output_patch_depths_to_visualizer_port.interrupt()
        self.output_patch_coords_to_controller_port.interrupt()
        self.output_patch_depths_to_controller_port.interrupt()
        return True
    
    def getPeriod(self):
        return 1/float(self.fps)

    def updateModule(self):
        
        pending_reads = self.input_rgb_port.getPendingReads()
        print(f"[ODOMETER] Pending reads: {pending_reads}")
        
        reset = self.ctrigger_port.read(shouldWait=False) 
        if reset is not None or self.pendingReset:
            
            if pending_reads > 0:
                print(f'[ODOMETER] Received reset. Consuming buffer ({pending_reads} pending reads).')
                if not self.pendingReset:
                    self.pendingReset = True # Delay the reset to end processing the entire coming sequence.
            else:
                print('[ODOMETER] Received reset. Resetting pose-graph.')
                self.slam = DPVO(self._cfg, 
                    self.network, 
                    self.device, 
                    ht=self.image_height,
                    wd=self.image_width,
                    viz=VIZ)

                self.t = 0              # Number of inference steps for DPVO.
                self.counter = 0        # Number of received RGB frames.
                self.pendingReset = False
                self.coordsSent = False  

        rgb = self.input_rgb_port.read()
        self.input_rgb_image.copy(rgb)
        assert(
            self.input_rgb_array.__array_interface__['data'][0] == \
            self.input_rgb_image.getRawImage().__int__()
        )

        rgb_array = np.copy(self.input_rgb_array[..., ::-1])
        print(f"[ODOMETER] Reading input rgb image {self.counter} - rgb.shape: {self.input_rgb_array.shape} - {time.time()}")
        self.counter += 1

        with torch.no_grad():
            image = torch.from_numpy(rgb_array).permute(2,0,1).to(self.device)
            
            if DEBUG:
                cv2.imwrite('tests/debug/odometerRGBInput.png', cv2.cvtColor(self.input_rgb_array, cv2.COLOR_RGB2BGR))

            res = self.slam(self.t, image, self.K)

            if self.attachTimestamp:
                self.timestamp = time.time()
                envelopeBottle = yarp.Bottle()
                envelopeBottle.addFloat64(self.timestamp)

            if res is not None:
                updateFlag, (poses, tstamps) = res
                if updateFlag:
                    x, y, inv_depths = self.slam.get_patch_depths()
                    print('[ODOMETER] Preparing output bottles.')
                    # Prepare bottle to controller (only bottling the last camera pose)
                    cameraBottle = yarp.Bottle()
                    for value in poses[-1]:
                        cameraBottle.addFloat32(value.item())

                    if self.attachTimestamp:
                        self.output_camera_to_controller_port.setEnvelope(envelopeBottle)
                    self.output_camera_to_controller_port.write(cameraBottle)
                    print(f"[ODOMETER] Last camera transform: {poses[-1]}")

                    if self.visualization:
                        # Prepare bottles to visualizer
                        camerasVisualizerBottle = self.output_cameras_to_visualizer_port.prepare()
                        camerasVisualizerBottle.clear() # NOTE: Here clearing is necessary, otherwise we will continue to append to the same bottle!

                        for frameIdx in range(poses.shape[0]):
                            for jIdx in range(poses.shape[1]):
                                camerasVisualizerBottle.addFloat32(poses[frameIdx][jIdx].item())
                        
                        if self.attachTimestamp:
                            self.output_cameras_to_visualizer_port.setEnvelope(envelopeBottle)
                        self.output_cameras_to_visualizer_port.write()
                        print(f"[ODOMETER] Writing output poses - poses.shape: {poses.shape}")      

                        if DEBUG:
                            self.outputFile.write(f"[ODOMETER] Last camera bottle sent: {camerasVisualizerBottle.toString()}\n")          
                
                    if not self.coordsSent:
                        # Prepare bottle for patch coords
                        coordsBottle = yarp.Bottle()
                        for (xCoord, yCoord) in zip(x, y):
                            coordsBottle.addFloat32(xCoord.item())
                            coordsBottle.addFloat32(yCoord.item())
                        
                        if self.attachTimestamp:
                            self.output_patch_coords_to_controller_port.setEnvelope(envelopeBottle)
                        self.output_patch_coords_to_controller_port.write(coordsBottle)
                        if self.visualization:
                            if self.attachTimestamp:
                                self.output_patch_coords_to_visualizer_port.setEnvelope(envelopeBottle)
                            self.output_patch_coords_to_visualizer_port.write(coordsBottle)

                        self.coordsSent = True

                    invdepthsBottle = yarp.Bottle()
                    for patchIdx in range(self.num_patches):
                        invdepthsBottle.addFloat32(inv_depths[patchIdx].item())

                    if self.attachTimestamp:
                        self.output_patch_depths_to_controller_port.setEnvelope(envelopeBottle)
                    self.output_patch_depths_to_controller_port.write(invdepthsBottle)

                    if self.visualization:
                        invdepthsVisualizerBottle = self.output_patch_depths_to_visualizer_port.prepare()
                        invdepthsVisualizerBottle.copy(invdepthsBottle)
                        if self.attachTimestamp:
                            self.output_patch_depths_to_visualizer_port.setEnvelope(envelopeBottle)
                        self.output_patch_depths_to_visualizer_port.write()

                    print(f"[ODOMETER] Written {self.num_patches} patches to output ports.")
                else:
                    print(f"[ODOMETER] Frame discarded for optimization. Not enough motion magnitude.")

            self.t += 1
        return True

    
if __name__ == '__main__':

    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.setDefaultContext("VisualOdometryRFModule")
    conffile = rf.find("from").asString()
    if not conffile: 

        print("Using default conf file")
        rf.setDefaultConfigFile("conf.ini")
    else:
        rf.setDefaultConfigFile(rf.find("from").asString())

    rf.configure(sys.argv)

    # Run module
    manager = VisualOdometryRFModule()
    try:
        manager.runModule(rf)
    finally:
        print('Closing VisualOdometryRFModule due to an error...')
        manager.cleanup() 
