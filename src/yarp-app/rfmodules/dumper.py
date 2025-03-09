import yarp
import sys
import yaml
import time
import copy
import numpy as np
from utils.io import IOHandler
from multiprocessing import Lock

'''
Yarp RFModule which receives data from selected modules (see src/yarp-app/configs/default.yaml) and dumps them.
'''

yarp.Network.init()

class DumperRFModule(yarp.RFModule):
    def configure(self, rf):
        self.module_name = "dumper"
        with open("src/yarp-app/configs/default.yaml", "r") as f:
            cfg = yaml.safe_load(f)

        self.fps = cfg['yarp']['fps'][self.module_name]
        self.active = cfg['global']['dump']['active']
        self.READER = self.active and cfg['global']['dump']['reader']
        self.DEPTHER = self.active and cfg['global']['dump']['depther']
        self.ODOMETER = self.active and cfg['global']['dump']['odometer']
        self.GRASPER = self.active and cfg['global']['dump']['grasper']
        self.CONTROLLER = self.active and cfg['global']['dump']['controller']
        self.IO = None

        self.root = str(cfg['global']['dump']['root'])
        self.width = int(cfg['global']['img_width'])
        self.height = int(cfg['global']['img_height'])
        self.num_patches = int(cfg['odometry']['dpvo_config']['PATCHES_PER_FRAME'])

        port_name = "/" + self.module_name + "/receive/command:i"
        self.input_command_port = yarp.BufferedPortBottle()
        self.input_command_port.open(port_name)
        print(f"{port_name} opened") 

        self.dump = False
        self.pendingStop = False
        self.delete = False

        if self.READER:
            port_name = "/" + self.module_name + "/receive/rgb/image:i"
            self.input_rgb_port = yarp.BufferedPortImageRgb()
            self.input_rgb_port.open(port_name)
            self.input_rgb_port.setStrict()
            print(f"{port_name} opened")    

            self.input_rgb_array = np.ones(
                (self.height, self.width, 3), dtype=np.uint8
            )
            self.input_rgb_image = yarp.ImageRgb()
            self.input_rgb_image.resize(
                self.input_rgb_array.shape[1], self.input_rgb_array.shape[0]
            )
            self.input_rgb_image.setExternal(
                self.input_rgb_array, self.input_rgb_array.shape[1], self.input_rgb_array.shape[0]
            )

        if self.DEPTHER:
            port_name = "/" + self.module_name + "/receive/depth/image:i"
            self.input_depth_port = yarp.BufferedPortImageFloat()
            self.input_depth_port.open(port_name)
            print(f"{port_name} opened")

            self.input_depth_array = np.ones(
                (self.height, self.width), dtype=np.float32
            )
            self.input_depth_image = yarp.ImageFloat()
            self.input_depth_image.resize(
                self.input_depth_array.shape[1], self.input_depth_array.shape[0]
            )
            self.input_depth_image.setExternal(
                self.input_depth_array, self.input_depth_array.shape[1], self.input_depth_array.shape[0]
            )
        
        if self.ODOMETER:
            port_name = "/" + self.module_name + "/receive/cameras/se3:i"
            self.input_cameras_port = yarp.BufferedPortBottle()
            self.input_cameras_port.open(port_name)
            self.input_cameras_port.setStrict()
            print(f"{port_name} opened")

            port_name = "/" + self.module_name + "/receive/patches/coords:i"
            self.input_patch_coords_port = yarp.BufferedPortBottle()
            self.input_patch_coords_port.open(port_name)
            self.input_patch_coords_port.setStrict()
            print(f"{port_name} opened")

            port_name = "/" + self.module_name + "/receive/patches/depths:i"
            self.input_patch_depths_port = yarp.BufferedPortBottle()
            self.input_patch_depths_port.open(port_name)
            self.input_patch_depths_port.setStrict()
            print(f"{port_name} opened")

            self.hasCoords = False
            self.patchCoordsX = np.zeros((self.num_patches, ), dtype=np.float32)
            self.patchCoordsY = np.zeros((self.num_patches, ), dtype=np.float32)
            self.invDepths = np.zeros((self.num_patches, ), dtype=np.float32)
            self.only_last_camera_pose = cfg['global']['dump']['only_last_camera_pose']

        if self.GRASPER:
            port_name = "/" + self.module_name + "/receive/grasps/se3:i"
            self.input_grasps_port = yarp.BufferedPortBottle()
            self.input_grasps_port.open(port_name)
            print(f"{port_name} opened")

        if self.CONTROLLER:
            port_name = "/" + self.module_name + "/receive/candidate:i"
            self.input_candidate_port = yarp.BufferedPortBottle()
            self.input_candidate_port.open(port_name)
            print(f"{port_name} opened") 

            port_name = "/" + self.module_name + "/receive/hannesJoints:i"
            self.input_joints_port = yarp.BufferedPortBottle()
            self.input_joints_port.open(port_name)
            print(f"{port_name} opened") 
            self.hasJoints = False

            port_name = "/" + self.module_name + "/receive/emg:i"
            self.input_emg_port = yarp.BufferedPortBottle()
            self.input_emg_port.open(port_name)
            print(f"{port_name} opened") 

            self.HANNES = cfg['hannes']['active']

        self.attachTimestamp = cfg['global']['dump']['timestamps']
        self.IO = IOHandler()

        return True

    def respond(self, command, reply):
        return super().respond(command, reply)

    def cleanup(self):
        self.input_command_port.close()
        if self.READER:
            self.input_rgb_port.close()
        if self.DEPTHER:
            self.input_depth_port.close()
        if self.ODOMETER:
            self.input_cameras_port.close()
            self.input_patch_coords_port.close()
            self.input_patch_depths_port.close()
        if self.GRASPER:
            self.input_grasps_port.close()
        if self.CONTROLLER:
            self.input_candidate_port.close()
            self.input_joints_port.close()
            self.input_emg_port.close()
    
    def interruptModule(self):
        self.input_command_port.interrupt()
        if self.READER:
            self.input_rgb_port.interrupt()
        if self.DEPTHER:
            self.input_depth_port.interrupt()
        if self.ODOMETER:
            self.input_cameras_port.interrupt()
            self.input_patch_coords_port.interrupt()
            self.input_patch_depths_port.interrupt()
        if self.GRASPER:
            self.input_grasps_port.interrupt()
        if self.CONTROLLER:
            self.input_candidate_port.interrupt()
            self.input_joints_port.interrupt()
            self.input_emg_port.interrupt()

        return True
    
    def getPeriod(self):
        return 1/float(self.fps)
    
    def clear(self):
        self.delete = False
        self.hasCoords = False
        self.hasJoints = False
        self.patchCoordsX = np.zeros((self.num_patches, ), dtype=np.float32)
        self.patchCoordsY = np.zeros((self.num_patches, ), dtype=np.float32)
        self.invDepths = np.zeros((self.num_patches, ), dtype=np.float32)

    def getIncomingBottles(self):
        incomingBottles = 0
        if self.READER:
            incomingBottles += self.input_rgb_port.getPendingReads()
        if self.DEPTHER:
            incomingBottles += self.input_depth_port.getPendingReads()
        if self.ODOMETER:
            incomingBottles += self.input_cameras_port.getPendingReads()
            incomingBottles += self.input_patch_coords_port.getPendingReads()
            incomingBottles += self.input_patch_depths_port.getPendingReads()
        if self.GRASPER:
            incomingBottles += self.input_grasps_port.getPendingReads()
        if self.CONTROLLER:
            incomingBottles += self.input_candidate_port.getPendingReads()
            incomingBottles += self.input_joints_port.getPendingReads()

            if self.HANNES:
                incomingBottles += self.input_emg_port.getPendingReads()

        return incomingBottles

    def getLastTimestamp(self, yarpPort):
        if self.attachTimestamp:
            envelope = yarp.Bottle()
            yarpPort.getEnvelope(envelope)
            tstamp = envelope.get(0).asFloat64()
        else:
            tstamp = None
        return tstamp
    
    def updateModule(self):

        commandBottle = self.input_command_port.read(shouldWait=False)
        if commandBottle is not None:
            action = commandBottle.get(0).asString()
            print(f"[DUMPER] Received command bottle with action: {action}")
            
            if action == 'start':
                self.IO.create_data_folder(self.root, time.strftime('%d_%m_%y'), time.strftime('%H_%M_%S'))
                self.dump = True
                self.pendingStop = False
                print("[DUMPER] Start dumping data...")

            elif action == 'stop':
                print("[DUMPER] Stop received. Will clear all buffers before stopping if necessary.")
                self.dump = False
                if self.getIncomingBottles() > 0:
                    if not self.pendingStop:
                        self.pendingStop = True
                else:
                    if self.IO.is_streaming():
                        print("[DUMPER] Saving recorded stream.")
                        self.IO.close_stream()
                    self.clear()

            elif action == 'delete':
                # NOTE: delete action is stop + delete recorded video. We still wait for all incoming buffers
                #       to be empty, to clear port buffers.
                self.dump = False
                self.delete = True
                
                if self.getIncomingBottles() > 0:
                    if not self.pendingStop:
                        self.pendingStop = True
                else:
                    
                    if self.IO.is_streaming():
                        print("[DUMPER] Deleting recorded stream.")
                        self.IO.delete_stream()
                    self.clear()
                    

        if self.dump or self.pendingStop:
            if self.READER:
                rgb = self.input_rgb_port.read(shouldWait=False)
                if rgb is not None:
                    self.input_rgb_image.copy(rgb)
                    tstamp = self.getLastTimestamp(self.input_rgb_port)
                    rgb_array = copy.deepcopy(self.input_rgb_array)
                    self.IO.save_rgb(rgb_array, tstamp)

            if self.DEPTHER:
                depth = self.input_depth_port.read(shouldWait=False)
                if depth is not None:
                    self.input_depth_image.copy(depth)
                    tstamp = self.getLastTimestamp(self.input_depth_port)
                    depth_array = copy.deepcopy(self.input_depth_array)
                    self.IO.save_depth(depth_array, tstamp)

            if self.ODOMETER:
                camerasDumperBottle = self.input_cameras_port.read(shouldWait=False)
                if camerasDumperBottle is not None:
                    tstamp = self.getLastTimestamp(self.input_cameras_port)
                    camerasBottleData = camerasDumperBottle.toString()
                    numCameras = len(camerasBottleData.split(' ')) // 7
                    cameraPoses = np.zeros((numCameras, 7), dtype=np.float32)

                    for frameIdx in range(numCameras):
                        for jIdx in range(7):
                            cameraPoses[frameIdx, jIdx] = camerasDumperBottle.get(frameIdx * 7 + jIdx).asFloat32()

                    if self.only_last_camera_pose:
                        self.IO.save_camera_poses(cameraPoses[-1], tstamp)
                    else:
                        self.IO.save_camera_poses(cameraPoses, tstamp)

                if not self.hasCoords:
                    coordsDumperBottle = self.input_patch_coords_port.read(shouldWait=False)
                    if coordsDumperBottle is not None:
                        tstamp = self.getLastTimestamp(self.input_patch_coords_port)
                        for patchIdx in range(self.num_patches):
                            self.patchCoordsX[patchIdx] = coordsDumperBottle.get(2 * patchIdx).asFloat32()
                            self.patchCoordsY[patchIdx] = coordsDumperBottle.get(2 * patchIdx + 1).asFloat32()
                        self.hasCoords = True
                        
                        self.IO.save_patch_coords(self.patchCoordsX, self.patchCoordsY, tstamp)

                invDepthsDumperBottle = self.input_patch_depths_port.read(shouldWait=False)
                if invDepthsDumperBottle is not None:
                    tstamp = self.getLastTimestamp(self.input_patch_depths_port)
                    for patchIdx in range(self.num_patches):
                        self.invDepths[patchIdx] = invDepthsDumperBottle.get(patchIdx).asFloat32()
                    
                    self.IO.save_inv_depths(self.invDepths, tstamp)
            
            if self.GRASPER:
                graspDumperBottle = self.input_grasps_port.read(shouldWait=False)
                if graspDumperBottle is not None:
                    tstamp = self.getLastTimestamp(self.input_grasps_port)
                    numGrasps = len(graspDumperBottle.toString().split(' ')) // 9
                    grasps = np.zeros((numGrasps, 9), dtype=np.float32)

                    for graspIdx in range(numGrasps):
                        for jIdx in range(9):
                            grasps[graspIdx, jIdx] = graspDumperBottle.get(graspIdx * 9 + jIdx).asFloat32()

                    self.IO.save_gripper_poses(grasps, tstamp)

            if self.CONTROLLER:
                emgBottle = self.input_emg_port.read(shouldWait=False)
                if emgBottle is not None:
                    tstamp = self.getLastTimestamp(self.input_emg_port)
                    emgChannels = np.zeros((6, ), dtype=np.float32)
                    for channelIdx in range(6):
                        emgChannels[channelIdx] = emgBottle.get(channelIdx).asFloat32()
                    self.IO.save_emg_channels(emgChannels, tstamp)

                candidateBottle = self.input_candidate_port.read(shouldWait=False)
                if candidateBottle is not None:
                    tstamp = self.getLastTimestamp(self.input_candidate_port)
                    candidateId = candidateBottle.get(0).asInt32()
                    self.IO.save_candidate_id(candidateId, tstamp)

                if not self.hasJoints:
                    jointsBottle = self.input_joints_port.read(shouldWait=False)
                    if jointsBottle is not None:
                        tstamp = self.getLastTimestamp(self.input_joints_port)
                        print("[DUMPER] Received Hannes joints configuration. Dumping it...")
                        wps = jointsBottle.get(0).asFloat32()
                        wfe = jointsBottle.get(1).asFloat32()
                        fingers = jointsBottle.get(2).asFloat32()
                        self.IO.save_joints_ref(wps, wfe, fingers, tstamp)
                        self.hasJoints = True

            if self.pendingStop:
                self.pendingStop = (self.getIncomingBottles() > 0)
                print(f"[DUMPER] self.getIncomingBottles() = {self.getIncomingBottles()}")
                print(f"[DUMPER] self.pendingStop = {self.pendingStop}")
                if not self.pendingStop:
                    print("[DUMPER] Buffers cleaned. Now actually stopping.")
                    # We were still waiting for bottles, but now we have processed them all.   
                    # So now we can finally reset the state of the dumper.
                    if self.delete:
                        print("[DUMPER] Deleting recorded stream.")
                        self.IO.delete_stream()
                    else:
                        print("[DUMPER] Saving recorded stream.")
                        self.IO.close_stream()
                    self.clear()
        return True
    

if __name__ == '__main__':
    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.setDefaultContext("DumperRFModule")
    conffile = rf.find("from").asString()
    if not conffile: 
        print("Using default conf file")
        rf.setDefaultConfigFile("conf.ini")
    else:
        rf.setDefaultConfigFile(rf.find("from").asString())

    rf.configure(sys.argv)

    # Run module
    manager = DumperRFModule()
    try:
        manager.runModule(rf)
    finally:
        print('Closing DumperRFModule due to an error...')
        manager.cleanup() 
