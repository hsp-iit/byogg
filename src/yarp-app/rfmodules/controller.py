import yarp
import sys
import yaml
import copy
import open3d as o3d
import numpy as np
import faulthandler
import requests

import torch
import os
from utils.camera import load_intrinsics
from utils.ops import compute_scale_factor
from pcd import make_point_cloud
import threading
from multiprocessing import Lock

import time
import matplotlib.pyplot as plt
import roma

import grasping.contact_graspnet.utils.mesh_utils as mesh_utils
from enum import Enum

from utils.hannes import HannesFingersController, HannesConf
from utils.emg import PTriggerType, EMGProcessor
from odometry.dpvo.lietorch import SE3

from utils.tobii import startTobiiRecording, stopTobiiRecording

VERBOSE = False
KNN = 1
DIST_THRESH = 0.3

yarp.Network.init()

def vprint(message):
    print(message if VERBOSE else "", end='\n' if VERBOSE else '')

def selectPtriggerType(action):
    actions = {
        'open-hold': PTriggerType.OPEN_HOLD,
        'open-double-peak': PTriggerType.OPEN_DOUBLE_PEAK,
        'cocontr': PTriggerType.COCONTR,
        'manual': PTriggerType.MANUAL
    }
    assert action in actions
    return actions[action]

def selectCtriggerType(ctrigger):
    criteria = {
        'auto-dist': CTriggerType.AUTO_DIST,
        'auto-still-hand': CTriggerType.AUTO_STILL_HAND,
        'manual-close-hold': CTriggerType.MANUAL_CLOSE_HOLD,
        'manual-close-double-peak': CTriggerType.MANUAL_CLOSE_DOUBLE_PEAK,
        'manual-cocontr': CTriggerType.MANUAL_COCONTR,
        'manual-rpc': CTriggerType.MANUAL_RPC
    }
    assert ctrigger in criteria
    return criteria[ctrigger]

class ControllerFSMState(Enum):
    IDLE = 0
    ODOMETRY = 1
    GRASPING = 2

class CTriggerType(Enum):
    AUTO_DIST = 0
    AUTO_STILL_HAND = 1             # TODO: Not implemented. Requires empirical analysis on iHannesDataset.
    MANUAL_CLOSE_HOLD = 2           # TODO: Not implemented. Should rely on new method for the EMGProcessor.
    MANUAL_CLOSE_DOUBLE_PEAK = 3    # TODO: Not implemented. Should rely on new method for the EMGProcessor.
    MANUAL_COCONTR = 4              # TODO: Not implemented. Should rely on new method for the EMGProcessor.
    MANUAL_RPC = 5

# Contoller RFModule to interface the Hannes prosthesis.
# NOTE: The controller should only keep in memory the 
class ControllerRFModule(yarp.RFModule):

    def configure(self, rf):
        self.module_name = "controller"
        with open("src/yarp-app/configs/default.yaml", "r") as f:
            cfg = yaml.safe_load(f)

        self.num_patches = cfg['odometry']['dpvo_config']['PATCHES_PER_FRAME']
        self.cfg = cfg
        self.flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        self.scale_transform = 2.0
        
        # Init gripper (panda) params
        gripper = mesh_utils.create_gripper('panda')
        gripper_control_points = gripper.get_control_point_tensor(1, False, convex_hull=False).squeeze()
        mid_point = 0.5*(gripper_control_points[1, :] + gripper_control_points[2, :])
        self.grasp_line_plot = np.array([np.zeros((3,)), mid_point, gripper_control_points[1], gripper_control_points[3], 
                                    gripper_control_points[1], gripper_control_points[2], gripper_control_points[4]])

        calib = cfg["global"]["calib"]
        self.K = load_intrinsics(calib)

        vprint(f"[CONTROLLER] Loading camera intrinsics: {self.K}")

        self.o3dK = o3d.camera.PinholeCameraIntrinsic(width=640, 
                                height=480, 
                                fx=self.K[0, 0], 
                                fy=self.K[1, 1], 
                                cx=self.K[0, 2], 
                                cy=self.K[1, 2])
        
        self.width = cfg["global"]["img_width"]
        self.height = cfg["global"]["img_height"]
        self.scaling_method = cfg["global"]["scaling_method"]
        self.fps = cfg["yarp"]["fps"]["controller"]
        self.viz = cfg["visualization"]["active"]
        self.dumper = cfg["global"]["dump"]["active"]

        self.controller_state = ControllerFSMState.IDLE
        self.device_name = cfg["hannes"]["device_name"]
        self.hannes_active = cfg["hannes"]["active"]
        self.full_buffer_behavior = cfg["hannes"]["when_buffer_is_full"]
        self.use_last_camera = cfg["candidate-selection"]["use-last-camera"]

        self.hannes = None
        self.ptrigger = selectPtriggerType(cfg['hannes']['ptrigger']['action'])
        self.ctrigger = selectCtriggerType(cfg['hannes']['ctrigger']['criteria'])

        if not self.hannes_active:
            # If Hannes is not available or disabled, automatically set (overwrite) ptrigger action to manual (RPC)
            self.ptrigger = PTriggerType.MANUAL 

        if self.hannes_active:
            from libs.pyHannesAPI.pyHannesAPI.pyHannes import Hannes
            from utils.hannes_utils import hannes_init
            ### Initialize Hannes hand communication
            self.hannes = Hannes(device_name=self.device_name)
            hannes_init(self.hannes, control_modality='CONTROL_UNITY')
            self.grasp_preshape = self.cfg['hannes']['preshape']['config'] # Preshape to apply during grasping stage.
            
            self.hannesHome()

            # NOTE: If this flag is False, emg readings are not used for fingers control.
            self.emg_fingers_control = self.cfg['hannes']['emg']['fingers']['active']
            if self.emg_fingers_control:
                ### Initialize Hannes fingers controller
                self.fingers_controller = HannesFingersController(
                    self.hannes,
                    self.cfg['hannes'],
                    fingers_cur_range=0,
                    ps_cur_range=0,
                    fe_cur_range=50
                )

        # NOTE: Should be initialized after pushing the home configuration to Hannes, if active.
        self.hannes_conf = HannesConf(
            hannes = self.hannes,
            hannes_cfg = self.cfg['hannes'],
            candidate_cfg = self.cfg['candidate-selection']
        )

        if self.ptrigger is not PTriggerType.MANUAL:
            ### Initialize Hannes EMG processor
            self.emg_processor = EMGProcessor(
                self.cfg['hannes'],
                ptrigger=self.ptrigger,
                fingers_cur_range=0,
                ps_cur_range=0,
                fe_cur_range=50
            )

        if self.ctrigger is CTriggerType.AUTO_DIST:
            # End ODOMETRY stage if Hannes inside the sphere of radius 1 and center on the gripper width middle point.
            self.auto_dist_thresh = float(cfg['hannes']['ctrigger']['auto-dist']['dist-thresh'])

        elif self.ctrigger is CTriggerType.AUTO_STILL_HAND:
            # NOTE: This check should be disabled during the first frames of the captured sequence.
            # Check that there is limited movement (Hannes is basically still).  
            # This can be done by comparing the last <self.auto_still_hand_thresh> cameras. 
            # TODO: Compute the difference between the camera poses of the last frames on the iHannesDataset and estimate 
            # a reasonable threshold, under which a camera pose can be considered sufficiently similar to the previous one. 
            self.auto_still_hand_frames = 15 

        ### Input ports
        port_name = "/" + self.module_name + "/receive/rgb/image:i"
        self.input_rgb_port = yarp.BufferedPortImageRgb()
        self.input_rgb_port.open(port_name)
        print(f"{port_name} opened") 

        port_name = "/" + self.module_name + "/receive/depth/image:i"
        self.input_depth_port = yarp.BufferedPortImageFloat()
        self.input_depth_port.open(port_name)
        print(f"{port_name} opened") 

        port_name = "/" + self.module_name + "/receive/camera/se3:i"
        self.input_camera_transform_port = yarp.BufferedPortBottle()
        self.input_camera_transform_port.open(port_name)
        print(f"{port_name} opened")

        port_name = "/" + self.module_name + "/receive/patches/coords:i"
        self.input_patch_coords_port = yarp.BufferedPortBottle()
        self.input_patch_coords_port.open(port_name)
        print(f"{port_name} opened")

        port_name = "/" + self.module_name + "/receive/patches/depths:i"
        self.input_patch_depths_port = yarp.BufferedPortBottle()
        self.input_patch_depths_port.open(port_name)
        print(f"{port_name} opened")

        port_name = "/" + self.module_name + "/receive/grasps/se3:i"
        self.input_grasps_port = yarp.BufferedPortBottle()
        self.input_grasps_port.open(port_name)
        print(f"{port_name} opened")

        ### Input port for communication with reader
        port_name = "/" + self.module_name + "/receive/from/reader/reset:i"
        self.reset_receive_reader_port = yarp.BufferedPortBottle()
        self.reset_receive_reader_port.open(port_name)
        print(f"{port_name} opened")

        ### Ouput port for communication with reader
        port_name = "/" + self.module_name + "/forward/to/reader/ptrigger:o"
        self.ptrigger_forward_port_reader = yarp.Port()
        self.ptrigger_forward_port_reader.open(port_name)
        print(f"{port_name} opened")

        ### Ouput port for communication with reader
        port_name = "/" + self.module_name + "/forward/to/reader/ctrigger:o"
        self.ctrigger_forward_port_reader = yarp.Port()
        self.ctrigger_forward_port_reader.open(port_name)
        print(f"{port_name} opened")        

        ### Ouput port for communication with odometer
        port_name = "/" + self.module_name + "/forward/to/odometer/ctrigger:o"
        self.ctrigger_forward_port_odometer = yarp.Port()
        self.ctrigger_forward_port_odometer.open(port_name)
        print(f"{port_name} opened")

        if self.dumper:
            port_name = "/" + self.module_name + "/forward/to/dumper/command:o"
            self.command_dumper_port = yarp.Port()
            self.command_dumper_port.open(port_name)
            print(f"{port_name} opened")

            port_name = "/" + self.module_name + "/forward/hannesJoints:o"
            self.joints_dumper_port = yarp.Port()
            self.joints_dumper_port.open(port_name)
            print(f"{port_name} opened")

            port_name = "/" + self.module_name + "/forward/emg:o"
            self.emg_channels_port = yarp.Port()
            self.emg_channels_port.open(port_name)
            print(f"{port_name} opened")

        ### Output ports for communication with visualizer
        if self.viz:
            port_name = "/" + self.module_name + "/forward/to/visualizer/clear:o"
            self.clear_viz_port = yarp.Port()
            self.clear_viz_port.open(port_name)
            print(f"{port_name} opened")        

            port_name = "/" + self.module_name + "/forward/to/visualizer/candidate:o"
            self.candidate_viz_port = yarp.Port()
            self.candidate_viz_port.open(port_name)
            print(f"{port_name} opened") 

        ### Command/RPC port
        self.cmd_port = yarp.Port()
        self.cmd_port.open('/' + self.module_name + '/command:i')
        print('{:s} opened'.format('/' + self.module_name + '/command:i'))
        self.attach(self.cmd_port)        

        ### Prepare input image, depth and pose buffers
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

        self.input_camera_transform_array = np.array([0., 0., 0., 0., 0., 0., 1.], dtype=np.float32)
        self.grasp_params = []

        self.patchCoordsX = np.zeros((self.num_patches, ), dtype=np.float32)
        self.patchCoordsY = np.zeros((self.num_patches, ), dtype=np.float32)
        self.invDepths = np.zeros((self.num_patches, ), dtype=np.float32)

        self.hasGrasps = False
        self.hasRGB = False
        self.hasDepth = False
        self.hasRGBD = False
        self.hasCoords = False
        self.hasScaleFactor = False
        self.candidates = []

        self.lock = Lock()

        self.attachTimestamp = cfg['global']['dump']['timestamps']
        self.candidateEnvelope = None
        self.jointsEnvelope = None
        self.emgEnvelope = None

        print(f"[CONTROLLER] Running at {self.fps} fps.")

        return True

    def respond(self, command, reply):
        with self.lock:
            dest, action = command.get(0).asString(), command.get(1).asString()
            if dest == "controller":
                print("[CONTROLLER] Received command via RPC:", dest, action)
                if action == "ptrigger":
                    self.startDumping()
                    self.initOdometryStage()
                    ### Switch controller FSM state to ODOMETRY
                    self.controller_state = ControllerFSMState.ODOMETRY
                    reply.addString("[CONTROLLER] Controller state set to ODOMETRY.")
                    print("[CONTROLLER] Controller state set to ODOMETRY.")
                elif action == "ctrigger":
                    self.endOdometryStage()
                    self.stopDumping(delete=True)
                    self.clear()
                    if self.hannes:
                        self.hannesHome()
                    ### Cleaning the reset port.
                    _ = self.reset_receive_reader_port.read(shouldWait=False)  
                    ### Switch controller FSM state back to IDLE
                    self.controller_state = ControllerFSMState.IDLE
                    reply.addString("[CONTROLLER] Controller state set back to IDLE.")
                    print("[CONTROLLER] Controller state set to IDLE.")
                else:
                    reply.addString("[CONTROLLER] Error: Unknown command to controller.")
            else:
                reply.addString("[CONTROLLER] RPC call ignored. Use 'controller <command>'.")
                
        return True

    def clearVisualizer(self):
        if not self.viz:
            return
        clearBottleToVisualizer = yarp.Bottle()
        clearBottleToVisualizer.addInt8(42)
        self.clear_viz_port.write(clearBottleToVisualizer)

    def sendCandidateToVisualizer(self, candidate_id, envelope=None):
        if not self.viz:
            return
        candidateBottleToVisualizer = yarp.Bottle()
        candidateBottleToVisualizer.addInt32(int(candidate_id))
        if envelope is not None:
            self.candidate_viz_port.setEnvelope(envelope)
        self.candidate_viz_port.write(candidateBottleToVisualizer)

    def sendJointsToDumper(self, wps_ref, wfe_ref, fingers_ref, envelope=None):
        if not self.dumper:
            return
        jointsBottle = yarp.Bottle()
        jointsBottle.addFloat32(float(wps_ref))
        jointsBottle.addFloat32(float(wfe_ref))
        jointsBottle.addFloat32(float(fingers_ref))
        if envelope is not None:
            self.joints_dumper_port.setEnvelope(envelope)
        self.joints_dumper_port.write(jointsBottle)
        print(f"[CONTROLLER] Sent Hannes joints configuration to Dumper!")

    def clear(self):
        # Cleanup stuff to prepare the next grasping sequence.
        self.hasGrasps = False
        self.hasRGB = False
        self.hasDepth = False
        self.hasRGBD = False
        self.hasCoords = False
        self.hasScaleFactor = False

        self.candidates = []
        self.grasp_params = []
        self.input_camera_transform_array = np.array([0., 0., 0., 0., 0., 0., 1.], dtype=np.float32)
        
        self.patchCoordsX = np.zeros((self.num_patches, ), dtype=np.float32)
        self.patchCoordsY = np.zeros((self.num_patches, ), dtype=np.float32)
        self.invDepths = np.zeros((self.num_patches, ), dtype=np.float32)

        # NOTE: The controller is responsible for clearing the visualizer, if visualization is active in the app config file.
        self.clearVisualizer()

    def cleanup(self):
        self.input_rgb_port.close()
        self.input_depth_port.close()
        self.input_camera_transform_port.close()
        self.input_patch_coords_port.close()
        self.input_patch_depths_port.close()
        self.input_grasps_port.close()
        self.ptrigger_forward_port_reader.close()
        self.ctrigger_forward_port_reader.close()  
        self.ctrigger_forward_port_odometer.close()
        self.reset_receive_reader_port.close()
        if self.dumper:
            self.command_dumper_port.close()
            self.joints_dumper_port.close()
            self.emg_channels_port.close()
        if self.viz:
            self.clear_viz_port.close()
            self.candidate_viz_port.close()

    def interruptModule(self):
        self.input_rgb_port.interrupt()
        self.input_depth_port.interrupt()
        self.input_camera_transform_port.interrupt()
        self.input_patch_coords_port.interrupt()
        self.input_patch_depths_port.interrupt()
        self.input_grasps_port.interrupt()
        self.ptrigger_forward_port_reader.interrupt()
        self.ctrigger_forward_port_reader.interrupt()
        self.ctrigger_forward_port_odometer.interrupt()
        self.reset_receive_reader_port.interrupt()
        if self.dumper:
            self.command_dumper_port.interrupt()
            self.joints_dumper_port.interrupt()
            self.emg_channels_port.interrupt()
        if self.viz:
            self.clear_viz_port.interrupt()
            self.candidate_viz_port.interrupt()
        return True
    
    def getPeriod(self):
        return 1/float(self.fps)

    def register_grasps(self, grasps, cam_pose=np.eye(4)):
        # Grasps are encoded as 9D vectors in the form:
        # [4D quaternion (unit), 3D trans, opening, score]
        gquats = grasps[:, :4]
        gtranslations = grasps[:, 4:7]
        gopenings = grasps[:, 7]
        gscores = grasps[:, 8]

        grots = roma.unitquat_to_rotmat(torch.tensor(gquats)).numpy()
        gposes = np.tile(np.eye(4), (gtranslations.shape[0], 1, 1))
        gposes[:, :3, :3] = grots
        gposes[:, :3, 3] = gtranslations

        # TODO: Wrap this code inside a function to import both in visualize_all.py script and here.
        # TODO: Avoid code duplication.

        for g_idx, (g,g_opening) in enumerate(zip(gposes, gopenings)):
            gripper_control_points_closed = self.grasp_line_plot.copy()
            gripper_control_points_closed[2:,0] = np.sign(self.grasp_line_plot[2:,0]) * g_opening/2
            
            pts = np.matmul(gripper_control_points_closed, g[:3, :3].T)
            pts += np.expand_dims(g[:3, 3], 0)
            pts_homog = np.concatenate((pts, np.ones((7, 1))),axis=1)
            pts = np.dot(pts_homog, cam_pose.T)[:,:3]

            # Store gripper params (center and opening) in order to select a candidate.
            self.grasp_params.append((g_idx, pts[1], g, g_opening))

        vprint("[CONTROLLER] Adding grasp geometries to grasp buffer.")

    def select_candidates(self):
        # Check if we already have grasp geometries.
        if len(self.grasp_params) == 0:
            return []
        
        vprint("[CONTROLLER] Selecting a grasp candidate...")
        M = len(self.grasp_params)
        hand_pos = self.input_camera_transform_array[:3]

        # Filter grippers based on the distance from the center of the last camera in the trajectory.
        if VERBOSE:
            vprint(f"Hand position: {hand_pos}")
            grippers = []
            for g_idx, center, pose, opening in self.grasp_params:
                dist_from_hand = np.linalg.norm(center - hand_pos)
                vprint(f"Gripper Params - index: {g_idx}, center: {center}, opening: {opening}, distance from hand: {dist_from_hand}")    
                if dist_from_hand < DIST_THRESH:
                    vprint("Selected as potential candidate grasp.")
                    grippers.append((g_idx, center, pose, opening))  
        else:
            grippers = [(g_idx, center, pose, opening) for g_idx, center, pose, opening in self.grasp_params if np.linalg.norm(center - hand_pos) < DIST_THRESH]
        
        grippers = sorted(grippers, key=lambda x: np.linalg.norm(x[1] - hand_pos))
        candidates = grippers[:KNN]
        return candidates
    
    def initOdometryStage(self):
        dummyBottle = yarp.Bottle()
        dummyBottle.addInt8(42)
        self.ptrigger_forward_port_reader.write(dummyBottle)
    
    def endOdometryStage(self):
        dummyBottle = yarp.Bottle()
        dummyBottle.addInt8(42)
        self.ctrigger_forward_port_reader.write(dummyBottle)
        self.ctrigger_forward_port_odometer.write(dummyBottle)

    def startDumping(self):
        if not self.dumper:
            return
        commandBottle = yarp.Bottle()
        commandBottle.addString('start')
        self.command_dumper_port.write(commandBottle)    

    def stopDumping(self, delete=False):
        if not self.dumper:
            return
        commandBottle = yarp.Bottle()
        if not delete:
            commandBottle.addString('stop')
        else:
            commandBottle.addString('delete')
        self.command_dumper_port.write(commandBottle)

    def hannesHome(self):
        if self.hannes:
            self.hannes.move_wristPS(0)
            time.sleep(0.2)
            self.hannes.move_wristFE(50)
            time.sleep(0.2)
            self.hannes.move_hand(0)
            time.sleep(0.2)
            self.hannes.move_thumb_lateral()
        else:
            raise ValueError('[CONTROLLER] Cannot map home config to Hannes: no available connection!')

    def hannesPreshape(self, preshape):
        assert preshape in ['lateral', 'pinch', 'power']
        if preshape == 'lateral':
            self.hannes.move_thumb_lateral()
        elif preshape == 'pinch':
            self.hannes.move_thumb_pinch()
        elif preshape == 'power':
            self.hannes.move_thumb_power()
        else:
            self.hannes.move_thumb_home()

    def selectNearestCandidate(self):
        vprint(f"[CONTROLLER] Selecting the nearest candidate grasp.")
        if len(self.candidates) > 0 and self.hasScaleFactor:
            nearest_candidate_id, nearest_candidate_center, nearest_candidate_pose, nearest_candidate_opening = self.candidates[0]
            # print(f"[CONTROLLER] Selected candidate center: {nearest_candidate_center}")
            last_camera_pos = self.input_camera_transform_array[:3]
            # print(f"[CONTROLLER] Camera position (optical center): {last_camera_pos}")
            # Compute distance as euclidean (l2) norm.
            dist = np.linalg.norm(last_camera_pos - nearest_candidate_center)

            print(f"[CONTROLLER] Last hand position: {last_camera_pos}")
            print(f"[CONTROLLER] Candidate Grasp - id: {nearest_candidate_id}, center: {nearest_candidate_center}, opening: {nearest_candidate_opening}, dist (from hand): {dist}")
            
            return nearest_candidate_id, nearest_candidate_center, nearest_candidate_pose, nearest_candidate_opening, dist
        return None

    def getCameraTransform(self):
        # Transform 3D trans + 4D unit quaternion to a camera transform
        if not self.hannes_active and self.use_last_camera:
            # TODO: Understand if we need to compute the inv before computing the matrix.
            cpose = SE3(torch.tensor(self.input_camera_transform_array)).inv().matrix().numpy()
        else: 
            cpose = np.eye(4)
        return cpose

    def updateModule(self):
        with self.lock:
            vprint(f"[CONTROLLER] Update module loop with controller state {self.controller_state}")
            if self.controller_state is ControllerFSMState.IDLE:
                ### Read EMG signals
                if self.hannes_active and self.hannes is not None:
                    channels = self.hannes.measurements_emg()

                    if self.attachTimestamp:
                        ts = time.time()
                        self.emgEnvelope = yarp.Bottle()
                        self.emgEnvelope.addFloat64(ts)

                    print(f"[CONTROLLER] EMG channels: {channels}")

                    channelsBottle = yarp.Bottle()
                    for channel in channels:
                        channelsBottle.addFloat32(channel)

                    if self.attachTimestamp:
                        self.emg_channels_port.setEnvelope(self.emgEnvelope)
                    self.emg_channels_port.write(channelsBottle)

                    if self.emg_fingers_control:
                        ### Let the user control fingers in IDLE state.
                        assert hasattr(self, 'fingers_controller')
                        self.fingers_controller.update(channels)

                    ### Check for prediction trigger.
                    if self.ptrigger is not PTriggerType.MANUAL:
                        if self.emg_processor.check_ptrigger(channels):
                            self.startDumping()
                            self.initOdometryStage()
                            ### Switch controller FSM state to ODOMETRY
                            self.controller_state = ControllerFSMState.ODOMETRY
                            print("[CONTROLLER] Controller state set to ODOMETRY.")
                    
            elif self.controller_state is ControllerFSMState.ODOMETRY:
                vprint("[CONTROLLER] Entering the odometry loop.")
                # Check if the reader has reset the odometer state to prevent memory issues
                # and possible overflows. In this case, we switch back to IDLE stage.
                # TODO: Understand if we can find some conditions under which is convenient
                #       to map a gripper pose to Hannes even if the ctrigger condition has not been triggered. 
                readerResetBottle = self.reset_receive_reader_port.read(shouldWait=False)
                if readerResetBottle is not None:
                    print("[CONTROLLER] Received reset bottle from reader.")
                    # NOTE: You can define this behavior from the configuration file.
                    # When reaching the full size of the buffer, you can either go back to the Home position,
                    # or map the nearest grasp (no guarantees it will be meaningful). 

                    if self.full_buffer_behavior == 'nearest-grasp':
                        # NOTE: In this case self.clear() is called while running the GRASPING stage, so it should not
                        #       be called here. This also allows to view the selected gripper pose in the Visualizer 
                        #       before it gets deleted.

                        # Push the nearest gripper pose to Hannes, after mapping.
                        nearest_candidate_info = self.selectNearestCandidate()

                        if self.attachTimestamp:
                            ts = time.time()
                            self.candidateEnvelope = yarp.Bottle()
                            self.candidateEnvelope.addFloat64(ts)

                        if nearest_candidate_info is not None:
                            (nearest_candidate_id, nearest_candidate_center, nearest_candidate_pose, nearest_candidate_opening, nearest_candidate_distance) = nearest_candidate_info
                    
                            self.sendCandidateToVisualizer(candidate_id=nearest_candidate_id, envelope=self.candidateEnvelope)

                            print(f"[CONTROLLER] Selecting nearest gripper pose without guarantees, ending odometry stage.")
                            self.endOdometryStage()
                            cpose = self.getCameraTransform()
                            vprint(f"[CONTROLLER] cpose = {cpose}")

                            self.wps_ref, self.wfe_ref, self.fingers_ref = self.hannes_conf.map_gripper_pose_to_hannes_config(
                                                                                        gpose = nearest_candidate_pose,
                                                                                        gopening = nearest_candidate_opening,
                                                                                        cpose = cpose)
                            if self.attachTimestamp:
                                ts = time.time()
                                self.jointsEnvelope = yarp.Bottle()
                                self.jointsEnvelope.addFloat64(ts)

                            self.sendJointsToDumper(self.wps_ref, self.wfe_ref, self.fingers_ref, envelope=self.jointsEnvelope)
                            self.stopDumping()
                            self.controller_state = ControllerFSMState.GRASPING
                            print("[CONTROLLER] Controller state set to GRASPING.")
                    else:
                        # Push home configuration back to Hannes.
                        if self.hannes is not None: 
                            self.hannesHome()
                        # self.clear()
                        self.controller_state = ControllerFSMState.IDLE
                        print("[CONTROLLER] Controller state set to IDLE.")

                if not self.hasGrasps:
                    vprint("[CONTROLLER] Waiting for grasps...")
                    graspBottle = self.input_grasps_port.read(shouldWait=False)
                    if graspBottle is not None:
                        numGrasps = len(graspBottle.toString().split(' ')) // 9
                        grasps = np.zeros((numGrasps, 9), dtype=np.float32)

                        for graspIdx in range(numGrasps):
                            for jIdx in range(9):
                                grasps[graspIdx, jIdx] = graspBottle.get(graspIdx * 9 + jIdx).asFloat32()
                        
                        self.register_grasps(grasps)
                        vprint("[CONTROLLER] Registered {numGrasps} gripper poses.")
                        self.hasGrasps = True

                if not self.hasRGBD:
                    vprint("[CONTROLLER] Waiting for reference rgb and depth...")
                    rgb = self.input_rgb_port.read(shouldWait=False)
                    if rgb is not None:
                        self.input_rgb_image.copy(rgb)
                        self.hasRGB = True
                    depth = self.input_depth_port.read(shouldWait=False)
                    if depth is not None:
                        self.input_depth_image.copy(depth)
                        self.hasDepth = True
                    
                    self.hasRGBD = self.hasRGB and self.hasDepth

                vprint("[CONTROLLER] Polling on camera transform...")
                # Read the last camera position, whenever it is available.
                cameraBottle = self.input_camera_transform_port.read(shouldWait=False)
                if cameraBottle is not None:
                    for idx in range(len(self.input_camera_transform_array)):
                        self.input_camera_transform_array[idx] = cameraBottle.get(idx).asFloat32()

                if not self.hasCoords:
                    vprint("[CONTROLLER] Polling on coords...")
                    coordsBottle = self.input_patch_coords_port.read(shouldWait=False)
                    if coordsBottle is not None:
                        vprint("[CONTROLLER] Unpacking coords bottle...")
                        for patchIdx in range(self.num_patches):
                            self.patchCoordsX[patchIdx] = coordsBottle.get(2 * patchIdx).asFloat32()
                            self.patchCoordsY[patchIdx] = coordsBottle.get(2 * patchIdx + 1).asFloat32()
                        self.hasCoords = True

                elif self.hasCoords and self.hasRGBD:
                    vprint("[CONTROLLER] Polling on inv. depths...")
                    # Read the updated inv. depths of the patches, whenever they are available.
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
                        vprint(f"[CONTROLLER] Predicted scale factor: {scale_factor}")
                        self.hasScaleFactor = True
                        # Camera transform is stored as 7D vector (3D trans + unit quaternion)
                        self.input_camera_transform_array[:3] *= scale_factor
                        vprint(f"[CONTROLLER] Last camera transform: {self.input_camera_transform_array}")

                # Update candidate selection. 
                self.candidates = self.select_candidates()
                nearest_candidate_info = self.selectNearestCandidate()

                if self.attachTimestamp:
                    ts = time.time()
                    self.candidateEnvelope = yarp.Bottle()
                    self.candidateEnvelope.addFloat64(ts)

                vprint(f"[CONTROLLER] nearest candidate info: {nearest_candidate_info}")
                
                if nearest_candidate_info is not None:
                    (nearest_candidate_id, nearest_candidate_center, nearest_candidate_pose, nearest_candidate_opening, nearest_candidate_distance) = nearest_candidate_info
                    vprint(f"[CONTROLLER] nearest candidate id: {nearest_candidate_id}")

                    # NOTE: Will also send candidate to Dumper module.
                    self.sendCandidateToVisualizer(candidate_id=nearest_candidate_id, envelope=self.candidateEnvelope)

                    if self.ctrigger is CTriggerType.AUTO_DIST and nearest_candidate_distance < self.auto_dist_thresh:
                        print(f"[CONTROLLER] Grasp pose selected, ending odometry stage.")
                        self.endOdometryStage()

                        cpose = self.getCameraTransform()
                        vprint(f"[CONTROLLER] cpose = {cpose}")

                        self.wps_ref, self.wfe_ref, self.fingers_ref = self.hannes_conf.map_gripper_pose_to_hannes_config(
                                                                                gpose = nearest_candidate_pose,
                                                                                gopening = nearest_candidate_opening,
                                                                                cpose = cpose)
                        if self.attachTimestamp:
                            ts = time.time()
                            self.jointsEnvelope = yarp.Bottle()
                            self.jointsEnvelope.addFloat64(ts)

                        self.sendJointsToDumper(self.wps_ref, self.wfe_ref, self.fingers_ref, envelope=self.jointsEnvelope)
                        self.stopDumping()
                        self.controller_state = ControllerFSMState.GRASPING
                        print("[CONTROLLER] Controller state set to GRASPING.")

            elif self.controller_state is ControllerFSMState.GRASPING:
                print(f"[CONTROLLER] Hannes config: WPS = {self.wps_ref}, WFE = {self.wfe_ref}, FING = {self.fingers_ref}")
                
                if self.hannes:
                    print("[CONTROLLER] Mapping selected pose to Hannes.")
                    self.hannes.move_wristPS(int(self.wps_ref))
                    time.sleep(0.1)
                    self.hannes.move_wristFE(int(self.wfe_ref))

                # Wait a given amount of time before automatically closing the fingers.
                time.sleep(float(self.cfg['global']['pipeline']['times']['closing_fingers']))

                if self.hannes:
                    self.hannesPreshape(self.grasp_preshape)
                    self.hannes.move_hand(int(self.fingers_ref))
                # Wait a given amount of time before switching back to IDLE state.
                time.sleep(float(self.cfg['global']['pipeline']['times']['reset']))

                self.clear()
                # Push home configuration back to Hannes.
                if self.hannes:
                    self.hannesHome()

                # Clean the reset port. (IMPORTANT) This is needed because the reader module can 
                # send a reset message which is never read by the controller (otherwise, i.e., without 
                # the following read from the reset port), as it can select a gripper pose before 
                # reading it. This behavior could cause some issues, as this "pending" reset would be 
                # read as the ODOMETRY stage is entered again by the controller FSM, i.e., during the next
                # approaching sequence. 
                vprint("[CONTROLLER] Cleaning the reset port.")
                _ = self.reset_receive_reader_port.read(shouldWait=False)

                self.controller_state = ControllerFSMState.IDLE
                print("[CONTROLLER] Controller state set to IDLE.")

        return True
    
if __name__ == '__main__':

    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.setDefaultContext("ControllerRFModule")
    conffile = rf.find("from").asString()
    if not conffile:
        print("Using default conf file")
        rf.setDefaultConfigFile("conf.ini")
    else:
        rf.setDefaultConfigFile(rf.find("from").asString())

    rf.configure(sys.argv)

    # Run module
    manager = ControllerRFModule()
    try:
        manager.runModule(rf)
    finally:
        print('Closing ControllerRFModule due to an error...')
        manager.cleanup() 
