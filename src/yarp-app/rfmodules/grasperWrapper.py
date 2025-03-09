import yarp
import sys
import yaml
import os
import numpy as np
import torch
import roma
import time
from utils.camera import load_intrinsics
from pcd import build_pcd_from_rgbd
import argparse
from multiprocessing import shared_memory, set_start_method

yarp.Network.init()

class GrasperWrapperRFModule(yarp.RFModule):

    def configure(self, rf):
        parser = argparse.ArgumentParser()
        parser.add_argument("--load", type=str, required=False)
        args = parser.parse_args()

        self.module_name = "grasper"
        with open("src/yarp-app/configs/default.yaml", "r") as f:
            cfg = yaml.safe_load(f)

        self.args = args
        self.fps = 30

        calib = cfg['global']['calib']
        self.K = load_intrinsics(calib)

        self.min_depth = cfg["mde"]["min_depth"]
        self.max_depth = cfg["mde"]["max_depth"]

        self.ckpt_dir = os.path.join(cfg['grasping']['model_ckpts_path'], 
                                     cfg['grasping']['model_name'])
        
        self.cfg = cfg['grasping']['contact_grasp_cfg']

        ### Output ports
        port_name = "/" + self.module_name + "/forward/grasps/se3:o"
        self.output_grasps_port = yarp.Port()
        self.output_grasps_port.open(port_name)
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
        
        ### Init output grasp poses buffer
        
        # Buffer dimension is set to support sending 10 grasps/Hz (to reduce communication to bare-minimum in case few grasps are generated).
        self.buffer_size = cfg['global']['grasp_buffer_size']
        self.output_grasps_array = np.ones(
            (self.buffer_size, 9), dtype=np.float32
        )
        self.output_grasps_buffer = yarp.ImageFloat()
        self.output_grasps_buffer.resize(
            self.output_grasps_array.shape[1], self.output_grasps_array.shape[0] 
        )
        self.output_grasps_buffer.setExternal(
            self.output_grasps_array, self.output_grasps_array.shape[1], self.output_grasps_array.shape[0] 
        )

        if args.load:
            # Load generated grasps.
            args.load = os.path.join(f'./results/predictions_pred_{args.load}.npz')
            grasp_data = np.load(args.load, allow_pickle=True)
            gposes = grasp_data['pred_grasps_cam'].item()[-1]
            gscores = grasp_data['scores'].item()[-1]
            gopenings = grasp_data['gripper_openings'].item()[-1] 

            self.generated_grasps = []
            self.tx = 0

            # Encode grasps in desired format.
            for idx in range(gposes.shape[0]):
                rot, trans = torch.from_numpy(gposes[idx][:3, :3]), gposes[idx][:3, 3]
                uquat = roma.rotmat_to_unitquat(rot).numpy()
                gopening = gopenings[idx]
                gscore = gscores[idx]

                self.generated_grasps.append(np.hstack([uquat, trans, np.float32(gopening), np.float32(gscore)]))
        else:

            set_start_method('spawn')
            ### Initialize shared memories for R/W communication with Contact-GraspNet process

            shm_rgb_buffer = shared_memory.SharedMemory('rgb', create=True, size=self.input_rgb_array.nbytes)
            shm_depth_buffer = shared_memory.SharedMemory('depth', create=True, size=self.input_depth_array.nbytes)
            
            print('[GRASPER] Waiting to read depth and rgb...')
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

            # Copy data (rgb and depth) into shared memory
            shm_rgb_array = np.ndarray(self.input_rgb_array.shape, dtype=self.input_rgb_array.dtype, buffer=shm_rgb_buffer.buf)
            # TODO: Understand if this is necessary (i.e., to use an additional NumPy array).
            # Can we do it directly using memoryview APIs? 
            np.copyto(shm_rgb_array, self.input_rgb_array) 

            shm_depth_array = np.ndarray(self.input_depth_array.shape, dtype=self.input_depth_array.dtype, buffer=shm_depth_buffer.buf)
            np.copyto(shm_depth_array, self.input_depth_array)

            # Run inference with contact-graspnet script to generate grasps.
            # NOTE: Why 20000 * 9 * 4 in size?
            # Because there are at most 20000 grasps, encoded as:
            #   unit quaternion (4d) + translations vector (3d) + gripper opening (1d) + grasp score (1d)
            # Each of this value is a float32, thus requiring 4 bytes.
            shm_grasp_buffer = shared_memory.SharedMemory('grasp', create=True, size=20000 * 9 * 4)    

            # Start Contact-GraspNet process with a system call.
            fx, fy, cx, cy = self.K[0, 0], self.K[1,1], self.K[0,2], self.K[1,2]
            contactgrasp_cmd = \
                f'CUDA_VISIBLE_DEVICES=1 python scripts/contact_graspnet_inference.py \
                    --use_shm --K "[{fx}, 0, {cx}, 0, {fy}, {cy}, 0, 0, 1]"'
            
            os.system(contactgrasp_cmd)

            # At this point, we can directly access the shared memory and copy its content (just for preventing inconsistencies, which should not happen).
            all_grasps_buffer_view = np.ndarray((20000, 9), dtype=np.float32, buffer=shm_grasp_buffer.buf)
            all_grasps_buffer = np.copy(all_grasps_buffer_view)
            
            self.generated_grasps = [] 
            self.tx = 0

            for idx in range(all_grasps_buffer.shape[0]):
                if not np.all(np.isnan(all_grasps_buffer[idx])) and not np.all(all_grasps_buffer[idx] == 0):
                    self.generated_grasps.append(all_grasps_buffer[idx])

            # We can now close shared memories.
            shm_rgb_buffer.close()
            shm_depth_buffer.close()
            shm_grasp_buffer.close()

            # shm_rgb_buffer.unlink()
            # shm_depth_buffer.unlink()
            # shm_grasp_buffer.unlink()

        print(f"Received {len(self.generated_grasps)} valid grasps.")

        # Note that 20000 is the max possible number of generated grasps, according to the sampling strategy adopted by Contact-GraspNet. 
        # However, practically speaking, this number is impossible to reach. Thus, we filter the invalid grasp poses.

        # print(f'[GRASPER] Predicted {len(pred_grasps_cam)} grasp poses.')
        print(f"[GRASPER] Running at {self.fps} fps.")
        return True

    def respond(self, command, reply):
        return super().respond(command, reply)

    def cleanup(self):
        self.input_rgb_port.close()
        self.input_depth_port.close()
        self.output_grasps_port.close()
        return True

    def interruptModule(self):
        self.input_rgb_port.interrupt()
        self.input_depth_port.interrupt()
        self.output_grasps_port.interrupt()
        return True
    
    def getPeriod(self):
        return 1/float(self.fps)

    def updateModule(self):
        # Fill the output grasp buffer
        # self.tx records the next grasp pose to be transmitted. 
        # NOTE: self.output_grasps_array is the np.array interface to access the yarp self.output_grasps_buffer buffer.
         
        if len(self.generated_grasps) - self.tx > 0:
            # Load the next burst
            # Check if we can load the entire buffer.
            if len(self.generated_grasps) - self.tx >= self.buffer_size:
                self.output_grasps_array[:, :] = np.vstack(self.generated_grasps[self.tx:self.tx+self.buffer_size])
                self.tx += self.buffer_size
                N = self.buffer_size
            else:
                self.output_grasps_array[:len(self.generated_grasps) - self.tx, :] = np.vstack(self.generated_grasps[self.tx:])
                self.tx += len(self.generated_grasps) - self.tx
                N = len(self.generated_grasps) - self.tx

            print(f"Writing {N} grasp poses to output port. Bufferized grasps: {len(self.generated_grasps) - self.tx}")
            self.output_grasps_port.write(self.output_grasps_buffer)

            # uquat, trans, gopening, gscore = self.all_grasps_buffer[idx][:4], self.all_grasps_buffer[idx][4:7], \
            #                                                 self.all_grasps_buffer[7], self.all_grasps_buffer[8] 
        
        return True

if __name__ == '__main__':
    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.setDefaultContext("GrasperWrapperRFModule")

    conffile = rf.find("from").asString()
    if not conffile:
        print("Using default conf file")
        rf.setDefaultConfigFile("conf.ini")
    else:
        rf.setDefaultConfigFile(rf.find("from").asString())

    rf.configure(sys.argv)

    # Run module
    manager = GrasperWrapperRFModule()
    try:
        manager.runModule(rf)
    finally:
        print('Closing GraspingRFModule due to an error...')
        manager.cleanup() 
