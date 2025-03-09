import os
import shutil
import numpy as np
import pickle as pkl
import csv
import cv2

def remove_folder(folder_path):
    try:
        shutil.rmtree(folder_path)
        print(f"Successfully removed {folder_path} and all its contents.")
    except Exception as e:
        print(f"Failed to remove {folder_path}. Reason: {e}")

class IOHandler():
    def __init__(self):
        self.root_folder = None
        self.save_folder = None
        self.rgb_folder = None
        self.depth_folder = None
        self.emg_buffer = None
        self._working = False
        self.dumps = ['rgb', 'depth', 'cameras', 'invdepths', 'coords', 'grasps', 'emg-channels']

    def create_data_folder(self, root, *subfolders):
        '''
        Create data folder using provided config metadata.
        Created folder will have the following tree:
        <folder_name>
            + rgb 
            |   | rgb_0.png
            |   | rgb_1.png
            |   |   ...
            + depth
            |   | depth_0.npy
            |   | depth_1.npy
            |   |   ...
            + cameras
            |   | cameras_0.pkl
            |   | cameras_1.pkl
            |   | ...
            + coords
            |   | dumps.pkl
            + invdepths
            |   | invdepths_0.pkl
            |   | invdepths_1.pkl
            |   | ...
            + grasps
            |   | dumps.pkl
            |   | ...
            + emg-channels
            |   | dump0.pkl
            |   | timestamps0.pkl
            |   | dump1.pkl
            |   | timestamps1.pkl
            |   | ...
            ...
        '''
        self.root_folder = root
        self.save_folder = os.path.join(self.root_folder, *subfolders)
        self.rgb_folder = os.path.join(self.save_folder, 'rgb')
        self.depth_folder = os.path.join(self.save_folder, 'depth')
        
        self.rgb_count = 0
        self.depth_count = 0
        self.camera_count = 0
        self.inv_depth_count = 0
        self.candidate_count = 0
        
        if os.path.exists(self.save_folder):
            raise ValueError(f'ERROR: Folder {self.save_folder} already exists.')
        else:
            os.makedirs(self.save_folder)
        for dump in self.dumps:
            os.makedirs(os.path.join(self.save_folder, dump))

        # Create candidate recording txt file.
        self.candidate_file = open(os.path.join(self.save_folder, 'candidates.csv'), 'w', newline='')
        self.candidate_writer = csv.writer(self.candidate_file)
        self.candidate_writer.writerow(['step', 'candidateId', 'timestamp'])

        # Create file to save final Hannes joints configration.
        self.joints_file = open(os.path.join(self.save_folder, 'hannesJoints.txt'), 'w')
        self.joints_file.write('Final Hannes Joints Configuration - ')

        self.ts_file = open(os.path.join(self.save_folder, 'timestamps.csv'), 'w', newline='')
        self.ts_writer = csv.writer(self.ts_file)
        self.ts_writer.writerow(['type', 'path', 'timestamp'])

        self.emg_buffer = np.zeros((200, 6), dtype=np.float32)
        self.emg_timestamps = np.zeros((200, ), dtype=np.float64)
        self.emg_count = 0
        self.emg_dump_count = 0
        self._working = True

    def save_camera_poses(self, poses, timestamp=None):
        '''
        Store camera poses at the last optimization step.
        '''
        if self.save_folder is None:
            return
            
        camera_pkl_path = os.path.join(self.save_folder, 'cameras', f'cameras_{self.camera_count}.pkl')
        with open(camera_pkl_path, 'wb') as camera_pkl_file:
            pkl.dump(poses, camera_pkl_file)
        self.camera_count += 1

        if timestamp is not None:
            self.ts_writer.writerow(['cameras', camera_pkl_path, timestamp])
        
    def save_inv_depths(self, invdepths, timestamp=None):
        '''
        Store patch inv. depths at the last optimization step.
        '''
        if self.save_folder is None:
            return

        invdepths_pkl_path = os.path.join(self.save_folder, 'invdepths', f'invdepths_{self.inv_depth_count}.pkl')
        with open(invdepths_pkl_path, 'wb') as invdepths_pkl_file:
            pkl.dump(invdepths, invdepths_pkl_file)
        self.inv_depth_count += 1

        if timestamp is not None:
            self.ts_writer.writerow(['invdepths', invdepths_pkl_path, timestamp])

    def save_patch_coords(self, xcoords, ycoords, timestamp=None):
        '''
        Store patch sampling coords.
        '''
        if self.save_folder is None:
            return
        coords_pkl_path = os.path.join(self.save_folder, 'coords', 'coords.pkl')
        with open(coords_pkl_path, 'wb') as coords_pkl_file:
            pkl.dump(np.stack([xcoords, ycoords], axis=0), coords_pkl_file)

        if timestamp is not None:
            self.ts_writer.writerow(['coords', coords_pkl_path, timestamp])

    def save_gripper_poses(self, poses, timestamp=None):
        '''
        Store gripper poses.
        '''
        if self.save_folder is None:
            return
        gripper_pkl_path = os.path.join(self.save_folder, 'grasps', 'grasps.pkl')
        with open(gripper_pkl_path, 'wb') as gripper_pkl_file:
            pkl.dump(poses, gripper_pkl_file)

        if timestamp is not None:
            self.ts_writer.writerow(['grasps', gripper_pkl_path, timestamp])

    def save_emg_channels(self, channels, timestamp=None, lastBuffer=False):
        if self.save_folder is None:
            return

        if timestamp is None:
            timestamp = -1
        
        if not lastBuffer:
            print(channels)
            print(type(channels))
            print(self.emg_count)
            self.emg_buffer[self.emg_count] = channels
            self.emg_timestamps[self.emg_count] = timestamp
            self.emg_count += 1

        if self.emg_count >= self.emg_buffer.shape[0] or lastBuffer:
            emg_pkl_path = os.path.join(self.save_folder, 'emg-channels', f'dump{self.emg_dump_count}.pkl')
            emg_timestamps_pkl_path = os.path.join(self.save_folder, 'emg-channels', f'timestamps{self.emg_dump_count}.pkl')

            with open(emg_pkl_path, 'wb') as emg_pkl_file:
                pkl.dump(self.emg_buffer, emg_pkl_file)
            with open(emg_timestamps_pkl_path, 'wb') as emg_timestamps_pkl_file:
                pkl.dump(self.emg_timestamps, emg_timestamps_pkl_file)

            self.emg_count = 0
            self.emg_dump_count += 1
            self.emg_buffer = np.zeros((200, 6), dtype=np.float32)
            self.emg_timestamps = np.zeros((200, ), dtype=np.float64)

    def save_rgb(self, rgb, timestamp=None):
        '''
        Store RGB frames.
        '''
        if self.rgb_folder is None:
            return

        rgb_path = os.path.join(self.rgb_folder, f"{self.rgb_count}.png")
        cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        self.rgb_count += 1

        if timestamp is not None:
            self.ts_writer.writerow(['rgb', rgb_path, timestamp])
    
    def save_depth(self, depth, timestamp=None):
        '''
        Store estimated/reconstructed depth.
        '''
        if self.depth_folder is None:
            return

        depth_path = os.path.join(self.depth_folder, f"{self.depth_count}.npy")
        with open(depth_path, 'wb') as depth_file:
            np.save(depth_file, depth)
        self.depth_count += 1

        if timestamp is not None:
            self.ts_writer.writerow(['depth', depth_path, timestamp])

    def save_candidate_id(self, candidateId, timestamp=None):
        '''
        Store gripper candidate id.
        Candidate id corresponds to the gripper index in the gripper list returned by Contact-GraspNet.
        '''
        if self.candidate_file.closed:
            return

        if timestamp is None:
            timestamp = -1
        self.candidate_writer.writerow([self.candidate_count, candidateId, timestamp])
        self.candidate_count += 1

    def save_joints_ref(self, wps, wfe, fingers, timestamp=None):
        '''
        Store final Hannes joints configuration.
        '''
        if self.joints_file.closed:
            return

        if timestamp is None:
            self.joints_file.write(f"WPS = {wps}, WFE = {wfe}, Fingers = {fingers}\n")
        else:
            self.joints_file.write(f"[{timestamp}] WPS = {wps}, WFE = {wfe}, Fingers = {fingers}\n")

    def is_streaming(self):
        return self._working

    def close_stream(self):
        self.save_emg_channels(channels=np.array([]), lastBuffer=True)
        self.root_folder = None
        self.save_folder = None
        self.rgb_folder = None
        self.depth_folder = None

        self.rgb_count = 0
        self.depth_count = 0
        self.camera_count = 0
        self.inv_depth_count = 0
        self.candidate_count = 0
        self.emg_count = 0
        self.emg_dump_count = 0
        
        self.candidate_file.close()
        self.joints_file.close()
        self.ts_file.close()
        self._working = False

    def delete_stream(self):
        self.rgb_count = 0
        self.depth_count = 0
        self.camera_count = 0
        self.inv_depth_count = 0
        self.candidate_count = 0
        self.emg_count = 0
        self.emg_dump_count = 0

        self.candidate_file.close()
        self.joints_file.close()
        self.ts_file.close()

        try:
            shutil.rmtree(self.save_folder)
            print(f"[I/O] Successfully removed {self.save_folder} and all its contents.")
        except Exception as e:
            print(f"[I/O] Failed to remove {self.save_folder}. Reason: {e}")

        self.root_folder = None
        self.save_folder = None
        self.rgb_folder = None
        self.depth_folder = None
        self._working = False