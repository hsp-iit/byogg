## Run inference with Contact-GraspNet model

import os
import sys
import argparse
import numpy as np
import time
import glob
import cv2
import yaml
from multiprocessing import shared_memory
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("available GPU devices: ", physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(os.path.join(BASE_DIR))
print("current PYTHONPATH: ", os.environ['PYTHONPATH'])

# NOTE (IMPORTANT): Always import PyTorch or PyTorch-based modules after importing and setting TFv1 on GPUs.
# For some obsure reasons, TFv1 will break if you don't do so (with an error that is not easy to fully understand, 
# i.e., corrupted size vs. prev_size)
import torch
import roma

import grasping.contact_graspnet.utils.config_utils as config_utils
from grasping.contact_graspnet.utils.data_utils import load_available_input_data
from grasping.contact_graspnet.model import GraspEstimator
from grasping.contact_graspnet.utils.visualization_utils import visualize_grasps, show_image

# ci_build_and_not_headless = False
# try:
#     from cv2.version import ci_build, headless
#     ci_and_not_headless = ci_build and not headless
# except:
#     pass
# if sys.platform.startswith("linux") and ci_and_not_headless:
#     os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
# if sys.platform.startswith("linux") and ci_and_not_headless:
#     os.environ.pop("QT_QPA_FONTDIR")

def inference(global_config, checkpoint_dir, input_paths, 
              K=None, local_regions=True, skip_border_objects=False, filter_grasps=True, 
              segmap_id=None, z_range=[0.2,1.8], forward_passes=1, 
              top_k=None, win_name=None, use_shm=False):
    """
    Predict 6-DoF grasp distribution for given model and input data
    
    :param global_config: config.yaml from checkpoint directory
    :param checkpoint_dir: checkpoint directory
    :param input_paths: .png/.npz/.npy file paths that contain depth/pointcloud and optionally intrinsics/segmentation/rgb
    :param K: Camera Matrix with intrinsics to convert depth to point cloud
    :param local_regions: Crop 3D local regions around given segments. 
    :param skip_border_objects: When extracting local_regions, ignore segments at depth map boundary.
    :param filter_grasps: Filter and assign grasp contacts according to segmap.
    :param segmap_id: only return grasps from specified segmap_id.
    :param z_range: crop point cloud at a minimum/maximum z distance from camera to filter out outlier points. Default: [0.2, 1.8] m
    :param forward_passes: Number of forward passes to run on each point cloud. Default: 1
    """
    
    # Build the model
    grasp_estimator = GraspEstimator(global_config)
    grasp_estimator.build_network()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver(save_relative_paths=True)

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    # Load weights
    grasp_estimator.load_weights(sess, saver, checkpoint_dir, mode='test')
    
    os.makedirs('results', exist_ok=True)

    if use_shm:
        shm_rgb_buffer = shared_memory.SharedMemory('rgb', create=False)
        shm_depth_buffer = shared_memory.SharedMemory('depth', create=False)

        # TODO: Better to copy these two multi-dim array on top of memoryviews to new arrays, to ensure consistency.
        # However, there are no race-conditions. For the same reasons, we don't use any sync primitive. 
        # So, ensuring consistency is not strictly necessary.
        shm_rgb_array = np.ndarray(shape=(480, 640, 3), dtype=np.uint8, buffer=shm_rgb_buffer.buf)
        shm_depth_array = np.ndarray(shape=(480, 640), dtype=np.float32, buffer=shm_depth_buffer.buf)

        assert(K) # use_shm also requires to pass camera intrinsics as argument to the script, to avoid using an additional buffer.
        cam_K = eval(K)
        cam_K = np.array(cam_K).reshape(3,3)

        print('Generating Grasps...')
        # start_time = tf.timestamp()
        pc_full, pc_segments, pc_colors = grasp_estimator.extract_point_clouds(shm_depth_array, cam_K, segmap=None, rgb=shm_rgb_array,
                                                                            skip_border_objects=False, z_range=z_range)
        pred_grasps_cam, scores, contact_pts, gripper_openings = grasp_estimator.predict_scene_grasps(sess, pc_full, pc_segments=pc_segments, 
                                                                                        local_regions=local_regions, filter_grasps=filter_grasps, forward_passes=forward_passes)  
        
        # end_time = tf.timestamp()
        # print("Time for predicting grasps: ", (end_time-start_time))
    
        shm_grasp_buffer = shared_memory.SharedMemory('grasp', create=False)
        shm_grasp_array = np.ndarray(shape=(20000, 9), dtype=np.float32, buffer=shm_grasp_buffer.buf)
        
        encoded_grasps = np.full(shape=(20000, 9), fill_value=np.nan, dtype=np.float32)
        np.copyto(shm_grasp_array, encoded_grasps)

        # print(f"Scores: {scores[-1].shape[0]}")
        # print(f"Openings: {gripper_openings[-1].shape[0]}")

        print(f"Generated {pred_grasps_cam[-1].shape[0]} grasps")

        for idx in range(pred_grasps_cam[-1].shape[0]):
            gpose = pred_grasps_cam[-1][idx]
            rot, trans = torch.from_numpy(gpose[:3, :3]), gpose[:3, 3]
            uquat = roma.rotmat_to_unitquat(rot).numpy()
            gopening = gripper_openings[-1][idx]
            gscore = scores[-1][idx]

            encoded_grasps[idx] = np.hstack([uquat, trans, np.float32(gopening),  np.float32(gscore)])
            
            # print(encoded_grasps[idx].nbytes)
            # print(f"Quaternion for grasp {idx}: {encoded_grasps[idx]}")

        # print(f"Generated grasps: {encoded_grasps}")
        np.copyto(shm_grasp_array, encoded_grasps)

        #### DEBUGGING ####
        # for idx in range(shm_grasp_array.shape[0]):
        #     if np.all(np.isnan(shm_grasp_array[idx])):
        #         break
        #     print(f"Quaternion for grasp {idx}: {shm_grasp_array[idx]}")

        shm_rgb_buffer.close()
        shm_depth_buffer.close()
        shm_grasp_buffer.close()
    else:
        # Process example test scenes
        for p in glob.glob(input_paths):
            print('Loading ', p)

            pc_segments = {}
            segmap, rgb, depth, cam_K, pc_full, pc_colors = load_available_input_data(p, K=K)
            
            if segmap is None and (local_regions or filter_grasps):
                raise ValueError('Need segmentation map to extract local regions or filter grasps')

            if pc_full is None:
                print('Converting depth to point cloud(s)...')
                pc_full, pc_segments, pc_colors = grasp_estimator.extract_point_clouds(depth, cam_K, segmap=segmap, rgb=rgb,
                                                                                        skip_border_objects=skip_border_objects, z_range=z_range,
                                                                                        x_range=[-0.5, 0.5], y_range=[-0.5, 0.5])

            start_time = tf.timestamp()

            # print('##### SHAPES #####')
            # print(pc_full.shape)
            # print(pc_colors.shape)
            # print(pc_full.dtype)
            # print(pc_colors.dtype)
            # print(type(pc_full))
            # print(type(pc_colors))
            
            print('Generating Grasps...')
            pred_grasps_cam, scores, contact_pts, gripper_openings = grasp_estimator.predict_scene_grasps(sess, pc_full, pc_segments=pc_segments, 
                                                                                            local_regions=local_regions, filter_grasps=filter_grasps, forward_passes=forward_passes)  

            print(pred_grasps_cam)
            print(gripper_openings)

            end_time = tf.timestamp()

            print("Time for predicting grasps: ", (end_time-start_time))
            # Save results
            np.savez('results/predictions_{}'.format(os.path.basename(p.replace('png','npz').replace('npy','npz'))), 
                    pred_grasps_cam=pred_grasps_cam, scores=scores, contact_pts=contact_pts, gripper_openings=gripper_openings)

            # Visualize results          
            # show_image(rgb, segmap)
            # cv2.imwrite('./example.png', rgb)
            visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors, gripper_openings=gripper_openings, top_k=top_k, win_name=win_name)
            
        if not glob.glob(input_paths):
            print('No files found: ', input_paths)
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--np_path', default='test_data/7.npy', help='Input data: npz/npy file with keys either "depth" & camera matrix "K" or just point cloud "pc" in meters. Optionally, a 2D "segmap"')
    parser.add_argument('--png_path', default='', help='Input data: depth map png in meters')
    parser.add_argument('--top_k', default=-1, type=int, help='Visualize top-K grasp contacts by score. top_k == -1 visualizes all the grasp contacts.')
    parser.add_argument('--win_name', type=str, help='Name to give to the Mayavi window for visualization.')
    parser.add_argument('--K', default=None, help='Flat Camera Matrix, pass as "[fx, 0, cx, 0, fy, cy, 0, 0 ,1]"')
    parser.add_argument('--z_range', default=[0.0, 2.0], help='Z value threshold to crop the input point cloud')
    parser.add_argument('--local_regions', action='store_true', default=False, help='Crop 3D local regions around given segments.')
    parser.add_argument('--filter_grasps', action='store_true', default=False,  help='Filter grasp contacts according to segmap.')
    parser.add_argument('--skip_border_objects', action='store_true', default=False,  help='When extracting local_regions, ignore segments at depth map boundary.')
    parser.add_argument('--forward_passes', type=int, default=1,  help='Run multiple parallel forward passes to mesh_utils more potential contact points.')
    parser.add_argument('--segmap_id', type=int, default=0,  help='Only return grasps of the given object id')
    parser.add_argument('--arg_configs', nargs="*", type=str, default=[], help='overwrite config parameters')
    parser.add_argument('--use_shm', action='store_true', default=False, help='Read inputs (RGBD) from shared /tmp/[rgb,depth] memory. ')
    FLAGS = parser.parse_args() 

    # Load yaml config file
    with open("src/yarp-app/configs/default.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    # Build path for loading inference model
    FLAGS.ckpt_dir = os.path.join(cfg['grasping']['model_ckpts_path'], cfg['grasping']['model_name'])

    global_config = config_utils.load_config(FLAGS.ckpt_dir, batch_size=FLAGS.forward_passes, arg_configs=FLAGS.arg_configs)
    print('GLOBAL CONFIG: ', str(global_config))
    print('pid: %s'%(str(os.getpid())))

    inference(global_config, FLAGS.ckpt_dir, FLAGS.np_path if not FLAGS.png_path else FLAGS.png_path, z_range=eval(str(FLAGS.z_range)),
                K=FLAGS.K, local_regions=FLAGS.local_regions, filter_grasps=FLAGS.filter_grasps, segmap_id=FLAGS.segmap_id, 
                forward_passes=FLAGS.forward_passes, skip_border_objects=FLAGS.skip_border_objects,
                top_k=None if FLAGS.top_k < 1 else FLAGS.top_k, win_name=FLAGS.win_name, use_shm=FLAGS.use_shm)
