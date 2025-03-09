import yaml
import torch
import os
import multiprocessing
import threading
import numpy as np
import cv2
import open3d as o3d

from multiprocessing import Process, Queue
from odometry.dpvo.dpvo import DPVO
from odometry.dpvo.stream import image_stream, transformed_image_stream
from odometry.dpvo.config import cfg as _cfg

from mde.io import load_model_setup
from mde.utils import build_dinov2_depther
from visualizer import non_blocking_visualizer

## DPVO Process ##
@torch.no_grad()
def dpvo(cfg, network, rgbQueue, poseQueue, cudaDevice):
    ## DPVO inference loop
    # Takes rgb as inputs from the input queue (rgbQueue) and 
    # outputs output poses on the output queue (poseQueue)
    # print("Optimizing poses and sparse patch depths...")
    slam = None
    while 1:
        (t, image, intrinsics) = rgbQueue.get()
        if t < 0:
            poseQueue.put((None, None)) 
            break

        image = torch.from_numpy(image).permute(2,0,1).to(cudaDevice)
        intrinsics = torch.from_numpy(intrinsics).to(cudaDevice)

        if slam is None:
            slam = DPVO(cfg, network, device=cudaDevice, ht=image.shape[1], wd=image.shape[2], viz=False)

        image = image.to(cudaDevice)
        intrinsics = intrinsics.to(cudaDevice)
        # print("####################### POSES #############################")
        res = slam(t, image, intrinsics)
        if res is not None:
            (poses, tstamps) = res
            poseQueue.put((tstamps, poses))

    for _ in range(12):
        slam.update()
    
## DINOv2 Process ##
@torch.no_grad()
def dense_depth_estimation(cfg, tensorQueue, pcdQueue, depthQueue, cudaDevice):
    # Camera intrinsics matrix definition
    intr_matr = np.zeros((3,3))
    intr_matr[0, 0] = 614.13299560546875
    intr_matr[1, 1] = 614.47918701171875
    intr_matr[0, 2] = 318.909332275390625
    intr_matr[1, 2] = 249.5451202392578125

    model_cfg = load_model_setup(cfg) 
    model = build_dinov2_depther(cfg,
                            backbone_size='small', 
                            head_type=model_cfg['head'], 
                            min_depth=0, 
                            max_depth=2,
                            inference=True,
                            checkpoint_name=cfg['mde']['model_name'],
                            use_depth_anything_encoder=True if cfg['mde']['backbone'] == 'depth-anything' else False)
    model.eval()
    model.to(cudaDevice)

    while 1:
        # print(tensorQueue)
        (t, rgb, rgb_path) = tensorQueue.get()

        if t < 0:
            pcdQueue.put((t, None, None, None))
            break

        rgb = torch.from_numpy(rgb).unsqueeze(0).to(cudaDevice)
        depth = model.whole_inference(rgb, img_meta=None, rescale=False)
        depth = depth.squeeze().cpu().numpy()

        camera_bgr = cv2.imread(rgb_path)
        depthQueue.put((t, np.array(camera_bgr), depth))
        depth = (depth * 1000).astype(np.uint16)
        pcdQueue.put((t, camera_bgr, depth, intr_matr))


def grasp_pose_generation(cfg, depthQueue, graspQueue, cudaDeviceIdx):
    # os.environ["CUDA_VISIBLE_DEVICES"]="1"
    
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()
    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # print("available GPU devices: ", physical_devices)

    # tf.config.experimental.set_visible_devices(physical_devices[cudaDeviceIdx], 'GPU')
    # tf.config.experimental.set_memory_growth(physical_devices[cudaDeviceIdx], True)

    import grasping.contact_graspnet.utils.config_utils as config_utils
    from grasping.contact_graspnet.model import GraspEstimator
    from pcd import build_pcd_from_rgbd

    # physical_devices[cudaDeviceIdx] is now mapped to logical_devices[0]
    # logical_devices = tf.config.experimental.list_logical_devices('GPU')
    # print(f"LOGICAL DEVICES: {logical_devices}")

    # with tf.device('/device:GPU:0'):
    # Camera intrinsics matrix definition
    K = np.zeros((3,3))
    K[0, 0] = 614.13299560546875
    K[1, 1] = 614.47918701171875
    K[0, 2] = 318.909332275390625
    K[1, 2] = 249.5451202392578125

    checkpoint_dir = os.path.join(cfg['grasping']['model_ckpts_path'], cfg['grasping']['model_name'])

    print(f"checkpoint_dir: {checkpoint_dir}")

    concact_grasp_cfg = cfg['grasping']['contact_grasp_cfg']
    contact_grasp_global_cfg = config_utils.load_config(checkpoint_dir, 
                                                        batch_size=concact_grasp_cfg['forward_passes'], 
                                                        arg_configs=concact_grasp_cfg['arg_configs'])

    # with tf.device('/device:GPU:0'):
        # Initialize grasp estimator
    
    grasp_estimator = GraspEstimator(contact_grasp_global_cfg)
    grasp_estimator.build_network()

    saver = tf.train.Saver(save_relative_paths=True)
    # Create a TF session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = "1"
    config.allow_soft_placement = True
    config.log_device_placement = True

    sess = tf.Session(config=config)

    # Load weights
    grasp_estimator.load_weights(sess, saver, checkpoint_dir, mode='test')

    while 1:
        # Get PCD from depthQueue
        (t, rgb, depth) = depthQueue.get()
        print("#### Getting depth from queue ####")
        if t < 0:
            break

        print(type(depth))
        print(type(rgb))

        pc_full, pc_colors = build_pcd_from_rgbd(rgb, depth, K)
        
        print(pc_full.shape)
        print(type(pc_full))
        # print(pc_colors.shape)

        pred_grasps_cam, scores, contact_pts, gripper_openings = \
            grasp_estimator.predict_scene_grasps(sess, pc_full, 
                                                pc_segments={}, 
                                                local_regions=concact_grasp_cfg['local_regions'], 
                                                filter_grasps=concact_grasp_cfg['filter_grasps'], 
                                                forward_passes=concact_grasp_cfg['forward_passes'])  
        print("########### Predicted Grasps ##############")
        print(t, len(pred_grasps_cam))

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    # Check if CUDA is available. This script only runs on 2 GPUs.
    # print(f"CUDA is available: {torch.cuda.is_available()}")
    # print("Available CUDA devices:")
    # for i in range(torch.cuda.device_count()):
    #     print(torch.cuda.get_device_properties(i).name)

    available_gpus = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    # Load yaml config file
    with open("src/yarp-app/configs/default.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # Build yacs config node from yaml cfg
    for key, value in cfg['odometry']['dpvo_config'].items():
        if key in _cfg:
            _cfg.merge_from_list([key, value])

    imagedir = '/docker_volume/example_images/rgb'

    calib = cfg['global']['camera_intrinsics']
    stride = cfg['odometry']['dpvo_config']['STRIDE']
    skip = cfg['odometry']['dpvo_config']['SKIP_FRAMES']
    network = os.path.join(cfg['odometry']['model_ckpts_path'], f"{cfg['odometry']['model_name']}.pth")

    rgbQueue = Queue(maxsize=8)
    tensorQueue = Queue(maxsize=8)
    poseQueue = Queue(maxsize=8)
    pcdQueue = Queue(maxsize=8)
    depthQueue = Queue(maxsize=8)
    graspQueue = Queue(maxsize=8)

    dpvoCudaDevice = available_gpus[0]
    mdeCudaDevice = available_gpus[1]
    graspCudaDeviceIdx = 1

    backbone=cfg['mde']['backbone']

    readerProcess = Process(target=image_stream, args=(rgbQueue, imagedir, calib, stride, skip))
    transReaderProcess = Process(target=transformed_image_stream, args=(tensorQueue, imagedir, backbone, stride, skip))
    dpvoProcess = Process(target=dpvo, args=(_cfg, network, rgbQueue, poseQueue, dpvoCudaDevice))
    mdeProcess = Process(target=dense_depth_estimation, args=(cfg, tensorQueue, pcdQueue, depthQueue, mdeCudaDevice))
    vizProcess = Process(target=non_blocking_visualizer, args=(pcdQueue, poseQueue))
    graspProcess = Process(target=grasp_pose_generation, args=(cfg, depthQueue, graspQueue, graspCudaDeviceIdx))

    graspProcess.start()
    mdeProcess.start()
    vizProcess.start()
    readerProcess.start()
    transReaderProcess.start()
    dpvoProcess.start()

    mdeProcess.join()
    dpvoProcess.join()
    graspProcess.join()
    vizProcess.join()
    readerProcess.join()
    transReaderProcess.join()

