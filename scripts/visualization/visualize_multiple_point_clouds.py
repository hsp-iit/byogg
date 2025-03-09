import os
os.environ['PYTHONPATH'] = '/workspace/hannes-graspvo/src'
import torch
import numpy as np
import cv2
import glob
import open3d as o3d
import torch.nn.functional as F
from PIL import Image
import argparse
import yaml
import threading
import time

from mde.utils import build_dinov2_depther
from mde.io import load_model_setup, read_depth
from mde.ops import make_depth_transform
from odometry.dpvo.utils import Timer
from odometry.dpvo.dpvo import DPVO
from odometry.dpvo.config import cfg as dpvo_cfg
from odometry.dpvo.stream import image_stream, video_stream, realsense_stream, camera_stream
from odometry.dpvo.plot_utils import plot_trajectory, save_trajectory_tum_format
from cameras.realsense import init_rs_camera

from pcd import get_pcd_geometry_with_camera_poses, make_point_cloud
from visualizer import non_blocking_visualizer, pcdQueue

# NOTE: In case --odometry is given to this script, it runs a DPVO instance and optimizes for poses and sparse patch depths.
# For running, it uses the rgb frames in the same directory as rgb_path.

parser = argparse.ArgumentParser(prog='Visualize multiple point clouds interactively.')
parser.add_argument('--rgb_path', type=str, required=True)
args = parser.parse_args()

# Load yaml config file
with open("src/yarp-app/configs/default.yaml", "r") as f:
    cfg = yaml.safe_load(f)

def main():
    # Camera intrinsics matrix definition
    intr_matr = np.zeros((3,3))
    intr_matr[0, 0] = 614.13299560546875
    intr_matr[1, 1] = 614.47918701171875
    intr_matr[0, 2] = 318.909332275390625
    intr_matr[1, 2] = 249.5451202392578125

    # Camera intrinsics in open3d format for interactive visualizer
    K = o3d.camera.PinholeCameraIntrinsic(width=640, 
                                            height=480, 
                                            fx=intr_matr[0, 0], 
                                            fy=intr_matr[1, 1], 
                                            cx=intr_matr[0, 2], 
                                            cy=intr_matr[1, 2])

    # Load model config from model name (model configs are indexed from model name)
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
    model.cuda()
        
    # Load RGB Input and apply transform to it
    transform = make_depth_transform(
        backbone=cfg['mde']['backbone']
    )

    images = glob.glob(os.path.join(args.rgb_path, '*.jpg'))
    images.sort()
    
    # Start visualization thread
    threading.Thread(target=non_blocking_visualizer, daemon=True).start()

    for image in images:
        rgb = transform(Image.open(image))
        
        # Run inference on RGB input
        with torch.inference_mode():
            rgb = rgb.unsqueeze(0)
            depth = model.whole_inference(rgb.cuda(), img_meta=None, rescale=False)
            depth = depth.squeeze().cpu().numpy()
        
        depth = (depth * 1000).astype(np.uint16)
        camera_bgr = cv2.imread(image)
        pcd, _ = make_point_cloud(camera_bgr, depth, K, convert_rgb_to_intensity=False)
        pcdQueue.put(pcd)

    pcdQueue.join()

if __name__ == '__main__':
    main()


