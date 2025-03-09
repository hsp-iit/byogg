import torch
import numpy as np
import cv2
import open3d as o3d
import argparse
import yaml
from PIL import Image

from mde.utils import build_dinov2_depther
from mde.io import load_model_setup
from mde.ops import make_depth_transform

from pcd import get_pcd_geometry_with_camera_frame

parser = argparse.ArgumentParser(prog='Visualize point cloud generated by depth estimation model.')
parser.add_argument('--rgb_path', type=str, required=True)

args = parser.parse_args()

# Load yaml config file
with open("src/yarp-app/configs/default.yaml", "r") as f:
    cfg = yaml.load(f, Loader=yaml.Loader)

def main():
    # Camera intrinsics matrix definition
    intr_matr = np.zeros((3,3))
    intr_matr[0, 0] = 614.13299560546875
    intr_matr[1, 1] = 614.47918701171875
    intr_matr[0, 2] = 318.909332275390625
    intr_matr[1, 2] = 249.5451202392578125

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
    rgb = transform(Image.open(args.rgb_path))
    
    # Run inference on RGB input
    with torch.inference_mode():
        rgb = rgb.unsqueeze(0)
        pred = model.whole_inference(rgb.cuda(), img_meta=None, rescale=False)
        pred = pred.squeeze().cpu() 

    pred_depth = (pred.numpy() * 1000).astype(np.uint16)

    # Retrieve camera bgr
    camera_bgr = cv2.imread(args.rgb_path)

    geometries = get_pcd_geometry_with_camera_frame(camera_bgr, pred_depth, intr_matr)
    o3d.visualization.draw_geometries(geometries)


if __name__ == '__main__':
    main()