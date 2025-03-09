import os
import cv2
import numpy as np
from PIL import Image
import torch
import yaml
import argparse

import sys
sys.path.append('/workspace/hannes-graspvo')
sys.path.append('/workspace/hannes-graspvo/src')
print(sys.path)

from mde.utils import build_dinov2_depther
from mde.io import load_model_setup, read_depth, render_depth
from mde.ops import make_depth_transform, resize, clamp

''' from /workspace/hannes-graspvo
python scripts/dumping/dump_point_cloud.py  --rgb_path /docker_volume/example_images/rgb/00000024.jpg \
                                            --save_id mustard24 \
                                            --format contact-graspnet
'''

parser = argparse.ArgumentParser(prog='Script to dump PCDs built with reconstructed depth.')
parser.add_argument('--rgb_path', type=str, required=True)
parser.add_argument('--seg_path', type=str, required=False)
parser.add_argument('--save_id', type=str, required=True)
parser.add_argument('--format', type=str, choices=['uois', 'contact-graspnet'], required=True)
args = parser.parse_args()

def depth2pc(depth, K, rgb):
    """
    Convert depth and intrinsics to point cloud and optionally point cloud color
    :param depth: hxw depth map in m
    :param K: 3x3 Camera Matrix with intrinsics
    :returns: (Nx3 point cloud, point cloud color)
    """

    mask = np.where(depth > 0)
    x,y = mask[1], mask[0]
    
    normalized_x = (x.astype(np.float32) - K[0,2])
    normalized_y = (y.astype(np.float32) - K[1,2])

    world_x = normalized_x * depth[y, x] / K[0,0]
    world_y = normalized_y * depth[y, x] / K[1,1]
    world_z = depth[y, x]

    rgb = rgb[y,x,:]
        
    pc = np.vstack((world_x, world_y, world_z)).T
    return (pc, rgb)

def get_gt_valid_mask(gt, min_depth, max_depth):
    # Substitute invalid values in depth map
    gt[torch.isinf(gt)] = 0
    gt[torch.isnan(gt)] = 0
    # Compute valid mask for gt depth
    valid_mask = torch.logical_and(gt > min_depth, 
                                    gt < max_depth)
    return valid_mask

def eval_step(model, model_setup, backbone, rgb_path):
    transform = make_depth_transform(backbone)

    rgb = Image.open(rgb_path)
    image = transform(rgb)

    with torch.inference_mode():
        # Add the batch dimension to the input image
        image = image.unsqueeze(0)
        pred = model.whole_inference(image.cuda(), img_meta=None, rescale=False)
        pred = pred.squeeze().cpu() # (B, H, W)

        # If head type is linear4, increase the resolution of the prediction before computing errors.
        if model_setup['head'] == 'linear4': 
            pred = resize(
                input=pred.unsqueeze(1), size=(480, 640), mode='bilinear', align_corners=False, warning=False
            )
            pred = pred.squeeze()

    return pred


if __name__  == "__main__":
    min_depth, max_depth = 1e-3, 2
    
    # Load yaml config file
    with open("src/yarp-app/configs/default.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    print(cfg['mde'])
    print(type(cfg['mde']))

    model_name  = cfg['mde']['model_name']
    backbone = cfg['mde']['backbone']

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

    # Camera Intrinsics Matrix
    # TODO: Build this intr_matrix from a camera config file.
    intr_matr = np.zeros((3, 3))
    # Correct RealSense values
    intr_matr[0, 0] = 614.13299560546875
    intr_matr[1, 1] = 614.47918701171875
    intr_matr[0, 2] = 318.909332275390625
    intr_matr[1, 2] = 249.5451202392578125

    print("#######")
    print("fx: ", intr_matr[0, 0])
    print("fy: ", intr_matr[1, 1])
    print("cx: ", intr_matr[0, 2])
    print("cy: ", intr_matr[1, 2])
    fovx, fovy, _, _, _ = cv2.calibrationMatrixValues(intr_matr, (640, 480), 0, 0)
    print("fovx: ", fovx)
    print("fovy: ", fovy)
    print("#######")

    rgb_path = args.rgb_path
    pred_depth = eval_step(model=model,
                    model_setup=model_cfg,
                    backbone=backbone,
                    rgb_path=rgb_path)
    
    # rs_depth = rgb_path.replace("rgb", "depth").replace(".jpg", ".float")
    # rs_depth = torch.tensor(read_depth(rs_depth))

    camera_bgr = cv2.imread(rgb_path)
    pred_depth = clamp(pred_depth, xmin=min_depth, xmax=max_depth)
    # rs_depth = clamp(rs_depth, xmin=min_depth, xmax=max_depth)

    # rs_pcd_save_path = os.path.join(cfg['global']['dump_rs_pcd_path'], f'realsense_{args.save_id}.npy')
    recon_pcd_save_path = os.path.join(cfg['global']['dump_pred_pcd_path'], f'pred_{args.save_id}.npy')
    
    if args.format == 'contact-graspnet':
        # Dumping pcd data in the format required by Contact-GraspNet
        if args.seg_path:
            segmap = cv2.imread(args.seg_path, cv2.IMREAD_GRAYSCALE)
            # Assign segmap_id = 1 to every non-zero pixel.
            # TODO: Test if dilation works better by "smoothing" the borders of a segmentation mask.
            segmap[segmap>0] = 1
            pred_pcd = dict(rgb=np.array(camera_bgr),
                            segmap=np.array(segmap), 
                            depth=np.array(pred_depth), 
                            K=intr_matr)
        else:
            pred_pcd = dict(rgb=np.array(camera_bgr), 
                            depth=np.array(pred_depth), 
                            K=intr_matr)
        # Dump and render predicted depth
        depth = render_depth(pred_depth)
        depth.save(f'./results/depth/{args.save_id}.png')
        # rs_pcd = dict(rgb=np.array(camera_bgr), depth=np.array(rs_depth), K=intr_matr)
    else:
        pred_pcd, _ = depth2pc(np.array(pred_depth), intr_matr, camera_bgr[..., ::-1])
        # rs_pcd, _ = depth2pc(np.array(rs_depth), intr_matr, camera_bgr[..., ::-1])

        pred_pcd = pred_pcd.reshape(camera_bgr.shape[0], camera_bgr.shape[1], 3) 
        # rs_pcd = rs_pcd.reshape(camera_bgr.shape[0], camera_bgr.shape[1], 3)

        pred_pcd = dict(rgb=camera_bgr[..., ::-1], xyz=pred_pcd)
        # rs_pcd = dict(rgb=camera_bgr[..., ::-1], xyz=rs_pcd)

    # np.save(rs_pcd_save_path, rs_pcd)
    # print(f"PCD succesfully saved at {rs_pcd_save_path}")
    np.save(recon_pcd_save_path, pred_pcd)
    print(f"PCD succesfully saved at {recon_pcd_save_path}")



