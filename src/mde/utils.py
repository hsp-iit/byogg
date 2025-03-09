import os
import sys
import torch
import torch.nn.functional as F
from functools import partial
import mmcv
from mmcv.runner import load_checkpoint
from mde.dinov2.eval.depth.models import build_depther
from mde.io import load_config_from_url, load_model_setup, write_model_setup

### utility methods to load/build a mde model based on DINOv2 ###

def build_dinov2_backbone(encoder, patch_size=16, img_h=480, img_w=640, use_depth_anything_encoder=True):
    
    if use_depth_anything_encoder:
        # NOTE: This only supports vits as backbone until now in my code
        print("[INFO] using depth-anything checkpoint for DINOv2 encoder")
        depth_anything_model = build_depth_anything(encoder, local_ckpt=True)
        # Extract the pre-trained encoder from the checkpoint 
        backbone_model = depth_anything_model.pretrained
    else:
        print("[INFO] using facebookresearch/dinov2 checkpoint for DINOv2 encoder")
        backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=f"dinov2_{encoder}14")

    # interpolate patch embedding filter from 14x14 to 16x16
    backbone_model.patch_embed.proj.weight = torch.nn.Parameter(
        F.interpolate(
            backbone_model.patch_embed.proj.weight, 
            (patch_size, patch_size), 
            mode="bicubic", align_corners=False
        )
    )

    # adjust parameters
    backbone_model.patch_embed.img_size = img_h, img_w
    backbone_model.patch_embed.patch_size = patch_size, patch_size
    backbone_model.patch_embed.patches_resolution = (
        img_h // patch_size, img_w // patch_size
    )
    backbone_model.patch_embed.num_patches = \
        img_h // patch_size * img_w // patch_size
    backbone_model.patch_embed.proj.kernel_size = backbone_model.patch_embed.patch_size
    backbone_model.patch_embed.proj.stride = backbone_model.patch_embed.patch_size
    backbone_model.patch_size = patch_size
    backbone_model.cuda()
    return backbone_model

def build_depth_anything(encoder, patch_size=16, img_h=480, img_w=640, local_ckpt=False):
    if local_ckpt:
        from mde.depth_anything.dpt import DPT_DINOv2
        #NOTE: these hyper-parameters only work for loading a vits-based checkpoint
        depth_anything = DPT_DINOv2(encoder, features=64, out_channels=[48, 96, 192, 384])
        ckpt = torch.load('models/checkpoints/depth_anything_vits14.pth')   # Directly a state_dict without nesting
        depth_anything.load_state_dict(ckpt)
    else:  
        from mde.depth_anything.dpt import DepthAnything
        depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{:}14'.format(encoder))  

    # interpolate patch embedding filter from 14x14 to 16x16
    depth_anything.pretrained.patch_embed.proj.weight = torch.nn.Parameter(
        F.interpolate(
            depth_anything.pretrained.patch_embed.proj.weight, 
            (patch_size, patch_size), 
            mode="bicubic", align_corners=False
        )
    )

    # depth_anything.pretrained is a DinoVisionTransformer
    # adjust parameters
    depth_anything.pretrained.patch_embed.img_size = img_h, img_w
    depth_anything.pretrained.patch_embed.patch_size = patch_size, patch_size
    depth_anything.pretrained.patch_embed.patches_resolution = (
        img_h // patch_size, img_w // patch_size
    )
    depth_anything.pretrained.patch_embed.num_patches = \
        img_h // patch_size * img_w // patch_size
    
    depth_anything.pretrained.patch_embed.proj.kernel_size = patch_size, patch_size
    depth_anything.pretrained.patch_embed.proj.stride = patch_size, patch_size
    depth_anything.pretrained.patch_size = patch_size

    print(depth_anything)

    depth_anything.cuda()

    return depth_anything

def build_dinov2_depther(cfg,
                         backbone_size, 
                         head_type, 
                         min_depth, 
                         max_depth, 
                         inference=True, 
                         loss_weights=None, 
                         checkpoint_name='[pretrained]', 
                         bins_strategy=None,
                         return_vanilla_head=False,
                         use_depth_anything_encoder=True):

    ''' Builds a Depth Estimation model with a DINOv2 backbone and its losses.'''
    backbone_archs = {
        "small": "vits",
        "base": "vitb",
        "large": "vitl",
        "giant": "vitg",
    }

    # bins_strategy should be set to None if head_type is DPT
    assert not (head_type == 'dpt' and bins_strategy)

    encoder = backbone_archs[backbone_size]
    backbone_model = build_dinov2_backbone(encoder, use_depth_anything_encoder=use_depth_anything_encoder)

    HEAD_DATASET = "nyu" # in ("nyu", "kitti")
    HEAD_TYPE = head_type # in ("linear", "linear4", "dpt")
    backbone_name = f"dinov2_{encoder}14"
    DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
    head_config_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_config.py"
    head_checkpoint_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_head.pth"
    
    head_cfg_str = load_config_from_url(head_config_url)
    head_cfg = mmcv.Config.fromstring(head_cfg_str, file_format=".py")    
    head_cfg.model.decode_head.min_depth = min_depth
    head_cfg.model.decode_head.max_depth = max_depth

    # NOTE: By default the model is loaded for inference
    if not inference:

        default_weights = {
            'grad_weight': 1.0,
            'sig_weight': 0.5,
            'scale_inv_weight': 0.85,  
            'mask_weight': 0.0
        }

        if not loss_weights:
            loss_weights = default_weights

        # modify model configuration for fine-tuning the decode_head
        head_cfg.model.test_cfg = {}
        head_cfg.model.train_cfg = dict(mode='whole')

        for idx, loss in enumerate(head_cfg.model.decode_head.loss_decode):
            if loss['type'] == 'GradientLoss':
                loss['loss_weight'] = loss_weights['grad_weight'] 
            elif loss['type'] == 'SigLoss':
                loss['loss_weight'] = loss_weights['sig_weight']
                loss['scale_inv_weight'] = loss_weights['scale_inv_weight']

            loss['max_depth'] = max_depth
            head_cfg.model.decode_head.loss_decode[idx] = loss

        mask_weight = float(loss_weights['mask_weight'])
        if mask_weight > 0:
            # If mask_weight > 0, add new loss term to training loss
            head_cfg.model.decode_head.loss_decode.append(
                dict(
                    type='MaskedSigLoss',
                    valid_mask=True,
                    loss_weight=mask_weight,
                    warm_up=True,
                    loss_name='loss_masked_depth',
                    max_depth=max_depth
                )
            )

    train_cfg = head_cfg.get("train_cfg")
    test_cfg = head_cfg.get("test_cfg")    

    if head_type in ['linear', 'linear4'] and not bins_strategy:
        # TODO: Implement AdaBins strategy
        # NOTE: When using linear or linear4 head, bins_strategy is set by default.
        print("Selecting NO bins for the bin strategy")
        head_cfg.model.decode_head.classify = False

    # print(train_cfg)
    # cfg.model.decode_head['scale_up'] = True

    model = build_depther(head_cfg.model, train_cfg=train_cfg, test_cfg=test_cfg)
    
    # This line here will make the saved state_dict to contain also encoder weights
    # Thus, should be executed only if interested in modifying the encoder weights.
    model.backbone = backbone_model

    model.backbone.forward = partial(
        backbone_model.get_intermediate_layers,
        n=head_cfg.model.backbone.out_indices,
        reshape=True,
        return_class_token=head_cfg.model.backbone.output_cls_token,
        norm=head_cfg.model.backbone.final_norm,
    )

    if checkpoint_name == '[pretrained]':
        if return_vanilla_head:
            # Get vanilla decoder before loading the given checkpoint
            decoder = model.get_submodule('decode_head')
            vanilla_decoder_sd = copy.deepcopy(decoder.state_dict())

        # Load checkpoint for the head 
        load_checkpoint(model, head_checkpoint_url, map_location="cpu", strict=False)
        
        if return_vanilla_head:
            decoder.load_state_dict(vanilla_decoder_sd)
            
    else:
        checkpoint = torch.load(os.path.join(cfg['mde']['model_ckpts_path'], 'best_{}.pt'.format(checkpoint_name)))
        model.load_state_dict(checkpoint['model.state_dict'], strict=False)

    print(f"[INFO] Loaded MDE-DINOv2 {checkpoint_name} checkpoint.")
    return model
