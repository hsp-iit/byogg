import torch
import torch.nn.functional as F
from torchvision import transforms
from mde.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

@torch.inference_mode()
def clamp(x, xmin, xmax):
    x[torch.isinf(x)] = xmin
    x[torch.isnan(x)] = xmin
    x[x < xmin] = xmin
    x[x > xmax] = xmax
    return x

# Resize depth prediction to the desired shape.
def resize(input, size=None, scale_factor=None, mode="nearest", align_corners=None, warning=False):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if (
                    (output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1)
                    and (output_h - 1) % (input_h - 1)
                    and (output_w - 1) % (input_w - 1)
                ):
                    warnings.warn(
                        f"When align_corners={align_corners}, "
                        "the output would more aligned if "
                        f"input size {(input_h, input_w)} is `x+1` and "
                        f"out size {(output_h, output_w)} is `nx+1`"
                    )
    return F.interpolate(input, size, scale_factor, mode, align_corners)

def make_depth_transform(backbone='dinov2', normalize=True) -> transforms.Compose:
    if backbone == 'depth-anything':
        return transforms.Compose([
            Resize(
                width=640,
                height=480,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=16,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(return_dict=False),
        ])
    else:
        if normalize:
            return transforms.Compose([
                transforms.ToTensor(),
                lambda x: 255.0 * x[:3], # Discard alpha component and scale by 255
                transforms.Normalize(
                    mean=(123.675, 116.28, 103.53),
                    std=(58.395, 57.12, 57.375),
                ),
            ])  
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                lambda x: 255.0 * x[:3], # Discard alpha component and scale by 255
            ])

def depth_clustering(depth, k=5):
    pass

def stacked_features_clustering(features, k=5):
    pass
    