import matplotlib
import matplotlib.pyplot as plt
import urllib
import struct
import numpy as np
from PIL import Image
import cv2
import json
import yaml


def load_model_setup(cfg):
    model_name = cfg['mde']['model_name']
    if model_name == '[pretrained]':
        # If 'pretrained' is provided by the user, assume that head = dpt.
        model_setup = {'head': 'dpt', 'bins_strategy': None}
        return model_setup

    with open(cfg['mde']['model_config_path'], 'r') as model_cfg:
        model_cfg = json.load(model_cfg)

        if model_name in model_cfg:
            model_setup = model_cfg[model_name]
            if "bins_strategy" in model_setup:
                model_setup["bins_strategy"] = True if model_setup['bins_strategy'] == 'Y' else False
            else:
                model_setup["bins_strategy"] = None
        else:
            print(f"No {model_name} model configuration found in {cfg['mde']['model_config_path']}.")
            sys.exit(-1)

    return model_setup 

def write_model_setup(cfg, model_name, head, bins_strategy=None):
    # Load model configs json
    with open(cfg['mde']['model_config_path'], 'r') as cfg_file:
        model_cfg = json.load(cfg_file)

        if model_name in model_cfg:
            print(f'Model with name {model_name} is already present in configs.json file. Model saved as {model_name}_dup.')
            model_name = model_name + '_dup'

    # Write new model setup to configs json
    model_cfg[model_name] = {'head': head}
    if bins_strategy:
        model_cfg[model_name]['bins_strategy'] =  bins_strategy

    with open(cfg['mde']['model_config_path'], 'w') as cfg_file:
        json.dump(model_cfg, cfg_file, indent=4)

def read_depth(filename):
    with open(filename, 'rb') as f:
        width = f.read(8)
        width = int.from_bytes(width, "little")
        assert width == 640
        height = f.read(8)
        height = int.from_bytes(height, "little")
        assert height == 480

        depth_img = []
        while (True):
            depthval_b = f.read(4)      # binary, little endian
            if not depthval_b:
                break
            depthval_m = struct.unpack("<f", depthval_b)    # depth val as meters
            depth_img.append(depthval_m)
        assert len(depth_img) == height * width

    depth_img = np.array(depth_img).reshape(height, width)

    return depth_img

def render_depth(values, colormap_name="magma_r", normalize=True) -> Image:
    if normalize:
        min_value, max_value = values.min(), values.max()
        normalized_values = (values - min_value) / (max_value - min_value)
    else:
        normalized_values = values

    colormap = matplotlib.colormaps[colormap_name]
    colors = colormap(normalized_values, bytes=True) # ((1)xhxwx4)
    colors = colors[:, :, :3] # Discard alpha component
    return Image.fromarray(colors)

def render_error(input, target, colormap_name='Reds'):
    '''
        input and target are expected to be already min-max normalized
    '''
    err = np.abs(input-target)/ (target + 1e-5)
    err = (err - err.min()) / (err.max() - err.min())
    colormap = matplotlib.colormaps[colormap_name]
    colors = colormap(err, bytes=True)
    colors = colors[:, :, :3]
    return Image.fromarray(colors)

def read_exr_map(exr_file_path, single_channel=True):
    import OpenEXR
    # Read the EXR file using OpenEXR library
    exr_file = OpenEXR.InputFile(exr_file_path)
    # Get the image dimensions
    dw = exr_file.header()['dataWindow']

    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # Read all channels from the EXR file
    r, g, b = [np.frombuffer(exr_file.channel(c), dtype=np.float32).reshape(height, width) for c in 'RGB']

    # Combine the channels into a single depth map
    depth_map = cv2.merge([r, g, b]) * exr_file.header()['pixelAspectRatio']
    if single_channel:
        # Extract the red channel as a single-channel depth map
        depth_map = depth_map[:, :, 0]  # Assuming red is the first channel (channel index 0)
    # # You can now work with the depth map using OpenCV functions
    # # For example, display the depth map
    # cv2.imshow('Depth Map', depth_map)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return depth_map

def render_with_mask_normalization(src, mask):
    import numpy.ma as ma
    masked_src = ma.masked_array(src, mask=mask)
    masked_src_min, masked_src_max = masked_src.min(), masked_src.max()
    src = (src - masked_src_min) / (masked_src_max - masked_src_min)
    src = render_depth(src, colormap_name='magma_r', normalize=False)
    return cv2.bitwise_and(np.array(src)[..., ::-1], cv2.bitwise_not(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)))

def cvSubplot(imgs,         # 2d np array of imgs (each img an np arrays of depth 1 or 3).
              pad=10,       # number of pixels to use for padding between images. must be even
              titles=None   # (optional) np array of subplot titles
              ):
    '''
    Makes cv2 based subplots. Useful to plot image in actual pixel size
    '''

    rows, cols = imgs.shape[0:2]
    subplot_shapes = np.array([list(map(np.shape, x)) for x in imgs])
    sp_height, sp_width, depth = np.max(np.max(subplot_shapes, axis=0), axis=0)

    title_pad = 30
    if titles is not None:
        pad_top = pad + title_pad
    else:
        pad_top = pad

    pad_left = pad_top * 4

    frame = np.zeros((rows*(sp_height+pad_top), pad_left + cols*(sp_width+pad), depth ))

    row_legend = ['Mask Pred.', 'Pred.', 'RGB, Err.']
    frame_h = frame.shape[0]
    legend_positions = np.int32(np.linspace((frame_h/rows)/2, frame_h - (frame_h/rows)/2, rows))

    for r in range(rows):
        frame = cv2.putText(frame, row_legend[r], (pad_left//16, legend_positions[r]), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255))

        for c in range(cols):
            img = imgs[r, c]
            h, w, _ = img.shape
            y0 = r * (sp_height+pad_top) + pad_top//2
            x0 = pad_left + c * (sp_width+pad) + pad//2
            frame[y0:y0+h, x0:x0+w, :] = img

            if titles is not None:
                frame = cv2.putText(frame, titles[r, c], (x0, y0-title_pad//4), cv2.FONT_HERSHEY_COMPLEX, .75, (255,255,255))

    return frame

def load_config_from_url(url: str) -> str:
    with urllib.request.urlopen(url) as f:
        return f.read().decode()

def load_image_from_url(url: str) -> Image:
    with urllib.request.urlopen(url) as f:
        return Image.open(f).convert("RGB")