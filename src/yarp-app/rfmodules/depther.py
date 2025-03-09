import yarp
import sys
import yaml
import torch
import time
import numpy as np
import cv2
from utils.camera import load_intrinsics
from mde.utils import build_dinov2_depther
from mde.io import load_model_setup, render_depth
from mde.ops import make_depth_transform, clamp
from utils.io import IOHandler as IO

DEBUG = False

yarp.Network.init()

class DepthRFModule(yarp.RFModule):

    def configure(self, rf):
        self.module_name = "depther"
        with open("src/yarp-app/configs/default.yaml", "r") as f:
            cfg = yaml.safe_load(f)

        self.cfg = cfg
        calib = cfg["global"]["calib"]
        K = load_intrinsics(calib)

        self.fps = cfg["yarp"]["fps"]["depther"]
        self.min_depth = cfg["mde"]["min_depth"]
        self.max_depth = cfg["mde"]["max_depth"]
        self.ckpt_name = cfg["mde"]["model_name"]

        model_config = load_model_setup(cfg)
        self.model = build_dinov2_depther(cfg, 
                                     backbone_size='small',
                                     head_type=model_config['head'],
                                     min_depth=self.min_depth,
                                     max_depth=self.max_depth,
                                     inference=True,
                                     checkpoint_name=self.ckpt_name,
                                     use_depth_anything_encoder=(cfg["mde"]["backbone"] == "depth-anything"))
        self.model.eval()
        self.model.cuda()

        self.transform = make_depth_transform(backbone=cfg["mde"]["backbone"])

        ### Input ports
        port_name = "/" + self.module_name + "/receive/rgb/image:i"
        self.input_rgb_port = yarp.BufferedPortImageRgb()
        self.input_rgb_port.open(port_name)
        print(f"{port_name} opened")

        port_name = "/" + self.module_name + "/receive/reference/image:i"
        self.input_reference_port = yarp.BufferedPortImageRgb()
        self.input_reference_port.open(port_name)
        print(f"{port_name} opened")

        ### Output ports
        port_name = "/" + self.module_name + "/forward/depth/image:o"
        self.output_depth_port = yarp.Port()
        self.output_depth_port.open(port_name)
        print(f"{port_name} opened")

        port_name = "/" + self.module_name + "/forward/depthcolor/image:o"
        self.output_render_depth_port = yarp.Port()
        self.output_render_depth_port.open(port_name)
        print(f"{port_name} opened")

        port_name = "/" + self.module_name + "/forward/reference/rgb/image:o"
        self.output_reference_rgb_port = yarp.Port()
        self.output_reference_rgb_port.open(port_name)
        print(f"{port_name} opened")

        port_name = "/" + self.module_name + "/forward/reference/depth/image:o"
        self.output_reference_depth_port = yarp.Port()
        self.output_reference_depth_port.open(port_name)
        print(f"{port_name} opened")
        
        ### Input images
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

        self.input_reference_array = np.ones(
            (cfg["global"]["img_height"], cfg["global"]["img_width"], 3), dtype=np.uint8
        )
        self.input_reference_image = yarp.ImageRgb()
        self.input_reference_image.resize(
            self.input_reference_array.shape[1], self.input_reference_array.shape[0]
        )
        self.input_reference_image.setExternal(
            self.input_reference_array, self.input_reference_array.shape[1], self.input_reference_array.shape[0]
        )

        ### Output images 
        self.output_depth_array = np.ones(
            (cfg["global"]["img_height"], cfg["global"]["img_width"]), dtype=np.float32
        )
        self.output_depth_image = yarp.ImageFloat()
        self.output_depth_image.resize(
            self.output_depth_array.shape[1], self.output_depth_array.shape[0]
        )
        self.output_depth_image.setExternal(
            self.output_depth_array, self.output_depth_array.shape[1], self.output_depth_array.shape[0]
        )

        self.output_render_depth_array = np.ones(
            (cfg["global"]["img_height"], cfg["global"]["img_width"], 3), dtype=np.uint8
        )
        self.output_render_depth_image = yarp.ImageRgb()
        self.output_render_depth_image.resize(
            self.output_render_depth_array.shape[1], self.output_render_depth_array.shape[0]
        )
        self.output_render_depth_image.setExternal(
            self.output_render_depth_array, self.output_render_depth_array.shape[1], self.output_render_depth_array.shape[0]
        )

        self.output_reference_rgb_array = np.ones(
            (cfg["global"]["img_height"], cfg["global"]["img_width"], 3), dtype=np.uint8
        )
        self.output_reference_rgb_image = yarp.ImageRgb()
        self.output_reference_rgb_image.resize(
            self.output_reference_rgb_array.shape[1], self.output_reference_rgb_array.shape[0]
        )
        self.output_reference_rgb_image.setExternal(
            self.output_reference_rgb_array, self.output_reference_rgb_array.shape[1], self.output_reference_rgb_array.shape[0]
        )

        self.output_reference_depth_array = np.ones(
            (cfg["global"]["img_height"], cfg["global"]["img_width"]), dtype=np.float32
        )
        self.output_reference_depth_image = yarp.ImageFloat()
        self.output_reference_depth_image.resize(
            self.output_reference_depth_array.shape[1], self.output_reference_depth_array.shape[0]
        )
        self.output_reference_depth_image.setExternal(
            self.output_reference_depth_array, self.output_reference_depth_array.shape[1], self.output_reference_depth_array.shape[0]
        )

        self.attachTimestamp = cfg['global']['dump']['timestamps']
        self.timestamp = None

        print(f"[DEPTHER] Running at {self.fps} fps.")
        return True

    def respond(self, command, reply):
        return super().respond(command, reply)

    def cleanup(self):
        self.input_rgb_port.close()
        self.output_depth_port.close()
        self.output_render_depth_port.close()
        self.output_reference_rgb_port.close()
        self.output_reference_depth_port.close()

    def interruptModule(self):
        self.input_rgb_port.interrupt()
        self.output_depth_port.interrupt()
        self.output_render_depth_port.interrupt()
        self.output_reference_rgb_port.interrupt()
        self.output_reference_depth_port.interrupt()
        return True
    
    def getPeriod(self):
        return 1/float(self.fps)
    
    def predictDepth(self, rgb):
        with torch.no_grad():
            # Feed image to depth model
            rgb = self.transform(rgb)
            rgb = rgb.unsqueeze(0).cuda()
            depth = self.model.whole_inference(rgb, img_meta=None, rescale=False)
            depth = clamp(depth, xmin=self.min_depth, xmax=self.max_depth)
            depth = depth.squeeze().cpu().numpy()
        return depth
    
    def updateModule(self):
        # Check if a frame can be read from the reference port.
        reference = self.input_reference_port.read(shouldWait=False)
        # Reference port has priority over standard input port.
        if reference is not None:
            self.input_reference_image.copy(reference)
            assert(
                self.input_reference_array.__array_interface__['data'][0] == \
                self.input_reference_image.getRawImage().__int__()
            )
            print(f"[DEPTHER] Reading input RGB image from reference port - rgb.shape: {self.input_reference_array.shape}")

            if DEBUG:
                cv2.imwrite('tests/debug/deptherRGBInput.png', cv2.cvtColor(self.input_reference_array, cv2.COLOR_RGB2BGR))

            depth = self.predictDepth(self.input_reference_array)
            if self.attachTimestamp:
                self.timestamp = time.time()

            self.output_reference_rgb_array[:, :] = self.input_reference_array
            self.output_reference_depth_array[:, :] = depth

            self.output_reference_rgb_port.write(self.output_reference_rgb_image)
            
            if self.attachTimestamp:
                envelopeBottle = yarp.Bottle()
                envelopeBottle.addFloat64(self.timestamp)
                self.output_reference_depth_port.setEnvelope(envelopeBottle)

            self.output_reference_depth_port.write(self.output_reference_depth_image)

        else:
            # Read image from input port
            image = self.input_rgb_port.read()
            self.input_rgb_image.copy(image)
            assert(
                self.input_rgb_array.__array_interface__['data'][0] == \
                self.input_rgb_image.getRawImage().__int__()
            )
            print(f"[DEPTHER] Reading input RGB image - rgb.shape: {self.input_rgb_array.shape}")
            depth = self.predictDepth(self.input_rgb_array)
            # Write depth and rgb to output ports
            self.output_depth_array[:, :] = depth
            self.output_depth_port.write(self.output_depth_image)
            # y,x = np.where((depth > 0) & (depth <= 2))
            # print(f"[DEPTHER] {y.shape, x.shape}")  
            self.output_render_depth_array[:, :] = render_depth(depth)
            self.output_render_depth_port.write(self.output_render_depth_image)

        return True

if __name__ == '__main__':

    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.setDefaultContext("DepthRFModule")
    conffile = rf.find("from").asString()
    if not conffile:
        print("Using default conf file")
        rf.setDefaultConfigFile("conf.ini")
    else:
        rf.setDefaultConfigFile(rf.find("from").asString())

    rf.configure(sys.argv)

    # Run module
    manager = DepthRFModule()
    try:
        manager.runModule(rf)
    finally:
        print('Closing DepthRFModule due to an error...')
        manager.cleanup() 
