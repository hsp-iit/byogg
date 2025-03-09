import yarp
import sys
import yaml
import numpy as np
import os
import cv2
from utils.camera import load_intrinsics
from itertools import chain
from pathlib import Path
from multiprocessing import Lock
import copy
import time

yarp.Network.init()

# NOTE: A reader is either a camera or a module which 
# streams images/videos from the file-system.

class ReaderRFModule(yarp.RFModule):

    def configure(self, rf):
        self.module_name = "reader"
        with open("src/yarp-app/configs/default.yaml", "r") as f:
            cfg = yaml.safe_load(f)

        ### rgb stream setup
        self.cfg = cfg
        self.counter = 0
        stream = cfg["yarp"]["stream"]
        self.stream = stream
        self.cap = None
        self.rgb_list = None
        self.fps = cfg["yarp"]["fps"]["reader"]
        calib = cfg["global"]["calib"]
        self.K = load_intrinsics(calib)
        self.fails = 0 # Counter for reading fails
        if "camera" in stream:
            # Read stream from camera
            camera = stream.split('_')[-1]
            try:
                self.cap = cv2.VideoCapture(int(camera))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["global"]["img_width"])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["global"]["img_height"])
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            except:
                raise FileExistsError(f"error: camera with idx {camera} is not connected or not currently visible.")

        elif os.path.isdir(stream):
            # Read image stream from image folder
            img_exts = ["*.png", "*.jpeg", "*.jpg"]
            self.rgb_list = sorted(chain.from_iterable(Path(stream).glob(e) for e in img_exts))
            self.rgb_idx = 0
        else:
            try:
                # Process input stream from video
                self.cap = cv2.VideoCapture(stream)
            except:
                raise FileExistsError(f"error: {stream} is neither a valid camera id, video or rgb folder.")

        ### Command/RPC port
        self.cmd_port = yarp.Port()
        self.cmd_port.open('/' + self.module_name + '/command:i')
        print('{:s} opened'.format('/' + self.module_name + '/command:i'))
        self.attach(self.cmd_port)

        ### Output ports
        port_name = "/" + self.module_name + "/forward/reference/image:o"
        self.output_reference_port = yarp.Port()
        self.output_reference_port.open(port_name)
        print(f"{port_name} opened")

        port_name = "/" + self.module_name + "/forward/rgb/image:o"
        self.output_rgb_port = yarp.Port()
        self.output_rgb_port.open(port_name)
        print(f"{port_name} opened")

        port_name = "/" + self.module_name + "/forward/to/odometer/rgb/image:o"
        self.output_rgb_to_odometer_port = yarp.BufferedPortImageRgb()
        self.output_rgb_to_odometer_port.open(port_name)
        self.output_rgb_to_odometer_port.writeStrict()
        print(f"{port_name} opened")

        ### Prepare output buffers
        self.output_reference_array = np.ones(    
            (cfg["global"]["img_height"], cfg["global"]["img_width"], 3), np.uint8
        )   
        self.output_reference_image = yarp.ImageRgb() 
        self.output_reference_image.resize(   
            self.output_reference_array.shape[1], self.output_reference_array.shape[0]
        )   
        self.output_reference_image.setExternal(  
            self.output_reference_array, self.output_reference_array.shape[1], self.output_reference_array.shape[0]
        )   

        self.output_rgb_array = np.ones(    
            (cfg["global"]["img_height"], cfg["global"]["img_width"], 3), np.uint8
        )   
        self.output_rgb_image = yarp.ImageRgb() 
        self.output_rgb_image.resize(   
            self.output_rgb_array.shape[1], self.output_rgb_array.shape[0]
        )   
        self.output_rgb_image.setExternal(  
            self.output_rgb_array, self.output_rgb_array.shape[1], self.output_rgb_array.shape[0]
        )  

        self.output_rgb_to_odometer_array = np.ones(    
            (cfg["global"]["img_height"], cfg["global"]["img_width"], 3), np.uint8
        )  
        self.output_rgb_to_odometer_image = yarp.ImageRgb()
        self.output_rgb_to_odometer_image.resize(
            self.output_rgb_to_odometer_array.shape[1], self.output_rgb_to_odometer_array.shape[0]
        )  
        self.output_rgb_to_odometer_image.setExternal(
            self.output_rgb_to_odometer_array, self.output_rgb_to_odometer_array.shape[1], self.output_rgb_to_odometer_array.shape[0]
        )

        ### Input ports for communication with controller
        self.controller_port_ptrigger = yarp.BufferedPortBottle()
        port_name = "/" + self.module_name + "/receive/ptrigger:i"
        self.controller_port_ptrigger.open(port_name)
        print(f"{port_name} opened")

        self.controller_port_ctrigger = yarp.BufferedPortBottle()
        port_name = "/" + self.module_name + "/receive/ctrigger:i"
        self.controller_port_ctrigger.open(port_name)
        print(f"{port_name} opened")

        ### Output port for stand-alone communication with odometer or other modules.
        # This port allows the reader module to announce its reset, even when the controller is not used in the loop.
        # It is only used when the reader processes a stream of limited and known size (i.e., video or rgb folder).
        # If connected to the ctigger port of the odometer, bottles sent over this port will make the odometer
        # reset its internal state. 

        self.signal_reset_port = yarp.Port()
        port_name = "/" + self.module_name + "/forward/reset:o"
        self.signal_reset_port.open(port_name)
        print(f"{port_name} opened")

        self.lock = Lock()  
        self.run = False
        self.odometry = False

        self.MAX_FRAMES = cfg['odometry']['dpvo_config']['BUFFER_SIZE']
        self.odometerCounter = 0
        self.attachTimestamp = cfg['global']['dump']['timestamps']
        self.timestamp = None

        print(f"[READER] Running at {self.fps} fps.")
        return True

    def respond(self, command, reply):
        with self.lock:
            comm = command.get(0).asString()
            source = command.get(1).asString()

            print(comm, source)
            if comm == "read":
                if source == "default":
                    # Do nothing, simply start to process the default rgb source path 
                    # defined in yarp-app/configs/default.yaml
                    path = self.cfg["yarp"]["stream"]
                    self.run = True
                    reply.addString(f"reading {path}")
                else:
                    # Read image stream from image folder
                    img_exts = ["*.png", "*.jpeg", "*.jpg"]
                    self.rgb_list = sorted(chain.from_iterable(Path(source).glob(e) for e in img_exts))
                    self.rgb_idx = 0
                    reply.addString(f"reading {source}")
                    self.run = True

            elif comm == 'set':
                # Command for setting a reading source different than default (only lazy setting, not reading)
                # NOTE: Can be useful if used in combination with a manually crafted controller ptrigger command
                #       to controller RPC port.
                img_exts = ["*.png", "*.jpeg", "*.jpg"]
                self.rgb_list = sorted(chain.from_iterable(Path(source).glob(e) for e in img_exts))
                self.rgb_idx = 0
                reply.addString(f"set {source} as reading source")

            elif comm == "stream":
                if source == "start":
                    # Start streaming from selected camera
                    assert 'camera' in self.stream and self.cap
                    self.run = True
                    reply.addString(f"streaming camera on")
                elif source == "stop":
                    # Stop streaming from selected camera
                    assert 'camera' in self.stream and self.cap
                    self.run = False
                    self.counter = 0
                    self.fails = 0
                    reply.addString(f"streaming camera off")
            else:
                reply.addString(f"error: other readings methods are not yet fully implemented.")
                
        return True

    def cleanup(self):
        if self.cap:
            self.cap.release()

        self.cmd_port.close()
        self.controller_port_ptrigger.close()
        self.controller_port_ctrigger.close()
        self.output_rgb_port.close()
        self.output_rgb_to_odometer_port.close()
        self.output_reference_port.close()
        self.signal_reset_port.close()

    def interruptModule(self):
        if self.cap:
            self.cap.release()
        
        self.cmd_port.interrupt()
        self.controller_port_ptrigger.interrupt()
        self.controller_port_ctrigger.interrupt()
        self.output_rgb_port.interrupt()
        self.output_rgb_to_odometer_port.interrupt()
        self.output_reference_port.interrupt()
        self.signal_reset_port.interrupt()
        
        return True
    
    def getPeriod(self):
        return 1/float(self.fps)

    def signalReset(self):
        resetBottle = yarp.Bottle()
        resetBottle.addInt8(42)
        self.signal_reset_port.write(resetBottle)

    # TODO: Check if we need to invert RGB (BGR to RGB) channels when reading frames from cap.
    def updateModule(self):
        with self.lock:
            ### If reading from a camera, always stream frames to yarpview. In case the reaching phase starts
            ### (i.e., the pipeline FSM enters the state ODOMETRY), also start to stream frames to buffered port.
            if 'camera' in self.stream:
                assert self.cap and not self.rgb_list
                ret, frame = self.cap.read()
                if self.attachTimestamp:
                    self.timestamp = time.time()
                    
                if not ret:
                    self.fails += 1
                    ### Reader module crashes after 10 failed attempts in reading camera feed or video stream.
                    if self.fails >= 10: 
                        raise ValueError(f"Failed to grab frames [{self.fails} failed attempts]")
                else:
                    self.fails = 0
                
                    frame = cv2.flip(frame, 0)
                    frame = cv2.flip(frame, 1)

                    self.output_rgb_array[:] = frame[..., ::-1]
                    self.output_rgb_port.write(self.output_rgb_image)

                    # Read port for bottle communication with controller (ptrigger).
                    ptrigger = self.controller_port_ptrigger.read(shouldWait=False)
                    if ptrigger is not None:
                        # NOTE: For the sake of simplicity, no parsing is performed here on the bottle.
                        # Basically, every bottle sent over this channel would trigger the ptrigger logic.
                        print('[READER] Received ptrigger. Sending reference.')
                        self.output_reference_array[:] = frame[..., ::-1]
                        self.output_reference_port.write(self.output_reference_image)
                        self.odometry = True
                    
                    if self.odometry:
                        print('[READER] Sending frame to odometer.')
                        output_rgb_to_odometer_image = self.output_rgb_to_odometer_port.prepare()
                        self.output_rgb_to_odometer_array[:] = frame[..., ::-1]
                        output_rgb_to_odometer_image.copy(self.output_rgb_to_odometer_image)

                        if self.attachTimestamp:
                            envelopeBottle = yarp.Bottle()
                            envelopeBottle.addFloat64(self.timestamp)
                            self.output_rgb_to_odometer_port.setEnvelope(envelopeBottle)

                        self.output_rgb_to_odometer_port.write()
                        self.odometerCounter += 1

                        # NOTE: To avoid overflowing the odometer buffer, we reset the odometer state 
                        # once we reach the max amount of frames in camera mode. 
                        # In this case, even in controller mode, the reader is in charge of notifying other modules.
                        # This is because the controller has no info about the number of frames ingested by odometer.
                        if self.odometerCounter >= self.MAX_FRAMES:
                            print('[READER] Reached MAX_FRAMES number. Resetting...')
                            self.odometerCounter = 0
                            self.odometry = False
                            self.signalReset()

                    # Read port for bottle communication with controller (ctrigger). 
                    ctrigger = self.controller_port_ctrigger.read(shouldWait=False)
                    if ctrigger is not None:
                        print('[READER] Received ctrigger.')
                        self.odometry = False
                        self.odometerCounter = 0
                        # NOTE: When ctrigger is received, the controller is in charge of resetting the state of the odometer. 
            else:
                ### If reading from disk (video or rgb folder), check if module is active. 
                if self.run:
                    ### If self.cap is not None, we are reading a video.
                    if self.cap:
                        assert not self.rgb_list
                        ret, frame = self.cap.read()
                        if not ret:
                            self.run = False
                            self.counter = 0
                            self.signalReset()

                    ### Otherwise, we are given an RGB folder.
                    elif self.rgb_list:
                        assert not self.cap
                        if self.rgb_idx >= len(self.rgb_list):
                            self.signalReset()
                            # Exit gracefully after ending processing of input rgb sequence.
                            print("[READER] End processing of input rgb sequence.")
                            self.run = False
                            self.rgb_idx = 0
                            self.counter = 0
                            return True

                        imfile = self.rgb_list[self.rgb_idx]
                        self.rgb_idx += 1
                        frame = cv2.imread(str(imfile))
                        if self.attachTimestamp:
                            self.timestamp = time.time()
                    
                    if self.counter == 0:
                        self.output_reference_array[:] = frame[..., ::-1]
                        self.output_reference_port.write(self.output_reference_image)

                    self.output_rgb_array[:] = frame[..., ::-1]
                    self.output_rgb_port.write(self.output_rgb_image)
                    
                    output_rgb_to_odometer_image = self.output_rgb_to_odometer_port.prepare()
                    self.output_rgb_to_odometer_array[:] = frame[..., ::-1]
                    output_rgb_to_odometer_image.copy(self.output_rgb_to_odometer_image)

                    if self.attachTimestamp:
                        envelopeBottle = yarp.Bottle()
                        envelopeBottle.addFloat64(self.timestamp)
                        self.output_rgb_to_odometer_port.setEnvelope(envelopeBottle)

                    self.output_rgb_to_odometer_port.write() 

                    print(f"[READER] Writing output rgb image {self.counter} - rgb.shape: {self.output_rgb_array.shape} - {time.time()}")  
                    self.counter += 1
                else:
                    # Check if a ptrigger bottle is received on the corresponding port.
                    # If yes, start streaming frames by setting self.run = True without having to 
                    # manually send an RPC call.
                    ptrigger = self.controller_port_ptrigger.read(shouldWait=False)
                    if ptrigger is not None:
                        self.run = True
        return True
    
if __name__ == '__main__':

    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.setDefaultContext("ReaderRFModule")
    conffile = rf.find("from").asString()
    if not conffile:
        print("Using default conf file")
        rf.setDefaultConfigFile("conf.ini")
    else:
        rf.setDefaultConfigFile(rf.find("from").asString())

    rf.configure(sys.argv)

    # Run module
    manager = ReaderRFModule()
    try:
        manager.runModule(rf)
    finally:
        manager.cleanup() 
