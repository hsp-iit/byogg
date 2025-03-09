import yarp
import sys
import yaml
import numpy as np
import os

import time
from utils.hannes import hannes_init
from libs.pyHannesAPI.pyHannesAPI.pyHannes import Hannes

yarp.Network.init()

class MultidofRFModule(yarp.RFModule):

    def configure(self, rf):
        self.module_name = "multidof"
        with open("src/yarp-app/configs/default.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        
        self.device_name = cfg["hannes"]["device_name"]
        self.hannes_active = cfg["hannes"]["active"]

        self.hannes = None
        if self.hannes_active:
            ### Initialize Hannes hand communication
            self.hannes = Hannes(device_name=self.device_name)
            hannes_init(self.hannes, control_modality='CONTROL_EMG')

        port_name = "/" + self.module_name + "/receive/command:i"
        self.input_command_port = yarp.BufferedPortBottle()
        self.input_command_port.open(port_name)
        print(f"{port_name} opened")

        port_name = "/" + self.module_name + "/forward/emg:o"
        self.emg_channels_port = yarp.Port()
        self.emg_channels_port.open(port_name)
        print(f"{port_name} opened")

        self.attachTimestamp = cfg['global']['dump']['timestamps']
        self.emgEnvelope = None

        self.dump = False
        self.period = 1/30
        return True

    def respond(self, command, reply): 
        return super().respond(command, reply)

    def cleanup(self):
        self.emg_channels_port.close()
        self.input_command_port.close()

    def interruptModule(self):
        self.emg_channels_port.interrupt()
        self.input_command_port.interrupt()
        return True
    
    def getPeriod(self):
        return self.period

    def updateModule(self):
        commandBottle = self.input_command_port.read(shouldWait=False)

        if commandBottle is not None:
            action = commandBottle.get(0).asString()
            if action == 'start':
                self.dump = True
                print('[MULTIDOF] Start dumping data...')
            elif action == 'stop':
                self.dump = False
                print('[MULTIDOF] Stop dumping data.')

        if self.hannes_active and self.hannes is not None and self.dump:
            channels = self.hannes.measurements_emg()

            if self.attachTimestamp:
                ts = time.time()
                self.emgEnvelope = yarp.Bottle()
                self.emgEnvelope.addFloat64(ts)

            print(f"[MULTIDOF] EMG channels: {channels}")

            channelsBottle = yarp.Bottle()
            for channel in channels:
                channelsBottle.addFloat32(channel)

            if self.attachTimestamp:
                self.emg_channels_port.setEnvelope(self.emgEnvelope)
            self.emg_channels_port.write(channelsBottle)

        return True
    
if __name__ == '__main__':

    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.setDefaultContext("MultidofRFModule")
    conffile = rf.find("from").asString()
    if not conffile:
        print("Using default conf file")
        rf.setDefaultConfigFile("conf.ini")
    else:
        rf.setDefaultConfigFile(rf.find("from").asString())

    rf.configure(sys.argv)

    # Run module
    manager = MultidofRFModule()
    try:
        manager.runModule(rf)
    finally:
        print('Closing MultidofRFModule due to an error...')
        manager.cleanup() 
