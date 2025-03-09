import time
import argparse
import yarp
import os
import sys
import yaml
import csv
from multiprocessing import Lock
from utils.tobii import startTobiiRecording, stopTobiiRecording, getRecordingFolder
from termcolor import colored, cprint

yarp.Network.init()

class CSVWriter(object):
    def __init__(self, csv_path, headline):
        self.csv_path = csv_path
        self.headline = headline
 
    def write(self, row):
        assert len(row) == len(self.headline)
 
        if not os.path.exists(self.csv_path):
            self.csvfile = open(self.csv_path, 'w', newline='')
            self.writer = csv.writer(self.csvfile)
            self.writer.writerow(self.headline)
        else:
            self.csvfile = open(self.csv_path, 'a', newline='')
            self.writer = csv.writer(self.csvfile)
 
        self.writer.writerow(row)
        self.csvfile.close()

class ChronoRFModule(yarp.RFModule):
    def configure(self, rf):
        with open("src/yarp-app/configs/default.yaml", "r") as f:
            cfg = yaml.safe_load(f)

        self.module_name = "chrono"

        self.period = 0.01 # Proportional to the error that we want to accept.
        self.sync_tobii = cfg['global']['pipeline']['chrono']['sync_tobii']
        self.tobii_uri = cfg['global']['pipeline']['tobii']['serial_number']
        self.dump_path = cfg['global']['pipeline']['chrono']['dump_path']
        self.MULTIDOF = cfg['global']['pipeline']['chrono']['multidof']

        self.table_headline = ['Trial', 'Subject', 'Object', 'Outcome', 'Total Time', 'Start', 'Stop', 'Folder']

        print(self.tobii_uri)

        self.writer = CSVWriter(self.dump_path, self.table_headline)

        # if not os.path.exists(self.dump_path):
        #     self.csvfile = open(self.dump_path, 'w', newline='')
        #     self.writer = csv.writer(self.csvfile)
        #     self.writer.writerow(self.table_headline)
        # else:
        #     self.csvfile = open(self.dump_path, 'a', newline='')
        #     self.writer = csv.writer(self.csvfile)

        self.command_port = yarp.Port()
        self.command_port.open('/' + self.module_name + '/command:i')
        print('{:s} opened'.format('/' + self.module_name + '/command:i'))
        self.attach(self.command_port)  

        if self.MULTIDOF:
            port_name = "/" + self.module_name + "/forward/to/dumper/command:o"
            self.dumper_port = yarp.Port()
            self.dumper_port.open(port_name)
            print(f"{port_name} opened") 

            port_name = "/" + self.module_name + "/forward/to/multidof/command:o"
            self.multidof_port = yarp.Port()
            self.multidof_port.open(port_name)
            print(f"{port_name} opened") 

        self.lock = Lock()
        self.init = None
        self.stop = None
        return True

    def cleanup(self):
        self.command_port.close()
        self.multidof_port.close()
    
    def interruptModule(self):
        self.csvfile.close()
        self.command_port.interrupt()
        self.multidof_port.interrupt()
        return True

    def startEMGDumping(self):
        if not self.MULTIDOF:
            return
        bottle = yarp.Bottle()
        bottle.addString('start')
        self.dumper_port.write(bottle)
        self.multidof_port.write(bottle)
    
    def stopEMGDumping(self):
        if not self.MULTIDOF:
            return
        bottle = yarp.Bottle()
        bottle.addString('stop')
        self.dumper_port.write(bottle)
        self.multidof_port.write(bottle)

    def respond(self, command, reply):
        with self.lock:
            action = command.get(0).asString()
            if action == 'start':
                assert self.init is None
                assert self.stop is None
                self.subject, self.obj, self.trial  =  command.get(1).asString(), \
                                                        command.get(2).asString(), \
                                                        command.get(3).asInt32()
                if self.MULTIDOF:
                    self.startEMGDumping()
                if self.sync_tobii:
                    self.init = startTobiiRecording(self.tobii_uri)
                    print('answer')
                    reply.addString("[CHRONO] Recording timestamps (sync with Tobii)...")
                else:
                    self.init = time.time()
                    reply.addString("[CHRONO] Reconding time...")
                
            elif action == 'stop' or action == 'S' or action == 'F':
                if action == 'S' or action == 'F':
                    outcome = action
                else:
                    outcome = command.get(1).asString()

                # Defining some shortcuts...
                if outcome == 'S':
                    outcome = 'success'
                elif outcome == 'F':
                    outcome = 'fail'

                # Stop immediately when RPC stop is called.
                self.stop = time.time()
                if self.sync_tobii:
                    recordingFolder = getRecordingFolder(self.tobii_uri)
                    tobiiStop = stopTobiiRecording(self.tobii_uri)
                else:
                    recordingFolder = ''

                if self.MULTIDOF:
                    self.stopEMGDumping()

                assert self.init is not None
                diff = self.stop - self.init
                newrow = [self.trial, self.subject, self.obj, outcome, diff, self.init, self.stop, recordingFolder]
                self.writer.write(newrow)
                self.init = None
                self.stop = None
                if outcome == '':
                    cprint('Stop recording!                             ', 'white', 'on_red')
                else:
                    cprint(f'Stop recording with outcome {outcome}      ', 'white', 'on_red')

                reply.addString(f"[CHRONO] Stop recording. Total time: {diff} s")
            else:
                reply.addString(f"[CHRONO] Wrong command. Use start or stop.")
        
        return True

    def getPeriod(self):  
        return self.period

    def updateModule(self):
        with self.lock:
            if self.init is not None:
                currentTime = time.time()
                cprint(f'Recording...      {currentTime - self.init}               ', 'white', 'on_green')
                if currentTime - self.init > 600:
                    self.stop = time.time()
                    tobiiStop = stopTobiiRecording(self.tobii_uri)
                    diff = self.stop - self.init
                    outcome = 'fail'
                    newrow = [self.trial, self.subject, self.obj, outcome, diff, self.init, self.stop]
                    self.writer.write(newrow)
                    cprint('Stop recording. Failing the trial because more than 600s have elapsed...', 'white', 'on_red')
        return True

if __name__ == '__main__':

    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.setDefaultContext("ChronoRFModule")
    conffile = rf.find("from").asString()
    if not conffile:
        print("Using default conf file")
        rf.setDefaultConfigFile("conf.ini")
    else:
        rf.setDefaultConfigFile(rf.find("from").asString())

    rf.configure(sys.argv)

    # Run module
    manager = ChronoRFModule()
    try:
        manager.runModule(rf)
    finally:
        print('Closing ChronoRFModule due to an error...')
        manager.cleanup() 
