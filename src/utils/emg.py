import numpy as np
from enum import Enum
from utils.misc import find_monotonous_subsequences
import time

class PTriggerType(Enum):
    OPEN_HOLD = 0
    OPEN_DOUBLE_PEAK = 1
    COCONTR = 2 
    MANUAL = 3

# TODO: Extend detect_double_peak, detect_open_hold and detect_cocontraction 
#       to also work on the close EMG channel (i.e., EMG channel used to close Hannes fingers).
#       This is required to provide implementation for CTriggerType.MANUAL_CLOSE_HOLD, 
#       CTriggerType.MANUAL_CLOSE_DOUBLE_PEAK and CTriggerType.MANUAL_COCONTR.
#       To do this, it is also required to duplicate buffers for recording EMG values and time deltas.

class EMGProcessor(object):
    def __init__(self, hannes_cfg, 
                        ptrigger=PTriggerType.OPEN_DOUBLE_PEAK, 
                        fingers_cur_range=0, 
                        ps_cur_range=0, 
                        fe_cur_range=0):

        self.hannes_cfg = hannes_cfg

        self.open_ch = int(self.hannes_cfg["emg"]["open_ch"])
        self.close_ch = int(self.hannes_cfg["emg"]["close_ch"])
        self.fingers_thresh = int(self.hannes_cfg["fingers"]["range_thresh"])

        self.ptrigger = ptrigger
        self.check_ptrigger = None
        if ptrigger is PTriggerType.OPEN_HOLD:
            self.hold_window = float(self.hannes_cfg["ptrigger"]["hold"]["keep_last_s"])
            self.hold_thresh = float(self.hannes_cfg["ptrigger"]["hold"]["emg_thresh"])
            self.check_ptrigger = self.detect_open_hold
        elif ptrigger is PTriggerType.OPEN_DOUBLE_PEAK:
            self.dp_window = float(self.hannes_cfg["ptrigger"]["double_peak"]["keep_last_s"])
            self.dp_prominence = float(self.hannes_cfg["ptrigger"]["double_peak"]["prominence"])
            self.check_ptrigger = self.detect_double_peak_and_fingers
        elif ptrigger is PTriggerType.COCONTR:
            self.cocontr_thresh = float(hannes_cfg["ptrigger"]["cocontr"]["cocontr_thresh"])
            self.check_ptrigger = self.detect_cocontraction
        elif ptrigger is PTriggerType.MANUAL:
            # NOTE: If trigger is manual, it can be activated via RPC call to controller.
            # Thus, the EMGProcessor always returns False when check_ptrigger is called.
            self.check_ptrigger = lambda x: False

        # Set the initial fingers position (default: hand completely open)
        self.fingers_cur_range = fingers_cur_range
        # Set the initial ps position 
        self.ps_cur_range = ps_cur_range # Not used for now.
        self.fe_cur_range = fe_cur_range # Not used for now.

        self.emg_buffer = []
        self.emg_record_deltas = []
        self.emg_record_last_time = time.time()

    def detect_double_peak_and_fingers(self, channels):
        ## if a double peak is detected and hand is open, trigger the change of state -1 -> 1.
        ## NOTE: In this way, a new prediction can be "requested" by the user without
        ## turning on the visual servoning.
        if self.detect_double_peak(channels) == True \
            and  self.fingers_cur_range <= self.fingers_thresh:
            return True
        return False

    def detect_double_peak(self, channels):
        # TODO: Add channel configuration to src/yarp-app/configs/default.yaml
        open_sig = channels[self.open_ch]
        self.emg_buffer = np.insert(self.emg_buffer, 0, open_sig)

        # keep only the last emg_open_double_peak.keep_last_s seconds of reading
        elapsed_time = time.time() - self.emg_record_last_time
        self.emg_record_last_time = time.time()
        self.emg_record_deltas = np.insert(
            self.emg_record_deltas, 0, elapsed_time
        )
        tot_time = 0
        remove_from_idx = None
        for idx, t in enumerate(self.emg_record_deltas):
            tot_time += t
            if tot_time > self.dp_window:
                remove_from_idx = idx
                break
        if remove_from_idx is not None:
            self.emg_record_deltas = self.emg_record_deltas[:remove_from_idx]
            self.emg_buffer = self.emg_buffer[:remove_from_idx]

        if not len(self.emg_buffer):
            return False

        subsequences = find_monotonous_subsequences(self.emg_buffer)
        peaks = []
        for start_idx, end_idx in subsequences:
            if self.emg_buffer[end_idx] - self.emg_buffer[start_idx] >=  self.dp_prominence:
                peaks.append(end_idx)

        if len(peaks) == 2:
            self.emg_record_deltas = np.array([])
            self.emg_buffer = np.array([])
            return True
        else:
            return False

    def detect_open_hold(self, channels):
        open_sig = channels[self.open_ch]
        self.emg_buffer = np.insert(self.emg_buffer, 0, open_sig)

        # keep only the last emg_open_double_peak.keep_last_s seconds of reading
        # TODO: Add to the config.yaml file a parameter that sets the interval span (by default 0.5s)
        
        elapsed_time = time.time() - self.emg_record_last_time
        self.emg_record_last_time = time.time()
        self.emg_record_deltas = np.insert(
            self.emg_record_deltas, 0, elapsed_time
        )
        tot_time = 0
        remove_from_idx = None
        for idx, t in enumerate(self.emg_record_deltas):
            tot_time += t
            if tot_time >= self.hold_window:
                remove_from_idx = idx
                break

        if remove_from_idx is not None:
            # NOTE: when checking self.emg_record_deltas, it is required to keep also the reading with 
            # delta time that possibly overflows cfg["emg_open_hold"]["keep_last_s"]. Otherwise, 
            # the sum of delta contained in self.emg_record_deltas would always be < keep_last_s,
            # leading to inconsistent "holding" checks. 
            # EXAMPLE: Suppose you are reading 
            # self.emg_record_deltas: [0.3, 0.3]
            # self.emg_buffer: [4.65, 0.8]
            # And that keep_last_s = 0.5, emg_thresh = 1.5.
            # If we truncate the self.emg_buffer array at the element remove_from_idx, we will get 
            # [0.3]
            # [4.65]
            # which will wrongly trigger the hold condition, as a value above the threshold has been recorded for
            # a range of time equal to 0.3 (lower than keep_last_s, which by default is set to 0.5).
            self.emg_record_deltas = self.emg_record_deltas[:remove_from_idx+1]
            self.emg_buffer = self.emg_buffer[:remove_from_idx+1]

        if not len(self.emg_buffer):
            return False

        if np.sum(self.emg_buffer > self.hold_thresh) \
                                    == len(self.emg_buffer):
            if self.hold_window <= tot_time <= self.hold_window * 2:
                self.emg_record_deltas = np.array([])
                self.emg_buffer = np.array([])
                return True
            else:
                # NOTE: This condition here is only possible when all values in self.emg_buffer are above threshold,
                # BUT the array also contains "old" readings, with a sum of deltas that is higher than self.hannes_cfg["emg_open_hold"]["keep_last_s"] * 2.
                self.emg_record_deltas = self.emg_record_deltas[:remove_from_idx]
                self.emg_buffer = self.emg_buffer[:remove_from_idx]
                return False
        else:
            return False

    def detect_cocontraction(self, channels):
        open_sig = channels[self.open_ch]
        close_sig = channels[self.close_ch]
        if open_sig > self.cocontr_thresh and close_sig > self.cocontr_thresh:     
            return True
        else:
            return False