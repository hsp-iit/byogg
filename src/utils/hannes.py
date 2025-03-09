import numpy as np
import signal
from utils.ops import scale_to_range

from pytransform3d.rotations import active_matrix_from_intrinsic_euler_zyx, axis_angle_from_matrix
from roboticstoolbox import PoERevolute, PoERobot
from spatialmath import SE3
import numpy as np

import torch
import roma
import math

def fix_ps_tick(ps_tick, tick_min, tick_max, tick_overflow):
    if ps_tick >= tick_min:
        return ps_tick - tick_overflow
    elif ps_tick <= tick_max:
        return ps_tick
    elif abs(ps_tick - tick_min) < abs(ps_tick - tick_max):
        return tick_min - tick_overflow
    else:
        return tick_max

class HannesConf(object):
    def __init__(self, hannes, hannes_cfg, candidate_cfg):
        self.hannes = hannes
        self.hannes_cfg = hannes_cfg
        self.candidate_cfg = candidate_cfg
        self.wps_range = (hannes_cfg['ps']['limits']['range']['min'], hannes_cfg['ps']['limits']['range']['max'])
        self.wps_eul_deg = (hannes_cfg['ps']['limits']['eul_deg']['min'], hannes_cfg['ps']['limits']['eul_deg']['max'])

        self.wfe_range = (hannes_cfg['fe']['limits']['range']['min'], hannes_cfg['fe']['limits']['range']['max'])
        self.wfe_eul_deg = (hannes_cfg['fe']['limits']['eul_deg']['min'], hannes_cfg['fe']['limits']['eul_deg']['max'])

        self.fingers_range = (hannes_cfg['fingers']['limits']['range']['min'], hannes_cfg['fingers']['limits']['range']['max'])
        self.gripper_width = 0.08
        
        # Setup kinematic chain
        self.L1 = int(hannes_cfg['robot']['L1'])
        self.L2 = int(hannes_cfg['robot']['L2'])
        self.robot = HannesRobot(L1=self.L1, L2=self.L2)

        self.lmbda_psfe = np.array([hannes_cfg["ps"]["lmbda"], 
                                    hannes_cfg["fe"]["lmbda"]])

        self.R_e2c =  active_matrix_from_intrinsic_euler_zyx(
            np.deg2rad([
                    hannes_cfg["robot"]["eul_e2c_deg"]["z"], 
                    hannes_cfg["robot"]["eul_e2c_deg"]["y"], 
                    hannes_cfg["robot"]["eul_e2c_deg"]["x"]
                ])
        )

        self.t_e2c = np.array([
            hannes_cfg["robot"]["t_e2c"]["x"], 
            hannes_cfg["robot"]["t_e2c"]["y"], 
            hannes_cfg["robot"]["t_e2c"]["z"]
        ])

        self.M_e2c = np.eye(4)
        self.M_e2c[:3, :3] = self.R_e2c
        self.M_e2c[:3, 3] = self.t_e2c

        self.Ad_c2e = SE3(np.linalg.inv(self.M_e2c)).Ad()

        theta_z = -math.pi/2
        self.rot_z = np.array([
            [math.cos(theta_z), -math.sin(theta_z), 0],
            [math.sin(theta_z), math.cos(theta_z), 0],
            [0, 0, 1]
        ])

        theta_y = -math.pi/4
        self.rot_y = np.array([
            [math.cos(theta_y), 0, math.sin(theta_y)],
            [0, 1, 0],
            [-math.sin(theta_y), 0, math.cos(theta_y)]
        ])

        self.rot_z180 = np.array([
            [math.cos(math.pi), -math.sin(math.pi), 0],
            [math.sin(math.pi), math.cos(math.pi), 0],
            [0, 0, 1]
        ])

        self.loop_freq = 1000
        self.dt = 1/self.loop_freq
        self.fing_off = 25
        self.active = hannes_cfg['active']
        self.read_encoders = hannes_cfg['active']

        self.always_pinch = candidate_cfg['always-pinch']['active']
        self.always_pinch_thresh = int(candidate_cfg['always-pinch']['threshold'])
        assert 0 <= self.always_pinch_thresh <= 100
        self.always_pinch_thresh = float(self.always_pinch_thresh / 100) * self.gripper_width

        self.stop_criteria = candidate_cfg['stop-prop-controller']['criteria']
        assert self.stop_criteria in ['max-steps', 'convergence']
        if self.stop_criteria == 'max-steps':
            self.N_STEPS = int(candidate_cfg['stop-prop-controller']['max-steps'])
        else:
            self.CONV_THRESH = float(candidate_cfg['stop-prop-controller']['conv-thresh'])

        if self.active and self.read_encoders:
            home_ps_tick = self.hannes.measurements_wristPS()["position"]
            home_fe_tick = self.hannes.measurements_wristFE()["position"]
            home_ps_eul_deg, home_fe_eul_deg = self.eul_degs_from_ticks(home_ps_tick, home_fe_tick)
        else:
            home_ps_eul_deg = float(hannes_cfg['home_configuration']['ps']['eul_deg'])
            home_fe_eul_deg = float(hannes_cfg['home_configuration']['fe']['eul_deg'])

        self.q_home = np.array([home_ps_eul_deg, home_fe_eul_deg], dtype=np.float64)

    def eul_degs_from_ticks(self, cur_ps_tick, cur_fe_tick):
        cur_ps_tick = fix_ps_tick(                         
            cur_ps_tick, 
            self.hannes_cfg["ps"]["limits"]["tick"]["min"], 
            self.hannes_cfg["ps"]["limits"]["tick"]["max"], 
            self.hannes_cfg["ps"]["limits"]["tick"]["overflow"], 
        )
        cur_fe_eul_deg = scale_to_range(
            cur_fe_tick, 
            self.hannes_cfg["fe"]["limits"]["tick"]["min"], 
            self.hannes_cfg["fe"]["limits"]["tick"]["max"], 
            self.hannes_cfg["fe"]["limits"]["eul_deg"]["min"], 
            self.hannes_cfg["fe"]["limits"]["eul_deg"]["max"], 
        )
        
        ps_min_tick = - self.hannes_cfg["ps"]["limits"]["tick"]["max"]
        ps_max_tick = self.hannes_cfg["ps"]["limits"]["tick"]["max"]
        
        cur_ps_eul_deg = scale_to_range(
            cur_ps_tick, 
            ps_min_tick, 
            ps_max_tick,
            self.hannes_cfg["ps"]["limits"]["eul_deg"]["min"], 
            self.hannes_cfg["ps"]["limits"]["eul_deg"]["max"], 
        )
        return cur_ps_eul_deg, cur_fe_eul_deg

    def check_stop_condition(self, step, res_error=None):
        if self.stop_criteria == 'max-steps':
            # Check if N_STEPS of optimization have been performed.
            if step >= self.N_STEPS:
                return True
        else:
            # Check if residual error is lower than threshold.
            if res_error is not None and res_error < self.CONV_THRESH:
                return True
        return False

    def map_gripper_pose_to_hannes_config(self, gpose, gopening, cpose=np.eye(4)):
        '''
            Transforms a parallel gripper SO(3)xR pose (rotation + opening width) to
            a configuration for the Hannes hand (wrist + fingers) via prop. control. 
            See scripts/visualization/gripper2hannes.py for an Open3D visualization.
            Gripper poses are parameterized and predicted in the frame of the first camera.
            Before running the prop. controller to find the configuration of joints which 
            allows to reach a desired gripper pose, we project gripper pose to the new estimated
            camera frame. 
        '''

        
        R_c_des = gpose[:3, :3] @ self.rot_z @ self.rot_y
        R_c_des_rotz = R_c_des @ self.rot_z180

        if self.hannes is not None and self.read_encoders:
            # Read hannes encoders
            cur_ps_tick = self.hannes.measurements_wristPS()["position"]
            cur_fe_tick = self.hannes.measurements_wristFE()["position"]

            cur_ps_eul_deg, cur_fe_eul_deg = self.eul_degs_from_ticks(cur_ps_tick, cur_fe_tick)

            q_current = np.array([cur_ps_eul_deg, cur_fe_eul_deg], dtype=np.float64)
            # b2e transform at home configuration.
            M_b2e_home = self.robot.forward_kinematics(self.q_home)
            # b2e transform for the current joint configuration.
            M_b2e_curr = self.robot.forward_kinematics(q_current)

            M_b2c_home = M_b2e_home @ self.M_e2c
            M_b2c_curr = M_b2e_curr @ self.M_e2c

            # Compute current camera pose in the initial (home) camera frame.
            M_c_curr = np.linalg.inv(M_b2c_home) @ M_b2c_curr
            R_c_curr = M_c_curr[:3, :3]

            R_c_des = (R_c_curr).T @ R_c_des
            R_c_des_rotz = (R_c_curr).T @ R_c_des_rotz
            # R_c_curr becomes the identity - we now run the prop.controller optimization in this frame.
            R_c_curr = np.eye(3)
        else:
            # If hannes is not available or we don't want to read from encoders.
            # Thus, in this case, we assume that the wrist is still in the home configuration 
            # w.r.t to the base frame, when the last frame before grasping the object is captured. 
            # This could be true when using Hannes in theory, but 
            # in practice it's not because the PS is controlled in closed loop, causing drifts.
            # Also, in this case, we cannot compute the camera rotation w.r.t. the home configuration.
            # But we can use the estimated rotation from the visual odometry module (cpose[:3, :3]).

            q_current = np.copy(self.q_home)
            R_c_curr = cpose[:3, :3]
        
            R_c_des = (R_c_curr).T @ R_c_des
            R_c_des_rotz = (R_c_curr).T @ R_c_des_rotz

            R_c_curr = np.eye(3)

        err1 = axis_angle_from_matrix(R_c_des)
        err2 = axis_angle_from_matrix(R_c_des_rotz)
        errx1 = np.abs(err1[0] * err1[3])
        errx2 = np.abs(err2[0] * err2[3])
        if errx2 < errx1:
            R_c_des = np.copy(R_c_des_rotz)
            
        # print(errx1, errx2)
        step = 0
        res_error = None
        # Run proportial controller for N_STEPS optimization steps or until convergence.
        while not self.check_stop_condition(step, res_error):

            M_b2e_prev = self.robot.forward_kinematics(q_current)
            # Compute Jacobian in the end-effector frame
            jacobian_e = self.robot.jacobian_endeffector(q_current)
            # Convert Jacobian velocities to camera frame by pre-multiplication with c2e adjoint
            jacobian_c = self.Ad_c2e @ jacobian_e
            # Take Jacobian rows corresponding to axis x and z angular velocities
            jacobian_c_rot = np.vstack([jacobian_c[3, :], jacobian_c[5, :]])

            # Compute x-z axis error
            # NOTE: equivalent of axis_angle_from_matrix(R_c_des @ np.eye(3).T),
            # because the camera rotation is always the identity in the camera frame.
            err = axis_angle_from_matrix(R_c_des)
            axis, angle = np.split(err, [3])
            err = np.hstack([axis[0] * angle, axis[2] * angle])

            # Update joint angles
            theta_dot_rads = self.lmbda_psfe * (np.linalg.pinv(jacobian_c_rot) @ err)
            delta_theta_rads = theta_dot_rads * self.dt
            delta_theta_degs = np.rad2deg(delta_theta_rads)
            q_current += delta_theta_degs

            # Compute new forward kinematics and camera pose in the base frame
            M_b2e_curr = self.robot.forward_kinematics(q_current)
            M_b2c_prev = M_b2e_prev @ self.M_e2c
            M_b2c_curr = M_b2e_curr @ self.M_e2c

            # Compute desired camera pose in the new camera frame
            M_c_curr2c_prev = np.linalg.inv(M_b2c_curr) @ M_b2c_prev
            R_c_des_prev = np.copy(R_c_des)
            R_c_des_curr = M_c_curr2c_prev[:3, :3] @ R_c_des
            
            # Delta in rotation - use for visualization.
            dR_c_des = R_c_des_curr @ R_c_des_prev.T
            R_c_des = np.copy(R_c_des_curr)

            # Update residual error and step counter.
            res_error = np.linalg.norm(err)
            step += 1

        # print(f"final joint config: {q_current}")
        # cut to joint limits
        ps_eul_deg = np.clip(
            q_current[0].item(),
            self.wps_eul_deg[0],
            self.wps_eul_deg[1],
        )
        fe_eul_deg = np.clip(
            q_current[1].item(),
            self.wfe_eul_deg[0],
            self.wfe_eul_deg[1],
        )

        # convert degrees to range
        wps_ref = scale_to_range(
            ps_eul_deg, 
            self.wps_eul_deg[0],
            self.wps_eul_deg[1],
            self.wps_range[0],
            self.wps_range[1],  
        )

        wfe_ref = scale_to_range(
            fe_eul_deg,
            self.wfe_eul_deg[0],
            self.wfe_eul_deg[1],
            self.wfe_range[0],
            self.wfe_range[1],  
        )        

        fingers_ref = scale_to_range(gopening.item(), 
                                    old_min=0, 
                                    old_max=self.gripper_width, 
                                    new_min=-self.fingers_range[1], 
                                    new_max=self.fingers_range[0])

        if self.always_pinch:
            if gopening.item() < self.always_pinch_thresh:
                wps_ref = 0
         
        # Convert to PS range [-100, 0] that we want for a right hand. 
        if wps_ref > 0:
            wps_ref = -wps_ref

        return wps_ref, wfe_ref, -fingers_ref + self.fing_off
    
class HannesFingersController(object):

    def __init__(self, hannes,
                    hannes_cfg,
                    fingers_cur_range=0,
                    ps_cur_range=0,
                    fe_cur_range=50):

        self.MAX_SPEED = float(hannes_cfg["fingers"]["max_speed"])
        self.OPEN_GAIN = float(hannes_cfg["fingers"]["open_gain"])
        self.CLOSE_GAIN = float(hannes_cfg["fingers"]["close_gain"])
        self.CH_RANGE = float(hannes_cfg["fingers"]["ch_range"])

        self.open_ch = int(hannes_cfg["emg"]["open_ch"])
        self.close_ch = int(hannes_cfg["emg"]["close_ch"])

        self.open_thresh = int(hannes_cfg["emg"]["open_thresh"])
        self.close_thresh = int(hannes_cfg["emg"]["close_thresh"])

        self.hannes = hannes

        self.fingers_cur_range = fingers_cur_range
        self.fingers_min_range = int(hannes_cfg["fingers"]["limits"]["range"]["min"])
        self.fingers_max_range = int(hannes_cfg["fingers"]["limits"]["range"]["max"])
        
        self.ps_cur_range = ps_cur_range
        self.fe_cur_range = fe_cur_range

        # Init the SIGINT signal handler
        def signal_handler(signal, frame):
            print('Gracefully terminating the hannes communication...')
            print('1. Pushing home configuration to Hannes...')
            self.hannes.move_wristPS(self.ps_cur_range)
            self.hannes.move_wristFE(self.fe_cur_range)
            self.hannes.move_hand(self.fingers_cur_range)
            print('2. Disconnecting from Hannes...')
            self.hannes.disconnect()
            exit(0)

        # Register the SIGINT signal handler
        signal.signal(signal.SIGINT, signal_handler)

    def update(self, channels):
        open_sig = channels[self.open_ch]
        close_sig = channels[self.close_ch]
        
        if open_sig > close_sig:    # open_sig has the priority
            if open_sig > self.open_thresh:
                speed = self.MAX_SPEED * self.OPEN_GAIN * (open_sig / self.CH_RANGE)
                self.fingers_cur_range -= speed
                self.fingers_cur_range = np.clip(
                    self.fingers_cur_range, 
                    self.fingers_min_range, 
                    self.fingers_max_range
                ).item()
                self.hannes.move_hand(int(np.round(self.fingers_cur_range)))
        else:
            if close_sig > self.close_thresh:
                speed = self.MAX_SPEED * self.CLOSE_GAIN * (close_sig / self.CH_RANGE)
                self.fingers_cur_range += speed
                self.fingers_cur_range = np.clip(
                    self.fingers_cur_range, 
                    self.fingers_min_range, 
                    self.fingers_max_range
                ).item()
                self.hannes.move_hand(int(np.round(self.fingers_cur_range)))

class HannesRobot(object):

    def __init__(self, L1: float, L2: float):
        # screw axes in the robot base frame
        ps = PoERevolute([0, 1, 0], [0, 0, 0], name="ps")
        fe = PoERevolute([0, 0, -1], [0, L1, 0], name="fe")
        # configuration of the end effector frame w.r.t. the base frame when the 
        # robot is at zero configuration
        self.M0_b2e = SE3.Ty(L1 + L2)
        self.robot = PoERobot([ps, fe], self.M0_b2e)

    def adjoint_b2e(self, M_b2e=None):
        if M_b2e is None:
            R_b2e = self.M0_b2e.R
        else:
            assert M_b2e.shape == (4, 4)
            R_b2e = M_b2e[:3, :3]

        Z = np.zeros((3, 3), dtype=R_b2e.dtype)
        return np.block([
            [R_b2e, Z],
            [Z, R_b2e]
        ])

    def jacobian_endeffector(self, q_deg):
        q_rad = np.deg2rad(q_deg)
        return self.robot.jacobe(q_rad)

    def jacobian_base(self, q_deg):
        # see PoERobot.jacob0
        
        q_rad = np.deg2rad(q_deg)
        """
        Jacobian in world frame

        :param q: joint configuration
        :type q: array_like(n)
        :return: Jacobian matrix
        :rtype: ndarray(6,n)
        """
        columns = []
        T = SE3()
        for link, qk in zip(self.robot, q_rad):
            columns.append(T.Ad() @ link.S.S)
            T *= link.S.exp(qk)
        T *= self.robot.T0
        J = np.column_stack(columns)

        # TODO lines below are copied from PoERobot.jacob0, however they seems
        #      wrong.. double-check!
        # convert Jacobian from velocity twist to spatial velocity
        #Jsv = np.eye(6)
        #Jsv[:3, 3:] = -skew(T.t)
        #return Jsv @ J

        return J

    def forward_kinematics(self, q_deg):
        q_rad = np.deg2rad(q_deg)
        return self.robot.fkine(q_rad).data[0]
