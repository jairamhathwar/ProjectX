from .traj_tracking_dyn import TrajTrackingDyn
from .traj_tracking_kin import TrajTrackingKin
import numpy as np
from scipy.interpolate import CubicSpline
from rc_control_msgs.msg import RCControl
from traj_msgs.msg import Trajectory 
from geometry_msgs.msg import PoseStamped
import rospy
from queue import Queue

class MPC:
    def __init__(self, T = 1, N = 10,
                    dyn_model = True, 
                    pose_topic = '/zed2/zed_node/pose',
                    ref_traj_topic = 'planning/trajectory',
                    controller_topic = 'control/servo_control',
                    params_file = 'modelparams.yaml'):
        
        '''
        Main class for the MPC trajectory tracking controller
        Input:
            freq: frequence to publish the control input to ESC and Servo
            T: prediction time horizon for the MPC
            N: number of integration steps in the MPC
        '''

        self.T =T
        self.N = N
        
        # set up the optimal control solver

        if dyn_model:
            self.ocp_solver = TrajTrackingDyn(self.T, self.N, params_file = params_file)
        else:
            self.ocp_solver = TrajTrackingKin(self.T, self.N, params_file = params_file)
        
        # set up subscriber to the reference trajectory and pose
        self.traj_sub = rospy.Subscriber(ref_traj_topic, Trajectory, self.traj_sub_callback)
        self.pose_sub = rospy.Subscriber(pose_topic, PoseStamped, self.pose_sub_callback)

        # set up publisher to the low-level ESC and servo controller
        self.control_pub = rospy.Publisher(controller_topic, RCControl, queue_size=10)

    def interp_traj(self):
        pass

    def traj_sub_callback(self):
        pass

    def pose_sub_callback(self):
        pass

    def publish_control(self):
        pass

    
