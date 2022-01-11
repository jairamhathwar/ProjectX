from .traj_tracking_dyn import TrajTrackingDyn
from .traj_tracking_kin import TrajTrackingKin
from .realtime_buffer import RealtimeBuffer
import numpy as np
from scipy.interpolate import CubicSpline
from rc_control_msgs.msg import RCControl
from traj_msgs.msg import Trajectory 
from geometry_msgs.msg import PoseStamped
import rospy
from queue import Queue


class RefTraj:
    def __init__(self, msg) -> None:
        '''
        Decode the ros message and apply cubic interpolation 
        '''
        self.t_0 = msg.header.stamp.to_sec()
        self.dt = msg.dt
        self.step = msg.step
        
        # discrete time step
        self.t = np.linspace(0, self.step*self.dt, self.step, endpoint=False) # unit of second
        
        self.x = CubicSpline(self.t, np.array(msg.x))
        self.y = CubicSpline(self.t, np.array(msg.y))
        
        self.psi = CubicSpline(self.t, np.array(msg.psi))
        self.vel = CubicSpline(self.t, np.array(msg.vel))
        
    def interp_traj(self, t_1, dt, n):
        t_interp = np.arange(n)*dt+(t_1.to_sec()-self.t_0)
        x_interp = self.x(t_interp)
        y_interp = self.y(t_interp)
        psi_interp = self.psi(t_interp)
        vel_interp = self.vel(t_interp)
        
        return t_interp, x_interp, y_interp, psi_interp, vel_interp

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
        self.traj_buffer = RealtimeBuffer()
        
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

    
