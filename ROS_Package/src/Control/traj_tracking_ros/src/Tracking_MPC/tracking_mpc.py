from copy import copy
from .traj_tracking_dyn import TrajTrackingDyn
from .realtime_buffer import RealtimeBuffer

import numpy as np
from copy import deepcopy
from scipy.interpolate import CubicSpline
import rospy
import threading
# https://petercorke.github.io/spatialmath-python/intro.html
from spatialmath.base import *
from rc_control_msgs.msg import RCControl
from traj_msgs.msg import Trajectory 
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TransformStamped


class RefTraj:
    def __init__(self, msg: Trajectory):
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
        
        x_ref = np.stack([x_interp, y_interp, psi_interp, vel_interp])
        
        return t_interp, x_ref

class Tracking_MPC:
    def __init__(self, T = 1, N = 10, replan_freq = 10,
                    pose_topic = '/zed2/zed_node/pose',
                    ref_traj_topic = '/planning/trajectory',
                    controller_topic = '/control/rc_control',
                    params_file = 'modelparams.yaml'):
        
        '''
        Main class for the MPC trajectory tracking controller
        Input:
            freq: frequence to publish the control input to ESC and Servo
            T: prediction time horizon for the MPC
            N: number of integration steps in the MPC
        '''

        # parameters for the ocp solver
        self.T =T
        self.N = N
        self.dt = T/N
        
        # set up the optimal control solver
        self.ocp_solver = TrajTrackingDyn(self.T, self.N, params_file = params_file)
        rospy.loginfo("Successfully initialized the solver with horizon "+str(T)+"s, and "+str(N)+" steps.")
        
        # objects to estimate current pose
        self.prev_control = np.zeros(2)
        
        # objects to schedule trajectory publishing 
        self.traj_buffer = RealtimeBuffer()
        self.cur_t = None
        self.cur_pose = None
        self.cur_pose_delta = None

        self.last_pub_t = None
        self.replan_dt = 1/replan_freq
        self.thread_lock = threading.Lock()
        
             
        # set up subscriber to the reference trajectory and pose
        self.traj_sub = rospy.Subscriber(ref_traj_topic, Trajectory, self.traj_sub_callback)
        self.pose_sub = rospy.Subscriber(pose_topic, PoseStamped, self.pose_sub_callback)

        # set up publisher to the low-level ESC and servo controller
        self.control_pub = rospy.Publisher(controller_topic, RCControl, queue_size=1)
        
        threading.Thread(target=self.control_pub_thread).start()

    def traj_sub_callback(self, msg: Trajectory):
        """
        Subscriber callback function of the reference trajectory
        """
        ref_traj = RefTraj(msg)
        self.traj_buffer.writeFromNonRT(ref_traj)

    def pose_sub_callback(self, msg: PoseStamped):
        """
        Subscriber callback function of the robot pose
        """
        # Convert the current pose msg into a SE3 Matrix
        cur_pose = transl(msg.translation.x, msg.translation.y, msg.translation.z)
        cur_pose[:3,:3] = q2r([msg.rotation.x, 
                        msg.rotation.y, 
                        msg.rotation.z, 
                        msg.rotation.w])
        
        # cur_pose = transl(msg.position.x, msg.position.y, msg.position.z)
        # cur_pose[:3,:3] = q2r([msg.orientation.x, 
        #                 msg.orientation.y, 
        #                 msg.orientation.z, 
        #                 msg.orientation.w])
        
        self.thread_lock.acquire()
        
        # make a copy of previous state
        prev_t = deepcopy(self.cur_t)
        prev_pose = np.array(self.cur_pose, copy=True)
        self.cur_pose = cur_pose
        self.cur_t = msg.header.stamp

        if prev_t is not None:
            # approximate the velocity
            dt = (self.cur_t - prev_t).to_sec()
            
            # use tr2delta https://petercorke.github.io/spatialmath-python/func_3d.html#spatialmath.base.transforms3d.tr2delta
            # [dx, dy, dz, dthetax, dthetay, dthetaz]
            self.cur_pose_delta = tr2delta(prev_pose, cur_pose)/dt
            
        self.thread_lock.release()
            
            
        
    def control_pub_thread(self):
        rospy.loginfo("Contol publishing thread started")
        while not rospy.is_shutdown():
            # determine if we need to publish
            self.thread_lock.acquire()
            
            since_last_pub = self.replan_dt if self.last_pub_t is None else (self.cur_t-self.last_pub_t).to_sec() 
            if since_last_pub >= self.replan_dt:
                # make a copy of the data
                cur_t = deepcopy(self.cur_t)
                cur_pose = np.array(self.cur_pose, copy=True)
                cur_pose_delta = np.array(self.cur_pose_delta, copy=True)      
                      
            self.thread_lock.release()
            
            if since_last_pub >= self.replan_dt and cur_t is not None:
                start_time = rospy.get_rostime()
                 # get current state State: [X, Y, Vx, Vy, psi: heading, omega: yaw rate, delta: steering angle]
                cur_state = np.array([cur_pose[0,-1], cur_pose[1,-1], 
                              cur_pose_delta[0], cur_pose_delta[1], 
                              tr2rpy(cur_pose)[-1], # heading
                              cur_pose_delta[-1], # yaw rate
                              self.prev_control[1] # steering angle ( assume the servo will reach that position)
                              ])
            
                # get the reference trajectory
                ref_traj = self.traj_buffer.readFromRT()
                _, x_ref = ref_traj.interp_traj(self, cur_t, self.dt, self.N)
                
                # solve the ocp
                sol_x, sol_u = self.ocp_solver.solve(x_ref, cur_state)
                end_time = rospy.get_rostime()
                # form the new control output by taking the first
                control = RCControl()
                control.header.stamp = cur_t
                control.throttle = sol_u[0,0]
                control.steer = sol_x[0,-1]+sol_u[0,1]*since_last_pub  
                control.reverse = False
                
                self.control_pub.publish(control)
                self.prev_control = [control.throttle, control.steer]
                self.last_pub_t = cur_t
                rospy.loginfo("Use "+str((end_time-start_time).to_sec())+" to solve")
            
        

    
