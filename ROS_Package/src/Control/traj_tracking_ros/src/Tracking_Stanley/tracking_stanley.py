from stanley_controller import StanleyTracker
import rospy
from traj_msgs.msg import Trajectory
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TransformStamped
from rc_control_msgs.msg import RCControl

import os
from IPython import display
import matplotlib.pyplot as plt
import numpy as np

from realtime_buffer import RealtimeBuffer
from copy import copy, deepcopy
import threading
from spatialmath.base import *

import cubic_spline_planner

class Course(object):
    """
    Creata lists of x, y, psi, vel using  fitting 
    based on the x, y, psi, vel of the Trajectory message
    """
    def __init__(self, msg: Trajectory):
        '''
        Decode the ros message and apply cubic interpolation 
        '''
        self.t_0 = msg.header.stamp.to_sec()
        self.dt = msg.dt
        self.step = msg.step

        # create course data
        self.x, self.y, self.yaw, k, s \
            = cubic_spline_planner.calc_spline_course(msg.x, msg.y, ds=self.dt)
    
    def calculate_target_index(self, state):
        """
        Input state of the car and calculate the index of the next point in the course
        """
        # Calc front axle position
        fx = state.x + state.L * np.cos(state.yaw)
        fy = state.y + state.L * np.sin(state.yaw)

        # Search nearest point index
        dx = [fx - icx for icx in self.x]
        dy = [fy - icy for icy in self.y]
        d = np.hypot(dx, dy)
        target_idx = np.argmin(d)

        # Project RMS error onto front axle vector
        front_axle_vec = [-np.cos(state.yaw + np.pi / 2),
                        -np.sin(state.yaw + np.pi / 2)]
        error_front_axle = np.dot([dx[target_idx], dy[target_idx]], front_axle_vec)

        return target_idx, error_front_axle
    
class Tracking_Stanley(object):
    def __init__(self,
                 pose_topic='/zed2/zed_node/pose',
                 ref_traj_topic='/planning/trajectory',
                 controller_topic='/control/rc_control',
                 vicon_pose=False
                ):

        self.vicon_pose = vicon_pose

        self.stanley_tracker = StanleyTracker()

        # objects to schedule trajectory publishing
        self.traj_buffer = RealtimeBuffer()
        self.cur_t = None
        self.cur_pose = None
        self.cur_pose_delta = None

        self.last_pub_t = None
        self.thread_lock = threading.Lock()
        self.plot_lock = threading.Lock()

        # set up subscriber to the reference trajectory and pose
        self.traj_sub = rospy.Subscriber(
            ref_traj_topic, Trajectory, self.traj_sub_callback
        )
        
        if self.vicon_pose:
            self.pose_sub = rospy.Subscriber(
                pose_topic, TransformStamped, self.pose_sub_callback
            )
        else:
            self.pose_sub = rospy.Subscriber(
                pose_topic, PoseStamped, self.pose_sub_callback
            )

        # set up publisher to the low-level ESC and servo controller
        self.control_pub = rospy.Publisher(
            controller_topic, RCControl, queue_size=1
        )

        threading.Thread(target=self.control_pub_thread).start()

    def traj_sub_callback(self, msg: Trajectory):
        """
        Subscriber callback function of the reference trajectory
        """
        course = Course(msg)
        self.trajectory_buffer.writeFromNonRT(course)

    def pose_sub_callback(self, msg: PoseStamped):
        """
        Subscriber callback function of the robot pose
        """
        # Convert the current pose msg into a SE3 Matrix
        if self.vicon_pose:
            # vicon use TransformStamped
            cur_pose = transl(msg.transform.translation.x,
                              msg.transform.translation.y,
                              msg.transform.translation.z)
            cur_pose[:3, :3] = q2r([
                msg.transform.rotation.w, msg.transform.rotation.x,
                msg.transform.rotation.y, msg.transform.rotation.z
            ])
        else:
            cur_pose = transl(msg.position.x, msg.position.y, msg.position.z)
            cur_pose[:3, :3] = q2r([
                msg.orientation.w, msg.orientation.x, msg.orientation.y,
                msg.orientation.z
            ])

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
            self.cur_pose_delta = tr2delta(prev_pose, cur_pose) / dt

        self.thread_lock.release()

    def control_pub_thread(self):
        rospy.loginfo("Contol publishing thread started")
        while not rospy.is_shutdown():
            self.thread_lock.acquire()
            cur_t = deepcopy(self.cur_t)
            cur_pose = np.array(self.cur_pose, copy=True)
            if self.cur_pose_delta is None:
                cur_pose_delta = None
            else:
                cur_pose_delta = np.array(self.cur_pose_delta, copy=True)
            self.thread_lock.release()

            if cur_pose_delta is not None:
                # get current state State: [X, Y, Vx, Vy, psi: heading, omega: yaw rate, delta: steering angle]
                cur_state = np.array([
                    cur_pose[0, -1],
                    cur_pose[1, -1],
                    max(0, cur_pose_delta[0]),
                    cur_pose_delta[1],
                    tr2rpy(cur_pose)[-1],  # heading
                    cur_pose_delta[-1],  # yaw rate
                ])

                # get the reference trajectory
                ref_traj = self.traj_buffer.readFromRT()

                if ref_traj is not None:
                    _, x_ref = ref_traj.interp_traj(cur_t, self.dt, self.N)

                    control = RCControl()
                    control.header.stamp = cur_t
                    control.throttle = min(sol_u[0, 0], 0.8)
                    control.steer = -sol_u[1, 0] / 0.35
                    control.reverse = False

                    self.control_pub.publish(control)
                    self.last_pub_t = cur_t

                    self.plot_lock.acquire()
                    self.prev_traj = x_ref
                    self.plot_lock.release()

    def run(self):
        rospy.spin()