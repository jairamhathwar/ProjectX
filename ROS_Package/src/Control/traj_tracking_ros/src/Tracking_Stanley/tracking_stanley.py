#!/usr/bin/env python3
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
# from spatialmath.base import *

import cubic_spline_planner
from car_state import CarState
import math
from scipy.spatial.transform import Rotation

class Course(object):
    """
    Creata lists of x, y, psi, vel using  fitting 
    based on the x, y, psi, vel of the Trajectory message
    """
    # def __init__(self, msg: Trajectory):
    def __init__(self, msg):
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
                 course_topic='/simple_trajectory_topic',
                 controller_topic='/control/rc_control',
                 vicon_pose=False
                ):

        rospy.init_node("stanley_tracker_node", anonymous=True)

        self.vicon_pose = vicon_pose
        self.stanley_tracker = None
        self.course = None

        # self.target_speed = 30.0 / 3.6  # [m/s]
        self.target_speed = 1.0 # [m/s]

        self.current_pose = None
        self.current_pose_delta = None
        self.current_yaw = None
        self.current_t = None
        self.last_pub_t = None

        # tuning params
        self.stanley_k=2.0

        self.thread_lock = threading.Lock()

        # set up subscriber to the reference trajectory and pose
        self.course_sub = rospy.Subscriber(
            course_topic, Trajectory, self.course_sub_callback
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

    # def course_sub_callback(self, msg: Trajectory):
    def course_sub_callback(self, msg):
        """
        Subscriber callback function of the reference trajectory
        """
        self.thread_lock.acquire()
        self.course = Course(msg)
        self.thread_lock.release()

    # def pose_sub_callback(self, msg: PoseStamped):
    def pose_sub_callback(self, msg):
        """
        Subscriber callback function of the robot pose
        """
        # Convert the current pose msg into a SE3 Matrix
        if self.vicon_pose:
            # vicon use TransformStamped
            raise NotImplementedError
            # current_pose = transl(
            #     msg.transform.translation.x,
            #     msg.transform.translation.y,
            #     msg.transform.translation.z
            # )
            
            # current_pose[:3, :3] = q2r([
            #     msg.transform.rotation.w, msg.transform.rotation.x,
            #     msg.transform.rotation.y, msg.transform.rotation.z
            # ])
        else:
            # current_pose = transl(msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
            # current_pose[:3, :3] = q2r([
            #     msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y,
            #     msg.pose.orientation.z
            # ])
            x = -msg.pose.position.y
            y = msg.pose.position.x
            z = msg.pose.position.z

            current_pose = np.array([x, y, z])
            
            r = Rotation.from_quat([
                msg.pose.orientation.x, msg.pose.orientation.y,
                msg.pose.orientation.z, msg.pose.orientation.w
            ])
            
            rot_vec = r.as_rotvec()
            current_yaw = rot_vec[2] + math.pi * 0.5

        self.thread_lock.acquire()

        # make a copy of previous state
        previous_t = deepcopy(self.current_t)
        previous_pose = np.array(self.current_pose, copy=True)
        self.current_pose = current_pose
        self.current_t = msg.header.stamp
        self.current_yaw = current_yaw

        if previous_t is not None:
            # approximate the velocity
            dt = (self.current_t - previous_t).to_sec()

            # use tr2delta https://petercorke.github.io/spatialmath-python/func_3d.html#spatialmath.base.transforms3d.tr2delta
            # [dx, dy, dz, dthetax, dthetay, dthetaz]
            # self.current_pose_delta = tr2delta(previous_pose, current_pose) / dt
            self.current_pose_delta = (current_pose - previous_pose) / dt
        
        if self.stanley_tracker is None and self.course is not None:
            car_state = CarState(
                x = self.current_pose[0], 
                y = self.current_pose[1],
                yaw = self.current_yaw
            )
            self.stanley_tracker = StanleyTracker(car_state, self.course, k=self.stanley_k)

        self.thread_lock.release()

    def control_pub_thread(self):
        while not rospy.is_shutdown():
            self.thread_lock.acquire()
            current_t = deepcopy(self.current_t)
            current_pose = np.array(self.current_pose, copy=True)
            if self.current_pose_delta is None:
                current_pose_delta = None
            else:
                current_pose_delta = np.array(self.current_pose_delta, copy=True)
            
            current_yaw = self.current_yaw
            self.thread_lock.release()
            
            if current_pose_delta is not None and current_pose is not None:
                if self.stanley_tracker is not None:
                    vel = math.sqrt(
                        current_pose_delta[0]*current_pose_delta[0] + 
                        current_pose_delta[1]*current_pose_delta[1]
                    )

                    car_state = CarState(
                        x = current_pose[0], 
                        y = current_pose[1],
                        yaw = current_yaw,
                        v = vel
                    )

                    acceleration, steering = self.stanley_tracker(
                                                    self.target_speed, car_state)
                    
                    control = RCControl()
                    control.header.stamp = current_t
                    #! MAP VALUE OF STANLEY OUTPUT TO THROTTLE AND STEERING
                    control.throttle = 0.1 * abs(acceleration)
                    control.steer = np.clip(-steering / np.radians(40.0), -1, 1)
                    if acceleration < 0.0:
                        control.reverse = True
                    else:
                        control.reverse = False

                    self.control_pub.publish(control)
                    self.last_pub_t = current_t

    def run(self):
        rospy.spin()

if __name__=="__main__":
    stanley_tracker_node = Tracking_Stanley()
    stanley_tracker_node.run()