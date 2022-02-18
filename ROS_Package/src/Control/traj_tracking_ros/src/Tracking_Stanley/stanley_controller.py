"""
Path tracking simulation with Stanley steering control and PID speed control.
author: Atsushi Sakai (@Atsushi_twi)
Ref:
    - [Stanley: The robot that won the DARPA grand challenge](http://isl.ecst.csuchico.edu/DOCS/darpa2005/DARPA%202005%20Stanley.pdf)
    - [Autonomous Automobile Path Tracking](https://www.ri.cmu.edu/pub_files/2009/2/Automatic_Steering_Methods_for_Autonomous_Automobile_Path_Tracking.pdf)
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import cubic_spline_planner
from utils import *

class StanleyController(object):
    def __init__(self, k=5.0):
        self.k = k

    def __call__(self, state, course, last_target_idx):
        """
        Stanley steering control.
        :param state: (State object)
        :param course: (Course object)
        :param last_target_idx: (int)
        :return: (float, int)
        """
        current_target_idx, error_front_axle = course.calculate_target_index(state)

        if last_target_idx >= current_target_idx:
            current_target_idx = last_target_idx

        # theta_e corrects the heading error
        theta_e = normalize_angle(course.yaw[current_target_idx] - state.yaw)
        # theta_d corrects the cross track error
        theta_d = np.arctan2(self.k * error_front_axle, state.v)
        # Steering control
        delta = theta_e + theta_d

        return delta, current_target_idx

class PIDController(object):
    def __init__(self, Kp=1.0):
        self.Kp = Kp

    def __call__(self, target, current):
        return self.Kp * (target - current)


class StanleyTracker(object):
    """
    A StanleyTracker has a Stanley controller, a PID controller 
    and a reference trajectory (course).

    Intialize the Stanley tracker with the course, the current position 
    of the car. The tracker will figure out the closest point on the course 
    wrt the intial point, and start outputting acceleration and steering
    to drive the car toward the course and stay on the course.
    """
    def __init__(self, state_0, course, k=5.0):
        """
        Stanley tracker
        :param state_0: (CarState)
        :param course: (RefTraj)
        """
        self.stanley_controller = StanleyController(k=k)
        self.pid_controller = PIDController()
        # reference trajectory (course)
        self.course = course
        # initial target index w.r.t to the fitted course
        self.target_idx, _ = course.calculate_target_index(state_0)
    
    def __call__(self, target_velocity, actual_state):
        """
        Stanley tracker
        :param target_velocity: (float)
        :param actual_state: (CarState)
        """
        # acceleration ai
        ai = self.pid_controller(target_velocity, actual_state.v)
        # steering di
        di, self.target_idx = self.stanley_controller(
            actual_state, self.course, self.target_idx)
        return ai, di