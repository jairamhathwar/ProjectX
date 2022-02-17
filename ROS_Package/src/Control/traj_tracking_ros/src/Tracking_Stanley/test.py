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
from stanley_controller import StanleyController, PIDController
from car_state import CarState
from utils import *

show_animation = True

def main():
    """Plot an example of Stanley steering control on a cubic spline."""
    stanley_controller = StanleyController()
    pid_controller = PIDController()
    #  target course
    ax = [0.0, 100.0, 100.0, 50.0, 60.0]
    ay = [0.0, 0.0, -30.0, -20.0, 0.0]

    # cx, cy, cyaw, ck, s: course information
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=0.1)

    target_speed = 30.0 / 3.6  # [m/s]

    max_simulation_time = 100.0

    # Initial state
    state = CarState(x=-0.0, y=5.0, yaw=np.radians(20.0), v=0.0)

    last_idx = len(cx) - 1
    time = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]
    target_idx, _ = calc_target_index(state, cx, cy)

    while max_simulation_time >= time and last_idx > target_idx:
        ai = pid_controller(target_speed, state.v)
        di, target_idx = stanley_controller(state, cx, cy, cyaw, target_idx)
        state.update(ai, di)

        time += state.dt

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(cx, cy, ".r", label="course")
            plt.plot(x, y, "-b", label="trajectory")
            plt.plot(cx[target_idx], cy[target_idx], "xg", label="target")
            plt.axis("equal")
            plt.grid(True)
            plt.title("Speed[km/h]:" + str(state.v * 3.6)[:4])
            plt.pause(0.001)

    # Test
    assert last_idx >= target_idx, "Cannot reach goal"

    if show_animation:  # pragma: no cover
        plt.plot(cx, cy, ".r", label="course")
        plt.plot(x, y, "-b", label="trajectory")
        plt.legend()
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.axis("equal")
        plt.grid(True)

        plt.subplots(1)
        plt.plot(t, [iv * 3.6 for iv in v], "-r")
        plt.xlabel("Time[s]")
        plt.ylabel("Speed[km/h]")
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    main()