import numpy as np
from utils import *

class CarState(object):
    """
    Class representing the state of a vehicle.
    :param x: (float) x-coordinate
    :param y: (float) y-coordinate
    :param yaw: (float) yaw angle
    :param v: (float) speed
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, dt=0.1, L=2.9, max_steer = np.radians(30.0)):
        """Instantiate the object."""
        super(CarState, self).__init__()
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.dt = dt
        self.L = L
        self.max_steer = max_steer

    def update(self, acceleration, delta):
        """
        Update the state of the vehicle.
        Stanley Control uses bicycle model.
        :param acceleration: (float) Acceleration
        :param delta: (float) Steering
        """
        delta = np.clip(delta, -self.max_steer, self.max_steer)

        self.x += self.v * np.cos(self.yaw) * self.dt
        self.y += self.v * np.sin(self.yaw) * self.dt
        self.yaw += self.v / self.L * np.tan(delta) * self.dt
        self.yaw = normalize_angle(self.yaw)
        self.v += acceleration * self.dt