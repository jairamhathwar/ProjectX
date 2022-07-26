#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped
from traj_msgs.msg import Trajectory
from IPython import display
from threading import Lock
import matplotlib.pyplot as plt
from tracking_stanley import Course
from scipy.spatial.transform import Rotation
import math

class TrackingVisualizeNode:
    def __init__(self):
        self.x_traj = []
        self.y_traj = []
        self.directional_arrow = [0.0, 0.0] # x, y
        self.start_listening()
        self.lock = Lock()
    
    def start_listening(self):
        rospy.init_node("tracking_visualize_node", anonymous=True)
        rospy.Subscriber("/zed2/zed_node/pose", PoseStamped, self.callback_pose)
        rospy.Subscriber("/simple_trajectory_topic", Trajectory, self.callback_trajectory)
    
    def callback_trajectory(self, data):
        self.lock.acquire()
        self.course = Course(data)
        self.lock.release()

    def callback_pose(self, data):
        x_value, y_value = -data.pose.position.y, data.pose.position.x
        
        r = Rotation.from_quat([
            data.pose.orientation.x, data.pose.orientation.y,
            data.pose.orientation.z, data.pose.orientation.w
        ])
        
        rot_vec = r.as_rotvec()
        current_yaw = rot_vec[2] + math.pi * 0.5

        endx = x_value + math.cos(current_yaw)
        endy = y_value + math.sin(current_yaw)

        self.lock.acquire()
        self.x_traj.append(x_value)
        self.y_traj.append(y_value)
        self.directional_arrow = [endx, endy]

        while len(self.x_traj) > 100:
            self.x_traj.pop(0)
            self.y_traj.pop(0)

        self.lock.release()
    
if __name__ == "__main__":
    listener = TrackingVisualizeNode()
    plt.ion()
    plt.show()
    plt.figure(figsize=(3, 5))

    rospy.wait_for_message("/simple_trajectory_topic", Trajectory)
    while not rospy.is_shutdown():
        display.clear_output(wait = True)
        display.display(plt.gcf())
        plt.clf()
        listener.lock.acquire()
        plt.plot(listener.course.x, listener.course.y, ".r", label="course", linewidth = 10.0)
        plt.plot(listener.x_traj, listener.y_traj, linewidth = 5.0, alpha=0.6)
        if len(listener.x_traj) > 0 and len(listener.directional_arrow) > 0 :
            plt.plot([listener.x_traj[-1], listener.directional_arrow[0]], [listener.y_traj[-1], listener.directional_arrow[1]], "g", alpha = 0.2)
        listener.lock.release()
        plt.xlim((-4, 2))
        plt.ylim((-2, 6))
        plt.pause(0.001)
        
