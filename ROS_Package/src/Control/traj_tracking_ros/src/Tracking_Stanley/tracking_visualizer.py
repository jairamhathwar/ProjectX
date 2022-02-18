#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
from traj_msgs.msg import Trajectory
from IPython import display
from threading import Lock
import matplotlib.pyplot as plt
from tracking_stanley import Course

class TrackingVisualizeNode:
    def __init__(self):
        self.x_traj = []
        self.y_traj = []
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
        self.lock.acquire()
        self.x_traj.append(x_value)
        self.y_traj.append(y_value)

        while len(self.x_traj) > 100:
            self.x_traj.pop(0)
            self.y_traj.pop(0)

        self.lock.release()
    
if __name__ == "__main__":
    listener = TrackingVisualizeNode()
    plt.ion()
    plt.show()
    plt.figure(figsize=(3, 5))

    while not rospy.is_shutdown():
        display.clear_output(wait = True)
        display.display(plt.gcf())
        plt.clf()
        listener.lock.acquire()
        plt.plot(listener.course.x, listener.course.y, ".r", label="course", linewidth = 10.0)
        plt.plot(listener.x_traj, listener.y_traj, linewidth = 10.0)
        listener.lock.release()
        plt.xlim((-4, 2))
        plt.ylim((-2, 6))
        plt.pause(0.001)
        
