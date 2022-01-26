#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
from IPython import display
import matplotlib.pyplot as plt

class PoseNode:
    def __init__(self):
        self.x_traj = []
        self.y_traj = []
        self.start_listening()
    
    def start_listening(self):
        rospy.init_node("rc_pose_listener", anonymous=True)
        rospy.Subscriber("/zed2/zed_node/pose", PoseStamped, self.callback)
    
    def callback(self, data):
        x_value, y_value = -data.pose.position.y, data.pose.position.x
        self.x_traj.append(x_value)
        self.y_traj.append(y_value)

        while len(self.x_traj) > 50:
            self.x_traj.pop(0)
            self.y_traj.pop(0)
    
if __name__ == "__main__":
    listener = PoseNode()
    plt.ion()
    plt.show()
    plt.figure(figsize=(5, 5))

    while not rospy.is_shutdown():
        display.clear_output(wait = True)
        display.display(plt.gcf())
        plt.clf()
        
        plt.scatter(listener.x_traj, listener.y_traj)
        plt.xlim((-5, 5))
        plt.ylim((-5, 5))
        plt.pause(0.001)
