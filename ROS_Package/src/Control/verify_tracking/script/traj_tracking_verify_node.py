#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
from IPython import display
from threading import Lock
import matplotlib.pyplot as plt
import yaml
import numpy as np

class VerifyNode:
    def __init__(self):
        self.x_traj = []
        self.y_traj = []
        self.start_listening()
        self.lock = Lock()
    
    def start_listening(self):
        rospy.init_node("traj_tracking_verify", anonymous=True)
        TrajTopic = rospy.get_param("~TrajTopic")
        PoseTopic = rospy.get_param("~PoseTopic")
        Horizon = rospy.get_param("~Horizon")
        Step = rospy.get_param("~Step")
        ParamsFile = rospy.get_param("~ParamsFile")

        rospy.Subscriber(PoseTopic, PoseStamped, self.callback)
    
    def callback(self, data):
        x_value, y_value = -data.pose.position.y, data.pose.position.x
        self.lock.acquire()
        self.x_traj.append(x_value)
        self.y_traj.append(y_value)

        while len(self.x_traj) > 200:
            self.x_traj.pop(0)
            self.y_traj.pop(0)
        self.lock.release()
    
class Bicycle_Model:
    def __init__(self, params_file, dt):
        self.dt = dt
        with open(params_file) as file:
            self.params = yaml.load(file, Loader= yaml.FullLoader)
            
    def step(self, x0, num_steps):
        x = np.zeros((4,num_steps+1))
        

if __name__ == "__main__":
    listener = VerifyNode()
    plt.ion()
    plt.show()
    plt.figure(figsize=(5, 5))

    while not rospy.is_shutdown():
        
        display.clear_output(wait = True)
        display.display(plt.gcf())
        plt.clf()
        listener.lock.acquire()
        plt.scatter(listener.x_traj, listener.y_traj)
        listener.lock.release()
        plt.xlim((-5, 5))
        plt.ylim((-5, 5))
        plt.pause(0.001)
        
        

        
