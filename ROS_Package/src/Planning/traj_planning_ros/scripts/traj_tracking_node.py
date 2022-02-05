#!/usr/bin/env python
import rospy
from std_msgs.msg import String
#from MPC import MPC
import sys, os


def main():
    rospy.init_node('traj_tracking_node')
    rospy.Subscriber("chatter", String, callback)
    rospy.spin()

    ## read parameters
    # TrajTopic = rospy.get_param("TrajTopic")
    # PoseTopic = rospy.get_param("PoseTopic")
    # ControllerTopic = rospy.get_param("ControllerTopic")
    # Horizon = rospy.get_param("Horizon")
    # Step = rospy.get_param("Step")

    # rospy.loginfo(Horizon)

    # tracker = MPC()

    # print("done")
def callback(data):
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)

    
if __name__ == '__main__':
    main()
    