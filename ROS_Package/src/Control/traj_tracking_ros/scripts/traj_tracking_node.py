#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from Tracking_MPC import Tracking_MPC
import sys, os


def main():
    rospy.init_node('traj_tracking_node')

    ## read parameters
    TrajTopic = rospy.get_param("~TrajTopic")
    PoseTopic = rospy.get_param("~PoseTopic")
    ControllerTopic = rospy.get_param("~ControllerTopic")
    Horizon = rospy.get_param("~Horizon")
    Step = rospy.get_param("~Step")
    ParamsFile = rospy.get_param("~ParamsFile")

    tracker = Tracking_MPC(T= Horizon, N = Step, replan_freq = 50,
                    pose_topic = PoseTopic,
                    ref_traj_topic = TrajTopic,
                    controller_topic = ControllerTopic,
                    params_file = ParamsFile)
    rospy.spin()
if __name__ == '__main__':
    main()
    