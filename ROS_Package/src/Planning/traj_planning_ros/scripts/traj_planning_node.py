#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from planning import Planning_MPC
import sys, os


def main():
    rospy.init_node('traj_planning_node')
    rospy.loginfo("Start trajectory planning node")
    ## read parameters
    TrajTopic = rospy.get_param("~TrajTopic")
    PoseTopic = rospy.get_param("~PoseTopic")
    Horizon = rospy.get_param("~Horizon")
    Step = rospy.get_param("~Step")
    ParamsFile = rospy.get_param("~ParamsFile")

    planner = Planning_MPC(T= Horizon, N = Step,
                    pose_topic = PoseTopic,
                    ref_traj_topic = TrajTopic,
                    params_file = ParamsFile)
    planner.run()

if __name__ == '__main__':
    main()
    