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
    Horizon = rospy.get_param("~PlanHorizon")
    Step = rospy.get_param("~PlanStep")
    ParamsFile = rospy.get_param("~PlanParamsFile")
    BasePath = rospy.get_param("~PlanBasePath")
    planner = Planning_MPC(T=Horizon,
                           N=Step,
                           pose_topic=PoseTopic,
                           ref_traj_topic=TrajTopic,
                           params_file=ParamsFile,
                           base_path=BasePath)
    planner.run()


if __name__ == '__main__':
    main()
