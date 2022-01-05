#!/usr/bin/env python
import rospy
from rc_control_msgs.msg import RCControl
import casadi
import sys, os


if __name__ == '__main__':
    print("Using the python from", 
        os.path.dirname(sys.executable))
    rospy.init_node('test_node')
    