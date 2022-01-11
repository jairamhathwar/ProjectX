#!/usr/bin/env python
import rospy
from MPC import MPC
import sys, os


if __name__ == '__main__':
    print("Using the python from", 
        os.path.dirname(sys.executable))
    rospy.init_node('test_node')
    