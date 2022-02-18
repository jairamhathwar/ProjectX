#!/usr/bin/env python
"""
A simple planning node to output a constant trajectory information 
that can be used with stanley controller to control the car. The information
will be packed into traj_msg type, with the x, y be used to create the fitted cubicspline
"""
from traj_msgs.msg import Trajectory
import rospy


def publish_course():
    publisher = rospy.Publisher("/simple_trajectory_topic", Trajectory, queue_size=1)
    rospy.init_node("simple_planning_node")
    rate = rospy.Rate(10)

    ax = [0.0, -0.5, -1.2, -1.9, -2.3, -2.5]
    ay = [2.0, 2.8, 3.0, 3.2, 3.8, 4.8]

    # ay = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    # ax = [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0]
    
    while not rospy.is_shutdown():
        message = Trajectory()
        message.x = ax
        message.y = ay
        message.dt = 0.1
        publisher.publish(message)
        rate.sleep()

if __name__=="__main__":
    try:
        publish_course()
    except rospy.ROSInterruptException:
        pass
    

