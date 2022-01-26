#!/usr/bin/env python
import rospy
from rc_control_msgs.msg import RCControl
from geometry_msgs.msg import Twist

class ControlNode:
    def __init__(self):

        rospy.init_node("keyboard_control_listener", anonymous=True)

        self.pub = rospy.Publisher("/control/servo_control", RCControl, queue_size=50)
        self.sub = rospy.Subscriber("/cmd_vel", Twist, self.callback)
        rospy.spin()
        
    def callback(self, data):
        msg2pub = RCControl()
        msg2pub.header.stamp = rospy.get_rostime()
        msg2pub.throttle = data.linear.x
        msg2pub.steer = data.angular.z
        self.pub.publish(msg2pub)
    

if __name__ == "__main__":
    talker = ControlNode()
