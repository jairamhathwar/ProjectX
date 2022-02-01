#!/bin/bash

source devel/setup.bash
source jetson.sh
roslaunch rc_control rc_control_node.launch ControllerTopic:=/rc_control_node/g29_control
