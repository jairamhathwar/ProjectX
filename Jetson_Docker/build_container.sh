#!/bin/bash

# define parameters
L4T_MINOR_VERSION="5.0"
JETPACK_MAJOR="4"
JETPACK_MINOR="5"

ZED_SDK_MAJOR="3"
ZED_SDK_MINOR="5"

# build docker container for ROS master
docker build --build-arg L4T_MINOR_VERSION=${L4T_MINOR_VERSION} \
        -f Dockerfile.ros.melodic \
        -t jetson/ros-melodic:r32.${L4T_MINOR_VERSION} .

docker build --build-arg L4T_MINOR_VERSION=${L4T_MINOR_VERSION} \
        --build-arg ZED_SDK_MAJOR=${ZED_SDK_MAJOR} \
        --build-arg ZED_SDK_MINOR=${ZED_SDK_MINOR} \
        --build-arg JETPACK_MAJOR=${JETPACK_MAJOR} \
        --build-arg JETPACK_MINOR=${JETPACK_MINOR} \
        -f Dockerfile.zed.devel \
        -t jetson/zed-ros-melodic:r32.${L4T_MINOR_VERSION} .
