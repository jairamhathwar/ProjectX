#! /bin/bash
virtual_python_dir=~/Documents/ACADOS_env/bin
#/home/zixu/Data/PythonVirtualEnv/ros_test/bin


# install dependence
# rosdep install --from-paths src --ignore-src -r -y

# first source the python env
source $virtual_python_dir/activate

# rm old build files
rm -rf build devel
# comple the catkin with python3
catkin_make -DPYTHON_EXECUTABLE:FILEPATH=$virtual_python_dir/python

source devel/setup.bash
