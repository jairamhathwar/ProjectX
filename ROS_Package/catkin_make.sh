#! /bin/sh
virtual_python_dir = ~/Documents/ACADOS_env/bin

# first source the python env
source $virtual_python_dir/activate

# rm old build files
rm -rf build devel
# comple the catkin with python3
catkin_make -DPYTHON_EXECUTABLE:FILEPATH=$virtual_python_dir/python

source devel/setup.bash
