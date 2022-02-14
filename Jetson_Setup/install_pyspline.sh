#!/bin/bash


work_dir=~/Documents
cd $work_dir
source ACADOS_env/bin/activate

pip install pyyaml rospkg empy sympy catkin_pkg

pip install spatialmath-python # pyclothoids

# install gfortran
sudo apt-get install -y gfortran
# install pyspline
git clone https://github.com/mdolab/pyspline.git
cd pyspline

cat << EOF > config/config.mk
# Config File for LINUX and GFORTRAN Compiler
AR       = ar
AR_FLAGS = -rvs
RM       = /bin/rm -rf

# Fortran compiler and flags
FF90        = gfortran
FF90_FLAGS  = -fdefault-real-8 -O2 -fPIC -std=f2008

# C compiler and flags
CC       = gcc
CC_FLAGS   = -O2 -fPIC

# Define potentially different python, python-config and f2py executables:
PYTHON = python3.8
PYTHON-CONFIG = python3.8-config # use python-config for python 2
F2PY = f2py

# Define additional flags for linking
LINKER_FLAGS = 
SO_LINKER_FLAGS =-fPIC -shared
EOF

make
pip install .
