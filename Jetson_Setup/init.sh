#!/bin/bash

# set cur)dir
cur_dir=$PWD
# work_dir 
work_dir=~/Documents

cd $work_dir
sudo apt-get update 
sudo apt-get -y upgrade

# install git
sudo apt install -y git-all

# install python related packages
sudo apt-get install python-pip python3-dev python3-pip -y
sudo -H pip2 install --upgrade pip

wget https://bootstrap.pypa.io/get-pip.py
sudo -H python3 get-pip.py
rm get-pip*

sudo -H pip3 install matplotlib scipy numpy virtualenv pyyaml

# install jetson stats
sudo -H pip install -U jetson-stats

sh $cur_dir/install_ros.sh
sh $cur_dir/install_zed.sh

# Install controller dependence
sudo apt-get install libusb-1.0-0-dev mono-runtime libmono-system-windows-forms4.0-cil -y


cd $cur_dir

