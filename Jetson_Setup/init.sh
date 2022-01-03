#!/bin/bash

# set cur)dir
cur_dir=$PWD
# work_dir 
work_dir=~/Documents

#cd $work_dir
sudo apt-get update 
sudo apt-get -y upgrade

# install git
sudo apt install -y git-all

# install python related packages
#sudo apt-get remove python-*
#sudo apt autoremove -y

sudo apt-get install python3.8-dev
#update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1


# wget https://bootstrap.pypa.io/pip/2.7/get-pip.py
# sudo -H python get-pip.py 
# rm get-pip*

wget https://bootstrap.pypa.io/get-pip.py
sudo -H python3.8 get-pip.py
rm get-pip*

sudo -H pip3 install matplotlib scipy numpy virtualenv pyyaml

# install jetson stats
sudo -H pip3 install -U jetson-stats

# sh $cur_dir/install_ros.sh
# sh $cur_dir/install_zed.sh

# Install controller dependence
sudo apt-get install libusb-1.0-0-dev mono-runtime libmono-system-windows-forms4.0-cil -y


cd $cur_dir



