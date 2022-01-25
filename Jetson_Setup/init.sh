#!/bin/bash



#cd $work_dir
sudo apt-get update 
sudo apt-get -y upgrade

# install git
sudo apt install -y git

# upgrade cmake
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
sudo apt-add-repository -y "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
sudo apt update
sudo apt install -y cmake

# install jetson stats
sudo -H pip install jetson-stats

sudo apt-get install -y python3.8-dev
#update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1


# wget https://bootstrap.pypa.io/pip/2.7/get-pip.py
# python get-pip.py --user
# rm get-pip*
sudo apt-get install -y python-pip python-numpy python-matplotlib python-scipy python-virtualenv

wget https://bootstrap.pypa.io/get-pip.py
python3.8 get-pip.py --user
rm get-pip*

# add pip3.8 to path
echo "export PATH="$HOME/.local/bin:$PATH"" >> ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"



sh install_ros.sh
sh install_zed.sh
sh install_acados.sh







