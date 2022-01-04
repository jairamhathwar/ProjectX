#!/bin/bash

# install ZED sdk
sudo -H pip3 install --upgrade cython 
sudo -H pip3 install --upgrade numpy opencv-python pyopengl

wget https://stereolabs.sfo2.cdn.digitaloceanspaces.com/zedsdk/3.6/ZED_SDK_Tegra_JP46_v3.6.2.run
chmod +x ZED_SDK_Tegra_JP46_v3.6.2.run
./ZED_SDK_Tegra_JP46_v3.6.2.run

rm ZED_SDK_Tegra_JP46_v3.6.2.run
