# PrincetonRaceCar

Internal private repo for the Princeton autonomous RC car

## ProjectX - Summer 2022
1. Used Stanley controller to move the robot truck
2. Vicon system for detecting box obstacles, truck, and crazyflie
3. Create 2D grid of 0s (empty) and 1s (obstacles), each node 1 square meter
4. Implemented A* algorithm for path planning
5. Convert axes and coordinate grid from vicon basis to truck basis 
6. Utilized crazyswarm to control crazyflie
7. Path planning for crazyflie using vicon as tracking system
8. Used crazyflie location to determine box visibilty for truck
9. Output visualizations of truck's path from A* and Stanley
10. Execute path for truck, avoiding obstacles

#### Launch files to use:
- `roslaunch traj_tracking_ros test_tracking_with_astar.launch` (Used with ACADOS_env)
- `roslaunch crazyswarm hover_swarm.launch` (Use chooser.py to select & reboot crazyflie)

#### Notes:
- Programming and testing conducted on Truck 6 (local repo hosted on NX-6 computer)
- Drone used: Crazyflie 231 (address `0xE7E7E7E7E7`)
- ACADOS_env location on NX-6: `~/Documents/ACADOS_env/bin/activate`
- Built upon this version of the PrincetonRaceCar repo: https://github.com/SafeRoboticsLab/PrincetonRaceCar/tree/74dab9eb5b7bbd57db69bf75651ee82e2c996ad8

#### Special Instructions for Installing Crazyswarm:
- The Crazyswarm repo can be found here: https://github.com/USC-ACTLab/crazyswarm
- Go to `ROS_Package/src/`. The `crazyswarm/` directory here is our modified version of the `USC-ACTLab/crazyswarm/ros_ws/src/crazyswarm/` directory from the Crazyswarm repo (https://github.com/USC-ACTLab/crazyswarm/tree/master/ros_ws/src/crazyswarm)
- In order to fly the drone, we need the rest of the dependencies in `USC-ACTLab/crazyswarm/ros_ws/src/` directory, such as `crazyflie_tools`
- So, we need to install and build the crazyswarm repo
1. Rename our `crazyswarm/` directory to `crazyswarm_main/` -- `mv crazyswarm crazyswarm_main`
2. Move this folder outside of the PrincetonRaceCar repo temporarily (to `Desktop/` for example)
3. Now, in `ROS_Package/src/` follow these instructions to install & build Crazyswarm: https://crazyswarm.readthedocs.io/en/latest/installation.html
4. After installation and building, the Crazyswarm repo should be located in `ROS_Package/src/crazyswarm/`
5. Go to `ROS_Package/src/crazyswarm/ros_ws/src/` and delete the `crazyswarm` directory -- we are going to replace it with our modified version `rm -r crazyswarm`
6. Now rename `ROS_Package/src/crazyswarm` to `ROS_Package/src/crazyswarm_repo` 
7. Move `crazyswarm_main/` back to `ROS_Package/src/`
8. Rename `ROS_Package/src/crazyswarm_main` to `ROS_Package/src/crazyswarm`
9. Now we can go to `ROS_Package/src/crazyswarm/scripts` to run `chooser.py` and work with drones
