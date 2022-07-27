#!/usr/bin/env python
"""
This is the drone flying node for the Project X demo.
Executes a simple point A to B trajectory for one crazyflie.
"""

from pycrazyswarm import Crazyswarm


TAKEOFF_DURATION = 2.5
HOVER_DURATION = 15
LANDING_DURATION = 4.0

def main():
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]

    timeHelper.sleep(2)

    cf.takeoff(targetHeight=1.0, duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION)

    # cf.goTo([-2.0, -2.0, 1.0], yaw=0, duration=HOVER_DURATION)  # Diagonal route
    cf.goTo([-2.0, 0.25, 1.0], yaw=0, duration=HOVER_DURATION)  # Center route
    timeHelper.sleep(HOVER_DURATION)

    cf.land(targetHeight=0.04, duration=LANDING_DURATION)
    timeHelper.sleep(LANDING_DURATION)


if __name__ == "__main__":
    main()  