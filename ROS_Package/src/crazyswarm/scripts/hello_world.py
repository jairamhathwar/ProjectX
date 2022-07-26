#!/usr/bin/env python
"""Takeoff-hover-land for one CF. Useful to validate hardware config."""

from pycrazyswarm import Crazyswarm


TAKEOFF_DURATION = 2.5
HOVER_DURATION = 4.0
LANDING_DURATION = 4.0

def main():
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]

    timeHelper.sleep(2)

    cf.takeoff(targetHeight=1.0, duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION)

    cf.goTo([0.0, 0.0, 1.0], yaw=0, duration=HOVER_DURATION)
    timeHelper.sleep(HOVER_DURATION)

    cf.goTo([-2.0, 0.0, 1.0], yaw=0, duration=HOVER_DURATION)
    timeHelper.sleep(HOVER_DURATION)

    cf.land(targetHeight=0.04, duration=LANDING_DURATION)
    timeHelper.sleep(LANDING_DURATION)


if __name__ == "__main__":
    main()  

