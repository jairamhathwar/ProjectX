#!/usr/bin/env python
"""Takeoff-hover-land for one CF. Useful to validate hardware config."""

from pycrazyswarm import Crazyswarm


TAKEOFF_DURATION = 2.5
HOVER_DURATION = 3.0


def main():
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]

    cf.takeoff(targetHeight=1.0, duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION)

    cf.goTo([1.0, 0.0, 1.0], yaw=0, duration=HOVER_DURATION)
    timeHelper.sleep(HOVER_DURATION)

    cf.land(targetHeight=0.04, duration=2.5)
    timeHelper.sleep(TAKEOFF_DURATION)


if __name__ == "__main__":
    main()
