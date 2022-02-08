from Track import Track
from MPCC import MPCC
import numpy as np
import matplotlib.pyplot as plt
import time

r = 3
theta = np.linspace(0, 2*np.pi, 100, endpoint=True)
x = r*np.cos(theta)
y = r*np.sin(theta)
track = Track(np.array([x,y]), 0.5, 0.5, True)
#print(track.length)
params_file = '/home/nvidia/Documents/PrincetonRaceCar/ROS_Package/src/Planning/traj_planning_ros/params/modelparams.yaml'

ocp_solver = MPCC(2, 20, track, params_file = params_file)


x_cur = np.array([2.53437472, 1.5608051,  2.51235582, 0.00422506, 1.65570005])

t0 = time.time()
sol_x, sol_u = ocp_solver.solve(x_cur)
print(time.time()-t0)

#print(sol_x)
#print(sol_u)


track.plot_track()
plt.plot(sol_x[0,:], sol_x[1,:])

plt.show()