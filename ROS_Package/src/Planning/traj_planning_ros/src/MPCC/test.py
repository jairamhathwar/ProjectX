from time import time
import numpy as np
from matplotlib import pyplot as plt
from pyspline.pyCurve import Curve
from track import *

from mpcc_kin import MPCC


if __name__ == '__main__':
        
    env = Environment()
    env.road.add_straight(30, [1,1], None, 9)
    env.road.add_curve(50, -50, -np.pi/2, [1,1], None, 9)
    env.road.add_curve(50, -50, np.pi/2, [1,1], None, 9)
    env.road.add_straight(100, [1,1], None, 9)

    waypoints, _ = env.road.reference_traj(0, 0, ds = 10)
    
    # interp the reference route into B-spline
    interp_path = Curve(x = waypoints[:,0], y = waypoints[:,1], k = 3)
    path_length = interp_path.getLength()/100.0
    
    T = 1
    n = 20
    num_itr = 5
    planner = MPCC(T, n)
    
    x_cur = np.array([0, 0, 0, 0, 0, 0])    
    x_init = np.zeros((n,6))
    u_init = np.zeros((n,3))
    theta= x_init[:,-1]

    t0 = time()
    for _ in range(num_itr):
        ref = np.zeros((11,n))
        s = theta/path_length
        ref[:2,:] = interp_path.getValue(s).T/100.0
        ref[2, :] = theta
        ref[4,:] = 0.09
        ref[5,:] = 0.09
        ref[6:8,:] = 0
        ref[8,:] = 1
        ref[9,:] = 0
        ref[10,:] = 1
        for i in range(n):
            deri = interp_path.getDerivative(s[i])
            ref[3,i] = np.arctan2(deri[1], deri[0])
        x_init, u_init = planner.solve_itr(ref, x_cur)#, x_init, u_init)
        theta = x_init[:,-1]
    # print(x_init)
    # print(u_init)
    print(time()-t0)

    print( interp_path.projectPoint(x_init[-1,:2]*100)[0]*path_length)
    
    env.plot_env(5)
    #env.scene_ax.plot(waypoints[:,0], waypoints[:,1], '--')
    env.scene_ax.plot(x_init[:,0]*100, x_init[:,1]*100, '-.')
    theta_xy = interp_path.getValue(theta/path_length)
    env.scene_ax.plot(theta_xy[:,0], theta_xy[:,1], '*')

    plt.figure()
    x = np.arange(n)
    plt.plot(x, x_init[:,2], label = 'psi')
    plt.plot(x, x_init[:,3], label = 'v')
    plt.plot(x, x_init[:,4], label = 'delta')
    plt.plot(x, x_init[:,5], label = 'theta')
    plt.plot(x, u_init[:,0], label = 'a')

    a_lat = np.abs(x_init[:,3]*x_init[:,3]*np.tan(x_init[:,4])/(0.063)) 
    plt.plot(x, a_lat, label = 'a_lat')
    plt.legend()

    
    plt.show()

        