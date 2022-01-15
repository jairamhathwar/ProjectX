from time import time
import numpy as np
from matplotlib import pyplot as plt
from pyspline.pyCurve import Curve
from track import *

from mpcc_kin import MPCC


if __name__ == '__main__':
        
    env = Environment()
    env.road.add_straight(100, [1,1], None, 9)
    env.road.add_curve(50, 50, np.pi/2, [1,1], None, 9)
    env.road.add_straight(100, [1,1], None, 9)

    waypoints, _ = env.road.reference_traj(4.5, 0, ds = 10)
    
    # interp the reference route into B-spline
    interp_path = Curve(x = waypoints[:,0], y = waypoints[:,1], k = 3)
    path_length = interp_path.getLength()/100.0
    
    T = 2
    n = 10
    num_itr = 1
    planner = MPCC(T, n)
    
    x_cur = np.array([0, -0.045, ])    
    x_init = np.zeros((n,7))
    u_init = np.zeros((n,3))
    theta= np.zeros(n)
    
    for _ in range(num_itr):
        ref = np.zeros((11,n))
        s = theta/path_length
        ref[:2,:] = interp_path.getValue(s).T/100.0
        ref[2, :] = theta
        ref[4,:] = 9
        ref[5,:] = 9
        ref[6:8,:] = 0
        ref[8,:] = 1
        ref[9,:] = 0
        ref[10,:] = 1
        for i in range(n):
            deri = interp_path.getDerivative(s[i])
            ref[3,i] = 0 #np.arctan2(deri[1], deri[0])
        print(ref[:4,:])        
        x_init, u_init = planner.solve(ref, x_cur, x_init, u_init)
        theta = x_init[:,-1]
     
    
    env.plot_env(5)
    #env.scene_ax.plot(waypoints[:,0], waypoints[:,1], '--')
    env.scene_ax.plot(x_init[:,0], x_init[:,1], '-')
    
    
    plt.show()