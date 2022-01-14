import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc, rcParams
import matplotlib.font_manager
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)



from carsim import *
import timeit
import pickle, time, os

rcParams.update({'font.size': 16})
rcParams['mathtext.fontset'] = 'cm'
rcParams['mathtext.rm'] = 'serif'
rcParams['font.family'] = 'sans-serif'

def plot_env(env, ax, t=None):
    env.road.plot(ax)
    if env.road_bound is not None:
        for bound in env.road_bound:
            ax.plot(bound[:,0], bound[:,1], '-', color='black',linewidth=1)
    for obs in env.static_obs:
        obs.plot(env.scene_ax)
    if t is not None:
        for (i,obs) in enumerate(env.dynamic_obs):
            obs.plot(env.scene_ax, t)

if __name__ == '__main__':
    folder  = os.getcwd()+'/test_results/'
    filename = 'test_20201206-234212'
    infile = open(folder+filename+'.pkl','rb')
    plan_set = pickle.load(infile)
    infile.close()
    
    ego = Vehicle(1.8, 4.9, 5, 35)
    env = Environment()
    env.road.add_straight(50, [1,1])
    env.road.add_curve(50, 30, np.pi/3, [1,1])
    env.road.add_curve(50, 30, -np.pi/3, [1,1])
    env.road.add_curve(50, -30, -np.pi/3, [1,1])
    env.road.add_curve(50, -30, np.pi/3, [1,1])
    #env.road.add_straight(100, [1,1])
    #env.road.add_curve(-100, -40, -np.pi/2, [1,1])
    #env.road.add_straight(150, [1,1])
    env.add_road_bound(dis1=-5, dis2=5)
    env.add_dynamic_obstacle(40, 250, 1.8, 20, 2.5, 20, 0.1)

    t_init = 0
    replan_T = 2.5
    valid_traj = np.empty((0, 7)) #[[t,s,l,v,x,y,t]]
    for i, (s_list,  v_list, t_list, waypoints, l_list) in enumerate(plan_set):
        if i < len(plan_set)-1:
            idx_replan = np.where(t_list==(t_init+replan_T))[0][0]
        else:
            idx_replan = -1
        t_init += replan_T
        cur_plan = t_list[:idx_replan].reshape((idx_replan,1))
        cur_plan = np.append(cur_plan,s_list[:idx_replan].reshape((idx_replan,1)), axis=1)
        cur_plan = np.append(cur_plan,l_list[:idx_replan].reshape((idx_replan,1)), axis=1)
        cur_plan = np.append(cur_plan,v_list[:idx_replan].reshape((idx_replan,1)), axis=1)
        cur_plan = np.append(cur_plan,waypoints[:idx_replan,:], axis=1)
        valid_traj = np.append(valid_traj, cur_plan, axis=0)

    valid_traj = valid_traj[:101,:]

    fig = plt.figure()
    gs = fig.add_gridspec(12, 1)
    ax_road = fig.add_subplot(gs[0:7, :])
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    
    plot_env(env, ax_road)
    #ax_road.plot(valid_traj[:,4], valid_traj[:,5], 'k')
    #print(valid_traj.shape)
    for i, waypoint in enumerate(valid_traj):
        if i%10 is 0:
            alpha = 0.7*(i/100)+0.15
            t = waypoint[0]
            for obs in env.dynamic_obs:
                obs.plot(ax_road, t,alpha=alpha)
            ego.plot(ax_road, waypoint[4:], alpha=alpha)

    #ax_road.axis('equal')
    ax_road.set_ylim((-10,70))
    ax_road.set_xlim((0,250))
    ax_road.set_xlabel('$X$ [m]', fontsize=18)
    ax_road.set_ylabel('$X$ [m]', fontsize=18)

    _, (ax_s, ax_l, ax_v) = plt.subplots(3,1,sharex=True, constrained_layout=True)
    ax_s.plot(valid_traj[:,0], valid_traj[:,1]-(40+20*valid_traj[:,0]), color='black')
    ax_l.plot(valid_traj[:,0], -valid_traj[:,2], color='black')
    ax_v.plot(valid_traj[:,0], valid_traj[:,3], color='black')
    ax_s.set_ylabel('$\Delta s$ [m]')
    ax_l.set_ylabel('Lat. Pos [m]')
    ax_l.set_ylim((-3,3))
    ax_v.set_ylabel('$v_{ego}$ [m/s]')
    ax_v.set_xlabel('$t$ [s]')
    ax_v.set_xlim((-0.1, 10.1))
    #ax_l.yaxis.set_minor_locator(MultipleLocator(5))
    #ax_l.xaxis.set_minor_locator(MultipleLocator(5))
    plt.show()

