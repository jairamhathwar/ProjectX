import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from .road_circle import RoadCircle
from .road_clothoid import RoadClothoid
from .obstacles import *
import os, imageio, time, shutil
from pygifsicle import optimize





class Environment:

    def __init__(self):
        self.road = RoadClothoid()
        self.static_obs = []
        self.n_static_obs = 0
        self.dynamic_obs = []
        self.n_dynamic_obs = 0
        self.scene_ax = None
        self.dynamic_obs_handle = []
        self.road_bound = None

    def add_road_bound(self, dis1=-3.6, dis2=3.6, road_bound = None):
        # road_bound = [bound1, bound2,...]
        # bound1 = [[x1,y1],[x2,y2],...]

        if road_bound is not None:
            self.road_bound = road_bound
        else:
            bound1, _ = self.road.reference_traj(dis1)
            bound2, _ = self.road.reference_traj(dis2)
            self.road_bound = [bound1, bound2]
    
    def add_obstacle_static(self, s, l, width, length):
        # input pos_sl =[s,l]
        x, y, theta,_ =  self.road.route_to_global(s, l)
        pos_xyt = [x, y, theta]
        self.static_obs.append(Obstacle_static(s, l, pos_xyt, width, length))
        self.n_static_obs += 1
    
    def add_dynamic_obstacle(self, s_init, s_end, l, v, width, length, dt):
        waypoints, s_list = self.road.reference_traj(l, s_init=s_init, s_end=s_end, v=v, dt=dt)
        self.dynamic_obs.append(Obstacle_dynamics(s_list, l, v, waypoints, width, length, dt))

    def get_vertex_position(self, vertex, pose):
        #pose [x,y, theta]
        T = np.array([[np.cos(pose[2]), -np.sin(pose[2]), pose[0]],
                        [np.sin(pose[2]), np.cos(pose[2]), pose[1]],
                        [0,0,1]])
        return np.matmul(T, vertex)

    def simulate_env(self, T, dt, ego = None, plan_set = None, valid_plan_set = None, save_gif = True, folder = None, filename=None):
        if self.scene_ax is None:
            _, (self.scene_ax, self.v_ax) = plt.subplots(2)
            self.scene_ax.axis('equal')
            self.road.plot(self.scene_ax)
            if self.road_bound is not None:
                for bound in self.road_bound:
                    self.scene_ax.plot(bound[:,0], bound[:,1], '-', color='black',linewidth=1)
            for obs in self.static_obs:
                obs.plot(self.scene_ax)
            mng = plt.get_current_fig_manager()
            mng.resize(*mng.window.maxsize())
            self.v_ax.set_ylim(5,40)
            self.v_ax.set_xlim(-1, valid_plan_set[-1][1][-1]+1)
            self.v_ax.set_xlabel('t [s]')
            self.v_ax.set_ylabel('v [m/s]')
        if ego is None:
            t_list = np.arange(0, T, dt)
            for t in t_list:
                self.plot_env(t,dt)
        else:
            if save_gif:
                if folder is None:
                    folder  = '/hdd/Git_Repo/CarSimPy/test_results/'
                temp_folder = folder+'temp'
                if not os.path.exists(temp_folder):
                    os.makedirs(temp_folder)
                
                itr  = 0
            
            for plan, valid_plan in zip(plan_set, valid_plan_set):
                waypoints_full = plan[3]
                v_list = valid_plan[0]
                t_list = valid_plan[1]
                waypoints = valid_plan[2]
                self.scene_ax.plot(waypoints_full[:,0], waypoints_full[:,1])
                self.v_ax.plot(t_list, v_list)
                for t, waypoint in zip(t_list, waypoints):
                    self.plot_env_ego(t, dt, ego, waypoint)
                    if save_gif:
                        plt.savefig(temp_folder+'/temp_{i}.jpg'.format(i=itr))
                        itr += 1

            if save_gif:
                if filename is None:
                    gif_path = folder+'test_{i}.gif'.format(i = time.strftime("%Y%m%d-%H%M%S"))
                else:
                    gif_path = folder+filename+'.gif'
                with imageio.get_writer(gif_path, mode='I') as writer:
                    for i in range(itr):
                        writer.append_data(imageio.imread(temp_folder+'/temp_{i}.jpg'.format(i=i)))
                #optimize(gif_path)
                shutil.rmtree(temp_folder)


    def plot_env(self, t,dt = None):
        if self.scene_ax is None:
            self.scene_ax = plt.axes()
            self.scene_ax.axis('equal')
            self.road.plot(self.scene_ax)
            if self.road_bound is not None:
                for bound in self.road_bound:
                    self.scene_ax.plot(bound[:,0], bound[:,1], '-', color='black',linewidth=1)
            for obs in self.static_obs:
                obs.plot(self.scene_ax)
        self.scene_ax.patches[len(self.static_obs):] = []
        for (i,obs) in enumerate(self.dynamic_obs):
            obs.plot(self.scene_ax, t)
        if dt is not None:
            plt.draw()
            plt.pause(dt/2)


    def plot_env_ego(self, t, dt, ego, waypoint):
        if self.scene_ax is None:
            _, (self.scene_ax, self.v_ax) = plt.subplots(2,1)
            self.scene_ax.axis('equal')
            self.road.plot(self.scene_ax)
            if self.road_bound is not None:
                for bound in self.road_bound:
                    self.scene_ax.plot(bound[:,0], bound[:,1], '-', color='black',linewidth=1)
            for obs in self.static_obs:
                obs.plot(self.scene_ax)
        self.scene_ax.patches[len(self.static_obs):] = []
        for (i,obs) in enumerate(self.dynamic_obs):
            obs.plot(self.scene_ax, t)
        ego.plot(self.scene_ax, waypoint)
        plt.draw()
        plt.pause(dt/2)
        




