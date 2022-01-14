import numpy as np
from matplotlib import pyplot as plt
from .road import *
from pyclothoids import Clothoid, SolveG2
#https://pyclothoids.readthedocs.io/en/latest/

class RoadClothoid(Road):
    def __init__(self):
        self.sections = []
        self.length_list = []
        self.n_sec = 0

    def add_straight(self, dis, num_lane, start = None, lane_width = 3.6):
        if start is None:
            if self.n_sec==0:
                start = [0,0,0]
            else:
                start = self.sections[-1].end
        end = [start[0]+np.cos(start[2])*dis, start[1]+np.sin(start[2])*dis, start[2]]
        self.sections.append(ClothoidSection(start, end, num_lane, lane_width))
        if self.n_sec == 0:
            self.length_list.append(self.sections[-1].length_list[-1])
        else:
            self.length_list.append(self.length_list[-1]+self.sections[-1].length_list[-1])
        self.n_sec += 1

    def add_curve(self, dx, dy, dtheta, num_lane, start = None, lane_width = 3.6):
        if start is None:
            if self.n_sec==0:
                    start = [0,0,0]
            else:
                    start = self.sections[-1].end
        end = [start[0]+dx, start[1]+dy, start[2]+dtheta]
        self.sections.append(ClothoidSection(start, end, num_lane, lane_width))
        if self.n_sec == 0:
            self.length_list.append(self.sections[-1].length_list[-1])
        else:
            self.length_list.append(self.length_list[-1]+self.sections[-1].length_list[-1])
        self.n_sec += 1
    
    def reference_traj(self, d_lateral, s_init=0, s_end=None, v=None, dt=None, ds=3):
        if s_end is None:
            s_end = self.length_list[-1]
        if v is not None and dt is not None:
            s_list = np.arange( s_init, s_end, v*dt)
        else:
            s_list = np.arange( s_init, s_end, ds)
            v = 0
        num_waypoint = len(s_list)
        waypoints = np.empty([num_waypoint, 5])
        idx = 0
        for (i,dis) in enumerate(s_list):
            if dis>self.length_list[idx]: idx+=1
            if idx>0:
                dis_temp = dis-self.length_list[idx-1]
            else:
                dis_temp = dis
            x, y, theta, kappa = self.sections[idx].route_to_global(dis_temp, d_lateral)
            waypoints[i,:] = [x, y, theta, kappa, v]

        return waypoints, s_list

    def plot(self, ax):
        for section in self.sections:
            section.plot(ax)

    def route_to_global(self, dis, lateral_dis):
        sec_idx = np.searchsorted(self.length_list, dis)
        if sec_idx >0:
            sec_dis = dis - self.length_list[sec_idx-1]
        else:
            sec_dis = dis
        x, y, theta, kappa = self.sections[sec_idx].route_to_global(sec_dis, lateral_dis)
        return x, y, theta, kappa




class ClothoidSection(Section):
    def __init__(self, start, end, num_lane, road_width = 3.6):
        # input start: list [x,y,theta, (kappa)] of road center
        #       end: list [x, y, theta, (kappa)] of road center

        # if no kappa provided, assume that enter/exit with 0 curvature
        if len(start)==3:
            start.append(0)
        if len(end)==3:
            end.append(0)

        self.start = start
        self.end = end
        self.num_lane = num_lane
        self.road_width = road_width

        # check if straight lane if enter and exit pose have same theta, and line between two points have same theta
        if start[2] == end[2] and np.arctan2(end[1]-start[1], end[0]-start[0]) == start[2]:
            self.clothoid_list = [Clothoid.G1Hermite(start[0], start[1], start[2], end[0], end[1], end[2])]
            self.num_clothoid = 1
            self.length_list = [self.clothoid_list[0].length]
        else:
            self.clothoid_list = SolveG2(start[0], start[1], start[2], start[3], end[0], end[1], end[2], end[3])
            self.num_clothoid = len(self.clothoid_list)
            self.length_list = np.cumsum([temp.length for temp in self.clothoid_list])
            
    def route_to_global(self, dis, lateral_dis):
        x_c, y_c, theta_c, kappa_c = self.cloest_point(dis)
        #print(lateral_dis, kappa_c)
        x = x_c+np.sin(theta_c)*lateral_dis
        y = y_c-np.cos(theta_c)*lateral_dis
        with np.errstate(divide='ignore'):
            kappa = np.float64(1.0)/(np.float64(1.0)/kappa_c+lateral_dis)
        return x, y, theta_c, kappa
            
    def cloest_point(self, dis):
        idx_prev = np.searchsorted(self.length_list, dis)
        if idx_prev == 0:
            dis_section = dis
        else:
            dis_section = dis - self.length_list[idx_prev-1]
        x = self.clothoid_list[idx_prev].X(dis_section)
        y = self.clothoid_list[idx_prev].Y(dis_section)
        theta = self.clothoid_list[idx_prev].Theta(dis_section)
        kappa = self.clothoid_list[idx_prev].KappaStart+dis_section*self.clothoid_list[idx_prev].dk
        return x,y,theta, kappa

    def plot(self, ax, ds = 0.2):
        s_list = np.arange(0, self.length_list[-1], ds)
        lane_mark_list = np.zeros((sum(self.num_lane)+1, s_list.shape[0], 4))
        for i, s in enumerate(s_list):
            x_c, y_c, theta_c, kappa_c = self.cloest_point(s)
            lane_mark_list[0,i,:] = [x_c, y_c, theta_c, kappa_c]

        # left of center line - lateral
        for l in range(self.num_lane[0]):
            lane_mark_list[l+1,:,0] = lane_mark_list[0,:,0]-np.sin(lane_mark_list[0,:,2])*(l+1)*self.road_width
            lane_mark_list[l+1,:,1] = lane_mark_list[0,:,1]+np.cos(lane_mark_list[0,:,2])*(l+1)*self.road_width
            lane_mark_list[l+1,:,2] = lane_mark_list[0,:,2]
            with np.errstate(divide='ignore'):
                lane_mark_list[l+1,:,3] = 1/(1/lane_mark_list[0,:,3]-np.sign(lane_mark_list[0,:,3])*self.road_width)

        # right of center line + lateral 
        for r in range(self.num_lane[1]):
            lane_mark_list[self.num_lane[0]+r+1,:,0] = lane_mark_list[0,:,0]+np.sin(lane_mark_list[0,:,2])*(r+1)*self.road_width
            lane_mark_list[self.num_lane[0]+r+1,:,1] = lane_mark_list[0,:,1]-np.cos(lane_mark_list[0,:,2])*(r+1)*self.road_width
            lane_mark_list[self.num_lane[0]+r+1,:,2] = lane_mark_list[0,:,2]
            with np.errstate(divide='ignore'):
                lane_mark_list[self.num_lane[0]+r+1,:,3] = 1/(1/lane_mark_list[0,:,3]+np.sign(lane_mark_list[0,:,3])*self.road_width)

        ax.plot(lane_mark_list[0,:,0], lane_mark_list[0,:,1], '--', color='yellow',linewidth=1.5)
        #ax.plot(lane_mark_list[0,:,0]-np.sin(lane_mark_list[0,:,2])*0.2, lane_mark_list[0,:,1]+np.cos(lane_mark_list[0,:,2])*0.2, '-', color='yellow',linewidth=1)
        #ax.plot(lane_mark_list[0,:,0]+np.sin(lane_mark_list[0,:,2])*0.2, lane_mark_list[0,:,1]-np.cos(lane_mark_list[0,:,2])*0.2, '-', color='yellow',linewidth=1)

        for lane in lane_mark_list[1:,:,:]:
            ax.plot(lane[:,0], lane[:,1], '--',color='gray',linewidth=1.5)





