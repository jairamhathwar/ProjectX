import numpy as np
from matplotlib import pyplot as plt
from .road import *


# abstract class for road_section
class RoadCircle(Road):
    
    def __init__(self):
        self.sections = []
        self.sec_dis = []
        self.n_sec = 0

    def add_straight(self, dis, num_lane, start = None):
        if start is None:
            if self.n_sec==0:
                    start = [0,0,0]
            else:
                    start = self.sections[-1].pos_e
                    start.append(self.sections[-1].theta_e)
        self.sections.append(StraightSection(start[:2], start[-1], dis, num_lane))
        self.n_sec += 1
        if self.n_sec is 1:
            self.sec_dis.append(self.sections[-1].sec_dis)
        else:
            self.sec_dis.append(self.sec_dis[-1]+self.sections[-1].sec_dis)

    def add_curve(self, phi, curvature, num_lane, start = None):
        if start is None:
            if self.n_sec==0:
                    start = [0,0,0]
            else:
                    start = self.sections[-1].pos_e
                    start.append(self.sections[-1].theta_e)
        self.sections.append(CurveSection(start[:2], start[-1], phi, curvature, num_lane))
        self.n_sec += 1
        if self.n_sec is 1:
            self.sec_dis.append(self.sections[-1].sec_dis)
        else:
            self.sec_dis.append(self.sec_dis[-1]+self.sections[-1].sec_dis)
    
    def reference_traj(self, d_lateral, dis_init, dis_end, v, dt):
        d_list = np.arange( dis_init, dis_end, v*dt)
        num_waypoint = len(d_list)
        waypoints = np.empty([num_waypoint, 4])
        idx = 0
        for (i,dis) in enumerate(d_list):
            if dis>self.sec_dis[idx]: idx+=1
            if idx>0:
                dis_temp = dis-self.sec_dis[idx-1]
            else:
                dis_temp = dis
            waypoint = self.sections[idx].route_to_global(dis, d_lateral)
            waypoints[i,:] = np.array([waypoint[0], waypoint[1], waypoint[2], v])
        return waypoints

    def plot(self, ax):
        for section in self.sections:
            section.plot(ax)


class StraightSection(Section):
    def __init__(self, pos_s, theta, len, num_lane):
        # represent the straight len center as ax+by+c = 0
        # pos_s = [x,y] theta = atan2(a,b) in rad
        theta = theta % (2*np.pi)

        self.theta_s = theta
        self.theta_e = theta
        self.pos_s = pos_s
        self.sec_dis = len
        self.num_lane = num_lane #[len opposite the direction of center, len follow the same direction center,]

        if abs(theta - np.pi/2)<0.001:
            a = -1
            b = 0
            c = pos_s[0]
        elif abs(theta + np.pi/2)<0.001:
            a = 1
            b = 0
            c = -pos_s[0]
        else:
            a = -np.tan(theta)
            b = 1
            c = -a*pos_s[0]-b*pos_s[1]
    
        self.pos_e = [pos_s[0]+np.cos(theta)*len, pos_s[1]+np.sin(theta)*len]
        
    def route_to_global(self, dis, lateral_dis):
        # input:
        #    dis: dis along reference route since pos_s of this section
        #    lateral_dis: lateral distance of the point on reference route. Pos mean the same direction  (right) of center line
        # output:
        #    [x,y,theta of tangent line, curvature]
        x = self.pos_s[0]+np.cos(self.theta_s)*dis+np.sin(self.theta_s)*lateral_dis
        y = self.pos_s[1]+np.sin(self.theta_s)*dis-np.cos(self.theta_s)*lateral_dis
        return [x, y, self.theta_s, 0]

    def plot(self, ax):
        ax.plot([self.pos_s[0]+np.sin(self.theta_s)*-self.num_lane[0]*3.6, self.pos_e[0]+np.sin(self.theta_s)*-self.num_lane[0]*3.6], 
                [self.pos_s[1]-np.cos(self.theta_s)*-self.num_lane[0]*3.6, self.pos_e[1]-np.cos(self.theta_s)*-self.num_lane[0]*3.6], '-', color='gray')
        for i in (min(-self.num_lane[0]+1,-1),-1):
            ax.plot([self.pos_s[0]+np.sin(self.theta_s)*i*3.6, self.pos_e[0]+np.sin(self.theta_s)*i*3.6], 
                [self.pos_s[1]-np.cos(self.theta_s)*i*3.6, self.pos_e[1]-np.cos(self.theta_s)*i*3.6], '--', color='gray')
        ax.plot([self.pos_s[0], self.pos_e[0]], 
                [self.pos_s[1], self.pos_e[1]], '-', color='gold')
        for i in (1,max(self.num_lane[1]-1,1)):
            ax.plot([self.pos_s[0]+np.sin(self.theta_s)*i*3.6, self.pos_e[0]+np.sin(self.theta_s)*i*3.6], 
                [self.pos_s[1]-np.cos(self.theta_s)*i*3.6, self.pos_e[1]-np.cos(self.theta_s)*i*3.6], '--', color='gray')
        ax.plot([self.pos_s[0]+np.sin(self.theta_s)*self.num_lane[1]*3.6, self.pos_e[0]+np.sin(self.theta_s)*self.num_lane[1]*3.6], 
                [self.pos_s[1]-np.cos(self.theta_s)*self.num_lane[1]*3.6, self.pos_e[1]-np.cos(self.theta_s)*self.num_lane[1]*3.6], '-', color='gray')

class CurveSection(Section):
    def __init__(self, pos_s, theta, phi, curvature, num_lane):
        # represent the straight len center as ax+by+c = 0
        # pos_s = [x,y] theta = atan2(a,b) in rad

        self.theta_s = theta % (2*np.pi)
        self.curvature = curvature
        self.pos_s = pos_s
        self.phi_s = theta-np.sign(curvature)*np.pi/2
        self.phi_e = self.phi_s+np.sign(curvature)*phi
        self.sec_dis = phi/curvature
        self.num_lane = num_lane #[len opposite the direction of center, len follow the same direction center,]
        self.curve_center = [self.pos_s[0]-np.sin(theta)/curvature, self.pos_s[1]+np.cos(theta)/curvature]
        self.pos_e = [self.curve_center[0]+np.cos(self.phi_e)/abs(self.curvature), self.curve_center[1]+np.sin(self.phi_e)/abs(self.curvature)]
        self.theta_e = self.phi_e+np.sign(curvature)*np.pi/2
        
    def route_to_global(self, dis, lateral_dis):
        # input:
        #    dis: dis along reference route since pos_s of this section
        #    lateral_dis: lateral distance of the point on reference route. Pos mean the same direction  (right) of center line
        # output:
        #    [x,y,theta of tangent line, curvature]

        phi = dis*self.curvature
        x = self.curve_center[0]+np.cos(self.phi_e+np.sign(self.curvature)*phi)*np.sign(self.curvature)*(1/self.curvature+lateral_dis)
        y = self.curve_center[1]+np.sin(self.phi_e+np.sign(self.curvature)*phi)*np.sign(self.curvature)*(1/self.curvature+lateral_dis)
        theta = self.phi_s+np.sign(self.curvature)*phi+np.sign(self.curvature)*np.pi/2
        return [x, y, theta, self.curvature]

    def plot(self, ax):
        phi_list = np.linspace(self.phi_s, self.phi_e, 1000, endpoint=False)
        ax.plot(self.curve_center[0]+np.cos(phi_list)/abs(self.curvature), self.curve_center[1]+np.sin(phi_list)/abs(self.curvature), color='gold')
        ax.plot(self.curve_center[0]+np.cos(phi_list)*np.sign(self.curvature)*(1/self.curvature-self.num_lane[0]*3.6), 
                self.curve_center[1]+np.sin(phi_list)*np.sign(self.curvature)*(1/self.curvature-self.num_lane[0]*3.6), color='gray')
        ax.plot(self.curve_center[0]+np.cos(phi_list)*np.sign(self.curvature)*(1/self.curvature+self.num_lane[1]*3.6), 
                self.curve_center[1]+np.sin(phi_list)*np.sign(self.curvature)*(1/self.curvature+self.num_lane[1]*3.6), color='gray')
        for i in (min(-self.num_lane[0]+1,-1),-1):
            ax.plot(self.curve_center[0]+np.cos(phi_list)*np.sign(self.curvature)*(1/self.curvature+i*3.6), 
                self.curve_center[1]+np.sin(phi_list)*np.sign(self.curvature)*(1/self.curvature+i*3.6), '--',color='gray')
        for i in (1,max(self.num_lane[1]-1,1)):
            ax.plot(self.curve_center[0]+np.cos(phi_list)*np.sign(self.curvature)*(1/self.curvature+i*3.6), 
                self.curve_center[1]+np.sin(phi_list)*np.sign(self.curvature)*(1/self.curvature+i*3.6), '--',color='gray')
        





    


    