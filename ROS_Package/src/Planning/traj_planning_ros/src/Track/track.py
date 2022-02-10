from time import time
import numpy as np
from matplotlib import pyplot as plt
from pyspline.pyCurve import Curve

class Track:
    def __init__(self, center_line=None, width_left=None, width_right=None, loop = True):
        '''
        Consider a track with fixed width
            center_line: 2D numpy array containing samples of track center line 
                        [[x1,x2,...], [y1,y2,...]]
            width_left: float, width of the track on the left side
            width_right: float, width of the track on the right side
            loop: Boolean. If the track has loop
        '''
        self.width_left = width_left
        self.width_right = width_right
        self.loop = loop

                
        if center_line is not None:
            self.center_line = Curve(x = center_line[0,:], y = center_line[1,:], k = 3)
            self.length = self.center_line.getLength()            
        else:
            self.length = None
            self.center_line = None

        # variables for plotting
        self.track_bound = None
    
    

    def _interp_s(self, s):
        '''
        Given a list of s (progress since start), return corresponing (x,y) points  
        on the track. In addition, return slope of trangent line on those points
        '''
        n = len(s)
        
        interp_pt =  self.center_line.getValue(s)
        slope = np.zeros(n)

        for i in range(n):
            deri = self.center_line.getDerivative(s[i])
            slope[i] = np.arctan2(deri[1], deri[0])
        return interp_pt.T, slope

    def interp(self, theta_list):
        '''
        Given a list of theta (progress since start), return corresponing (x,y) points  
        on the track. In addition, return slope of trangent line on those points
        '''
        if self.loop:
            s = np.remainder(theta_list, self.length)/self.length
        else:
            s = np.array(theta_list)/self.length
            s[s>1] = 1
        return self._interp_s(s)
    
    def get_closest_pts(self, points):
        '''
        Points have [2xn] shape
        '''
        s, _ = self.center_line.projectPoint(points.T, eps=1e-3)
        return self._interp_s(s)
        
    def project_point(self, point):
        s, _ = self.center_line.projectPoint(point,eps=1e-3)
        return s*self.length
        

    def plot_track(self):
        if self.track_bound is None:
            theta_sample = np.linspace(0,1,200, endpoint=False)*self.length
            interp_pt, slope = self.interp(theta_sample)
            
            self.track_bound = np.zeros((4,200))

            self.track_bound[0,:] = interp_pt[0,:] - np.sin(slope)*self.width_left
            self.track_bound[1,:] = interp_pt[1,:] + np.cos(slope)*self.width_left

            self.track_bound[2,:] = interp_pt[0,:] + np.sin(slope)*self.width_right
            self.track_bound[3,:] = interp_pt[1,:] - np.cos(slope)*self.width_right
        
        plt.plot(self.track_bound[0,:], self.track_bound[1,:], 'k-')
        plt.plot(self.track_bound[2,:], self.track_bound[3,:], 'k-')
        
    def load_from_file(self):
        pass
    







