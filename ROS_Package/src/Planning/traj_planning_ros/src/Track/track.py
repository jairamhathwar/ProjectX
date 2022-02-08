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

                
        if center_line:
            self.center_line = Curve(x = center_line[0,:], y = center_line[1,:], k = 3)
            self.length = self.center_line.getLength()            
        else:
            self.length = None
            self.center_line = None

        # variables for plotting
        self.figure = None
        self.track_bound = None

    def interp(self, theta_list):
        '''
        Given a list of theta (progress since start), return corresponing (x,y) points  
        on the track. In addition, return slope of trangent line on those points
        '''
        if self.loop:
            s = np.remainder(theta_list, self.length)
        else:
            s = np.array(theta_list)/self.length
            s[s>1] = 1
        
        n = len(s)

        interp_pt =  self.center_line.getValue(s)
        slope = np.zeros(n)

        for i in range(n):
            deri = self.center_line.getDerivative(s[i])
            slope[i] = np.arctan2(deri[1], deri[0])
        return interp_pt, slope
    
    def project_points(self, points):
        n = points
        theta = np.zeros(n)
        
    def project_point(self, point):
        return self.center_line.projectPoint(point)*self.length
        

    def plot_track(self):
        if self.figure is None:
            plt.ioff()
            self.figure = plt.figure()

            theta_sample = np.linspace(0,1,200)*self.length
            interp_pt, slope = self.interp(theta_sample)

            self.track_bound = np.zeros(4,200)

            self.track_bound[0,:] = interp_pt[0,:] - np.sin(slope)*self.width_left
            self.track_bound[1,:] = interp_pt[1,:] + np.cos(slope)*self.width_left

            self.track_bound[2,:] = interp_pt[0,:] + np.sin(slope)*self.width_right
            self.track_bound[3,:] = interp_pt[1,:] - np.cos(slope)*self.width_right

        self.figure.plot(self.track_bound[0,:], self.track_bound[1,:], 'k-')
        self.figure.plot(self.track_bound[2,:], self.track_bound[3,:], 'k-')
        
    def load_from_file(self):
        pass
    







