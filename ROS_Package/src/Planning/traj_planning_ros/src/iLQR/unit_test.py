from dynamics import Dynamics
from cost import Cost
from ilqr import iLQR
from track import Track
import numpy as np
import csv


test_param = {}
test_param['N=5']
test_param['T=1']


class UnitTest:
    def __init__(self):
        params = {'L': 0.257, 'm': 2.99, 'track_width': 0.4,
                'delta_min': -0.35, 'delta_max': 0.35,
                'v_max': 4, 'v_min': 0, 'a_min': -3.5,
                'a_max': 3.5, 'alat_max': 5, 'w_vel': 2,
                'w_contour': 100, 'w_theta': 1, 'w_accel': 1,
                'w_delta': 1, 'q1_v': 0.4, 'q2_v': 2,
                'q1_road': 0.4, 'q2_road': 2, 'q1_lat': 0.4,
                'q2_lat': 2, 'T': 2, 'N': 11, 'max_itr': 50}

        # define a test track
        x = []
        y = []

        with open('racetrack/IMS.csv', newline='') as f:
            spamreader = csv.reader(f, delimiter=',')
            for i, row in enumerate(spamreader):
                if i>0:
                    x.append(float(row[0]))
                    y.append(float(row[1]))

        x = np.array(x)/25.0
        y = np.array(y)/25.0
        center_line = np.array([x,y])
        self.track = Track(center_line = center_line, width_left = 0.4, width_right = 0.4)
        
        self.dynamics = Dynamics(params)
        self.cost = Cost(params)
        self.iLQR = iLQR(self.track, params)
        
    def test_forward_step(self):
        
        
        pass
    
    def test_AB_matrix(self):
        pass
    
    def test_cost_deriv_state(self):
        pass
    
    def test_cost_deriv_control(self):
        pass
        
        