import numpy as np
from ..Track import Track

class Cost:
    def __init__(self, params, ref_path: Track):
        self.ref_path = ref_path
        self.Q_pos = params['Q_pos']
        self.Q_vel = params['Q_vel']
        
        self.R_accel = params['R_accel']
        self.R_delta = params['R_delta']
        
        self.Q = np.array([[self.Q_pos, 0, 0],
                           [0, self.Q_vel, 0],
                           [0,0,0]])
        
        self.R = np.array([[self.R_accel, 0], [self.R_delta]])
        
    def get_derivatives(self, nominal_states, nominal_controls, theta_guess = None):
        '''
        nominal_states: [d=4xN] array
        '''
        theta = self.ref_path.ge
        
        pass
    def _get_control_cost_jacobian(self, )
        
    
        