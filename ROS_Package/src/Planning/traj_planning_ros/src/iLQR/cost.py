import numpy as np
from constraints import Constraints
from ..Track import Track

class Cost:
    def __init__(self, params, ref_path: Track):
        self.soft_constraints = Constraints(params, ref_path)
        self.ref_path = ref_path
        self.Q_pos = params['Q_pos']
        self.Q_vel = params['Q_vel']
        
        self.R_accel = params['R_accel']
        self.R_delta = params['R_delta']
        
        self.Q = np.array([[self.Q_pos, 0],
                           [0, self.Q_vel]])
        
        self.R = np.array([[self.R_accel, 0], [self.R_delta]])
        
        self.zeros = np.zeros((self.N))
        self.ones = np.ones((self.N))
        
        
    def get_derivatives(self, nominal_states, nominal_controls, v_ref):
        '''
        nominal_states: [d=4xN] array
        '''
        
        pass
    
    def _get_cost_state_derivative(self, nominal_states, v_ref):
        '''
        nominal_states: [d=4xN] array
        '''
        closest_pt, slope = self.ref_path.get_closest_pts(nominal_states[:2,:])
        
        transform = np.array([[np.sin(slope), -np.cos(slope), self.zeros, self.zeros], 
                              [self.zeros, self.zeros, self.ones, self.zeros]])
        
        ref_states = np.zeros_like(nominal_states)
        ref_states[:2, :] = closest_pt
        ref_states[:2, :] = v_ref
        
        error = nominal_states - ref_states
        Q_trans = np.einsum('dbn, bdn->ddn', np.einsum('dan, abn -> dbn', transform.transpose(1,0,2), self.Q), transform)
        
        # shape [4xN]
        L_x = np.einsum('abn, bn->an', Q_trans, error)
        # shape [4x4xN]
        L_xx = Q_trans
        
        return L_x, L_xx
    
    def _get_cost_control_derivative(self, nominal_controls):
        '''
        nominal_control: [d=2xN] array
        '''
        L_u = np.einsum('abn, bn->an', self.R, nominal_controls)
        L_uu = self.R
        
        return L_u, L_uu
        
    
    
        
    
        