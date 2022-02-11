import numpy as np
from constraints import Constraints

# state: [x,y,v,psi]
class Cost:
    def __init__(self, params, ref_path):
        
        self.soft_constraints = Constraints(params, ref_path)
        self.ref_path = ref_path
        
        self.T = params['T']
        self.N = params['N']
        
        self.Q_pos = params['Q_pos']
        self.Q_vel = params['Q_vel']
        
        self.R_accel = params['R_accel']
        self.R_delta = params['R_delta']

        self.v_ref = params['v_max']
        
        self.Q = np.array([[self.Q_pos, 0],
                            [0, self.Q_vel]])
        
        self.R = np.array([[self.R_accel, 0], [0, self.R_delta]])
        
        self.zeros = np.zeros((self.N))
        self.ones = np.ones((self.N))
    
    def get_cost(self, states, controls):
        closest_pt, slope = self.ref_path.get_closest_pts(states[:2,:])
        transform = np.array([[np.sin(slope), -np.cos(slope), self.zeros, self.zeros], 
                                [self.zeros, self.zeros, self.ones, self.zeros]])
        ref_states = np.zeros_like(states)
        ref_states[:2, :] = closest_pt
        ref_states[2, :] = self.v_ref
        
        error = states - ref_states
        Q_trans = np.einsum('abn, bcn->acn', np.einsum('dan, ab -> dbn', transform.transpose(1,0,2), self.Q), transform)
        
        L_state = np.einsum('an, an->n', error, np.einsum('abn, bn->an', Q_trans, error))
        
        L_control = np.einsum('an, an->n', controls, np.einsum('ab, bn->an', self.R, controls))
        
        L_constraint = self.soft_constraints.get_cost(states, controls, closest_pt, slope)

        J  = np.sum(L_state + L_constraint + L_control)

        return J, closest_pt, slope

        
    def get_derivatives(self, nominal_states, nominal_controls, closest_pt, slope):
        '''
        nominal_states: [d=4xN] array
        '''
        L_x, L_xx = self._get_cost_state_derivative(nominal_states, closest_pt, slope)
        L_u, L_uu = self._get_cost_control_derivative(nominal_controls)
        
        return L_x, L_xx, L_u, L_uu
    
    def _get_cost_state_derivative(self, nominal_states, closest_pt, slope):
        '''
        nominal_states: [d=4xN] array
        '''
        #closest_pt, slope = self.ref_path.get_closest_pts(nominal_states[:2,:])

        L_x_rd, L_xx_rd = self.soft_constraints.road_boundary_derivate(nominal_states, closest_pt, slope)
        L_x_vel, L_xx_vel = self.soft_constraints.velocity_bound_derivate(nominal_states)

        transform = np.array([[np.sin(slope), -np.cos(slope), self.zeros, self.zeros], 
                                [self.zeros, self.zeros, self.ones, self.zeros]])
        
        ref_states = np.zeros_like(nominal_states)
        ref_states[:2, :] = closest_pt
        ref_states[2, :] = self.v_ref
        
        error = nominal_states - ref_states
        Q_trans = np.einsum('abn, bcn->acn', np.einsum('dan, ab -> dbn', transform.transpose(1,0,2), self.Q), transform)
        
        # shape [4xN]
        L_x = np.einsum('abn, bn->an', Q_trans, error)
        # shape [4x4xN]
        L_xx = Q_trans
        
        L_x = L_x  + L_x_vel+ L_x_rd
        L_xx = L_xx   + L_xx_vel+L_xx_rd

        return L_x, L_xx
    
    def _get_cost_control_derivative(self, nominal_controls):
        '''
        nominal_control: [d=2xN] array
        '''
        L_u = np.einsum('ab, bn->an', self.R, nominal_controls)
        
        L_uu = np.repeat(self.R[:,:,np.newaxis], self.N, axis=2)

        L_u_steer, L_uu_steer = self.soft_constraints.steering_bound_derivative(nominal_controls)
        
        L_u_accel, L_uu_accel = self.soft_constraints.accel_bound_derivative(nominal_controls)
        L_u = L_u + L_u_steer+L_u_accel
        L_uu = L_uu + L_uu_steer+L_uu_accel
        
        return L_u, L_uu
        
    
    
        
    
        