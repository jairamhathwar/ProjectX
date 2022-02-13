import numpy as np
from constraints import Constraints

# state: [x,y,v,psi]
class Cost:
    def __init__(self, params, ref_path):
        
        
        
        self.soft_constraints = Constraints(params, ref_path)
        self.ref_path = ref_path
        
        self.T = params['T']
        self.N = params['N']
        
        self.Q_l = params['Q_l']
        self.Q_c = params['Q_c']
        self.Q_theta = params['Q_theta']
        
        # self.q_theta = np.array([self.zeros, self.zeros, self.zeros, 
        #                          self.zeros, -params['Q_theta']*self.ones])
        
        self.R_accel = params['R_accel']
        self.R_delta = params['R_delta']

        self.v_ref = params['v_max']
        
        self.Q = np.array([[self.Q_c, 0],
                            [0, self.Q_l]])
        
        self.R = np.array([[self.R_accel, 0, 0], [0, self.R_delta, 0], [0, 0, 0]])
        
        # useful parameters
        self.zeros = np.zeros((self.N))
        self.ones = np.ones((self.N))

    def get_cost(self, states, controls, closest_pt, slope):
        
        transform = np.array([[np.sin(slope), -np.cos(slope), self.zeros, self.zeros, self.zeros], 
                                [-np.cos(slope), -np.sin(slope), self.zeros, self.zeros, self.zeros]])
        ref_states = np.zeros_like(states)
        ref_states[:2, :] = closest_pt
        # ref_states[2, :] = self.v_ref
        
        error = states - ref_states
        Q_trans = np.einsum('abn, bcn->acn', np.einsum('dan, ab -> dbn', transform.transpose(1,0,2), self.Q), transform)
        
        L_state = np.einsum('an, an->n', error, np.einsum('abn, bn->an', Q_trans, error))
        
        L_progress = -self.Q_theta*states[-1,:]
        
        L_control = np.einsum('an, an->n', controls, np.einsum('ab, bn->an', self.R, controls))
        
        L_constraint = self.soft_constraints.get_cost(states, controls, closest_pt, slope)

        J  = np.sum(L_state + L_constraint + L_control + L_progress)

        return J

        
    def get_derivatives(self, nominal_states, nominal_controls, closest_pt, slope):
        '''
        nominal_states: [d=4xN] array
        '''
        
        L_x_rd, L_xx_rd = self.soft_constraints.road_boundary_derivate(nominal_states, closest_pt, slope)
        L_x_vel, L_xx_vel = self.soft_constraints.velocity_bound_derivate(nominal_states)
        L_x_cost, L_xx_csot = self._get_cost_state_derivative(nominal_states, closest_pt, slope)
        L_x = L_x_rd+L_x_vel+L_x_cost
        L_xx = L_xx_rd+L_xx_vel+L_xx_csot
        
        L_u_steer, L_uu_steer = self.soft_constraints.steering_bound_derivative(nominal_controls)
        L_u_accel, L_uu_accel = self.soft_constraints.accel_bound_derivative(nominal_controls)
        L_u_dtheta, L_uu_dtheta = self.soft_constraints.dtheta_bound_derivate(nominal_controls)
        L_u_cost, L_uu_cost = self._get_cost_control_derivative(nominal_controls)
        L_u = L_u_steer+L_u_accel+L_u_dtheta+L_u_cost
        L_uu = L_uu_steer+L_uu_accel+L_uu_dtheta+L_uu_cost
        
        return L_x, L_xx, L_u, L_uu
    
    def _get_cost_state_derivative(self, nominal_states, closest_pt, slope):
        '''
        nominal_states: [d=4xN] array
        '''
        #closest_pt, slope = self.ref_path.get_closest_pts(nominal_states[:2,:])

        transform = np.array([[np.sin(slope), -np.cos(slope), self.zeros, self.zeros, self.zeros], 
                                [-np.cos(slope), -np.sin(slope), self.zeros, self.zeros, -self.ones]])
        
        ref_states = np.zeros_like(nominal_states)
        ref_states[:2, :] = closest_pt
        # ref_states[2, :] = self.v_ref
        
        error = nominal_states - ref_states
        error[-1,:] = 0
        Q_trans = np.einsum('abn, bcn->acn', np.einsum('dan, ab -> dbn', transform.transpose(1,0,2), self.Q), transform)
        
        
        # shape [4xN]
        L_x = np.einsum('abn, bn->an', Q_trans, error)
        L_x[-1,:] = self.Q_l*(np.cos(slope)*error[0,:]+np.sin(slope)*error[1,:])-self.Q_theta
        # shape [4x4xN]
        L_xx = Q_trans
        L_xx[-1,-1,:] = self.Q_l
        
        # L_x = L_x  + L_x_vel+ L_x_rd
        # L_xx = L_xx   + L_xx_vel+L_xx_rd

        return L_x, L_xx
    
    def _get_cost_control_derivative(self, nominal_controls):
        '''
        nominal_control: [d=2xN] array
        '''
        L_u = np.einsum('ab, bn->an', self.R, nominal_controls)
        
        L_uu = np.repeat(self.R[:,:,np.newaxis], self.N, axis=2)

        
        # L_u = L_u + L_u_steer+L_u_accel
        # L_uu = L_uu + L_uu_steer+L_uu_accel
        
        return L_u, L_uu
        
    
    
        
    
        