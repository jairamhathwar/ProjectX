import numpy as np
from constraints import Constraints
class Cost:
    def __init__(self, params, ref_path):
        self.soft_constraints = Constraints(params, ref_path)
        self.ref_path = ref_path
              
        # load parameters
        self.T = params['T'] # Planning Time Horizon
        self.N = params['N'] # number of planning steps
        self.dt = self.T/(self.N-1) # time step for each planning step
        self.v_max = params['v_max'] # max velocity

        # cost 
        self.w_vel = params['w_vel'] 
        self.w_contour = params['w_contour']
        self.w_theta = params['w_theta']
        self.w_accel = params['w_accel']
        self.w_delta = params['w_delta']
        
        self.W_state = np.array([[self.w_contour, 0],
                            [0, self.w_vel]])
        self.W_control = np.array([[self.w_accel, 0], [0, self.w_delta]])
        
        # useful constants
        self.zeros = np.zeros((self.N))
        self.ones = np.ones((self.N))

    def get_cost(self, states, controls, closest_pt, slope, theta):
        '''
        Given planned states and controls, calculate the cost
            states: 4xN array of planned trajectory
            controls: 2xN array of planned control
            closest_pt: 2xN array of each state's closest point [x,y] on the track
            slope: 
        '''
        
        transform = np.array([[np.sin(slope), -np.cos(slope), self.zeros, self.zeros], 
                                [self.zeros, self.zeros, self.ones, self.zeros]])
        ref_states = np.zeros_like(states)
        ref_states[:2, :] = closest_pt
        ref_states[2, :] = self.v_max
        
        error = states - ref_states
        Q_trans = np.einsum('abn, bcn->acn', np.einsum('dan, ab -> dbn', transform.transpose(1,0,2), self.W_state), transform)
        
        c_state = np.einsum('an, an->n', error, np.einsum('abn, bn->an', Q_trans, error))
        c_progress = -self.w_theta*np.sum(theta) #(theta[-1] - theta[0])

        c_control = np.einsum('an, an->n', controls, np.einsum('ab, bn->an', self.W_control, controls))

        c_control[-1] = 0
        
        c_constraint = self.soft_constraints.get_cost(states, controls, closest_pt, slope)

        J  = np.sum(c_state + c_constraint + c_control) + c_progress

        return J
        
    def get_derivatives(self, nominal_states, nominal_controls, closest_pt, slope):
        '''
        nominal_states: [d=4xN] array
        '''
        c_x_lat, c_xx_lat, c_u_lat, c_uu_lat, c_ux = \
            self.soft_constraints.lat_accel_bound_derivative(nominal_states, nominal_controls)
        c_x_rd, c_xx_rd = self.soft_constraints.road_boundary_derivate(nominal_states, closest_pt, slope)
        c_x_vel, c_xx_vel = self.soft_constraints.velocity_bound_derivate(nominal_states)
        c_x_cost, c_xx_cost = self._get_cost_state_derivative(nominal_states, closest_pt, slope)
        
        c_x = c_x_rd+c_x_vel+c_x_cost+ c_x_lat
        c_xx = c_xx_rd+c_xx_vel+c_xx_cost + c_xx_lat
        
        c_u_cost, c_uu_cost = self._get_cost_control_derivative(nominal_controls)
        c_u = c_u_cost+ c_u_lat
        c_uu = c_uu_cost+c_uu_lat
        
        return c_x, c_xx, c_u, c_uu, c_ux
    
    def _get_cost_state_derivative(self, nominal_states, closest_pt, slope):
        '''
        nominal_states: [d=4xN] array
        '''
        #closest_pt, slope = self.ref_path.get_closest_pts(nominal_states[:2,:])

        transform = np.array([[np.sin(slope), -np.cos(slope), self.zeros, self.zeros], 
                        [self.zeros, self.zeros, self.ones, self.zeros]])
        ref_states = np.zeros_like(nominal_states)
        ref_states[:2, :] = closest_pt
        ref_states[2, :] = self.v_max
        
        error = nominal_states - ref_states
        Q_trans = np.einsum('abn, bcn->acn', np.einsum('dan, ab -> dbn', transform.transpose(1,0,2), self.W_state), transform)
        
        
        # shape [4xN]
        c_x = np.einsum('abn, bn->an', Q_trans, error)

        c_x_progress = -self.w_theta*np.array([np.cos(slope), np.sin(slope), self.zeros, self.zeros])
        c_x = c_x + c_x_progress
        c_xx = Q_trans
        
        return c_x, c_xx
    
    def _get_cost_control_derivative(self, nominal_controls):
        '''
        nominal_control: [d=2xN] array
        '''
        c_u = np.einsum('ab, bn->an', self.W_control, nominal_controls)
        
        c_uu = np.repeat(self.W_control[:,:,np.newaxis], self.N, axis=2)        
        return c_u, c_uu