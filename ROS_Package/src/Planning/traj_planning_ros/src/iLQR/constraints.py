from matplotlib import transforms
from matplotlib.pyplot import close
import numpy as np

class Constraints:
    def __init__(self, params, track):
        self.track = track
        self.T = params['T']
        self.N = params['N']
        self.dt = self.T/(self.N-1)
        
        self.L = params['l_r']+params['l_f']
        self.delta_min = params['delta_min']
        self.delta_max = params['delta_max']
        self.a_min = params['a_min']
        self.a_max = params['a_max']
        self.v_max = params['v_max']

        self.alat_max = params['alat_max']
        self.alat_min = params['alat_min']
        
        self.dtheta_max = 1.5*self.v_max

        # parameters for barrier functions
        self.q1_accel = params['q1_accel']
        self.q2_accel = params['q2_accel']

        self.q1_delta = params['q1_delta']
        self.q2_delta = params['q2_delta']

        self.q1_v = params['q1_v']
        self.q2_v = params['q2_v']

        self.q1_road = params['q1_road']
        self.q2_road = params['q2_road']

        self.q1_lat = params['q1_lat']
        self.q2_lat = params['q2_lat']
    
        self.zeros = np.zeros((self.N))
        self.ones = np.ones((self.N))


    def get_cost(self, states, controls, closest_pt, slope):
        '''Road Boundary cost'''
        dx = states[0,:] - closest_pt[0,:]
        dy = states[1,:] - closest_pt[1,:]
        d = np.sin(slope)*dx - np.cos(slope)*dy

        L_boundary = self.q1_road*np.exp(self.q2_road*(d-self.track.width_right)) \
                        + self.q1_road*np.exp(self.q2_road*(self.track.width_left-d))
        
        L_vel = self.q1_v*np.exp(self.q2_v*(states[2,:] - self.v_max)) \
                        + self.q1_v*np.exp(-states[2,:]*self.q2_v)

        # loss due to control
        L_steer = self.q1_delta*np.exp(self.q2_delta*(controls[0,:] - self.delta_max)) \
                        + self.q1_delta*np.exp(self.q2_delta*(self.delta_min -controls[0,:]))
        L_accel = self.q1_accel*np.exp(self.q2_accel*(controls[1,:] - self.a_max)) \
                        + self.q1_accel*np.exp(self.q2_accel*(self.a_min -controls[1,:]))
        L_dtheta = self.q1_v*np.exp(self.q2_v*(controls[-1,:] - self.dtheta_max)) \
                        + self.q1_v*np.exp(-controls[-1,:]*self.q2_v)
        # terminal state does not have a control loss
        L_steer[-1] = 0
        L_accel[-1] = 0
        L_dtheta[-1] = 0
        
        return L_accel+L_vel+L_steer+L_boundary+L_dtheta

    def road_boundary_derivate(self, nominal_states, closest_pt, slope):
        ''' Road Boundary '''
        # constraint due to right road boundary. smaller than right_width             
        transform = np.array([np.sin(slope), -np.cos(slope), self.zeros, self.zeros])
        
        ref_states = np.zeros_like(nominal_states)
        ref_states[:2, :] = closest_pt
        
        error = nominal_states - ref_states

        c = np.einsum('an,an->n', transform, error) - self.track.width_right
        L_x_u, L_xx_u = self.barrier_function(self.q1_road, self.q2_road, c, transform)

        # constraint due to left road boundary. larger than -left_width
        c = - self.track.width_left - np.einsum('an,an->n', transform, error) 
        L_x_l, L_xx_l = self.barrier_function(self.q1_road, self.q2_road, c, -transform)

        return L_x_u+L_x_l, L_xx_u+L_xx_l
        
    def velocity_bound_derivate(self, nominal_states):
        '''
        nominal_states: [d=4xN] array
        '''
        '''Velocity bound'''
        # less than V_max
        transform = np.array([self.zeros, self.zeros, self.ones, self.zeros])
        c = nominal_states[2,:] - self.v_max
        L_x_u, L_xx_u = self.barrier_function(self.q1_v, self.q2_v, c, transform)

        # larger than 0
        c = -nominal_states[2,:]
        L_x_l, L_xx_l = self.barrier_function(self.q1_v, self.q2_v, c, -transform)

        return L_x_u+L_x_l, L_xx_u+L_xx_l
    
    def dtheta_bound_derivate(self, nominal_controls):
        '''
        nominal_states: [d=4xN] array
        '''
        '''Velocity bound'''
        # less than V_max
        transform = np.array([self.zeros, self.zeros, self.ones])
        c = nominal_controls[-1,:] - self.dtheta_max
        L_u_u, L_uu_u = self.barrier_function(self.q1_v, self.q2_v, c, transform)

        # larger than 0
        c = -nominal_controls[-1,:]
        L_u_l, L_uu_l = self.barrier_function(self.q1_v, self.q2_v, c, -transform)

        return L_u_l+L_u_u, L_uu_l+L_uu_u
    
    def steering_bound_derivative(self, nominal_controls):
        '''
        nominal_control: [d=2xN] array
        '''
        
        ''' Steering Delta Bound'''
        # delta upper bound
        transform = np.array([self.zeros, self.ones])
        c = nominal_controls[1,:] - self.delta_max
        L_u_u, L_uu_u = self.barrier_function(self.q1_delta, self.q2_delta, c, transform)
        
        #print(L_uu_u)
        # delta lower bound
        c =  self.delta_min - nominal_controls[1,:]
        L_u_l, L_uu_l = self.barrier_function(self.q1_delta, self.q2_delta, c, -transform)

        return L_u_l+L_u_u, L_uu_l+L_uu_u

    def accel_bound_derivative(self, nominal_controls):
        ''' Acceleration Bound'''
        # upper bound  a_max
        transform = np.array([self.ones, self.zeros])
        c = nominal_controls[0,:] - self.a_max
        L_u_u, L_uu_u = self.barrier_function(self.q1_accel, self.q2_accel, c, transform)

        c = self.a_min - nominal_controls[0,:]
        L_u_l, L_uu_l = self.barrier_function(self.q1_accel, self.q2_accel, c, -transform)

        return L_u_l+L_u_u, L_uu_l+L_uu_u

    def lat_accel_bound_derivative(self, nominal_states, nominal_controls):
        ''' Lateral Acceleration '''
        L_x = np.zeros((4,self.N))
        L_xx = np.zeros((4,4,self.N))
        L_u = np.zeros((2,self.N))
        L_uu =np.zeros((2,2,self.N))
        L_ux = np.zeros((2,4,self.N))

        # calculate the acceleration
        accel = nominal_states[2,:]**2*np.tan(nominal_controls[1,:])/self.L
        
        error_ub = accel - self.alat_max
        error_lb = self.alat_min - accel

        b_ub = self.q1_lat*np.exp(self.q2_lat*error_ub)
        b_lb = self.q1_lat*np.exp(self.q2_lat*error_lb)

        da_dx = 2*nominal_states[2,:]*np.tan(nominal_controls[1,:])/self.L
        da_dxx = 2*np.tan(nominal_controls[1,:])/self.L

        da_du = nominal_states[2,:]**2/(np.cos(nominal_controls[1,:])**2*self.L)
        da_dux = nominal_states[2,:]**2*np.sin(nominal_controls[1,:])/(np.cos(nominal_controls[1,:])**3*self.L)

    def barrier_function(self, q1, q2, c, c_dot):
        '''
        c = [n] array
        c_dot = [dxn] array
        '''
        b = q1*np.exp(q2*c)
        b_dot = np.einsum('n,an->an', q2*b, c_dot)
        b_ddot = np.einsum('n,abn->abn', (q2**2)*b, np.einsum('an,bn->abn',c_dot, c_dot))
        return b_dot, b_ddot