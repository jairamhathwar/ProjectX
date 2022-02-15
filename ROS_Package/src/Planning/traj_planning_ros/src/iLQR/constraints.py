from matplotlib import transforms
from matplotlib.pyplot import close
import numpy as np

class Constraints:
    def __init__(self, params):
        # load parameters
        self.T = params['T'] # Planning Time Horizon
        self.N = params['N'] # number of planning steps
        self.dt = self.T/(self.N-1) # time step for each planning step
        
        self.L = params['L'] # vehicle chassis length
        self.delta_min = params['delta_min'] # min steering angle rad
        self.delta_max = params['delta_max'] # max steering angle rad
        self.a_min = params['a_min'] # min longitudial accel
        self.a_max = params['a_max'] # max longitudial accel
        self.v_min = params['v_min'] # min velocity
        self.v_max = params['v_max'] # max velocity
        self.alat_max = params['alat_max'] # max lateral accel
        self.alat_min = params['alat_min'] # min lateral accel

        # parameter for barrier functions
        self.q1_v = params['q1_v']
        self.q2_v = params['q2_v']

        self.q1_road = params['q1_road']
        self.q2_road = params['q2_road']

        self.q1_lat = params['q1_lat']
        self.q2_lat = params['q2_lat']
    
        # useful constants
        self.zeros = np.zeros((self.N))
        self.ones = np.ones((self.N))


    def get_cost(self, states, controls, closest_pt, slope):
        '''Road Boundary cost'''
        dx = states[0,:] - closest_pt[0,:]
        dy = states[1,:] - closest_pt[1,:]
        d = np.sin(slope)*dx - np.cos(slope)*dy

        c_boundary = self.q1_road*np.exp(self.q2_road*(d-self.track.width_right)) \
                        + self.q1_road*np.exp(self.q2_road*(self.track.width_left-d))
        
        c_vel = self.q1_v*np.exp(self.q2_v*(states[2,:] - self.v_max)) \
                        + self.q1_v*np.exp(-states[2,:]*self.q2_v)

        # calculate the acceleration
        accel = states[2,:]**2*np.tan(controls[1,:])/self.L
        error_ub = accel - self.alat_max
        error_lb = self.alat_min - accel

        b_ub = self.q1_lat*np.exp(self.q2_lat*error_ub)
        b_lb = self.q1_lat*np.exp(self.q2_lat*error_lb)
        c_lat = b_lb+b_ub
                
        return c_vel+c_boundary+c_lat 

    def get_derivatives(self, states, controls, closest_pt, slope):
        ''' 
        Calculate the Jacobian and Hessian of soft constraint cost
            states: 4xN array of planned trajectory
            controls: 2xN array of planned control
            closest_pt: 2xN array of each state's closest point [x,y] on the track
            slope: 1xN array of track's slopes (rad) at closest points
        '''

        c_x_lat, c_xx_lat, c_u_lat, c_uu_lat, c_ux_lat = \
            self._lat_accec_bound_derivative(states, controls)
        c_x_rd, c_xx_rd = self._road_boundary_derivate(states, closest_pt, slope)  
        c_x_vel, c_xx_vel = self._velocity_bound_derivate(states)
        c_x_cons = c_x_rd+c_x_vel+ c_x_lat
        c_xx_cons = c_xx_rd+c_xx_vel+ c_xx_lat
        c_u_cons = c_u_lat
        c_uu_cons = c_uu_lat
        c_ux_cons = c_ux_lat

        return c_x_cons, c_xx_cons, c_u_cons, c_uu_cons, c_ux_cons

    def _road_boundary_derivate(self, states, closest_pt, slope):
        ''' 
        Calculate the Jacobian and Hessian of road boundary soft constraint cost
            states: 4xN array of planned trajectory
            closest_pt: 2xN array of each state's closest point [x,y] on the track
            slope: 1xN array of track's slopes (rad) at closest points
        '''
        # constraint due to right road boundary. smaller than right_width             
        transform = np.array([np.sin(slope), -np.cos(slope), self.zeros, self.zeros])
        
        ref_states = np.zeros_like(states)
        ref_states[:2, :] = closest_pt
        
        error = states - ref_states

        c = np.einsum('an,an->n', transform, error) - self.track.width_right
        c_x_u, c_xx_u = self.barrier_function(self.q1_road, self.q2_road, c, transform)

        # constraint due to left road boundary. larger than -left_width
        c = - self.track.width_left - np.einsum('an,an->n', transform, error) 
        c_x_l, c_xx_l = self.barrier_function(self.q1_road, self.q2_road, c, -transform)

        return c_x_u+c_x_l, c_xx_u+c_xx_l
        
    def _velocity_bound_derivate(self, states):
        ''' 
        Calculate the Jacobian and Hessian of velocity soft constraint cost
            states: 4xN array of planned trajectory
        '''
        # less than V_max
        transform = np.array([self.zeros, self.zeros, self.ones, self.zeros])
        c = states[2,:] - self.v_max
        c_x_u, c_xx_u = self.barrier_function(self.q1_v, self.q2_v, c, transform)

        # larger than 0
        c = -states[2,:]
        c_x_l, c_xx_l = self.barrier_function(self.q1_v, self.q2_v, c, -transform)

        return c_x_u+c_x_l, c_xx_u+c_xx_l
        
    def _lat_accec_bound_derivative(self, states, controls):
        ''' 
        Calculate the Jacobian and Hessian of Lateral Acceleration soft constraint cost
            states: 4xN array of planned trajectory
            controls: 2xN array of planned control
        '''
        c_x = np.zeros((4,self.N))
        c_xx = np.zeros((4,4,self.N))
        c_u = np.zeros((2,self.N))
        c_uu =np.zeros((2,2,self.N))
        c_ux = np.zeros((2,4,self.N))

        # calculate the acceleration
        accel = states[2,:]**2*np.tan(controls[1,:])/self.L
        
        error_ub = accel - self.alat_max
        error_lb = self.alat_min - accel

        b_ub = self.q1_lat*np.exp(self.q2_lat*error_ub)
        b_lb = self.q1_lat*np.exp(self.q2_lat*error_lb)

        da_dx = 2*states[2,:]*np.tan(controls[1,:])/self.L
        da_dxx = 2*np.tan(controls[1,:])/self.L

        da_du = states[2,:]**2/(np.cos(controls[1,:])**2*self.L)
        da_duu = states[2,:]**2*np.sin(controls[1,:])/(np.cos(controls[1,:])**3*self.L)

        da_dux = 2*states[2,:]/(np.cos(controls[1,:])**2*self.L)

        c_x[2,:] = self.q2_lat*(b_ub-b_lb)*da_dx
        c_u[1,:] = self.q2_lat*(b_ub-b_lb)*da_du

        c_xx[2,2,:] = self.q2_lat**2*(b_ub+b_lb)*da_dx**2 + self.q2_lat*(b_ub-b_lb)*da_dxx
        c_uu[1,1,:] = self.q2_lat**2*(b_ub+b_lb)*da_du**2 + self.q2_lat*(b_ub-b_lb)*da_duu

        c_ux[1,2,:] = self.q2_lat**2*(b_ub+b_lb)*da_dx*da_du + self.q2_lat*(b_ub-b_lb)*da_dux
        return c_x, c_xx, c_u, c_uu, c_ux

    def barrier_function(self, q1, q2, c, c_dot):
        '''
        c = [n] array
        c_dot = [dxn] array
        '''
        b = q1*np.exp(q2*c)
        b_dot = np.einsum('n,an->an', q2*b, c_dot)
        b_ddot = np.einsum('n,abn->abn', (q2**2)*b, np.einsum('an,bn->abn',c_dot, c_dot))
        return b_dot, b_ddot