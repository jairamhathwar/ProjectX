import numpy as np


class Dynamics:
    """
    A vehicle model with 4 dof. 
    State - [x, y, vel, theta]
    Control - [acc, yaw_rate]
    """
    def __init__(self, params):
        

        self.dim_x = 5
        self.dim_u = 3
        
        # load parameters
         # Planning Time Horizon
        self.T = params['T']
        # number of planning steps
        self.N = params['N']
        self.dt = self.T/(self.N-1)
        
        self.L = params['l_r']+params['l_f']
        self.delta_min = params['delta_min']
        self.delta_max = params['delta_max']
        self.a_min = params['a_min']
        self.a_max = params['a_max']
        self.v_max = params['v_max']
        
        self.dtheta_max = 3*self.v_max
        
        self.zeros = np.zeros((self.N))
        self.ones = np.ones((self.N))
        
    def forward_step(self, state, control, step = 1):
        """
        Find the next state of the vehicle given the current state and control input
        """
        # Clips the controller values between min and max accel and steer values
        accel = np.clip(control[0], self.a_min, self.a_max)
        delta = np.clip(control[1], self.delta_min, self.delta_max)
        dtheta = np.clip(control[2], 0, self.dtheta_max)
        control_clip = np.array([accel, delta, dtheta])       
        next_state = state
    
        dt = self.dt/step
        for _ in range(step):
            # State: [x, y, psi, v]
            d_x = (next_state[2]*dt+0.5*accel*dt**2)*np.cos(next_state[3])
            d_y = (next_state[2]*dt+0.5*accel*dt**2)*np.sin(next_state[3])
            d_v = accel*dt
            d_psi = dt*next_state[3]*np.tan(delta)/self.L
            next_state = next_state + np.array([d_x, d_y, d_v, d_psi, dtheta*dt])
            next_state[2] = max(0, next_state[2])
            #next_state[2] = max(0, next_state[2])
        return next_state, control_clip

    def get_AB_matrix(self, nominal_states, nominal_controls):
        """
        Returns the linearized 'A' matrix of the ego vehicle 
        model for all states in backward pass. 
        """        
        # State has dimension [d=4,N]
        # State: [x, y, psi, v]
        
        v = nominal_states[2,:]
        psi = nominal_states[3,:]
        # Control have dimension [d=2,N]
        accel = nominal_controls[0,:]
        delta = nominal_controls[1,:]
        # A matrix has dimension [d=4,d=4,N]
        A = np.array([[self.ones, self.zeros, np.cos(psi)*self.dt, -(v*self.dt + 0.5*accel*self.dt**2)*np.sin(psi), self.zeros],
                      [self.zeros, self.ones, np.sin(psi)*self.dt, (v*self.dt + 0.5*accel*self.dt**2)*np.cos(psi), self.zeros],
                      [self.zeros, self.zeros, self.ones, self.zeros, self.zeros],
                      [self.zeros, self.zeros, np.tan(delta)/self.L, self.ones, self.zeros],
                      [self.zeros, self.zeros, self.zeros, self.zeros, self.ones]])
        
        B = np.array([[self.dt**2*np.cos(psi)/2, self.zeros, self.zeros],
                      [self.dt**2*np.sin(psi)/2, self.zeros, self.zeros],
                      [self.dt*self.ones, self.zeros, self.zeros], 
                      [self.zeros, v/(self.L*np.cos(delta)**2), self.zeros],
                      [self.zeros, self.zeros, self.ones*self.dt]])
        return A, B

    