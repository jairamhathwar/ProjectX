import numpy as np


class Dynamics:
    """
    A vehicle model with 4 dof. 
    State - [x, y, vel, theta]
    Control - [acc, yaw_rate]
    """
    def __init__(self, T, N, params):
        

        self.dim_x = 4
        self.dim_u = 2
        
        # load parameters
        self.L = params['l_r']+params['l_f']
        self.delta_min = params['delta_min']
        self.delta_max = params['delta_max']
        self.a_min = params['a_min']
        self.a_max = params['a_max']
        self.v_max = params['v_max']
        
        # Planning Time Horizon
        self.T = T
        # number of planning steps
        self.N = N
        self.dt = T/N
        self.zeros = np.zeros((self.N))
        self.ones = np.ones((self.N))
        
    def forward_step(self, state, control, step = 1):
        """
        Find the next state of the vehicle given the current state and control input
        """
        # Clips the controller values between min and max accel and steer values
        accel = np.clip(control[0], self.a_min, self.a_max)
        delta = np.clip(control[1], self.delta_min, self.delta_max)
        
        next_state = state
    
        dt = self.dt/step
        for _ in range(step):
            # State: [x, y, psi, v]
            d_x = (next_state[3]*dt+0.5*accel*dt**2)*np.cos(next_state[2])
            d_y = (next_state[3]*dt+0.5*accel*dt**2)*np.sin(next_state[2])
            d_psi = dt*next_state[2]*np.tan(delta)/self.L
            d_v = accel*dt
            next_state = next_state + np.array([d_x, d_y, d_psi, d_v])  
        return next_state

    def get_AB_matrix(self, nominal_states, nominal_controls):
        """
        Returns the linearized 'A' matrix of the ego vehicle 
        model for all states in backward pass. 
        """        
        # State has dimension [d=4,N]
        # State: [x, y, psi, v]
        psi = nominal_states[2,:]
        v = nominal_states[3,:]
        # Control have dimension [d=2,N]
        accel = nominal_controls[0,:]
        delta = nominal_controls[1,:]
        # A matrix has dimension [d=4,d=4,N]
        A = np.array([[self.ones, self.zeros, -(v*self.dt + 0.5*accel*self.dt**2)*np.sin(psi), np.cos(psi)*self.dt,]
                      [self.zeros, self.ones,  (v*self.dt + 0.5*accel*self.dt**2)*np.cos(psi), np.sin(psi)*self.dt],
                      [self.zeros, self.zeros, self.ones, np.tan(delta)/self.L],
                      [self.zeros, self.zeros, self.zeros, self.ones]])
        
        B = np.array([[self.dt**2*np.cos(psi)/2, self.zeros],
                      [self.dt**2*np.sin(psi)/2, self.zeros],
                      [self.zeros, v/(self.L*np.cos(delta)**2)],
                      [self.dt*self.ones, self.zeros]])
        return A, B

    