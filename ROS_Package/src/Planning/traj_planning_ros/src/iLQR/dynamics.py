import numpy as np
class Dynamics:
    def __init__(self, params):
        self.dim_x = 4
        self.dim_u = 2
        
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

        # useful constants
        self.zeros = np.zeros((self.N))
        self.ones = np.ones((self.N))
        
    def forward_step(self, state, control):
        """
        Find the next state of the vehicle given the current state and 
        control input state

            state: 4x1 array [X, Y, V, psi]
            control: 2x1 array [a, delta]
        """
        # Clips the controller values between min and max accel and steer values
        accel = np.clip(control[0], self.a_min, self.a_max)
        delta = np.clip(control[1], self.delta_min, self.delta_max)
        control_clip = np.array([accel, delta])       
        next_state = state
            
        # State: [x, y, psi, v]
        d_x = (next_state[2]*self.dt+0.5*accel*self.dt**2)*np.cos(next_state[3])
        d_y = (next_state[2]*self.dt+0.5*accel*self.dt**2)*np.sin(next_state[3])
        d_v = accel*self.dt
        d_psi = self.dt*next_state[2]*np.tan(delta)/self.L
        next_state = next_state + np.array([d_x, d_y, d_v, d_psi])

        # Clip the velocity
        next_state[2] = min(max(self.v_min, next_state[2]), self.v_max)
        
        return next_state, control_clip

    def get_AB_matrix(self, nominal_states, nominal_controls):
        """
        Returns the linearized 'A' and 'B' matrix of the ego vehicle around 
        nominal states and controls

          nominal_states: 4xN array
          nominal_controls: 2xN array
        """        
        v = nominal_states[2,:]
        psi = nominal_states[3,:]
        accel = nominal_controls[0,:]
        delta = nominal_controls[1,:]

        A = np.array([[self.ones, self.zeros, np.cos(psi)*self.dt, -(v*self.dt + 0.5*accel*self.dt**2)*np.sin(psi)],
                      [self.zeros, self.ones, np.sin(psi)*self.dt, (v*self.dt + 0.5*accel*self.dt**2)*np.cos(psi)],
                      [self.zeros, self.zeros, self.ones, self.zeros],
                      [self.zeros, self.zeros, np.tan(delta)*self.dt/self.L, self.ones]])
        
        B = np.array([[self.dt**2*np.cos(psi)/2, self.zeros],
                      [self.dt**2*np.sin(psi)/2, self.zeros],
                      [self.dt*self.ones, self.zeros], 
                      [np.tan(delta)*self.dt**2/(2*self.L), v*self.dt/(self.L*np.cos(delta)**2)]])
        return A, B