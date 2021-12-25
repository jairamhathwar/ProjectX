import casadi
from casadi import SX
import yaml
import numpy as np

class KinematicController:
    def __init__(self, model_name, params):
        self.model = casadi.types.SimpleNamespace()
        self.constraints = casadi.types.SimpleNamespace()
        
        self.model.name = model_name

        with open(params) as file:
            self.params = yaml.load(file, Loader= yaml.FullLoader)

        # define system dynmics
        self.system_dyn()



    def system_dyn(self):
        '''
        This function defines the system dynamics of the kinematic bicycle model
            State: [X, Y, psi: heading, V: speed]
            Control: [d: motor duty cycle, delta: steering input]  

                X_dot = V*cos(psi+beta)
                Y_dot = V*sin(psi+beta)
                psi_dot = V/l_r*sin(beta)
                V_dot = accel

            with:
                accel = [(C_m1-C_m2*V*cos(beta))*d-C_r-C_d*(V*cos(beta))^2]/m
                beta = atan(tan(delta)*l_r/(l_f+l_r))
        '''
        # Define parameters in Casadi
        l_f = self.params['l_f']
        l_r = self.params['l_r']

        C_m1 = self.params['C_m1']
        C_m2 = self.params['C_m2']
        C_r = self.params['C_r']
        C_d = self.params['C_d']

        m = self.params['m']

        # Position
        pos_X = SX.sym('pos_X')
        pos_X_dot = SX.sym('pos_X_dot')
        pos_Y = SX.sym('pos_Y')
        pos_Y_dot = SX.sym('pos_Y_dot')

        # heading
        psi = SX.sym('psi')
        psi_dot = SX.sym('psi_dot')
        
        # speed
        vel = SX.sym('vel')
        vel_dot = SX.sym('vel_dot')

        # progress
        theta = SX.sym('theta')
        theta_dot = SX.sym('theta_dot')

        # motor duty cycle
        d = SX.sym('d')
        delta = SX.sym('delta')

        # slip angle
        beta = casadi.atan(casadi.tan(delta)*l_r/(l_f+l_r))
        vel_x = vel*casadi.cos(beta)
        
        # state vector
        x = casadi.vertcat(pos_X, pos_Y, psi, vel, theta)
        x_dot = casadi.vertcat(pos_X_dot, pos_Y_dot, psi_dot, vel_dot, theta_dot)
        
        # control input
        u = casadi.vertcat(d, delta, theta_dot)

        # system dynamics
        f_expl = casadi.vertcat(
            vel*casadi.cos(psi+beta), # X_dot
            vel*casadi.sin(psi+beta), # Y_dot
            vel/l_r*casadi.sin(beta), # psi_dot
            ((C_m1-C_m2*vel_x)*d-C_r-C_d*vel_x**2)/m, # accel
            theta_dot
        )

        self.model.x = x
        self.model.x_dot = x_dot
        self.model.u = u
        self.model.f_expl = f_expl
        self.model.f_impl_expr = x_dot - f_expl
    
    def cost_func(self):
        """
        This function defines the cost function of the model predicative contouring control
        """


