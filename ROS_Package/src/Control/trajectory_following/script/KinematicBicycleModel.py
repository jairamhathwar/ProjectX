import casadi
from casadi import SX
import yaml
import numpy as np

class KinematicBicycleModel:
    def __init__(self, params_file):
        # System Dynamics of Kinematic Bicycle Model
        self.model = casadi.types.SimpleNamespace()
        self.model.name = 'KinematicBicycleModel'

        # Constraints of Kinematic Bicycle Model
        self.constraints = casadi.types.SimpleNamespace()

        # Systems specific parameters for the Kinematic Bicycle Model
        # Load parameters from YAML file
        with open(params_file) as file:
            self.params = yaml.load(file, Loader= yaml.FullLoader)
        

        # Define system dynamics
        self.define_sys_dyn()
        self.define_cost_func()
        self.define_constraint()

    def define_sys_dyn(self):
        '''
        This function defines the trajectroy tracking controller with the kinematic bicycle model
        '''
                
        # Load system parameters
        # chassis length
        l_f = self.params['l_f']
        l_r = self.params['l_r']

        # motor coefficient
        C_m1 = self.params['C_m1']
        C_m2 = self.params['C_m2']
        C_r = self.params['C_r']
        C_d = self.params['C_d']
        
        # weight
        m = self.params['m']

        '''
        Step 2:

        Define State, Control, and System Dynamics 
            State: [X, Y, psi: heading, V: speed, delta: steering angle]
            Control: [d: motor duty cycle, delta_dot: steering rate]  

                X_dot = V*cos(psi+beta)
                Y_dot = V*sin(psi+beta)
                psi_dot = V/l_r*sin(beta)
                V_dot = accel
                delta_dot = delta_dot

            with:
                accel = [(C_m1-C_m2*V*cos(beta))*d-C_r-C_d*(V*cos(beta))^2]/m
                beta = atan(tan(delta)*l_r/(l_f+l_r))
        '''

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

        # motor duty cycle
        d = SX.sym('d')

        # Steering angle
        delta = SX.sym('delta')
        delta_dot = SX.sym('delta_dot')

        # slip angle
        beta = casadi.atan(casadi.tan(delta)*l_r/(l_f+l_r))
        vel_x = vel*casadi.cos(beta)
        
        # state vector
        x = casadi.vertcat(pos_X, pos_Y, psi, vel, delta)
        x_dot = casadi.vertcat(pos_X_dot, pos_Y_dot, psi_dot, vel_dot, delta_dot)
        
        # control input
        u = casadi.vertcat(d, delta_dot)

        # system dynamics
        f_expl = casadi.vertcat(
            vel*casadi.cos(psi+beta), # X_dot
            vel*casadi.sin(psi+beta), # Y_dot
            vel/l_r*casadi.sin(beta), # psi_dot
            ((C_m1-C_m2*vel_x)*d-C_r-C_d*vel_x*vel_x)/m, # accel
            delta_dot
        )

        self.model.x = x
        self.model.xdot = x_dot
        self.model.u = u
        self.model.f_expl_expr = f_expl
        self.model.f_impl_expr = x_dot - f_expl

    def define_cost_func(self):
        '''
        Define Cost function of OCP 
        Consider a simple L2 cost for the state w.r.t to the reference trajectory
        J(t) = (X_ref-X)*Q_pos*(X_ref-X) + (Y_ref-Y)*Q_pos*(Y_ref-Y)
                + (Psi_ref - Psi)*Q_psi*(Psi_ref-Psi) + (Vel_ref-Vel)*Q_vel*(Vel_ref-Vel)
        '''

        # load cost function weight
        Q_pos = self.params['Q_pos']
        Q_psi = self.params['Q_psi']
        Q_vel = self.params['Q_vel']

        # Reference Position, heading and speed
        pos_X_ref = SX.sym('pos_X_ref')
        pos_Y_ref = SX.sym('pos_Y_ref')
        psi_ref = SX.sym('psi_ref')
        vel_ref = SX.sym('vel_ref')

        # weight 
        Q_pos = self.params["Q_pos"]
        Q_psi = self.params["Q_psi"]
        Q_vel = self.params["Q_vel"]


        # Assign reference and weight to the parameters
        p = casadi.vertcat(
                pos_X_ref,
                pos_Y_ref,
                psi_ref,
                vel_ref
            )

        self.model.p = p

        delta_pos_x = self.model.x[0] - pos_X_ref
        delta_pos_y = self.model.x[1] - pos_Y_ref
        delta_psi = casadi.mod((self.model.x[2] - psi_ref+np.pi/2),2*np.pi) - np.pi/2
        delta_vel = self.model.x[3] - vel_ref

        self.model.cost = delta_pos_x*Q_pos*delta_pos_x + delta_pos_y*Q_pos*delta_pos_y \
            + delta_psi*Q_psi*delta_psi + delta_vel*Q_vel*delta_vel
        

    def define_constraint(self):
        ''' 
        Define Constraints for the optimal control problem
        '''

        # State Constraint
        # initial constraint
        self.constraints.x0 = np.zeros(5)
        # State: [X, Y, psi: heading, V: speed, delta: steering angle]
        v_min = self.params['v_min']
        v_max = self.params['v_max']

        delta_min = self.params['delta_min']
        delta_max = self.params['delta_max']

        self.constraints.idxbx = np.array([3, 4])
        self.constraints.lbx = np.array([v_min, delta_min])
        self.constraints.ubx = np.array([v_max, delta_max])

        # Control input constraints
        # Control: [d: motor duty cycle, delta_dot: steering rate]

        d_min = self.params['d_min']
        d_max = self.params['d_max']

        deltadot_min = self.params['deltadot_min']  # minimum steering angle cahgne[rad/s]
        deltadot_max = self.params['deltadot_max']  # maximum steering angle cahgne[rad/s]

        self.constraints.idxbu = np.array([0,1])
        self.constraints.lbu = np.array([d_min, deltadot_min])
        self.constraints.ubu = np.array([d_max, deltadot_max])