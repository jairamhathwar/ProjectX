import casadi
from casadi import SX
import yaml
import numpy as np

class DynamicBicycleModel:
    def __init__(self, params_file):
        # System Dynamics of Kinematic Bicycle Model
        self.model = casadi.types.SimpleNamespace()
        self.model.name = 'DynamicBicycleModel'

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
        C_roll = self.params['C_roll']
        C_d = self.params['C_d']

        # tire coefficient
        B_f = self.params['B_f']
        B_r = self.params['B_r']
        C_f = self.params['C_f']
        C_r = self.params['C_r']
        D_f = self.params['D_f']
        D_r = self.params['D_r']

        
        # weight
        m = self.params['m']
        Iz = self.params['Iz']

        '''
        Step 2:

        Define State, Control, and System Dynamics 
            State: [X, Y, Vx, Vy, psi: heading, omega: yaw rate, delta: steering angle]
            Control: [d: motor duty cycle, delta_dot: steering rate]  

                X_dot = Vx*cos(psi)-Vy*sin(psi)
                Y_dot = Vx*sin(psi)+Vy*cos(psi)
                Vx_dot = (F_x+F_x*cos(delta)-F_fy*sin(delta)+m*Vy*omega)/m, # Vx_dot
                Vy_dot = (F_ry+F_x*sin(delta)+F_fy*cos(delta)-m*Vy*omega)/m, # Vy_dot
                psi_dot = omega
                omega = ((F_fy*cos(delta)+F_x*sin(delta))*l_f-F_ry*l_r)/Iz
                delta_dot = delta_dot

            with:
                F_fx = [(C_m1-C_m2*V*cos(beta))*d-C_r-C_d*(V*cos(beta))^2]/2
                F_fy = D_f*sin(C_f*atan(B_f*alpha_f))
                F_ry = D_r*sin(C_r*atan(B_r*alpha_r))

                alpha_f = delta-atan((omega*l_f+Vy)/Vx)
                alpha_r = atan((omega*l_r-Vy)/Vx)
        '''

        # Position
        pos_X = SX.sym('pos_X')
        pos_X_dot = SX.sym('pos_X_dot')
        pos_Y = SX.sym('pos_Y')
        pos_Y_dot = SX.sym('pos_Y_dot')

        # heading
        psi = SX.sym('psi')
        psi_dot = SX.sym('psi_dot')

        # angular velocity        
        omega = SX.sym('omega')
        omega_dot = SX.sym('omega_dot')

        # speed
        Vx = SX.sym('Vx')
        Vx_dot = SX.sym('Vx_dot')

        Vy = SX.sym('Vy')
        Vy_dot = SX.sym('Vy_dot')
        
        # motor duty cycle
        d = SX.sym('d')

        # Steering angle
        delta = SX.sym('delta')
        delta_dot = SX.sym('delta_dot')

        # slip angle
        alpha_f = delta-casadi.atan2((omega*l_f+Vy), Vx)
        alpha_r = casadi.atan2((omega*l_r-Vy), Vx)

        # Tire force
        F_x = ((C_m1-C_m2*Vx)*d-C_roll-C_d*Vx*Vx)/2
        F_fy = D_f*casadi.sin(C_f*casadi.atan(B_f*alpha_f))
        F_ry = D_r*casadi.sin(C_r*casadi.atan(B_r*alpha_r))

        # state vector
        x = casadi.vertcat(pos_X, pos_Y, Vx, Vy, psi, omega, delta)
        x_dot = casadi.vertcat(pos_X_dot, pos_Y_dot, Vx_dot, Vy_dot, psi_dot, omega_dot, delta_dot)
        
        # control input
        u = casadi.vertcat(d, delta_dot)

        # system dynamics
        f_expl = casadi.vertcat(
            Vx*casadi.cos(psi)-Vy*casadi.sin(psi), # X_dot
            Vx*casadi.sin(psi)+Vy*casadi.cos(psi), # Y_dot
            (F_x+F_x*casadi.cos(delta)-F_fy*casadi.sin(delta)+m*Vy*omega)/m, # Vx_dot
            (F_ry+F_x*casadi.sin(delta)+F_fy*casadi.cos(delta)-m*Vy*omega)/m, # Vy_dot
            omega,
            ((F_fy*casadi.cos(delta)+F_x*casadi.sin(delta))*l_f-F_ry*l_r)/Iz,
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
                + (Psi_ref - Psi)*Q_psi*(Psi_ref-Psi) + ||U||_2
        '''

        # load cost function weight
        Q_pos = self.params['Q_pos']
        Q_psi = self.params['Q_psi']
        Q_vel = self.params['Q_vel']
        Q_u = self.params['Q_u']

        # Reference Position, heading and speed
        pos_X_ref = SX.sym('pos_X_ref')
        pos_Y_ref = SX.sym('pos_Y_ref')
        psi_ref = SX.sym('psi_ref')
        vel_ref = SX.sym('vel_ref')

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
        delta_psi = casadi.mod((self.model.x[4] - psi_ref+np.pi/2),2*np.pi) - np.pi/2
        delta_vel = self.model.x[2] - vel_ref

        self.model.cost = delta_pos_x*Q_pos*delta_pos_x + delta_pos_y*Q_pos*delta_pos_y \
                + self.model.u[0]*Q_u*self.model.u[0] + self.model.u[1]*Q_u*self.model.u[1] \
                + delta_psi*Q_psi*delta_psi + delta_vel*Q_vel*delta_vel
        

    def define_constraint(self):
        ''' 
        Define Constraints for the optimal control problem
        '''

        # State Constraint
        # initial constraint
        self.constraints.x0 = np.zeros(7)

        #State: [X, Y, Vx, Vy, psi: heading, omega: yaw rate, delta: steering angle]
        v_min = self.params['v_min']
        v_max = self.params['v_max']

        delta_min = self.params['delta_min']
        delta_max = self.params['delta_max']

        self.constraints.idxbx = np.array([6])
        self.constraints.lbx = np.array([delta_min])
        self.constraints.ubx = np.array([delta_max])

        # Control input constraints
        # Control: [d: motor duty cycle, delta_dot: steering rate]

        d_min = self.params['d_min']
        d_max = self.params['d_max']

        deltadot_min = self.params['deltadot_min']  # minimum steering angle cahgne[rad/s]
        deltadot_max = self.params['deltadot_max']  # maximum steering angle cahgne[rad/s]

        self.constraints.idxbu = np.array([0,1])
        self.constraints.lbu = np.array([d_min, deltadot_min])
        self.constraints.ubu = np.array([d_max, deltadot_max])

        # non-linear constraints
        self.constraints.con_h_expr = casadi.vertcat(self.model.x[2]**2+self.model.x[3]**2)
        self.constraints.lh = np.array([v_min])
        self.constraints.uh = np.array([v_max])
