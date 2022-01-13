import casadi
from casadi import SX
import numpy as np
import yaml
from planning_base import PlanningBase


class TrajPlanningKin(PlanningBase):
    def __init__(self, Tf, N, params_file = 'modelparams.yaml'):
        """
        Base Class of the trajectory tracking controller using ACADOS
        Input: 
            Tf - Time Horizon
            N - Number of discrete step in the time Horizon
        """
        super().__init__(Tf, N)
        
        with open(params_file) as file:
            self.params = yaml.load(file, Loader= yaml.FullLoader)
        self.acados_model.name = "traj_planning_kin"
        self.ACADOS_setup()
        
        
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
        
        # weight
        m = self.params['m']

        '''
        Step 2:

        Define State, Control, and System Dynamics 
            State: [X, Y, psi: heading, V: speed, d: motor duty cycle, delta: steering angle, theta: progress]
            Control: [d_dot: motor duty cycle change rate, delta_dot: steering rate, theta_dot: progress rate]  

                X_dot = V*cos(psi+beta)
                Y_dot = V*sin(psi+beta)
                psi_dot = V/l_r*sin(beta)
                V_dot = accel
                d_dot = d_dot
                delta_dot = delta_dot
                theta_dot = theta_dot
            with:
                accel = [(C_m1-C_m2*V*cos(beta))*d-C_roll-C_d*(V*cos(beta))^2]/m
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
        d_dot = SX.sym('d_dot')

        # Steering angle
        delta = SX.sym('delta')
        delta_dot = SX.sym('delta_dot')
        
        # progress
        theta = SX.sym('theta')
        theta_dot = SX.sym('theta_dot')
        
        # slip angle
        beta = casadi.atan2(casadi.tan(delta)*l_r, (l_f+l_r))
        vel_x = vel*casadi.cos(beta)
        
        # state vector
        x = casadi.vertcat(pos_X, pos_Y, psi, vel, d, delta, theta)
        x_dot = casadi.vertcat(pos_X_dot, pos_Y_dot, psi_dot, 
                               vel_dot, d_dot, delta_dot, theta_dot)
        
        # control input
        u = casadi.vertcat(d_dot, delta_dot, theta_dot)
        F_x = (casadi.if_else(d>0,(C_m1-C_m2*vel_x),C_m2*vel_x)*d-C_roll-C_d*vel_x*vel_x)

        # system dynamics
        f_expl = casadi.vertcat(
            vel*casadi.cos(psi+beta), # X_dot
            vel*casadi.sin(psi+beta), # Y_dot
            vel/l_r*casadi.sin(beta), # psi_dot
            F_x/m, # accel
            d_dot,
            delta_dot,
            theta_dot
        )

        self.acados_model.x = x
        self.acados_model.xdot = x_dot
        self.acados_model.u = u
        self.acados_model.f_expl_expr = f_expl
        self.acados_model.f_impl_expr = x_dot - f_expl

    def define_constrint(self):
        ''' 
        Define Constraints for the optimal control problem
        '''

        # State Constraint
        # initial constraint
        self.ocp.constraints.x0 = np.zeros(5)
        # State: [X, Y, psi: heading, V: speed, d: motor duty cycle, delta: steering angle, theta: progress]
        v_min = self.params['v_min']
        v_max = self.params['v_max']
        
        d_min = self.params['d_min']
        d_max = self.params['d_max']

        delta_min = self.params['delta_min']
        delta_max = self.params['delta_max']

        self.ocp.constraints.idxbx = np.array([3, 4, 5])
        self.ocp.constraints.lbx = np.array([v_min, d_min, delta_min])
        self.ocp.constraints.ubx = np.array([v_max, d_max, delta_max])

        # Control input constraints
        # Control: [d_dot: motor duty cycle, delta_dot: steering rate]
        
        # since we can contorl d directly, just assign a large value to allow immediate change of d
        ddot_min = -100.0
        ddot_max = 100.0
        
        deltadot_min = self.params['deltadot_min']  # minimum steering angle cahgne[rad/s]
        deltadot_max = self.params['deltadot_max']  # maximum steering angle cahgne[rad/s]
        
        thetadot_min = 0.0 # do not allow progress to decrease
        thetadot_max = 100.0

        self.ocp.constraints.idxbu = np.array([0,1,2])
        self.ocp.constraints.lbu = np.array([ddot_min, deltadot_min, thetadot_min])
        self.ocp.constraints.ubu = np.array([ddot_max, deltadot_max, thetadot_max])

        # non-linear constraints
        self.acados_model.con_h_expr = None
        self.ocp.constraints.lh = np.array([])
        self.ocp.constraints.uh = np.array([])

    def define_cost(self):
        '''
        Define Cost function of OCP 
        Consider a simple L2 cost for the state w.r.t to the reference trajectory
        J(t) = (X_ref-X)*Q_pos*(X_ref-X) + (Y_ref-Y)*Q_pos*(Y_ref-Y)
                + (Psi_ref - Psi)*Q_psi*(Psi_ref-Psi) + (Vel_ref-Vel)*Q_vel*(Vel_ref-Vel)
        '''

        # load cost function weight
        Q_c = self.params['Q_c'] # contouring error cost
        Q_l = self.params['Q_l'] # lag error cost
        Q_theta = self.params['Q_theta'] # progress cost
        
        R_d = self.params['R_d'] #regulization on duty cycle rate
        R_delta = self.params['R_d'] # regulization on the steering rate

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

        self.acados_model.p = p

        delta_pos_x = self.acados_model.x[0] - pos_X_ref
        delta_pos_y = self.acados_model.x[1] - pos_Y_ref
        delta_psi = casadi.mod((self.acados_model.x[2] - psi_ref+np.pi/2),2*np.pi) - np.pi/2
        delta_vel = self.acados_model.x[3] - vel_ref

        self.acados_model.cost_expr_ext_cost = delta_pos_x*Q_pos*delta_pos_x + delta_pos_y*Q_pos*delta_pos_y \
                + delta_psi*Q_psi*delta_psi + delta_vel*Q_vel*delta_vel \
                + self.acados_model.u[0]*Q_u*self.acados_model.u[0] + self.acados_model.u[1]*Q_u*self.acados_model.u[1]
        self.ocp.cost.cost_type = "EXTERNAL"
        

    
if __name__ == '__main__':

    Tf = 1
    N = Tf*20

    angle = np.linspace(0, np.pi/6, N)
    r = 2
    
    x_ref = r*np.cos(angle)
    y_ref = r*np.sin(angle)

    psi_ref = angle + np.pi/2
    vel_ref =r*np.pi/6/Tf*np.ones_like(angle)

    ref_traj = np.stack([x_ref, y_ref, psi_ref, vel_ref])


    x_0 = np.array([1.98, -0.02,  np.pi/2, vel_ref[0]*0.98, 0])

    # v = 1
    # x_ref = np.linspace(0, v*Tf, N,endpoint=False)
    # y_ref = np.zeros_like(x_ref)
    # psi_ref = np.zeros_like(x_ref)
    # v_ref = np.ones_like(x_ref)*v

    # ref_traj = np.stack([x_ref, y_ref, psi_ref, v_ref])
    # x_0 = np.array([0,0,1,0,0,0,0])

    ocp = TrajTrackingKin(Tf, N)
    ocp.solve(ref_traj, x_0)