from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
import numpy as np
from matplotlib import pyplot as plt
import casadi
from casadi import SX
import yaml

class MPCC():
    def __init__(self, Tf, N, params_file = 'modelparams.yaml'):
        """
        Base Class of the trajectory planning controller using ACADOS
        Input: 
            Tf - Time Horizon
            N - Number of discrete step in the time Horizon
        """        
        # define OCP in ACADOS
        self.ocp = AcadosOcp()
        self.acados_model = AcadosModel()

        # time horzion in s
        self.Tf = Tf
        # number of discrete step
        self.N = N

        self.acados_solver = None

        with open(params_file) as file:
            self.params = yaml.load(file, Loader= yaml.FullLoader)
        self.acados_model.name = "traj_planning_kin"
        self.ACADOS_setup()

    def define_sys(self):
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

        
        '''
        Define Cost function of OCP 
        '''

        # load cost function weight
        Q_c = self.params['Q_c'] # contouring error cost
        Q_l = self.params['Q_l'] # lag error cost
        Q_theta = self.params['Q_theta'] # progress cost
        
        R_d = self.params['R_d'] #regulization on duty cycle rate
        R_delta = self.params['R_d'] # regulization on the steering rate

        # projection point on the contour
        x_d_ref = SX.sym('x_d_ref')
        y_d_ref = SX.sym('y_d_ref')
        theta_ref = SX.sym('theta_ref')
        phi_ref = SX.sym('phi_ref')
        
        # linearize around the reference point on the countour 
        # and approximate x_d, y_d on the contour
        x_d = x_d_ref + casadi.cos(phi_ref)*(theta - theta_ref)
        y_d = y_d_ref + casadi.sin(phi_ref)*(theta - theta_ref)

        # calculate the contour and lag error
        e_c = casadi.sin(phi_ref)*(pos_X - x_d) - casadi.cos(phi_ref)*(pos_Y - y_d)
        e_l = -casadi.cos(phi_ref)*(pos_X - x_d) - casadi.sin(phi_ref)*(pos_Y - y_d)

        self.acados_model.cost_expr_ext_cost = e_c*Q_c*e_c + e_l*Q_l*e_l \
                                    - Q_theta*theta_dot + d_dot*R_d*d_dot \
                                    + delta_dot*R_delta*delta_dot

        self.ocp.cost.cost_type = "EXTERNAL"

        ''' 
        Define Constraints for the optimal control problem
        '''
        # State Constraint
        # initial constraint
        self.ocp.constraints.x0 = np.zeros(7)
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
        thetadot_max = 1e15

        self.ocp.constraints.idxbu = np.array([1,2])
        self.ocp.constraints.lbu = np.array([deltadot_min, thetadot_min])
        self.ocp.constraints.ubu = np.array([deltadot_max, thetadot_max])

        ''' Set external constraints'''
        # First external constraints is the road bound constraint
        # Assume e_c is a good approximation to the contour error
        # we want to make sure the vehicle stay in the road where
        # the right edge is the upper bound, and left edge is the 
        # lower bound
        
        rd_right = SX.sym('rd_right') # upper bound 
        rd_left = SX.sym('rd_left') #negative lower bound
        con_right = rd_right - e_c # should be positive
        con_left = rd_left + e_c # should be positive

        # Object avodiance
        obj_1_x = SX.sym('obj_1_x')
        obj_1_y = SX.sym('obj_1_y')
        obj_1_a = SX.sym('obj_1_a')
        obj_1_b = SX.sym('obj_1_b')
        obj_1_c = SX.sym('obj_1_c')
        
        dx_1 = (obj_1_x - pos_X)
        dy_1 = (obj_1_y - pos_Y)
        con1 = dx_1*dx_1*obj_1_a + 2*obj_1_b*dx_1*dy_1 + dy_1*dy_1*obj_1_c

        # self.acados_model.con_h_expr = casadi.vertcat(
        #         con_right, con_left#, con1
        #     )   
        # self.ocp.constraints.lh = np.array([0,0])
        # self.ocp.constraints.uh = np.array([1e15, 1e15])

        '''
        use "p" variable in the acados to handle time-varing variables,
        such as contouring reference points and external constraints
        '''
        p = casadi.vertcat(
                x_d_ref,
                y_d_ref,
                theta_ref,
                phi_ref,
                rd_right,
                rd_left,
                obj_1_x,
                obj_1_y,
                obj_1_a,
                obj_1_b,
                obj_1_c
            )

        self.acados_model.p = p

    def ACADOS_setup(self):
        '''
        This function setup parameters in ACADOS
        '''
        self.define_sys()

        self.ocp.model = self.acados_model

        # solver setting
        # set QP solver and integration
        self.ocp.dims.N = self.N
        self.ocp.solver_options.tf = self.Tf
        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'#'FULL_CONDENSING_QPOASES'#
        self.ocp.solver_options.nlp_solver_type = "SQP_RTI" #"SQP"# 
        self.ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        #self.ocp.solver_options.levenberg_marquardt = 1e-3
        self.ocp.solver_options.integrator_type = "ERK"

        # initial value for p
        self.ocp.parameter_values = np.zeros(self.acados_model.p.size()[0])

        # self.ocp.solver_options.nlp_solver_max_iter = 50
        # self.ocp.solver_options.qp_solver_iter_max = 100

        self.ocp.solver_options.tol = 1e-4

        self.acados_solver = AcadosOcpSolver(self.ocp, json_file="traj_tracking_acados.json")

    def solve(self, ref, x_cur, x_init = None, u_init = None):
        for stageidx  in range(self.N):
            p_val = ref[:,stageidx]
            self.acados_solver.set(stageidx, "p", p_val)
            
            # warm start
            if x_init is not None:
                self.acados_solver.set(stageidx, "x", x_init[stageidx,:])
            else:
                self.acados_solver.set(stageidx, "x", x_cur)

            if u_init is not None:
                self.acados_solver.set(stageidx, "u", u_init[stageidx,:])

        # set initial state
        self.acados_solver.set(0, "lbx", x_cur)
        self.acados_solver.set(0, "ubx", x_cur)

        # solve the system
        self.acados_solver.solve()
        self.acados_solver.print_statistics()
        
        x_sol = []
        u_sol = []
        for stageidx in range(self.N):
            x_sol.append(self.acados_solver.get(stageidx, 'x'))
            u_sol.append(self.acados_solver.get(stageidx, 'u'))

        x_sol = np.array(x_sol)
        u_sol = np.array(u_sol)
        return x_sol, u_sol                                     
                             

    