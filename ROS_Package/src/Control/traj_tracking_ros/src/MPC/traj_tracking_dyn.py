import casadi
from casadi import SX
import numpy as np
import yaml
from traj_tracking_base import TrajTrackingBase
import matplotlib.pyplot as plt

class TrajTrackingDyn(TrajTrackingBase):
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
        self.acados_model.name = "traj_tracking_dyn"
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
        F_x = (casadi.if_else(d>0,(C_m1-C_m2*Vx),C_m2*Vx)*d-C_roll-C_d*Vx*Vx)/2
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

        self.acados_model.x = x
        self.acados_model.xdot = x_dot
        self.acados_model.u = u
        self.acados_model.f_expl_expr = f_expl
        self.acados_model.f_impl_expr = x_dot - f_expl

    def define_constrint(self):
        ''' 
        Define ocp.constraints for the optimal control problem
        '''

        # State Constraint
        # initial constraint
        self.ocp.constraints.x0 = np.zeros(7)

        #State: [X, Y, Vx, Vy, psi: heading, omega: yaw rate, delta: steering angle]
        v_min = self.params['v_min']
        v_max = self.params['v_max']

        delta_min = self.params['delta_min']
        delta_max = self.params['delta_max']

        self.ocp.constraints.idxbx = np.array([2,6])
        self.ocp.constraints.lbx = np.array([v_min, delta_min])
        self.ocp.constraints.ubx = np.array([v_max, delta_max])

        # Control input ocp.constraints
        # Control: [d: motor duty cycle, delta_dot: steering rate]

        d_min = self.params['d_min']
        d_max = self.params['d_max']

        deltadot_min = self.params['deltadot_min']  # minimum steering angle cahgne[rad/s]
        deltadot_max = self.params['deltadot_max']  # maximum steering angle cahgne[rad/s]

        self.ocp.constraints.idxbu = np.array([0,1])
        self.ocp.constraints.lbu = np.array([d_min, deltadot_min])
        self.ocp.constraints.ubu = np.array([d_max, deltadot_max])

        # non-linear ocp.constraints
        self.acados_model.con_h_expr = None 
        self.ocp.constraints.lh  = np.array([])
        self.ocp.constraints.uh  = np.array([])
        # casadi.vertcat(
        #         self.acados_model.x[2]**2+self.acados_model.x[3]**2
        #     )
        # self.ocp.constraints.lh = np.array([v_min**2])
        # self.ocp.constraints.uh = np.array([v_max**2])

    def define_cost(self):
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

        self.acados_model.p = p

        delta_pos_x = self.acados_model.x[0] - pos_X_ref
        delta_pos_y = self.acados_model.x[1] - pos_Y_ref
        delta_psi = casadi.mod((self.acados_model.x[4] - psi_ref+np.pi/2),2*np.pi) - np.pi/2
        delta_vel = self.acados_model.x[2] - vel_ref

        self.acados_model.cost_expr_ext_cost = delta_pos_x*Q_pos*delta_pos_x + delta_pos_y*Q_pos*delta_pos_y \
                + self.acados_model.u[0]*Q_u*self.acados_model.u[0] + self.acados_model.u[1]*Q_u*self.acados_model.u[1] \
                + delta_psi*Q_psi*delta_psi + delta_vel*Q_vel*delta_vel
                
        self.ocp.cost.cost_type = "EXTERNAL"
        

    
if __name__ == '__main__':

    Tf = 1
    step = 20
    N = int(Tf*step)

    # angle = np.linspace(0, np.pi/6, N)
    # r = 2
    
    # x_ref = r*np.cos(angle)
    # y_ref = r*np.sin(angle)

    # psi_ref = angle + np.pi/2
    # vel_ref =r*np.pi/6/Tf*np.ones_like(angle)

    # ref_traj = np.stack([x_ref, y_ref, psi_ref, vel_ref])
    

    # x_0 = np.array([1.98, -0.02,  vel_ref[0]*0.98, 0, np.pi/2, 0, 0])
    ''' turn and slow down '''
    dt = 1.0/step
    dangle = np.linspace(np.pi/4/N, 0, N)
    angle = np.cumsum(dangle)
    angle = np.insert(angle[:-1], 0, 0)
    r = 2
    
    
    x_ref = r*np.cos(angle)
    y_ref = r*np.sin(angle)

    psi_ref = angle + np.pi/2
    vel_ref =r*dangle/dt

    ref_traj = np.stack([x_ref, y_ref, psi_ref, vel_ref])
    

    x_0 = np.array([2, 0,  vel_ref[0], 0, np.pi/2, 0, 0])
    

    '''GO straight'''
    
    # v = 1
    # x_ref = np.linspace(0, v*Tf, N,endpoint=False)
    # y_ref = np.zeros_like(x_ref)
    # psi_ref = np.zeros_like(x_ref)
    # v_ref = np.ones_like(x_ref)*v

    # ref_traj = np.stack([x_ref, y_ref, psi_ref, v_ref])
    # x_0 = np.array([0,0,1,0,0,0,0])

    ''' slow down '''
    # v_0 = 5
    # v_ref = np.linspace(v_0,v_0-3,N)
    # dt = 1.0/step
    # x_ref = np.cumsum(v_ref)*dt
    # x_ref = np.insert(x_ref[:-1], 0, 0)
    # y_ref = np.zeros_like(x_ref)
    # psi_ref = np.zeros_like(x_ref)
    # ref_traj = np.stack([x_ref, y_ref, psi_ref, v_ref])
    # x_0 = np.array([0,0,v_0,0,0,0,0])

    # x_init = np.cumsum(np.arange(N))*dt
    # v_init = v_0*np.ones_like(x_init)
    # state_init = np.stack([x_init, y_ref, v_init, y_ref, y_ref, y_ref, y_ref])
    # u_init = np.zeros((2,N))


    ocp = TrajTrackingDyn(Tf, N)
    x_sol, u_sol = ocp.solve(ref_traj, x_0)#, state_init, u_init)

    #print(x_sol[:,0] - ref_traj[0,:])
    # print(u_sol)
    #print(ref_traj.T)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(x_sol[:,0], x_sol[:,1],'-.')
    ax1.plot(ref_traj[0,:], ref_traj[1,:], '.')
    
    ax2.plot(x_sol[:,2],'-') # vx
    ax2.plot(ref_traj[-1,:],'-') # vx
    ax2.plot(x_sol[:,3],'.') # vy
    ax2.plot(u_sol[:,0],'-.') # d
    ax2.plot(u_sol[:,1],'--') # delta
    plt.show()