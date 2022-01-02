import casadi
from casadi import SX
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from KinematicBicycleModel import KinematicBicycleModel
from DynamicBicycleModel import DynamicBicycleModel
import numpy as np
from matplotlib import pyplot as plt
import time

class TrajTrackingACADOS:
    def __init__(self, Tf, N, model_params = "modelparams.yaml"):
        
        bicycle_model = DynamicBicycleModel(model_params)
        self.model = bicycle_model.model
        self.constraints = bicycle_model.constraints

        # define OCP in ACADOS
        self.ocp = AcadosOcp()
        self.acados_model = AcadosModel()

        # time horzion in s
        self.Tf = Tf
        # number of discrete step
        self.N = N

        self.acados_solver = None

        self.acados_setting()
    
    def acados_setting(self):
        '''
        This function paste model, cost, and constraints into ACADOS model template
        '''

        # set system dynamics 
        self.acados_model.name = self.model.name
        self.acados_model.f_expl_expr = self.model.f_expl_expr
        self.acados_model.f_impl_expr = self.model.f_impl_expr

        self.acados_model.x = self.model.x
        self.acados_model.xdot = self.model.xdot

        self.acados_model.u = self.model.u
        self.acados_model.p = self.model.p
        

        # set cost function
        self.acados_model.cost_expr_ext_cost = self.model.cost
        self.ocp.cost.cost_type = "EXTERNAL"
        
        

        # set constraint
        self.ocp.constraints.idxbu = self.constraints.idxbu
        self.ocp.constraints.lbu = self.constraints.lbu
        self.ocp.constraints.ubu = self.constraints.ubu

        self.ocp.constraints.idxbx = self.constraints.idxbx
        self.ocp.constraints.lbx = self.constraints.lbx
        self.ocp.constraints.ubx = self.constraints.ubx

        self.acados_model.con_h_expr = self.constraints.con_h_expr
        self.ocp.constraints.lh = self.constraints.lh
        self.ocp.constraints.uh = self.constraints.uh

        self.ocp.constraints.x0 = self.constraints.x0

        self.ocp.dims.N = self.N

        self.ocp.model = self.acados_model

        # solver setting
        # set QP solver and integration
        self.ocp.solver_options.tf = self.Tf
        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'#'FULL_CONDENSING_QPOASES'
        self.ocp.solver_options.nlp_solver_type = "SQP"#"SQP_RTI" #
        self.ocp.solver_options.hessian_approx = "EXACT"
        self.ocp.solver_options.levenberg_marquardt = 0.1
        self.ocp.solver_options.integrator_type = "ERK"
        self.ocp.parameter_values = np.zeros(self.model.p.size()[0])

        self.ocp.solver_options.nlp_solver_max_iter = 100
        self.ocp.solver_options.qp_solver_iter_max = 100
        self.ocp.solver_options.tol = 1e-3
        #ocp.solver_options.print_level = 1
        # ocp.solver_options.nlp_solver_tol_comp = 1e-1

        self.acados_solver = AcadosOcpSolver(self.ocp, json_file="acados_ocp_dynamic.json")


    def solve(self, ref_traj, x_cur):
        t0 = time.time()
        for stageidx  in range(self.N):
            p_val = ref_traj[:,stageidx]
            self.acados_solver.set(stageidx, "p", p_val)
            self.acados_solver.set(stageidx, "x", x_cur)
        
        # set initial state
        self.acados_solver.set(0, "lbx", x_cur)
        self.acados_solver.set(0, "ubx", x_cur)
        status = self.acados_solver.solve()
        t1 = time.time()
        print(t1-t0)
        x_sol = []
        u_sol = []
        for stageidx in range(self.N):
            x_sol.append(self.acados_solver.get(stageidx, 'x'))
            u_sol.append(self.acados_solver.get(stageidx, 'u'))
        self.acados_solver.print_statistics()
        x_sol = np.array(x_sol)
        u_sol = np.array(u_sol)
        #print(x_sol)
        # #print(u_solution)
        #print(ref_traj.T)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(x_sol[:,0], x_sol[:,1],'-.')
        ax1.plot(ref_traj[0,:], ref_traj[1,:], '.')
        
        ax2.plot(x_sol[:,3],'-')
        ax2.plot(x_sol[:,4],'.')
        ax2.plot(u_sol[:,0],'-.')
        ax2.plot(u_sol[:,1],'--')
        plt.show()
        return x_sol, u_sol

            

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


    x_0 = np.array([1.98, -0.02, vel_ref[0]*0.98, 0, np.pi/2, 0.1, 0])

    # v = 1
    # x_ref = np.linspace(0, v*Tf, N,endpoint=False)
    # y_ref = np.zeros_like(x_ref)
    # psi_ref = np.zeros_like(x_ref)
    # v_ref = np.ones_like(x_ref)*v

    # ref_traj = np.stack([x_ref, y_ref, psi_ref, v_ref])
    # x_0 = np.array([0,0,1,0,0,0,0])

    ocp = TrajTrackingACADOS(Tf, N)
    ocp.solve(ref_traj, x_0)

    

