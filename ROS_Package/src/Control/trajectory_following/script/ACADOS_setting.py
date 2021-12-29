import casadi
from casadi import SX
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from KinematicBicycleModel import KinematicBicycleModel
import numpy as np
from matplotlib import pyplot as plt

class TrajTrackingACADOS:
    def __init__(self, Tf, N, model_params = "modelparams.yaml"):
        
        bicycle_model = KinematicBicycleModel(model_params)
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
        
        self.ocp.model = self.acados_model

        # set constraint
        self.ocp.constraints.idxbu = self.constraints.idxbu
        self.ocp.constraints.lbu = self.constraints.lbu
        self.ocp.constraints.ubu = self.constraints.ubu

        self.ocp.constraints.idxbx = self.constraints.idxbx
        self.ocp.constraints.lbx = self.constraints.lbx
        self.ocp.constraints.ubx = self.constraints.ubx

        self.ocp.constraints.x0 = self.constraints.x0

        self.ocp.dims.N = self.N

        # solver setting
        # set QP solver and integration
        self.ocp.solver_options.tf = self.Tf
        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        #ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        self.ocp.solver_options.nlp_solver_type = "SQP"#"SQP_RTI" #
        self.ocp.solver_options.hessian_approx = "EXACT"
        #self.ocp.solver_options.levenberg_marquardt = 0.1
        self.ocp.solver_options.integrator_type = "ERK"
        self.ocp.parameter_values = np.zeros(self.model.p.size()[0])

        #self.ocp.solver_options.nlp_solver_step_length = 0.01
        self.ocp.solver_options.nlp_solver_max_iter = 50
        self.ocp.solver_options.tol = 1e-4
        #ocp.solver_options.print_level = 1
        # ocp.solver_options.nlp_solver_tol_comp = 1e-1

        self.acados_solver = AcadosOcpSolver(self.ocp, json_file="acados_ocp_dynamic.json")


    def solve(self, ref_traj, x_cur):
        for stageidx  in range(self.N):
            p_val = ref_traj[:,stageidx]
            self.acados_solver.set(stageidx, "p", p_val)
            self.acados_solver.set(stageidx, "x", x_cur)
        
        # set initial state
        self.acados_solver.set(0, "lbx", x_cur)
        self.acados_solver.set(0, "ubx", x_cur)
        status = self.acados_solver.solve()
        
        x_solution = []
        u_solution = []
        for stageidx in range(self.N):
            x_solution.append(self.acados_solver.get(stageidx, 'x'))
            u_solution.append(self.acados_solver.get(stageidx, 'u'))
        self.acados_solver.print_statistics()
        x_solution = np.array(x_solution)
        u_solution = np.array(u_solution)
        print(x_solution)
        #print(u_solution)
        print(ref_traj.T)
        plt.plot(x_solution[:,0], x_solution[:,1],'-.')
        plt.plot(ref_traj[0,:], ref_traj[1,:], '-')
        plt.show()


            

if __name__ == '__main__':

    Tf = 6
    N = Tf*10

    angle = np.linspace(0, np.pi/3, N)
    r = 2
    
    x_ref = r*np.cos(angle)
    y_ref = r*np.sin(angle)

    psi_ref = angle + np.pi/2
    vel_ref =r*np.pi/3/Tf*np.ones_like(angle)

    ref_traj = np.stack([x_ref, y_ref, psi_ref, vel_ref])

    x_0 = np.array([2.1, -0.1, np.pi/2, 0, 0])

    v = 1
    # x_ref = np.linspace(0, v*Tf, N,endpoint=False)
    # y_ref = np.zeros_like(x_ref)
    # psi_ref = np.zeros_like(x_ref)
    # v_ref = np.ones_like(x_ref)*v

    # ref_traj = np.stack([x_ref, y_ref, psi_ref, v_ref])
    # x_0 = np.array([0, 0, 0 , 0, 0])

    ocp = TrajTrackingACADOS(Tf, N)
    ocp.solve(ref_traj, x_0)

    
