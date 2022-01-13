from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from abc import ABC, abstractmethod
import numpy as np
from matplotlib import pyplot as plt

class PlanningBase(ABC):
    def __init__(self, Tf, N):
        """
        Base Class of the trajectory planning controller using ACADOS
        Input: 
            Tf - Time Horizon
            N - Number of discrete step in the time Horizon
        """
        super().__init__()
        
        # define OCP in ACADOS
        self.ocp = AcadosOcp()
        self.acados_model = AcadosModel()

        # time horzion in s
        self.Tf = Tf
        # number of discrete step
        self.N = N

        self.acados_solver = None

    @abstractmethod
    def define_sys_dyn(self):
        pass

    @abstractmethod
    def define_constrint(self):
        pass

    @abstractmethod
    def define_cost(self):
        pass

    def ACADOS_setup(self):
        '''
        This function setup parameters in ACADOS
        '''
        self.define_sys_dyn()
        self.define_cost()
        self.define_constrint()

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

        self.ocp.solver_options.tol = 1e-2

        self.acados_solver = AcadosOcpSolver(self.ocp, json_file="traj_tracking_acados.json")

    def solve(self, ref_traj, x_cur, x_init = None, u_init = None):
        for stageidx  in range(self.N):
            p_val = ref_traj[:,stageidx]
            self.acados_solver.set(stageidx, "p", p_val)
            
            # warm start
            if x_init is not None:
                self.acados_solver.set(stageidx, "x", x_init[:, stageidx])
            else:
                self.acados_solver.set(stageidx, "x", x_cur)

            if u_init is not None:
                self.acados_solver.set(stageidx, "u", u_init[:, stageidx])

        # set initial state
        self.acados_solver.set(0, "lbx", x_cur)
        self.acados_solver.set(0, "ubx", x_cur)

        # solve the system
        self.acados_solver.solve()
        
        x_sol = []
        u_sol = []
        for stageidx in range(self.N):
            x_sol.append(self.acados_solver.get(stageidx, 'x'))
            u_sol.append(self.acados_solver.get(stageidx, 'u'))

        x_sol = np.array(x_sol)
        u_sol = np.array(u_sol)
        return x_sol, u_sol                                     
                             

    