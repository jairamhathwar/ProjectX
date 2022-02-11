from weakref import ref
import numpy as np
from cost import Cost
from dynamics import Dynamics

class iLQR():
    def __init__(self, T, N, ref_path, params):
        self.T = T
        self.N = N
        self.ref_path = ref_path

        self.steps = 100
        self.line_search_step = 0.5
        self.tol = 1e-5
        self.dynamics = Dynamics(T, N, params)

        self.alphas = 1.1**(-np.arange(10)**2)

        self.dim_x = self.dynamics.dim_x
        self.dim_u = self.dynamics.dim_u

        self.cost = Cost(params, ref_path)
        
    def forward_pass(self, nominal_states, nominal_controls, K_closed_loop, k_open_loop, alpha):
        X = np.zeros_like(nominal_states)
        U = np.zeros_like(nominal_controls)
        
        X[:,0] = nominal_states[:,0]
        for i in range(self.N-1):
            K = K_closed_loop[:,:,i]
            k = k_open_loop[:,i]
            U[:,i] = nominal_controls[:,i]+alpha*k+ K[:, :, i] @ (X[:, i] - nominal_states[:, i])
            X[:,i+1] = self.dynamics.forward_step(X[:,i], U[:,i], step=10)

        J, closest_pt, slope = self.cost.get_cost(X, U)
        
        return X, U, J, closest_pt, slope
        
    def backward_pass(self, nominal_states, nominal_controls, closest_pt, slope):
        L_x, L_xx, L_u, L_uu = self.cost.get_derivatives(nominal_states, nominal_controls, closest_pt, slope)
        fx, fu = self.dynamics.get_AB_matrix(nominal_states, nominal_controls)

        k_open_loop = np.zeros((self.dim_u, self.N-1))
        K_closed_loop = np.zeros((self.dim_u, self.dim_x, self.N-1))

        # derivative of value function at final step
        V_x = L_x[:,-1]
        V_xx = L_xx[:,:,-1]

        for i in range(self.N-1, -1, -1):
            Q_x = L_x[:,i] + fx[:,:,i].T @ V_x
            Q_u = L_u[:,i] + fu[:,:,i].T @ V_x
            Q_xx = L_xx[:,:,i] + fx[:,:,i].T @ V_xx @ fx[:,:,i] 
            Q_ux =  fu[:,:,i].T @ V_xx @ fx[:,:,i] # L_uxis 0
            Q_uu = L_uu[:,:,i] + fu[:,:,i].T @ V_xx @ fu[:,:,i]

            k_open_loop[:,i] = -np.linalg.lstsq(Q_uu, Q_u)
            K_closed_loop[:, i] = -np.linalg.lstsq(Q_uu, Q_ux)

            # Update value function derivative for the previous time step
            V_x = Q_x - K_closed_loop[:,:,i].T @ Q_uu @ k_open_loop[:,i]
            V_xx = Q_xx - K_closed_loop[:,:,i].T @ Q_uu @ K_closed_loop[:,:,i]

        return K_closed_loop, k_open_loop

    def solve(self, cur_state, controls = None):
        if controls is None:
            controls = np.zeros((self.dim_u, self.N))
        
        states = np.zeros((self.dim_x, self.N))
        states[:,0] = cur_state
        for i in range(1,self.N):
            states[:,i] = self.dynamics.forward_step(states[:,i-1], controls[:,i-1], step = 10)

        J, closest_pt, slope = self.cost.get_cost(states, controls)

        converged = False
        for _ in range(self.steps):
            K_closed_loop, k_open_loop = self.backward_pass(states, controls, closest_pt, slope)

            for alpha in self.alphas :
                X_new, U_new, J_new, closest_pt_new, slope_new = self.forward_pass(states, controls, K_closed_loop, k_open_loop, alpha)
                if J_new<J:
                    if np.abs((J - J_new) / J) < self.tol:
                        converged = True
                    J = J_new
                    states = X_new
                    controls = U_new
                    closest_pt = closest_pt_new
                    slope = slope_new
                    break
            if converged:
                break

        return states, controls





        
        

