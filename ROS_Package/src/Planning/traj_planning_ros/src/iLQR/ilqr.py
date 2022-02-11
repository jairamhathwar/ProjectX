from os import stat
from weakref import ref
import numpy as np
from cost import Cost
from dynamics import Dynamics
import matplotlib.pyplot as plt
import time

class iLQR():
    def __init__(self, ref_path, params):
        
        self.T = params['T']
        self.N = params['N']
        
        self.ref_path = ref_path

        self.steps = 500
        #self.line_search_step = 0.5
        self.tol = 1e-5
        self.dynamics = Dynamics(params)
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
            #print('k', i, k)
            #print('K', i, K)
            u = nominal_controls[:,i]+alpha*k+ K @ (X[:, i] - nominal_states[:, i])
            #print('U', i,  U[:,i])
            X[:,i+1], U[:,i] = self.dynamics.forward_step(X[:,i], u, step=10)

        J, closest_pt, slope = self.cost.get_cost(X, U)
        
        return X, U, J, closest_pt, slope
        
    def backward_pass(self, nominal_states, nominal_controls, closest_pt, slope):
        L_x, L_xx, L_u, L_uu = self.cost.get_derivatives(nominal_states, nominal_controls, closest_pt, slope)
        #print(L_u)
        fx, fu = self.dynamics.get_AB_matrix(nominal_states, nominal_controls)

        k_open_loop = np.zeros((self.dim_u, self.N-1))
        K_closed_loop = np.zeros((self.dim_u, self.dim_x, self.N-1))

        # derivative of value function at final step
        V_x = L_x[:,-1]
        V_xx = L_xx[:,:,-1]
        for i in range(self.N-2, -1, -1):
            Q_x = L_x[:,i] + fx[:,:,i].T @ V_x
            Q_u = L_u[:,i] + fu[:,:,i].T @ V_x
            Q_xx = L_xx[:,:,i] + fx[:,:,i].T @ V_xx @ fx[:,:,i] + 0.05*np.eye(4)
            Q_ux =  fu[:,:,i].T @ V_xx @ fx[:,:,i] # L_uxis 0
            Q_uu = L_uu[:,:,i] + fu[:,:,i].T @ V_xx @ fu[:,:,i] + 0.1*np.eye(2)
            
            k_open_loop[:,i] = -np.linalg.lstsq(Q_uu, Q_u, rcond=None)[0]            
            K_closed_loop[:, :, i] = -np.linalg.lstsq(Q_uu, Q_ux, rcond=None)[0]

            # Update value function derivative for the previous time step
            V_x = Q_x - K_closed_loop[:,:,i].T @ Q_uu @ k_open_loop[:,i]
            V_xx = Q_xx - K_closed_loop[:,:,i].T @ Q_uu @ K_closed_loop[:,:,i]

        return K_closed_loop, k_open_loop

    def solve(self, cur_state, controls = None):
        time0 = time.time()
        if controls is None:
            controls = np.zeros((self.dim_u, self.N))
            #controls[0,:] = 1
        
        states = np.zeros((self.dim_x, self.N))
        states[:,0] = cur_state
        for i in range(1,self.N):
            states[:,i],_ = self.dynamics.forward_step(states[:,i-1], controls[:,i-1], step = 10)

        J, closest_pt, slope = self.cost.get_cost(states, controls)

        converged = False
        for i in range(self.steps):
            
            #print("step", i)
            K_closed_loop, k_open_loop = self.backward_pass(states, controls, closest_pt, slope)
            J_local_opt = 1e15
            use_local_opt = False
            for alpha in self.alphas :
                X_new, U_new, J_new, closest_pt_new, slope_new = self.forward_pass(states, controls, K_closed_loop, k_open_loop, alpha)
                
                if J_new<J:
                    if np.abs((J - J_new) / J) < self.tol:
                        converged = True
                    #print("step ", i, "reduce the cost to ", J_new)
                    J = J_new
                    states = X_new
                    controls = U_new
                    closest_pt = closest_pt_new
                    slope = slope_new
                    use_local_opt = False
                    break
                elif J_new<J_local_opt:
                    use_local_opt = True
                    J_local_opt = J_new
                    states_local = X_new
                    controls_local = U_new
                    closest_pt_local = closest_pt_new
                    slope_local = slope_new
                    if np.abs((J_new - J_local_opt)) < 1e-2:
                        
                        break
                
            if use_local_opt:
                J = J_local_opt
                states = states_local
                controls = controls_local
                closest_pt = closest_pt_local
                slope = slope_local
            #print('Step ', i, "with cost ", J) 
            
            # self.ref_path.plot_track()
            # plt.plot(states[0,:], states[1,:])
            # plt.axis('equal')

            # # plt.figure()
            # # plt.plot(states[2,:], label='v')
            # # plt.plot(states[3,:], label='psi')
            # # plt.plot(controls[0,:], '--', label='a')
            # # plt.plot(controls[1,:], '--', label='delta')
            # # plt.legend()
            # plt.show()
            if converged:
                print("converged")
                break
        print(time.time()-time0)
        print("exit at step", i, "with final cost ", J)
        return states, controls





        
        

