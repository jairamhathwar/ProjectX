import numpy as np

class Constraints:
    def __init__(self, params, track) -> None:
        self.track = track
        
        self.delta_min = params['delta_min']
        self.delta_max = params['delta_max']
        self.a_min = params['a_min']
        self.a_max = params['a_max']
        self.v_max = params['v_max']
    
        self.zeros = np.zeros((self.N))
        self.ones = np.ones((self.N))
        
    def get_constraint_state_derivative(self, nominal_states, closest_pt, slope):
        '''
        nominal_states: [d=4xN] array
        '''
        L_x = np.zeros((4,self.N))
        L_xx = np.zeros((4,4,self.N))
        
        # cost due to right road boundary                
        transform = np.array([np.sin(slope), -np.cos(slope), self.zeros, self.zeros])
        
        ref_states = np.zeros_like(nominal_states)
        ref_states[:2, :] = closest_pt
        ref_states[:2, :] = v_ref
        
        error = nominal_states - ref_states
        Q_trans = np.einsum('dbn, bdn->ddn', np.einsum('dan, abn -> dbn', transform.transpose(1,0,2), self.Q), transform)
        
        # shape [4xN]
        L_x = np.einsum('abn, bn->an', Q_trans, error)
        # shape [4x4xN]
        L_xx = Q_trans
        
        return L_x, L_xx
    
    def get_constraint_control_derivative(self, nominal_controls):
        '''
        nominal_control: [d=2xN] array
        '''
        L_u = np.einsum('abn, bn->an', self.R, nominal_controls)
        L_uu = self.R
        
        return L_u, L_uu
    
    
    def barrier_function(self, q1, q2, c, c_dot):
        '''
        c = [n] array
        c_dot = [dxn] array
        '''
		b = q1*np.exp(q2*c)
		b_dot = np.einsum('n,an->an', q2*b, c_dot)
		b_ddot = np.einsum('n,abn->abn', (q2**2)*b, np.einsum('an,bn->abn',c_dot, c_dot))

		return b, b_dot, b_ddot