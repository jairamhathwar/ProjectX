import numpy as np

class Vehicle:
    def __init__(self, width, length, max_a, max_v):
        self.width = width
        self.length = length
        self.a_max = max_a
        self.v_max = max_v
        self.vertex_center = np.array([[-length/2.0, length/2.0, length/2.0, -length/2.0], 
                        [width/2.0, width/2.0, -width/2.0, -width/2.0],[1,1,1,1]])

    def plot(self, ax, state, alpha=1):
        # state = [x,y,theta,......]

        T = np.array([[np.cos(state[2]), -np.sin(state[2]), state[0]],
                        [np.sin(state[2]), np.cos(state[2]), state[1]],
                        [0,0,1]])
        vertex_t = np.matmul(T, self.vertex_center)
        return ax.fill(vertex_t[0,:], vertex_t[1,:],color='blue', alpha=alpha)