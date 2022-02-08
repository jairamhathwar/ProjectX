import numpy as np

class Obstacle_static:
    def __init__(self, s, l, pos_xyt, width, length):
        self.width = width
        self.length = length
        self.s = s
        self.l = l
        self.pos_xyt = pos_xyt
        self.pose = np.array([[np.cos(pos_xyt[2]), -np.sin(pos_xyt[2]), pos_xyt[0]],
                        [np.sin(pos_xyt[2]), np.cos(pos_xyt[2]), pos_xyt[1]],
                        [0,0,1]])
        self.vertex_center = np.array([[-length/2.0, length/2.0, length/2.0, -length/2.0], 
                        [width/2.0, width/2.0, -width/2.0, -width/2.0],[1,1,1,1]])
        self.vertex = np.matmul(self.pose, self.vertex_center)

    def plot(self, ax):
        ax.fill(self.vertex[0,:], self.vertex[1,:],color='red')


class Obstacle_dynamics:
    def __init__(self, s_list, l, v, waypoints, width, length, dt):
        self.s_list = s_list
        self.l = l
        self.v = v
        self.width = width
        self.length = length
        self.waypoints = waypoints
        self.dt = dt
        self.vertex_center = np.array([[-length/2.0, length/2.0, length/2.0, -length/2.0], 
                        [width/2.0, width/2.0, -width/2.0, -width/2.0],[1,1,1,1]])

    def get_pos(self, t):
        idx_l = np.floor(t/self.dt).astype(np.int32)
        s = self.s_list[idx_l]
        waypoint = self.waypoints[idx_l,:]
        T = np.array([[np.cos(waypoint[2]), -np.sin(waypoint[2]), waypoint[0]],
                        [np.sin(waypoint[2]), np.cos(waypoint[2]), waypoint[1]],
                        [0,0,1]])
        vertex_t = np.matmul(T, self.vertex_center)
        return waypoint, T, vertex_t, s

    def plot(self, ax, t, alpha=1):
        _, _, vertex_t, _ = self.get_pos(t)
        return ax.fill(vertex_t[0,:], vertex_t[1,:],color='orange', alpha=alpha)

    


        





