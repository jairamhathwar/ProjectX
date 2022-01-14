import numpy as np


class Sensor:
    def __init__(self, env, range, resolution):
        self.env = env
        self.range = range
        self.resolution = resolution
        self.fov = [-np.pi/2, np.pi/2]
        self.theta_list = np.arange(self.fov[0], self.fov[1]+resolution, resolution)

    def sense(self, xyt_ego, t):
        depth_map_road = self.sense_road(xyt_ego)
        depth_map = depth_map_road
        for obs in self.env.static_obs:
            vertex = obs.vertex
            depth_map = self.update_obstacle(depth_map, xyt_ego, vertex)
        
        for obs in self.env.dynamic_obs:
            _, _, vertex, _ = obs.get_pos(t)
            depth_map = self.update_obstacle(depth_map, xyt_ego, vertex)
        
        return depth_map


    def sense_road(self, xyt_ego):
        # calculate the angle and distance between 
        polar_set = [self.trans2polar(xyt_ego, bound) for bound in self.env.road_bound]
        depth_map = np.empty((self.theta_list.shape[0],4)) #[[theta, r, x, y]]
        depth_map[:,0] = self.theta_list

        for theta_idx, theta in enumerate(self.theta_list):
            intersection = np.empty((0,3)) #[[r, x, y]]
            for polar, bound in polar_set:
                # find T[i-1]<theta<=T[i]
                idx_g = np.where(theta <= polar[:,0])[0]
                idx_l = np.where(polar[:,0] < theta )[0]+1
                cross_idx = np.intersect1d(idx_l, idx_g).tolist()
                # find T[i-1]>theta>=T[i]
                idx_l = np.where(theta >= polar[:,0])[0]
                idx_g = np.where(polar[:,0] > theta)[0]+1
                cross_idx += np.intersect1d(idx_l, idx_g).tolist()
                for cross in cross_idx:
                    pt1_xy = bound[cross-1,:]
                    pt2_xy = bound[cross, :]
                    pt1_polar = polar[cross-1, :]
                    pt2_polar = polar[cross, :]
                    a = (theta - pt2_polar[0])/(pt1_polar[0]-pt2_polar[0]) # a*theta1+(1-a)*theta2 = theta
                    x = a*pt1_xy[0]+(1-a)*pt2_xy[0]
                    y = a*pt1_xy[1]+(1-a)*pt2_xy[1]
                    r = ((x-xyt_ego[0])**2+(y-xyt_ego[1])**2)**(1/2)
                    if r>0 and r<=self.range:
                        intersection = np.append(intersection, [[r,x,y]], axis=0)
            #print(intersection)
            # find the smallest range
            if intersection.shape[0] == 0:
                r = self.range
                x = xyt_ego[0]+r*np.cos(theta+xyt_ego[2])
                y = xyt_ego[1]+r*np.sin(theta+xyt_ego[2])
                depth_map[theta_idx, 1:] = [r,x,y]
            else:
                smallest_idx = np.argmin(intersection[:,0])
                depth_map[theta_idx, 1:] = intersection[smallest_idx,:]
        
        return depth_map




    def trans2polar(self, xyt_ego, bound, trim = True):
        #polar = [[theta1, r1], [theta2, r2], ....]
        polar = np.empty((bound.shape[0],2))
        polar[:,0] = np.arctan2(bound[:,1]-xyt_ego[1], bound[:,0 ]-xyt_ego[0])-xyt_ego[2] 
        
        # make sure in -pi to pi
        idx_to_mod = polar[:,0]>np.pi
        polar[idx_to_mod,0] -= 2*np.pi
        idx_to_mod = polar[:,0]<-np.pi
        polar[idx_to_mod,0] += 2*np.pi

        polar[:,1] = np.linalg.norm(bound[:,0:2]-xyt_ego[0:2], axis=1)
        if trim:
            idx_valid = np.where((polar[:,0]>=(self.fov[0]-5*self.resolution) ) & (polar[:,0]<=(self.fov[1]+5*self.resolution)))[0]
            polar = polar[idx_valid,:]
            bound_valid = bound[idx_valid,:]
        else:
            bound_valid = bound
        return polar, bound_valid

    def plot_sense(self, depth_map, ax):
        ax.plot(depth_map[:,2], depth_map[:,3],'-', color='blue',linewidth=2)
        
    def update_obstacle(self, depth_map, xyt_ego, vertex):
        vertex_circ = vertex[0:2,:].T
        vertex_circ = np.append(vertex_circ, [vertex_circ[0,:]], axis=0)
        polar = np.empty((5,2))
        polar[:,0] = np.arctan2(vertex_circ[:,1]-xyt_ego[1], vertex_circ[ :,0]-xyt_ego[0])-xyt_ego[2] 
        
        idx_to_mod = polar[:,0]>np.pi
        polar[idx_to_mod,0] -= 2*np.pi
        idx_to_mod = polar[:,0]<-np.pi
        polar[idx_to_mod,0] += 2*np.pi
        polar[:,1] = np.linalg.norm(vertex_circ-xyt_ego[0:2], axis=1)
        for i in range(1,5):
            theta_idx = np.where((depth_map[:,0]<=max(polar[i,0], polar[i-1, 0])) & (depth_map[:,0]>=min(polar[i,0], polar[i-1, 0])))[0]
            for idx in theta_idx:
                if polar[i-1,0]-polar[i,0] == 0:
                    if polar[i-1,1]<polar[i,1]:
                        x = vertex_circ[i-1, 0]
                        y = vertex_circ[i-1, 1]
                        r = polar[i-1,1]
                    else:
                        x = vertex_circ[i, 0]
                        y = vertex_circ[i, 1]
                        r = polar[i,1]
                else:
                    a = ((depth_map[idx,0] - polar[i, 0])/(polar[i-1,0]-polar[i,0]))
                    x = a*vertex_circ[i-1, 0]+(1-a)*vertex_circ[i, 0]
                    y = a*vertex_circ[i-1, 1]+(1-a)*vertex_circ[i, 1]
                    r = ((x-xyt_ego[0])**2+(y-xyt_ego[1])**2)**(1/2)
                if r < depth_map[idx,1]:
                    depth_map[idx, 1:] = [r,x,y]
        
        return depth_map

    def find_gap(self, depth_map):
        # return gap =[[theta, r1, r2, x1, y1, x2, y2], .....]

        D_diff  = abs(depth_map[0:-1, 2] - depth_map[1:, 2])
        gap_idx = np.where((D_diff/depth_map[0:-1, 2])>=0.3)[0]
        gap = np.empty((gap_idx.shape[0], 7))
        for i,idx in enumerate(gap_idx):
            if depth_map[idx,1] > depth_map[idx+1,1]:
                gap[i,:] = [depth_map[idx,0], depth_map[idx+1,1], depth_map[idx,1], 
                    depth_map[idx+1,2], depth_map[idx+1,3], depth_map[idx,2], depth_map[idx,3]]
            else:
                gap[i,:] = [depth_map[idx+1,0], depth_map[idx,1], depth_map[idx+1,1], 
                    depth_map[idx,2], depth_map[idx,3], depth_map[idx+1,2], depth_map[idx+1,3]]
        return gap


    def find_frontier(self, xyt_ego, gap_list, reference_line):
        # reference_line ( [x,y,theta], s)
        # return [s, x, y]

        s_list = reference_line[1]
        xy_list = reference_line[0]
        num_gap = gap_list.shape[0]

        polar, _ = self.trans2polar(xyt_ego, xy_list, trim=False)
        
        if num_gap == 0: # No gap, find the point cloest to the range
            idx = np.argmin(abs(polar[:,1]-self.range))
            s = s_list[idx]
            x = xy_list[idx,0]
            y = xy_list[idx,1]
            return [s,x,y]


        intersection = np.empty((0,4)) #[[r,s, x, y]]

        for gap in gap_list:
            # find T[i-1]<theta<=T[i]
            idx_g = np.where(gap[0] <= polar[:,0])[0]
            idx_l = np.where(polar[:,0] < gap[0] )[0]+1
            cross_idx = np.intersect1d(idx_l, idx_g).tolist()
            # find T[i-1]>theta>=T[i]
            idx_l = np.where(gap[0] >= polar[:,0])[0]
            idx_g = np.where(polar[:,0] > gap[0])[0]+1
            cross_idx += np.intersect1d(idx_l, idx_g).tolist()
            for cross in cross_idx:
                pt1_xy = xy_list[cross-1,:]
                pt2_xy = xy_list[cross, :]
                pt1_polar = polar[cross-1, :]
                pt2_polar = polar[cross, :]
                a = (gap[0] - pt2_polar[0])/(pt1_polar[0]-pt2_polar[0]) # a*theta1+(1-a)*theta2 = theta
                x = a*pt1_xy[0]+(1-a)*pt2_xy[0]
                y = a*pt1_xy[1]+(1-a)*pt2_xy[1]
                s = a*s_list[cross-1]+(1-a)*s_list[cross]
                r = ((x-xyt_ego[0])**2+(y-xyt_ego[1])**2)**(1/2)
                if r>gap[1] and r<gap[2]:
                    intersection = np.append(intersection, [[r,s,x,y]], axis=0)
        if intersection.shape[0] == 0:
            idx = np.argmin(abs(polar[:,1]-self.range))
            s = s_list[idx]
            x = xy_list[idx,0]
            y = xy_list[idx,1]
            return [s,x,y]

        idx_min_r = np.argmin(intersection[:,0])

        return intersection[idx_min_r, 1:]



        