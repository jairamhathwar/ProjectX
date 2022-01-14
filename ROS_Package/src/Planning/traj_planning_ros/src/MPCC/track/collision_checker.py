import numpy as np
from matplotlib import pyplot as plt
from .sensing import Sensor


class CollisionChecker:
    def __init__(self, env, ego):
        self.env = env
        self.ego = ego
        self.buffer_long = self.ego.length/2+3
        self.buffer_side = self.ego.width/2+0.2
        self.v_lat_max = 2
        self.road_width = 3.6

    def check_safe(self, traj, incomming_state):
        # ego_state = [s,l,v,t]
        # check static obs
        _, _, _, t, waypoints = traj.gen_waypoints(0.1)
        for t_i, waypoint in zip(t, waypoints):
            p = np.array([waypoint[0], waypoint[1], 1])
            for static_ob in self.env.static_obs:
                p_rel = np.matmul(np.linalg.inv(static_ob.pose), p)
                if self.check_collision(p_rel[0], -p_rel[1], static_ob.length, static_ob.width):
                    #print(t_i, p_rel, static_ob.s, static_ob.l)
                    return False
            
            for dynamic_ob in self.env.dynamic_obs:
                _, obs_pose, _,_ = dynamic_ob.get_pos(t_i)
                p_rel = np.matmul(np.linalg.inv(obs_pose), p)
                if self.check_collision(p_rel[0], -p_rel[1], dynamic_ob.length, dynamic_ob.width):
                    return False

        slv_ego = traj.sl_end+[traj.v_end]
        for static_ob in self.env.static_obs:
            slv_obs = [static_ob.s, static_ob.l, 0]
            if not self.check_ICS(slv_ego, slv_obs, static_ob.length, static_ob.width, incomming_state):
                return False
            
        for dynamic_ob in self.env.dynamic_obs:
            _, _, _,s_dyn = dynamic_ob.get_pos(t[-1])
            slv_obs = [s_dyn, dynamic_ob.l, dynamic_ob.v]
            if not self.check_ICS(slv_ego, slv_obs, dynamic_ob.length, dynamic_ob.width, incomming_state):
                return False
        return True

    def check_collision(self, rel_s, rel_l, ob_length, ob_width):
        region = self.get_region(rel_s, rel_l, ob_length, ob_width)
        if region == 0:
            return True
        else:
            return False

    def check_ICS(self, slv_ego, slv_obs, ob_length, ob_width, incomming_state):
        rel_s = slv_ego[0] - slv_obs[0]
        rel_l = slv_ego[1] - slv_obs[1]
        rel_v = slv_ego[2] - slv_obs[2]

        region = self.get_region(rel_s, rel_l, ob_length, ob_width)
        if region == 0:
            return False
        
        if region == 4:
            safe, dis_ahead = self.calc_front_ICS(rel_s, rel_v, ob_length)
            if not safe:
                return False
        
        if region == 5:
            safe, dis_behind = self.calc_rear_ICS(rel_s, rel_v, ob_length)
            if not safe:
                return False
        t_evade_front, dis_travel_front  = self.evade_overtake_front(slv_ego, slv_obs, ob_length, ob_width)
        dis_require_front  = dis_travel_front - incomming_state[1]*t_evade_front+self.buffer_long
        t_evade_rear, dis_travel_rear  = self.evade_overtake_rear(slv_ego, slv_obs, ob_length, ob_width)
        dis_require_rear  = dis_travel_rear - incomming_state[1]*t_evade_rear+self.buffer_long

        # if slv_ego[1]<=0:
        #    print(incomming_state, slv_ego, dis_require_front, dis_require_rear)

        if max(0, incomming_state[0])<min(dis_require_rear, dis_require_front):
            return False

        
        
        return True
            
    def get_region(self, rel_s, rel_l, ob_length, ob_width):
        # wrt obj 0) collision 1) front_side 2) side 3) rear_side 4)front 5) rear
        on_side = abs(rel_l)>=(ob_width/2+self.buffer_side)
        front_rear = 0 # 1 for front 2 for rear
        if rel_s >= (ob_length/2+self.buffer_long):
            front_rear = 1
        elif rel_s <= -(ob_length/2+self.buffer_long):
            front_rear = 2
        
        region = 0

        if on_side==0:
            if front_rear == 1:
                # front
                region = 4
            elif front_rear == 2:
                # rear side
                region = 5
        else:
            if front_rear == 0:
                # side 
                region = 2
            elif front_rear == 1:
                # front side
                region = 1
            else:
                # rear side
                region = 3
        return region

    def calc_front_ICS(self, rel_s, rel_v, ob_length):
        if rel_v>=0:
            dis_require = self.buffer_long+ob_length/2
            return True, dis_require
        else:
            t_evade  = abs(rel_v/self.ego.a_max)
            dis_ahead = abs(rel_v*t_evade+0.5*self.ego.a_max*t_evade**2)
            dis_require = self.buffer_long+ob_length/2+dis_ahead
            if rel_s>=dis_require:
                return True, dis_require
            else:
                return False, dis_require

    def calc_rear_ICS(self, rel_s, rel_v,ob_length):
        if rel_v<=0:
            dis_require = -self.buffer_long-ob_length/2
            return True, dis_require
        else:
            t_evade  = abs(rel_v/self.ego.a_max)
            dis_behind = rel_v*t_evade-0.5*self.ego.a_max*t_evade**2
            dis_require = -self.buffer_long-ob_length/2-dis_behind
            if rel_s<=(-self.buffer_long-ob_length/2-dis_behind):
                return True, dis_require
            else:
                return False, dis_require


    def evade_overtake_front(self, slv_ego, slv_obs, ob_length, ob_width):
        # slv_* = [s, l, v]
        if slv_ego[1]<slv_obs[1]: # left to right
            lat_goal = (np.floor(abs(slv_obs[1])/self.road_width)*self.road_width+self.ego.width/2)*np.sign(slv_obs[1])
        else: # right to left
            lat_goal = (np.ceil(abs(slv_obs[1])/self.road_width)*self.road_width-self.ego.width/2)*np.sign(slv_obs[1])
                
        t_merge_total = abs(slv_ego[1]-lat_goal)/self.v_lat_max

        if t_merge_total < 0: # alread behind/ front of obstacle
            t_evade = 0
            dis_evade = 0
            return t_evade, dis_evade

        # Time since the car has been right on the side the obstacle
        if slv_ego[1]<slv_obs[1]: # left to right
            t_merge_2 = (self.ego.width/2 - (abs(slv_ego[1])%self.road_width-(self.buffer_side+ob_width/2)))/self.v_lat_max
        else: # right to left
            t_merge_2 = ((abs(slv_ego[1])%self.road_width+(self.buffer_side+ob_width/2))-(3.6-self.ego.width/2))/self.v_lat_max

        if t_merge_2>t_merge_total:
            t_merge_2 = t_merge_total
        t_merge_1 = t_merge_total - t_merge_2 # time for the car to move laterally to the side of the obstacle

        v_rel = slv_ego[2]-slv_obs[2]
        dis_catch = slv_obs[0] - slv_ego[0] +  self.buffer_long + ob_length/2       
        t_catch = np.max(np.roots([0.5*self.ego.a_max, v_rel, -dis_catch]))
        if not np.isreal(t_catch):
            t_catch = 0
        t2maxV = min(t_catch, max(0,(self.ego.v_max - slv_ego[2])/self.ego.a_max))
        dis_travel = slv_ego[2]*t2maxV+0.5*self.ego.a_max*t2maxV**2
        t_catch = t2maxV+(dis_catch-(dis_travel-slv_obs[2]*t2maxV))/(self.ego.v_max-slv_obs[2])
        dis_travel += (t_catch-t2maxV)*self.ego.v_max
        v_end = slv_ego[2]+t2maxV*self.ego.a_max

        if t_merge_1 > t_catch:
            t_evade = t_merge_total
            dis_travel += v_end*(t_evade-t_catch)
        else:
            t_evade = t_catch+t_merge_2
            dis_travel += v_end*t_merge_2
        
        return t_evade, dis_travel


    def evade_overtake_rear(self, slv_ego, slv_obs, ob_length, ob_width):
        # slv_* = [s, l, v]
        if slv_ego[1]<slv_obs[1]: # left to right
            lat_goal = (np.floor(abs(slv_obs[1])/self.road_width)*self.road_width+self.ego.width/2)*np.sign(slv_obs[1])
        else: # right to left
            lat_goal = (np.ceil(abs(slv_obs[1])/self.road_width)*self.road_width-self.ego.width/2)*np.sign(slv_obs[1])
        
        t_merge_total = abs(slv_ego[1]-lat_goal)/self.v_lat_max

        if t_merge_total < 0: # alread behind/ front of obstacle
            t_evade = 0
            dis_evade = 0
            return t_evade, dis_evade

        # Time since the car has been right on the side the obstacle
        if slv_ego[1]<slv_obs[1]: # left to right
            t_merge_2 = (self.ego.width/2 - (abs(slv_ego[1])%self.road_width-(self.buffer_side+ob_width/2)))/self.v_lat_max
        else: # right to left
            t_merge_2 = ((abs(slv_ego[1])%self.road_width+(self.buffer_side+ob_width/2))-(3.6-self.ego.width/2))/self.v_lat_max

        if t_merge_2>t_merge_total:
            t_merge_2 = t_merge_total
        t_merge_1 = t_merge_total - t_merge_2 # time for the car to move laterally to the side of the obstacle

        v_rel = slv_ego[2]-slv_obs[2]
        dis_catch = slv_ego[0]-slv_obs[0] + self.buffer_long + ob_length/2       
        t_catch = np.max(np.roots([-0.5*self.ego.a_max, v_rel, dis_catch]))
        if not np.isreal(t_catch):
            t_catch = 0


        t2maxV = min(t_catch, max(0,(slv_ego[2])/self.ego.a_max))
        dis_travel = slv_ego[2]*t2maxV-0.5*self.ego.a_max*t2maxV**2
        t_catch = t2maxV+(dis_catch-(slv_obs[2]*t2maxV-dis_travel))/(slv_obs[2])
        v_end = slv_ego[2]-t2maxV*self.ego.a_max

        if t_merge_1 > t_catch:
            t_evade = t_merge_total
            dis_travel += v_end*(t_evade-t_catch)
        else:
            t_evade = t_catch+t_merge_2
            dis_travel += v_end*t_merge_2
        
        return t_evade, dis_travel

        
        


        
   
