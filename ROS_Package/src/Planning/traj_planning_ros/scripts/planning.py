#!/usr/bin/env python

import threading
import rospy
from copy import deepcopy
import numpy as np
from IPython import display
import matplotlib.pyplot as plt
# https://petercorke.github.io/spatialmath-python/intro.html
from spatialmath.base import *
from Track import Track
from MPCC import MPCC
from iLQR import iLQR
from traj_msgs.msg import Trajectory 
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TransformStamped


class Planning_MPC():
    def __init__(self, T = 1, N = 10, 
                    track_file = None,
                    solver_type = "mpcc",
                    vicon_pose = True,
                    pose_topic = '/zed2/zed_node/pose',
                    ref_traj_topic = '/planning/trajectory',
                    params_file = 'modelparams.yaml'):
        
        '''
        Main class for the MPC trajectory planner
        Input:
            freq: frequence to publish the control input to ESC and Servo
            T: prediction time horizon for the MPC
            N: number of integration steps in the MPC
        '''
        # parameters for the ocp solver
        self.T =T
        self.N = N
        self.replan_dt = T/N
        self.vicon_pose = vicon_pose
        
        # previous trajectory for replan
        self.prev_plan = None
        self.prev_control = None
        
        # create track
        if track_file is None:
            # make a circle with 1.5m radius
            r = 3
            theta = np.linspace(0, 2*np.pi, 100, endpoint=True)
            x = r*np.cos(theta)
            y = r*np.sin(theta)
            self.track = Track(np.array([x,y]), 0.5, 0.5, True)
        else:
            self.track = Track()
            self.track.load_from_file(track_file)
            
        # set up the optimal control solver
        if solver_type == "mpcc":
            self.ocp_solver = MPCC(self.T, self.N, self.track, params_file = params_file)
        else:
            self.ocp_solver = iLQR(self.T, self.N, params_file = params_file)
        
        rospy.loginfo("Successfully initialized the solver with horizon "+str(T)+"s, and "+str(N)+" steps.")
                
        # objects to schedule trajectory publishing 
        self.cur_t = None
        self.cur_pose = None
        self.cur_state = None

        self.last_pub_t = None
        self.thread_lock = threading.Lock()
        
        # objects for visualization
        self.traj_x = []
        self.traj_y = []
        self.traj_lock = threading.Lock()
        
        
        # set up publiser to the reference trajectory and subscriber to the pose
        self.traj_pub = rospy.Publisher(ref_traj_topic, Trajectory, queue_size=1)
        if self.vicon_pose:
            self.pose_sub = rospy.Subscriber(pose_topic, TransformStamped, self.pose_sub_callback)
        else:
            self.pose_sub = rospy.Subscriber(pose_topic, PoseStamped, self.pose_sub_callback)
        
        threading.Thread(target=self.control_pub_thread).start()

    def pose_sub_callback(self, msg):
        """
        Subscriber callback function of the robot pose
        """
        # Convert the current pose msg into a SE3 Matrix
        if self.vicon_pose:
            # vicon use TransformStamped
            cur_pose = transl(msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z)
            cur_pose[:3,:3] = q2r([msg.transform.rotation.w,
                            msg.transform.rotation.x, 
                            msg.transform.rotation.y, 
                            msg.transform.rotation.z])
        else:
            cur_pose = transl(msg.position.x, msg.position.y, msg.position.z)
            cur_pose[:3,:3] = q2r([msg.orientation.w,
                            msg.orientation.x, 
                            msg.orientation.y, 
                            msg.orientation.z])
        
        self.thread_lock.acquire()
        
        # make a copy of previous state
        prev_t = deepcopy(self.cur_t)
        prev_pose = np.array(self.cur_pose, copy=True)
        self.cur_pose = cur_pose
        self.cur_t = msg.header.stamp

        if prev_t is not None:
            # approximate the velocity
            dt = (self.cur_t - prev_t).to_sec()
            
            # use tr2delta https://petercorke.github.io/spatialmath-python/func_3d.html#spatialmath.base.transforms3d.tr2delta
            # [dx, dy, dz, dthetax, dthetay, dthetaz]
            delta = tr2delta(prev_pose, cur_pose)/dt
            vel = (delta[0]**2 + delta[1]**2)**0.5
            self.cur_state = np.array([cur_pose[0,-1], cur_pose[1,-1], 
                                    tr2rpy(cur_pose)[-1], # heading
                                    vel, 0])
        self.thread_lock.release()
        
        self.traj_lock.acquire()
        self.traj_x.append(msg.transform.translation.x)
        self.traj_y.append(msg.transform.translation.y)
        self.traj_lock.release()

    def control_pub_thread(self):
        rospy.loginfo("Planning publishing thread started")
        while not rospy.is_shutdown():
            # determine if we need to publish
            self.thread_lock.acquire()
            
            since_last_pub = self.replan_dt if self.last_pub_t is None else (self.cur_t-self.last_pub_t).to_sec() 
            if since_last_pub >= self.replan_dt:
                # make a copy of the data
                cur_t = deepcopy(self.cur_t)
                if self.cur_state is not None:
                    cur_state = np.array(self.cur_state, copy=True)                      
                else:
                    cur_state = None
            self.thread_lock.release()
            
            if since_last_pub >= self.replan_dt and cur_state is not None:
                
                # TODO: use previous plan as initial guess
                cur_state[-1] = self.track.project_point(cur_state[:2])
                
                if self.prev_plan is None:
                    x_init = None
                    u_init = None
                else:
                    x_init = np.zeros((5,self.N))
                    x_init[:,1:-1] = self.prev_plan[:,2:]
                    x_init[:,0] = cur_state
                    x_init[:,-1] = self.prev_plan[:,-1]
                    
                    u_init = np.zeros((3,self.N))
                    u_init[:,:-1] = self.prev_control[:,1:]
                    
                start_time = rospy.get_rostime()
                sol_x, sol_u = self.ocp_solver.solve(cur_state, x_init, u_init)
                end_time = rospy.get_rostime()
            
                # contruct the new planning
                plan = Trajectory()
                plan.header.stamp = cur_t
                plan.dt = self.replan_dt
                plan.step = self.N
                plan.x = sol_x[0,:].tolist()
                plan.y = sol_x[1,:].tolist()
                plan.psi = sol_x[2,:].tolist()
                plan.vel = sol_x[3,:].tolist()

                plan.throttle = sol_u[0,:].tolist()
                plan.steering = sol_u[1,:].tolist()

                self.traj_pub.publish(plan)

                self.last_pub_t = cur_t
                
                # self.traj_lock.acquire()
                self.prev_plan = sol_x
                self.prev_control = sol_u
                # self.traj_lock.release()

                rospy.loginfo("Use "+str((end_time-start_time).to_sec())+" to plan")
                
    def run(self):
        # plt.ion()
        # plt.show()
        # plt.figure(figsize=(5, 5))        
        # while not rospy.is_shutdown():
        #     display.clear_output(wait = True)
        #     display.display(plt.gcf())
        #     plt.clf()
        #     self.track.plot_track()
        #     self.traj_lock.acquire()
        #     plt.plot(self.traj_x, self.traj_y, '--r')
        #     if self.prev_plan is not None:
        #         plt.plot(self.prev_plan[0,:], self.prev_plan[1,:], '-.b')
        #     self.traj_lock.release()            
        #     plt.xlim((-5, 5))
        #     plt.ylim((-5, 5))
        #     plt.pause(0.001)
        rospy.spin()