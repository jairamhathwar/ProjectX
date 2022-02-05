#!/usr/bin/env python

from MPCC import MPCC
from iLQR import iLQR
from traj_msgs.msg import Trajectory 
from geometry_msgs.msg import PoseStamped


class Planning_MPC():
    def __init__(self, T = 1, N = 10, 
                    mode = "MPCC",
                    pose_topic = '/zed2/zed_node/pose',
                    ref_traj_topic = '/planning/trajectory',
                    controller_topic = '/control/rc_control',
                    params_file = 'modelparams.yaml'):
        '''
        Main class for the MPC trajectory tracking controller
        Input:
            freq: frequence to publish the control input to ESC and Servo
            T: prediction time horizon for the MPC
            N: number of integration steps in the MPC
        '''

        self.T =T
        self.N = N
        self.dt = T/N
        self.prev_pose = None
        self.prev_t = None
        self.prev_control = np.zeros(2)
        
        # set up the optimal control solver
        #if dyn_model:
        self.ocp_solver = TrajTrackingDyn(self.T, self.N, params_file = params_file)

        rospy.loginfo("Successfully initialized the solver")
        
        # set up subscriber to the reference trajectory and pose
        self.traj_sub = rospy.Subscriber(ref_traj_topic, Trajectory, self.traj_sub_callback, queue_size=1)
        self.pose_sub = rospy.Subscriber(pose_topic, PoseStamped, self.pose_sub_callback, queue_size=1)

        # set up publisher to the low-level ESC and servo controller
        self.control_pub = rospy.Publisher(controller_topic, RCControl, queue_size=1)

    def traj_sub_callback(self, msg: Trajectory):
        """
        Subscriber callback function of the reference trajectory
        """
        ref_traj = RefTraj(msg)
        self.traj_buffer.writeFromNonRT(ref_traj)

    def pose_sub_callback(self, msg: PoseStamped):
        """
        Subscriber callback function of the robot pose
        """
        
        # Convert the current pose msg into a SE3 Matrix
        cur_pose = transl(msg.position.x, msg.position.y, msg.position.z)
        cur_pose[:3,:3] = q2r([msg.orientation.x, 
                        msg.orientation.y, 
                        msg.orientation.z, 
                        msg.orientation.w])


        if self.prev_pose is not None:
            # approximate the velocity
            dt = (msg.header.stamp - self.prev_t).to_sec()
            
            # use tr2delta https://petercorke.github.io/spatialmath-python/func_3d.html#spatialmath.base.transforms3d.tr2delta
            # [dx, dy, dz, dthetax, dthetay, dthetaz]
            delta = tr2delta(self.prev_pose, cur_pose)

            # get current state State: [X, Y, Vx, Vy, psi: heading, omega: yaw rate, delta: steering angle]
            x_cur = np.array([msg.position.x, msg.position.y, 
                              delta[0]/dt, delta[1]/dt, 
                              tr2rpy(cur_pose)[-1], # heading
                              delta[-1]/dt, # yaw rate
                              self.prev_control[1] # steering angle ( assume the servo will reach that position)
                              ])
            
            # get the reference trajectory
            ref_traj = self.traj_buffer.readFromRT()
            _, x_ref = ref_traj.interp_traj(self, msg.header.stamp, self.dt, self.N)
            
            # solve the ocp
            sol_x, sol_u = self.ocp_solver.solve(x_ref, x_cur)
            
            # form the new control output by taking the first
            control = RCControl()
            control.header = msg.header
            control.throttle = sol_u[0,0]
            control.steer = sol_x[0,-1]+sol_u[0,1]*dt  # assume pose receieved at a fixed rate
            control.reverse = False
            
            self.control_pub.publish(control)
            self.prev_control = np.array([control.throttle, control.steer])          
            
        self.prev_pose = cur_pose
        self.prev_t = msg.header.stamp
