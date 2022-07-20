#!/usr/bin/env python3
"""
This planning node uses the A* algorithm to output constant trajectory information 
that can be used with stanley controller to control the car. The information
will be packed into traj_msg type, with the x, y be used to create the fitted cubicspline

A* algorithm modified from:
https://github.com/atomoclast/realitybytes_blogposts
https://realitybytes.blog/2018/08/17/graph-based-path-planning-a/
"""
from traj_msgs.msg import Trajectory
import rospy

from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import math

COLOR_MAP = (0, 8)

class AStarPlanner:
    def __init__(self, grid, visual=False):
        """
        Constructor of the AStarPlanner Class.
        :param grid: List of lists that represents the
        occupancy map/grid. List should only contain 0's
        for open nodes and 1's for obstacles/walls.
        :param visual: Boolean to determine if Matplotlib
        animation plays while path is found.
        """
        self.grid = grid
        self.visual = visual
        self.heuristic = None
        self.goal_node = None

    def calc_heuristic(self):
        """
        Function will create a list of lists the same size
        of the occupancy map, then calculate the cost from the
        goal node to every other node on the map and update the
        class member variable self.heuristic.
        :return: None.
        """
        row = len(self.grid)
        col = len(self.grid[0])

        self.heuristic = [[0 for x in range(col)] for y in range(row)]
        
        # Calculating heuristic using euclidin distance to goal point
        for i in range(row - 1 , -1, -1):
            for j in range(col):
                distance = math.sqrt( (i - self.goal_node[0])**2 + (j - self.goal_node[1])**2 )
                self.heuristic[row - 1 - i][j] = distance

        # print("Heuristic:")
        # for i in range(len(self.heuristic)):
        #     print(self.heuristic[i])



    def a_star(self, start_cart, goal_cart):
        """
        A* Planner method. Finds a plan from a starting node
        to a goal node if one exits.
        :param init: Initial node in an Occupancy map. [x, y].
        Type: List of Ints.
        :param goal: Goal node in an Occupancy map. [x, y].
        Type: List of Ints.
        :return: Found path or -1 if it fails.
        """
        goal = [goal_cart[1], goal_cart[0]]
        self.goal_node = goal
        init = [start_cart[1], start_cart[0]]

        # Calculate the Heuristic for the map
        self.calc_heuristic()

        # Reverse the order of rows so it is acending upwards
        self.grid = np.flip(self.grid, axis=0)

        print (init, goal)

        if self.visual:
            viz_map = deepcopy(self.grid)
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(111)
            ax.set_title('Occupancy Grid')
            plt.xticks(visible=False)
            plt.yticks(visible=False)
            plt.imshow(viz_map, origin='lower', interpolation='none', clim=COLOR_MAP)
            ax.set_aspect('equal')
            plt.pause(2)
            viz_map[init[0]][init[1]] = 5  # Place Start Node
            viz_map[goal[0]][goal[1]] = 6
            plt.imshow(viz_map, origin='lower', interpolation='none', clim=COLOR_MAP)
            plt.pause(2)

        # Encode movements (including diagonals)
        delta = [[1, 0],   # go up
                 [0, -1],  # go left
                 [-1, 0],  # go down
                 [0, 1],   # go right
                 [1, -1],  # upper left
                 [-1, -1], # lower left
                 [1, 1],   # upper right
                 [-1, 1]]  # lower right
        delta_name = ['^ ', '< ', 'v ', '> ', 'UL', 'LL', 'UR', 'LR']

        # Heavily used from some of the A* Examples by Sebastian Thrun:

        num_rows = len(self.grid)
        num_cols = len(self.grid[0])

        closed = [[0 for col in range(len(self.grid[0]))] for row in range(len(self.grid))]
        shortest_path = [['  ' for _ in range(len(self.grid[0]))] for _ in range(len(self.grid))]
        closed[init[0]][init[1]] = 1

        expand = [[-1 for col in range(len(self.grid[0]))] for row in range(len(self.grid))]
        delta_tracker = [[-1 for _ in range(len(self.grid[0]))] for _ in range(len(self.grid))]

        cost = 1
        x = init[0]
        y = init[1]
        g = 0
        f = g + self.heuristic[x][y]
        open = [[f, g, x, y]]

        found = False  # flag that is set when search is complete
        resign = False  # flag set if we can't find expand
        count = 0
        while not found and not resign:
            if len(open) == 0:
                resign = True
                if self.visual:
                    plt.text(2, 10, s="No path found...", fontsize=18, style='oblique', ha='center', va='top')
                    plt.imshow(viz_map, origin='lower', interpolation='none', clim=COLOR_MAP)
                    plt.pause(5)
                return -1
            else:
                open.sort()
                open.reverse()
                next = open.pop()
                x = next[2]
                y = next[3]
                g = next[1]
                expand[x][y] = count
                count += 1

                if x == goal[0] and y == goal[1]:
                    found = True
                    if self.visual:
                        viz_map[goal[0]][goal[1]] = 7
                        plt.text(2, 10, s="Goal found!", fontsize=18, style='oblique', ha='center', va='top')
                        plt.imshow(viz_map, origin='lower', interpolation='none', clim=COLOR_MAP)
                        plt.pause(2)
                else:
                    for i in range(len(delta)):
                        x2 = x + delta[i][0]
                        y2 = y + delta[i][1]
                        if len(self.grid) > x2 >= 0 <= y2 < len(self.grid[0]):
                            if closed[x2][y2] == 0 and self.grid[x2][y2] == 0:
                                g2 = g + cost
                                f = g2 + self.heuristic[num_rows - 1 - x2][y2]
                                open.append([f, g2, x2, y2])
                                closed[x2][y2] = 1
                                delta_tracker[x2][y2] = i
                                if self.visual:
                                    viz_map[x2][y2] = 3
                                    plt.imshow(viz_map, origin='lower', interpolation='none', clim=COLOR_MAP)
                                    plt.pause(.5)

        current_x = goal[0]
        current_y = goal[1]
        shortest_path[current_x][current_y] = '* '
        full_path = []
        while current_x != init[0] or current_y != init[1]:
            previous_x = current_x - delta[delta_tracker[current_x][current_y]][0]
            previous_y = current_y - delta[delta_tracker[current_x][current_y]][1]
            shortest_path[previous_x][previous_y] = delta_name[delta_tracker[current_x][current_y]]
            full_path.append((current_x, current_y))
            current_x = previous_x
            current_y = previous_y
        full_path.reverse()
        
        print("Found the goal in {} iterations.".format(count))
        print("full_path: ", full_path[:-1])
        for i in range(len(shortest_path)):
            print (shortest_path[num_rows - 1 - i])

        if self.visual:
            for node in full_path:
                viz_map[node[0]][node[1]] = 7
                plt.imshow(viz_map, origin='lower', interpolation='none', clim=COLOR_MAP)
                plt.pause(.5)

            # Animate reaching goal:
            viz_map[goal[0]][goal[1]] = 8
            plt.imshow(viz_map, origin='lower', interpolation='none', clim=COLOR_MAP)
            plt.pause(5)

        # Return full_path -- a list of tuples representing the path coords
        return full_path[:-1]

def plan_course(grid, start, goal):
    # Create publisher    
    publisher = rospy.Publisher("/simple_trajectory_topic", Trajectory, queue_size=1)
    rate = rospy.Rate(10)

    # Create an instance of the AStarPlanner class:
    planner = AStarPlanner(grid, False)

    # Plan a path
    planned_path = planner.a_star(start, goal)
    
    # Convert planned_path (list of tuples) to 2 arrays with y and x points
    result = list(map(list, zip(*planned_path)))
    ay, ax = result

    # Insert start and goal points to the trajectory
    ax.insert(0, start[0])
    ay.insert(0, start[1])
    ax.append(goal[0])
    ay.append(goal[1])

    # Shift values of x-axis down in order to work with Stanley controller
    x_axis_center = len(grid[0]) // 2
    ax = [n - x_axis_center for n in ax]

    # Publish trajectory
    while not rospy.is_shutdown():
        message = Trajectory()
        message.x = ax
        message.y = ay
        message.dt = 0.1
        publisher.publish(message)
        rate.sleep()
    
    rospy.spin()


def insert_obstacle(grid, x, y):
    grid[len(grid) - 1 - y][x] = 1
    return grid

def create_grid(using_vicon=False):
    rospy.init_node("astar_planning_node")

    if using_vicon:
        # Get truck and box data from Vicon system
        truck_data = rospy.wait_for_message("/vicon/truck/truck", TransformStamped, timeout=2)
        box_data = rospy.wait_for_message("/vicon/box/box", TransformStamped, timeout=2)

        # Get coordinates of truck and box (vicon frame --> ground frame)
        truck_x = truck_data.transform.translation.y
        truck_y = -1 * truck_data.transform.translation.x
        box_x = box_data.transform.translation.y
        box_y = -1 * box_data.transform.translation.x
        box_z = box_data.transform.translation.z

        # Round the values for use in grid
        truck_x = round(truck_x)
        truck_y = round(truck_y)
        box_x = round(box_x)
        box_y = round(box_y)

        print(
            "initial vicon: \n",
            "truck: ", str([truck_x, truck_y]),
            "\n box: ", str([box_x, box_y])
        )

        # Create empty grid
        grid = np.zeros((10,9))

        # Find the middle of the x-axis on the grid
        x_axis_center = len(grid[0]) // 2

        # Set starting and ending points on grid [x,y]
        start = [x_axis_center, 0]
        goal = [x_axis_center, 6]

        # Shift coordinates of box so that truck begins at origin [0, 0]
        box_x = box_x - truck_x
        box_y = box_y - truck_y

        print(
            "after shifting: \n",
            "truck: ", str([0, 0]),
            "\n box: ", str([box_x, box_y])
        )

        # Find orientation of truck
        truck_qz = truck_data.transform.rotation.z
        truck_qw = truck_data.transform.rotation.w

        # Rotate box coords to align to truck's orientation
        r = Rotation.from_quat([0, 0, truck_qz, truck_qw])

        v = [box_x, box_y, box_z]
        print("\nv1:", str(v))

        v = r.apply(v, inverse=True)
        print("v1:", str(v))

        box_x = round(v[0])
        box_y = round(v[1])

        print(
            "\nafter rotation: \n",
            "truck: ", str([0, 0]),
            "\n box: ", str([box_x, box_y])
        )

        # Shift box according to the truck's start position on grid
        box_x = box_x + start[0]
        box_y = box_y + start[1]

        print(
            "after rotation + shift: \n",
            "truck: ", str(start),
            "\n box: ", str([box_x, box_y])
        )

        # Insert box location onto the empty grid
        grid = insert_obstacle(grid, box_x, box_y)
        print("\n" + str(grid))

    else:
        # Create test grid and start & end points
        grid = [[0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0]]

        # Find the middle of the x-axis on the grid
        x_axis_center = len(grid[0]) // 2

        # Set starting and ending points on grid [x,y]
        start = [x_axis_center, 0] # [x, y]
        goal = [x_axis_center, 6]   # [x, y]

    plan_course(grid, start, goal)

if __name__=="__main__":
    try:
        create_grid(using_vicon = True)
    except rospy.ROSInterruptException:
        pass
    

