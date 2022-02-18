from dynamics import Dynamics
from cost import Cost
from ilqr import iLQR
from track import Track
import numpy as np
import csv
import pickle


class UnitTest_Gen:
    def __init__(self):
        params = {'L': 0.257, 'm': 2.99, 'track_width': 0.4,
                'delta_min': -0.35, 'delta_max': 0.35,
                'v_max': 4, 'v_min': 0, 'a_min': -3.5,
                'a_max': 3.5, 'alat_max': 5, 'w_vel': 2,
                'w_contour': 100, 'w_theta': 1, 'w_accel': 1,
                'w_delta': 1, 'q1_v': 0.4, 'q2_v': 2,
                'q1_road': 0.4, 'q2_road': 2, 'q1_lat': 0.4,
                'q2_lat': 2, 'T': 1, 'N': 6, 'max_itr': 50}

        # define a test track
        x = []
        y = []

        with open('racetrack/IMS.csv', newline='') as f:
            spamreader = csv.reader(f, delimiter=',')
            for i, row in enumerate(spamreader):
                if i>0:
                    x.append(float(row[0]))
                    y.append(float(row[1]))

        x = np.array(x)/25.0
        y = np.array(y)/25.0
        center_line = np.array([x,y])
        self.ref_path = Track(center_line = center_line, width_left = 0.4, width_right = 0.4)
        
        self.dynamics = Dynamics(params)
        self.cost = Cost(params)
        self.iLQR = iLQR(self.ref_path, params)
        
        self.plan = pickle.load(open("test.p","rb"))[0]
        self.step = self.plan.shape[-1]
        
        self.forward_step_sol = None
        self.get_AB_matrix_sol = None
        self.get_cost_sol = None
        self.cost_state_deriv_sol = None
        self.cost_control_deriv_sol = None
        
        self.test_forward_step()
        self.test_AB_matrix()
        self.test_get_cost()
        self.test_cost_state_deriv()
        self.test_cost_control_deriv()
        
        print(self.plan.shape)
        
        pickle.dump([self.plan, self.forward_step_sol, 
                     self.get_AB_matrix_sol, 
                     self.get_cost_sol,
                     self.cost_state_deriv_sol,
                     self.cost_control_deriv_sol], 
                    open("unit_test_sol.p", "wb"))
        
    def test_forward_step(self):
        input = self.plan[:,0,:]
        solution = np.empty((4,self.step))
        
        for i in range(self.step):
            solution[:,i], _ = self.dynamics.forward_step(input[:4, i], input[4:,i])
        
        self.forward_step_sol = solution
        
    
    def test_AB_matrix(self):
        input = self.plan
        solution = []
        for i in range(self.step):
            A, B = self.dynamics.get_AB_matrix(input[:4,:, i], input[4:,:,i])
            solution.append((A,B))
        self.get_AB_matrix_sol = solution
            
    def test_get_cost(self):
        input = self.plan
        solution = np.empty(self.step)
        
        for i in range(self.step):
            closest_pt, slope, theta = self.ref_path.get_closest_pts(input[:2,:, i])
            solution[i] = self.cost.get_cost(input[:4,:, i], input[4:,:,i], closest_pt, slope, theta)
        self.get_cost_sol = solution

    def test_cost_state_deriv(self):
        input = self.plan
        solution = []
        for i in range(self.step):
            closest_pt, slope, theta = self.ref_path.get_closest_pts(input[:2,:, i])
            c_x, c_xx = self.cost._get_cost_state_derivative(input[:4,:, i], closest_pt, slope)
            solution.append((c_x, c_xx))
        self.cost_state_deriv_sol = solution
    
    def test_cost_control_deriv(self):
        input = self.plan
        solution = []
        for i in range(self.step):
            c_u, c_uu = self.cost._get_cost_control_derivative(input[4:,:, i])
            solution.append((c_u, c_uu))
        self.cost_control_deriv_sol = solution
        

class UnitTest:
    def __init__(self):
        self.params = {'L': 0.257, 'm': 2.99, 'track_width': 0.4,
                'delta_min': -0.35, 'delta_max': 0.35,
                'v_max': 4, 'v_min': 0, 'a_min': -3.5,
                'a_max': 3.5, 'alat_max': 5, 'w_vel': 2,
                'w_contour': 100, 'w_theta': 1, 'w_accel': 1,
                'w_delta': 1, 'q1_v': 0.4, 'q2_v': 2,
                'q1_road': 0.4, 'q2_road': 2, 'q1_lat': 0.4,
                'q2_lat': 2, 'T': 1, 'N': 6, 'max_itr': 50}

        # define a test track
        x = []
        y = []

        with open('racetrack/IMS.csv', newline='') as f:
            spamreader = csv.reader(f, delimiter=',')
            for i, row in enumerate(spamreader):
                if i>0:
                    x.append(float(row[0]))
                    y.append(float(row[1]))

        x = np.array(x)/25.0
        y = np.array(y)/25.0
        center_line = np.array([x,y])
        
        
        self.ref_path = Track(center_line = center_line, width_left = 0.4, width_right = 0.4)
        
        solutions = pickle.load(open("unit_test_sol.p","rb"))
        self.input = solutions[0]

        self.step = self.input.shape[-1]
        
        self.forward_step_sol = solutions[1]
        self.get_AB_matrix_sol = solutions[2]
        self.get_cost_sol = solutions[3]
        self.cost_state_deriv_sol = solutions[4]
        self.cost_control_deriv_sol = solutions[5]
        

    def choose_test_case(self, i):
        if i>=0 and i<self.step:
            print("Using testcase ",i)
            return i
        else:
            i_new = np.random.randint(0, self.step)
            print("Testcase ", i, " not found. Use randomly selected testcase ", i_new)
            return i_new      
    
    def check_solution(self, name, test, solution):
        print("Expected solution for ", name, " is:")
        print(solution)
        print("Your function output is:")
        print(test)
        print("with max error ", np.max(np.abs(test-solution)))     
          
    def test_forward_step(self, dyn_class, i):
        dynamics = dyn_class(self.params)
        i = self.choose_test_case(i)
        input = self.input[:,0,i]
        solution = self.forward_step_sol[:,i]
        test, _ = dynamics.forward_step(input[:4], input[4:])
        self.check_solution('test_forward', test, solution)
            
    def test_AB_matrix(self, dyn_class, i):
        dynamics = dyn_class(self.params)
        i = self.choose_test_case(i)
        input = self.input[:,:,i]
        solution_A, solution_B = self.get_AB_matrix_sol[i]
        A, B = dynamics.get_AB_matrix(input[:4,:], input[4:,:])
        self.check_solution('A_k', A, solution_A)
        self.check_solution('B_k', B, solution_B)
            
    def test_get_cost(self, cost_class, i):
        cost = cost_class(self.params)
        i = self.choose_test_case(i)
        input = self.input[:,:,i]
        solution = self.get_cost_sol[i]
        
        closest_pt, slope, theta = self.ref_path.get_closest_pts(input[:2,:])
        J = cost.get_cost(input[:4,:], input[4:,:], closest_pt, slope, theta)
        self.check_solution('J', J, solution)

    def test_cost_state_deriv(self, cost_class, i):
        cost = cost_class(self.params)
        i = self.choose_test_case(i)
        input = self.input[:,:,i]
        solution_c_x, solution_c_xx = self.cost_state_deriv_sol[i]
        closest_pt, slope, theta = self.ref_path.get_closest_pts(input[:2,:])
        c_x, c_xx = cost._get_cost_state_derivative(input[:4,:], closest_pt, slope)
        
        self.check_solution('c_x', c_x, solution_c_x)
        self.check_solution('c_xx', c_xx, solution_c_xx)
    
    def test_cost_control_deriv(self, cost_class, i):
        cost = cost_class(self.params)
        i = self.choose_test_case(i)
        input = self.input[:,:,i]
        solution_c_u, solution_c_uu = self.cost_control_deriv_sol[i]
        c_u, c_uu = cost._get_cost_control_derivative(input[4:,:])
        self.check_solution('c_u', c_u, solution_c_u)
        self.check_solution('c_uu', c_uu, solution_c_uu)
        


        
        
if __name__ == '__main__':
    test = UnitTest()
    test.test_cost_state_deriv(Cost, -1)
    
        