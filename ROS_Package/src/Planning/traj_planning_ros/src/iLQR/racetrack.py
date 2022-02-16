import numpy as np
from track import  Track
from matplotlib import pyplot as plt
from matplotlib import cm
from ilqr import iLQR
import yaml
import csv


filename = 'racetrack/Nuerburgring.csv'

params_file = 'modelparams.yaml'
with open(params_file) as file:
    params = yaml.load(file, Loader= yaml.FullLoader)
    
# Load parameters
L = params['l_r']+params['l_f']
delta_min = params['delta_min']
delta_max = params['delta_max']
a_min = params['a_min']
a_max = params['a_max']
v_max = params['v_max']
T = params['T']
# number of planning steps
N = params['N']
dt = T/(N-1)

sigma = np.array([0,0,0,0])

def simulate_kin(state, control, step = 10):

    accel = np.clip(control[0], a_min, a_max)
    delta = np.clip(control[1], delta_min, delta_max)
    
    next_state = state

    dt_step = dt/step
    for _ in range(step):
        # State: [x, y, psi, v]
        d_x = (next_state[2]*dt_step+0.5*accel*dt_step**2)*np.cos(next_state[3])
        d_y = (next_state[2]*dt_step+0.5*accel*dt_step**2)*np.sin(next_state[3])
        d_v = accel*dt_step
        d_psi = (next_state[2]*dt_step+0.5*accel*dt_step**2)*np.tan(delta)/L
        next_state = next_state + np.array([d_x, d_y, d_v, d_psi])+sigma*np.random.normal(size=(4))*dt_step
        next_state[2] = max(0, next_state[2])
    return next_state

x = []
y = []
width_r = []
width_l = []

with open(filename, newline='') as f:
    spamreader = csv.reader(f, delimiter=',')
    for i, row in enumerate(spamreader):
        if i>0:
            x.append(float(row[0]))
            y.append(float(row[1]))
            width_r.append(float(row[2]))
            width_l.append(float(row[3]))

x = np.array(x)/25.0
y = np.array(y)/25.0
center_line = np.array([x,y])
track = Track(center_line = center_line, width_left = 0.4, width_right = 0.4)



solver = iLQR(track, params)

itr_max = 500
state_hist = np.zeros((4,itr_max+1))
control_hist = np.zeros((2,itr_max))
plan_hist = np.zeros((4, N, itr_max))

pos0, psi0 = track.interp([0])
x_cur =np.array([pos0[0], pos0[1], 0, psi0[0]])

#state_hist[:,0] = x_cur
init_control = np.zeros((2,11))

# 
t_total = 0

for i in range(itr_max):
    
    states, controls, t_process, status, theta \
            = solver.solve(x_cur, controls = init_control) 
    
    plan_hist[:,:, i] = states
    
    state_hist[:,i] = states[:,0]
    control_hist[:,i] = controls[:,0]
    init_control[:,:-1] = controls[:,1:]
    
    x_cur = simulate_kin(x_cur, controls[:,0])
    t_total += t_process
   
    
    if theta[1]<theta[0]:
        break

print(t_total, i)

# show results of the run
# 

plt.figure(figsize=(15, 15))
state_hist = state_hist[:, :i+2]
control_hist = control_hist[:,:i+1]
plan_hist = plan_hist[:,:, :i+1]

for j in range(i+1):
    plt.clf()
    track.plot_track()
    plt.plot(plan_hist[0,:, j], plan_hist[1,:, j], linewidth= 4)
    #plt.plot(state_hist[0,j], state_hist[1,j], '*', markersize=20)
    sc = plt.scatter(state_hist[0, :j+1], state_hist[1,:j+1], s = 80, 
                c=state_hist[2,:j+1], cmap=cm.jet, 
                vmin=0, vmax=params['v_max'], edgecolor='none', marker='o')
    cbar = plt.colorbar(sc)
    cbar.set_label(r"velocity [$m/s$]", size=20)
    plt.axis('equal')
    plt.pause(0.01)
plt.close()

# raceline
plt.figure(figsize=(15, 15))
track.plot_track()
plt.plot(state_hist[0,:], state_hist[1,:], linewidth= 5)
plt.title("Trajectory")

# velocity
plt.figure(figsize=(15, 15))
track.plot_track()
sc = plt.scatter(state_hist[0, :-1], state_hist[1,:-1], s = 80, 
                c=state_hist[2,:-1], cmap=cm.jet, 
                vmin=params['v_min'], vmax=params['v_max'], edgecolor='none', marker='o')
cbar = plt.colorbar(sc)
cbar.set_label(r"Velocity [$m/s$]", size=20)

# Longitudinal Accel
plt.figure(figsize=(15, 15))
track.plot_track()
sc = plt.scatter(state_hist[0, :-1], state_hist[1,:-1], s = 80, 
                c=control_hist[0,:], cmap=cm.jet, 
                vmin=params['a_min'], vmax=params['a_max'], edgecolor='none', marker='o')
cbar = plt.colorbar(sc)
cbar.set_label(r"Longitudinal Accel [$m/s^2$]", size=20)
plt.show()



