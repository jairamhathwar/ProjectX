import csv
import numpy as np
from track import  Track
from matplotlib import pyplot as plt
from matplotlib import cm
from ilqr import iLQR
from IPython import display
import yaml

filename = 'racetrack/Catalunya.csv'

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

params_file = 'modelparams.yaml'
with open(params_file) as file:
    params = yaml.load(file, Loader= yaml.FullLoader)

solver = iLQR(track, params)

itr = 350
history = np.zeros((4,itr+1))

pos0, theta0 = track.interp([0])
x_cur =np.array([pos0[0], pos0[1], 0, theta0])
history[:,0] = x_cur
init_control = np.zeros((2,11))

plt.ion()
plt.show()
plt.figure(figsize=(5, 5))

for i in range(itr):
    states, controls = solver.solve(x_cur, controls = init_control)
    x_cur = states[:,1]
    history[:,i+1] = x_cur
    init_control[:,:-1] = controls[:,1:]

    display.clear_output(wait = True)
    display.display(plt.gcf())
    plt.clf()
    #print(controls)
    track.plot_track()
    plt.plot(states[0,:], states[1,:])
    sc = plt.scatter(history[0, :i+1], history[1,:i+1], s = 15, 
                c=history[2,:i+1], cmap=cm.jet, 
                vmin=0, vmax=4, edgecolor='none', marker='o')
    cbar = plt.colorbar(sc)
    cbar.set_label("velocity [m/s]")
    #plt.plot(history[0,:i+1], history[1,:i+1], '--')
    # plt.xlim((x_cur[0]-10, x_cur[0]+10))
    # plt.ylim((x_cur[1]-10, x_cur[1]+10))
    plt.axis('equal')


    # plt.figure()
    # plt.plot(states[2,:], label='v')
    # plt.plot(states[3,:], label='psi')
    # #plt.plot(states[4,:], label='theta')
    # plt.plot(controls[0,:], '--', label='a')
    # plt.plot(controls[1,:], '--', label='delta')
    # plt.plot(states[2,:]**2*np.tan(controls[1,:])/0.3, '-.',label='lat accel')
    # #plt.plot(controls[2,:], '--', label='d_theta')
    # plt.legend()
    plt.pause(0.01)

    #plt.show()