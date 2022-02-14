from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from ilqr import iLQR
from track import Track
import yaml
from IPython import display




y1 = np.linspace(0,10,100,endpoint=False)
x1 = np.zeros_like(y1)

x2 = np.linspace(1,2,100, endpoint=False)
y2 = np.zeros_like(x2)+11

y3 = -np.linspace(1,2,100,endpoint=False)+11
x3 = np.zeros_like(y3)+3

x4 = np.linspace(1,10,100,endpoint=False)+3
y4 = np.zeros_like(x3)+8


x = np.concatenate([x1,x2, x3, x4])
y = np.concatenate([y1,y2, y3, y4])

track = Track(np.array([x,y]), 0.5, 0.5, False)

params_file = 'modelparams.yaml'
with open(params_file) as file:
    params = yaml.load(file, Loader= yaml.FullLoader)

solver = iLQR(track, params)


itr = 35
history = np.zeros((4,itr+1))

x_cur =np.array([1, 0, 3, np.pi/2])
history[:,0] = x_cur
init_control = np.zeros((2,11))

plt.ion()
plt.show()
plt.figure(figsize=(5, 5))

for i in range(itr):
    states, controls = solver.solve(x_cur, controls = init_control,debug=False)
    x_cur = states[:,1]
    history[:,i+1] = x_cur
    init_control[:,:-1] = controls[:,1:]

    display.clear_output(wait = True)
    display.display(plt.gcf())
    plt.clf()
    #print(controls)
    track.plot_track()
    plt.plot(states[0,:], states[1,:])
    plt.plot(history[0,:i+1], history[1,:i+1], '--')
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
    plt.pause(0.5)

    #plt.show()



