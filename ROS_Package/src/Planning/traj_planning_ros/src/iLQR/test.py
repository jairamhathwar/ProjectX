from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from ilqr import iLQR
from track import Track
import yaml



y1 = np.linspace(0,1,100,endpoint=False)
x1 = np.zeros_like(y1)

x2 = -np.linspace(1,5,100, endpoint=False)
y2 = np.zeros_like(x2)+2

# y3 = np.linspace(1,4,100,endpoint=False)+5
# x3 = np.zeros_like(y3)+3

x = np.concatenate([x1,x2])
y = np.concatenate([y1,y2])

track = Track(np.array([x,y]), 0.5, 0.5, False)

params_file = 'modelparams.yaml'
with open(params_file) as file:
    params = yaml.load(file, Loader= yaml.FullLoader)

solver = iLQR(track, params)

x_cur =np.array([1, 0, 3, np.pi/2])
states, controls = solver.solve(x_cur, debug=False)
#print(controls)
track.plot_track()
plt.plot(states[0,:], states[1,:])
plt.axis('equal')

plt.figure()
plt.plot(states[2,:], label='v')
plt.plot(states[3,:], label='psi')
#plt.plot(states[4,:], label='theta')
plt.plot(controls[0,:], '--', label='a')
plt.plot(controls[1,:], '--', label='delta')
plt.plot(states[2,:]**2*np.tan(controls[1,:])/0.3, '-.',label='lat accel')
#plt.plot(controls[2,:], '--', label='d_theta')
plt.legend()


plt.show()



