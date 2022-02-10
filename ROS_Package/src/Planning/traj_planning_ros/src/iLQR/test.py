from pyspline.pyCurve import Curve
import numpy as np


theta = np.linspace(0, 2*np.pi, 1000)
x = np.cos(theta)
y = np.sin(theta)

c = Curve(x = x, y = y, k=3)

points = np.array([0.2,0.2])

s, _ = c.projectPoint(points, eps=1e-3)

pt_ref = c.getValue(s)
deri = c.getDerivative(s)
slope = np.arctan2(deri[1], deri[0])

T = np.array([np.sin(slope), -np.cos(slope)])
print(np.dot(T, (points-pt_ref)))


# Left is negative
# Right is positive