from pyspline.pyCurve import Curve
import numpy as np


theta = np.linspace(0, 2*np.pi, 1000)
x = np.cos(theta)
y = np.sin(theta)

c = Curve(x = x, y = y, k=3)

points = np.array([[1,1], [-1,1], [-1,-1],[1,-1]])

s, _ = c.projectPoint(points, eps=1e-3)

pt_ref = c.getValue(s)
slope = np.zeros_like(s)
for i in range(4):
    deri = c.getDerivative(s[i])
    slope[i] = np.arctan2(deri[1], deri[0])

T = np.array([np.sin(slope), -np.cos(slope)])

error = (points - pt_ref).T
c = np.einsum('an,an->n', T, error)

b = np.exp(c)
b_dot = np.einsum('n,an->an', b, T)
b_ddot = np.einsum('n,abn->abn', (1**2)*b, np.einsum('an,bn->abn',T, T))

print(b_ddot[:,:,0])

# Left is negative
# Right is positive