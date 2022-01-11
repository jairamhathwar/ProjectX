from scipy.interpolate import CubicSpline, interp1d
import numpy as np
import matplotlib.pyplot as plt
import time

x = np.arange(101)
y = np.sin(x)

start = time.time()
cs = CubicSpline(x, y)
end = time.time()
print(end - start)

start = time.time()
linear = interp1d(x,y, kind='cubic')
end = time.time()
print(end - start)

xs = np.arange(0.5, 99.6, 0.1)

start = time.time()
cs(xs)
linear(xs)
end = time.time()
print(end - start)


fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(x, y, 'o', label='data')
ax.plot(xs, np.sin(xs), label='true')
ax.plot(xs, cs(xs), label="S")
ax.plot(xs, linear(xs), label="S linear")
ax.plot(xs, cs(xs, 1), label="S'")
ax.plot(xs, cs(xs, 2), label="S''")
ax.plot(xs, cs(xs, 3), label="S'''")
ax.set_xlim(-0.5, 9.5)
ax.legend(loc='lower left', ncol=2)
plt.show()