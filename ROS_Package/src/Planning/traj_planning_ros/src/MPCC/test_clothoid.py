from numpy import pi
import matplotlib.pyplot as plt
from pyclothoids import Clothoid, SolveG2

clothoid_list = SolveG2(200, 0, 0, 0, 250, 50, 0.25*pi, 0)
plt.figure()
for i in clothoid_list:
        plt.plot( *i.SampleXY(500) )
        print(i.KappaEnd)



plt.show()
