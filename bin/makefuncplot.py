import numpy as np

import matplotlib.pyplot as plt

import math

x = np.linspace(-2.5*math.pi,2.5*math.pi,100)
#y = 1./300*x**2*(x-2*np.pi)**2*(x+2*np.pi)**2
y = x*x



fig, ax = plt.subplots()
ax.plot(x, y, 'k')

# set ticks and tick labels
ax.set_xlim((-2.5*np.pi, 2.5*np.pi))
ax.set_xticks([-2*np.pi,-np.pi,0, np.pi, 2*np.pi])
ax.set_xticklabels(['$-2\pi$', '$-\pi$','0', '$\pi$', '2$\pi$'])

plt.xlabel('Variance (prediction - target)')

plt.ylabel('Cost function penalty')

plt.savefig('Squaredcostfunc.png')

plt.show()
