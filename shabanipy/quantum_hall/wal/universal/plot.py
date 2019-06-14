import matplotlib.pyplot as plt
from magnetoconductivity import MC
import numpy as np
from math import pow, pi

L_phi = 10
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set(xlim=[-2, 1], title=rf'$L_\phi = {L_phi}$',
       ylabel=r'$\Delta\sigma / (2e^2 / h)$', xlabel='$log(B/B_t)$')
n_s = np.arange(3, 5001)
F = np.sum(1 / (n_s - 2))

for theta_R in [0, 1, 2, 4, 8]:
    x = np.linspace(-2, 1, 301)
    y = []

    for i in range(0, len(x)):
        x1 = pow(10, x[i])
        y.append(-F * MC(x1, L_phi, theta_R) / (2 * pi))

    ax.plot(x, y, label = rf"$\theta_R = {theta_R*0.25}\pi$")
plt.legend(loc='upper left')
plt.show()
