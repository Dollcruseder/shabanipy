import matplotlib.pyplot as plt
from magnetoconductivity import MC
import numpy as np
from math import pow, pi

L_phi = int(input("L_phi="))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set(title=rf'$L_\phi = {L_phi}$',
       ylabel=r'$\sigma / (2e^2 / h)$', xlabel='$B/B_t$',
       xscale='log')
n_s = np.arange(3, 5001)
F = np.sum(1 / (n_s - 2))
for N_orbit in [147818]:
    for theta_R in [0, 1, 2, 4, 8]:
        x = np.linspace(-2, 1, 301)
        x1 = 10**x

        y = np.empty(len(x))

        for i in range(0, len(x)):
            y[i] = -F * MC(x1[i], L_phi, theta_R, N_orbit) / (2 * pi)

        ax.plot(x1, y, label = rf"$\theta_R = {theta_R*0.25}\pi$")


plt.legend(loc='upper left')
plt.show()
