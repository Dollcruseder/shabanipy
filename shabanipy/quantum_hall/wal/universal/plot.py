import matplotlib.pyplot as plt
from shabanipy.quantum_hall.wal.universal.magnetoconductivity import MC
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

alpha = pi
beta1 = 0
beta3 = 0
N_orbit = 40000
k = 1
hvf = 1
x = np.linspace(-2, 1, 301)
x1 = 10**x
y = np.empty(len(x))
for i in range(0, len(x)):
    y[i] = -F * MC(x1[i], L_phi, alpha, beta1, beta3, N_orbit, k, hvf) / (2 * pi)
ax.plot(x1, y)
