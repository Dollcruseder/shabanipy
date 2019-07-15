import matplotlib.pyplot as plt
from shabanipy.quantum_hall.wal.universal.magnetoconductivity import MC
from shabanipy.quantum_hall.wal.universal.create_data_file import get_data, get_data_for_MC_cal
from shabanipy.quantum_hall.wal.universal.find_trace import compute_traces
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

Alpha = [0, pi/8, pi/4, pi/2, pi]
beta1 = 0
beta3 = 0
N_orbit = 40000
k = 1
hvf = 1
x = np.linspace(-2, 1, 301)
x1 = 10**x
index, l, c_phi, c_3phi, s_phi, s_3phi = get_data("data_for_trace_cal")
S, L, cosj = get_data_for_MC_cal(N_orbit)

for alpha in Alpha:
    T = compute_traces(index, l, c_phi, c_3phi, s_phi, s_3phi, alpha, beta1, beta3, k, hvf, N_orbit)
    y = np.empty(len(x))
    for i in range(0, len(x)):
        y[i] = -F * MC(x1[i], L_phi, T, S, L, cosj) / (2 * pi)
    ax.plot(x1, y)
plt.show()
