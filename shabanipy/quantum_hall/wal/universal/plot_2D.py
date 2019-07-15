import os
os.environ['MKL_NUM_THREADS'] = '1'
import time
import sys
import matplotlib.pyplot as plt
from shabanipy.quantum_hall.wal.universal.magnetoconductivity import MC
from shabanipy.quantum_hall.wal.universal.create_data_file import get_data, get_data_for_MC_cal
from shabanipy.quantum_hall.wal.universal.find_trace import compute_traces
import numpy as np
from math import pow, pi

L_phi = 100
N_orbit = 40000
n_s = np.arange(3, 5001)
F = np.sum(1 / (n_s - 2))

x1 = np.linspace(-2, 1, 301)
x = 10**x1
a = np.linspace(-2, 1, 16)
y = (10**a) * pi / 2
X,Y=np.meshgrid(x1,a)
Z = []
Zm = []
beta3 = pi / 32
k = 1
hvf = 1
index, l, c_phi, c_3phi, s_phi, s_3phi = get_data("data_for_trace_cal")
S, L, cosj = get_data_for_MC_cal(N_orbit)

tic = time.time()
for i in y:

    T = compute_traces(index, l, c_phi, c_3phi, s_phi, s_3phi, i, i, beta3, k, hvf, N_orbit)

    z = []
    for j in x:
        z.append( -F * MC(j, L_phi, T, S, L, cosj) / (2 * pi))
    Z.append(z)
    zm = x1[z.index(min(z))]
    Zm.append(zm)
print('Computation', time.time() - tic)

cs = plt.contourf(X, Y, Z, 20)
#plt.pcolormesh(X, Y, Z)
plt.contour(cs, colors='k')
lb = plt.contour(cs, levels=sorted([-0.1*i for i in range(1, 7)]), colors='k',
            linestyles='solid')
plt.plot(Zm, a, color = 'red')
# plt.contour(X, Y, Z, [-0.1*i for i in range(1, 7)], linestyle='dashed')
plt.colorbar(cs)
plt.clabel(lb, inline=True, fontsize=10)
plt.show()
