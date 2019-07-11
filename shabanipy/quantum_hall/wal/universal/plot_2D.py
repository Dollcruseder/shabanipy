import matplotlib.pyplot as plt
from shabanipy.quantum_hall.wal.universal.magnetoconductivity import MC
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



for i in y:
    z = []
    for j in x:
        z.append( -F * MC(j, L_phi, i, i, beta3, N_orbit, k, hvf) / (2 * pi))
    Z.append(z)
    zm = x1[z.index(min(z))]
    Zm.append(zm)

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
