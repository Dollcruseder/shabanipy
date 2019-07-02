import matplotlib.pyplot as plt
from magnetoconductivity_n import MC
import numpy as np
from math import pow, pi

L_phi = 100
N_orbit = 147818
n_s = np.arange(3, 5001)
F = np.sum(1 / (n_s - 2))

x1 = np.linspace(-2, 1, 301)
x = 10**x1
a = np.linspace(-2, 1, 16)
y = 10**a
X,Y=np.meshgrid(x1,a)
Z = []


for i in a:
    z = []
    for j in x1:
        z.append(-F * MC(10 ** j, L_phi, i, N_orbit) / (2 * pi))
    Z.append(z)

cs = plt.contourf(X, Y, Z, 20)
#plt.pcolormesh(X, Y, Z)
plt.contour(cs, colors='k')
plt.contour(cs, levels=sorted([-0.1*i for i in range(1, 7)]), colors='k',
            linestyles='solid')
# plt.contour(X, Y, Z, [-0.1*i for i in range(1, 7)], linestyle='dashed')
plt.show()
