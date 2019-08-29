import matplotlib.pyplot as plt
from shabanipy.quantum_hall.wal.universal.magnetoconductivity import MC
from shabanipy.quantum_hall.wal.universal.create_data_file import get_data, get_data_for_MC_cal
from shabanipy.quantum_hall.wal.universal.find_trace import compute_traces_Zeeman, compute_traces_theta
import numpy as np
from math import pow, pi

#L_phi = int(input("L_phi="))
L_phi = 48
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set(title=rf'$L_\phi = {L_phi}$',
       ylabel=r'$\sigma / (2e^2 / h)$', xlabel='$B/B_t$')
n_s = np.arange(3, 5001)
F = np.sum(1 / (n_s - 2))

theta_alpha = 1.5
theta_beta1 = 0
theta_beta3 = 1.1
B_ZY = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1, 5 ,10]
B_zx = 0
N_orbit = 147818
x = np.linspace(0, -10, 401)

index, l, c_phi, c_3phi, s_phi, s_3phi = get_data("data_for_trace_cal")
S, L, cosj = get_data_for_MC_cal(N_orbit)

for B_zy in B_ZY:
    B_zy = -B_zy
    T = compute_traces_Zeeman(index, l, c_phi, c_3phi, s_phi, s_3phi, theta_alpha, theta_beta1, theta_beta3, N_orbit, B_zx, B_zy)
    y = np.empty(len(x))
    for i in range(0, len(x)):
        y[i] = -F * MC(x[i], L_phi, T, S, L, cosj) / (2 * pi)
    ax.plot(x, y, label = f"{B_zy}")
plt.legend(loc='upper left')
plt.show()
