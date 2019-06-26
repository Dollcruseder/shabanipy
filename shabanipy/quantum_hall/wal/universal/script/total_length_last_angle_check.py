import pandas as pd
from trajectories import generate_trajectory, identify_trajectory
from trajectory_parameter import find_each_length
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

f = open('paper_data.txt','r')

dt = pd.read_csv(f, delim_whitespace = True)

number = dt['n']
seed = dt['seed']
n_scat = dt['n_scat']
L_paper = dt['L']
S = dt['S']
cosj_paper = dt["cosj'"]

n_scat_max = 5000
d = 2.5e-5
n_scat_cal = np.empty(len(number))
cosj_cal = np.empty(len(number))
L_cal = np.empty(len(number))
dL = np.empty(len(number))
dcosj = np.empty(len(number))
k = 0


for i in range(0, len(number)):

    n_scat_cal[i] = identify_trajectory(seed[i], n_scat_max, d)
    if n_scat_cal[i] == n_scat[i]:
        tj = generate_trajectory(seed[i], n_scat[i])
        x = tj[0]
        y = tj[1]
        x[-1] = 0
        y[-1] = 0

        l = find_each_length(x, y)

        L_cal[i] = np.sum(l)
        dL[i] = L_cal[i] - L_paper[i]
        if abs(L_cal[i] - L_paper[i]) > 0.005:
            print(f"The length of term {i+1} is wrong. The difference is{L_cal[i] - L_paper[i]}")
            k += 1

        cosj_cal[i] = (- x[-2]) / (sqrt(x[-2] ** 2 + y[-2] ** 2))
        dcosj[i] = cosj_cal[i] - cosj_paper[i]
        if abs(cosj_cal[i] - cosj_paper[i]) > 5e-7:
            print(f"The last angle of term {i+1} is wrong. The difference is{cosj_cal[i] - cosj_paper[i]}")
            k += 1

if k == 0:
    print("All the term are right.")
n = range(0, len(dL))
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.scatter(number, dL, s = 0.01)
ax2.scatter(number, dcosj, s = 0.01)
ax1.set(ylabel="difference of total length", xlabel="number")
ax2.set(ylim=[-6e-7, 6e-7], ylabel="difference of cosj", xlabel="number")

plt.show()


"""
dft = {'cosj_cal': cosj_cal,
      'cosj_paper': cosj_paper,
      'difference': abs(cosj_cal - cosj_paper)}

df = pd.DataFrame(dft)

print(df)
"""
