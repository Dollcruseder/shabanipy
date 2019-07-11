import pandas as pd
import h5py
from shabanipy.quantum_hall.wal.universal.trajectories import generate_trajectory, identify_trajectory
from shabanipy.quantum_hall.wal.universal.trajectory_parameter import *


f1 = h5py.File(f"cal_data.hdf5", "w")
f2 = open('paper_data.txt','r')

dt = pd.read_csv(f2, delim_whitespace = True)

number = dt['n']
seed = dt['seed']
n_scat = dt['n_scat']
L = dt['L']
S = dt['S']
cosj = dt["cosj'"]

n_scat_max = 5000
d = 2.5e-5

n_scat_cal = np.empty(len(number), dtype=np.int)
#n1 = 1
#n2 =100
k = 1
for i in range(0, len(number)):
#for i in range(n1 - 1, n2):
    #g1 = f1.create_group(f"n={i + 1}")
    n_scat_cal[i] = identify_trajectory(seed[i], n_scat_max, d)
    if n_scat_cal[i] == n_scat[i]:
        g1 = f1.create_group(f"n={k}")
        tj = generate_trajectory(seed[i], n_scat[i])
        x = tj[0]
        y = tj[1]
        x[-1] = 0
        y[-1] = 0

        l = find_each_length(x, y)
        angle = find_each_angle(x, y)

        dset = g1.create_dataset("x", (len(x),))
        dset[...] = x

        dset = g1.create_dataset("y", (len(y),))
        dset[...] = y

        dset = g1.create_dataset("l", (len(l),))
        dset[...] = l

        dset = g1.create_dataset("angle", (len(angle),))
        dset[...] = angle

        g1.attrs["Length"] = L[i]
        g1.attrs["n_scat"] = n_scat[i]
        g1.attrs["seed"] = seed[i]
        g1.attrs["Surface"] = S[i]
        g1.attrs["cosj'"] = cosj[i]
        k += 1
