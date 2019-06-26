import h5py
from trajectories import generate_trajectory, identify_trajectory
from trajectory_parameter import find_each_length, find_each_angle
import numpy as np


"""



"""

f1 = h5py.File(f"data_for_trace_cal.hdf5", "w")
f2 = h5py.File("paper_data.hdf5", "r")

number = f2["n"][:]
seed = f2["seed"][:]
n_scat = f2["n_scat"][:]

n_scat_max = 5000
d = 2.5e-5
n_scat_cal = np.empty(len(number), dtype=np.int)
k = 0
j = 0

for i in range(0, len(number)):

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

        dset = g1.create_dataset("l", (len(l),))
        dset[...] = l

        dset = g1.create_dataset("angle", (len(angle),))
        dset[...] = angle

        k += 1
    else:
        j += 1


print(k)
print(j)
