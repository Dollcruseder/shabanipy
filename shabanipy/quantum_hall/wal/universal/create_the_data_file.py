import h5py
from trajectories import generate_trajectory
from trajectory_parameter import *

#n1 = 1
#n2 = 100
#f1 = h5py.File(f"cal_n({n1}-{n2})_data.hdf5", "w")
f1 = h5py.File(f"cal_data.hdf5", "w")
f2 = h5py.File("paper_data.hdf5", "r")

#print(list(f2.keys()))

number = f2["n"][:]
seed = f2["seed"][:]
n_scat = f2["n_scat"][:]
L = f2["L"][:]
S = f2["S"][:]
cosjf = f2["cosj'"][:]

for i in range(0, len(number)):
#for i in range(n1 - 1, n2):
    g1 = f1.create_group(f"n={i + 1}")

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
    g1.attrs["cosj'"] = cosjf[i]
