import h5py
import numpy as np
from numpy import *
import pandas as pd
from trajectories import generate_trajectory
from trajectory_parameter import *
from math import sqrt

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

    tj1 = generate_trajectory(seed[i], n_scat[i])
    tj2 = pd.DataFrame(tj1, columns = ['x', 'y'])
    x = tj2["x"]
    y = tj2["y"]
    l = find_each_length(seed[i], n_scat[i])
    angle = find_each_angle(seed[i], n_scat[i])
    lx = find_each_length_x(seed[i], n_scat[i])
    ly = find_each_length_y(seed[i], n_scat[i])

    dset = g1.create_dataset("x", (len(x),))
    dset[...] = x

    dset = g1.create_dataset("y", (len(y),))
    dset[...] = y

    dset = g1.create_dataset("l", (len(l),))
    dset[...] = l

    dset = g1.create_dataset("l_x", (len(lx),))
    dset[...] = lx

    dset = g1.create_dataset("l_y", (len(ly),))
    dset[...] = ly

    dset = g1.create_dataset("angle", (len(angle),))
    dset[...] = angle

    T = find_trace(lx, ly, angle)

    g1.attrs["Length"] = L[i]
    g1.attrs["n_scat"] = n_scat[i]
    g1.attrs["seed"] = seed[i]
    g1.attrs["Surface"] = S[i]
    g1.attrs["cosj'"] = cosjf[i]
    g1.attrs["trace"] = T
