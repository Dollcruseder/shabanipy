import h5py
import numpy as np
import random
from math import pi

# Setting the random generator seed

N_orbit = 147818
f1 = h5py.File("data_for_trace_cal.hdf5", "r")
f2 = h5py.File("load_data.hdf5", "w")
L = []
A = []
index = []
i = 0

for n in range (0, N_orbit):
    l = f1[f"n={n}"]["l"][:]
    angle = f1[f"n={n}"]["angle"][:] + random.uniform(0, 2 * pi)
    L.extend(l)
    A.extend(angle)
    index.append((i, i +len(l)))
    #index.append(i)
    i += len(l)

print(index)
dset1 = f2.create_dataset("l", (len(L),))
dset1[...] = L

dset2 = f2.create_dataset("angle", (len(A),))
dset2[...] = A

dset3 = f2.create_dataset("index", (len(index),2), dtype = "int64")
dset3[...] = index
