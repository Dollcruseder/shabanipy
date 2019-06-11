import h5py
from math import *

f1 = h5py.File("cal_data.hdf5", "r")

N_orbit = 147885
S = []
L = []
cosj = []
T = []

for n in range(1, N_orbit + 1):

    S_a = f1[f"n={n}"].attrs["Surface"]
    S.append(S_a)

    L_a = f1[f"n={n}"].attrs["Length"]
    L.append(L_a)

    cosj_a = f1[f"n={n}"].attrs["cosj'"]
    cosj.append(cosj_a)

    T_a = f1[f"n={n}"].attrs["trace"]
    T.append(T_a)

f2 = h5py.File("useful_data.hdf5", "w")

dset1 = f2.create_dataset("Surface", (len(S),))
dset1[...] = S

dset2 = f2.create_dataset("Length", (len(L),))
dset2[...] = L

dset3 = f2.create_dataset("cosj'", (len(cosj),))
dset3[...] = cosj

dset4 = f2.create_dataset("Trace", (len(T),), dtype = complex)
dset4[...] = T
