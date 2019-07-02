import os
os.environ['MKL_NUM_THREADS'] = '1'

import h5py
from find_trace import find_trace
from math import pi
import numpy as np
import random
from numba import njit, prange
import time
import sys

N_orbit = int(input("N_orbit = "))
f1 = h5py.File("load_data.hdf5", "r")
f2 = h5py.File(f"trace_data_n_{N_orbit}.hdf5", "w")

l = f1["l"][:]
angle = f1["angle"][:]
index = f1["index"][:]

beta3 = pi / 32
k = 1
hvf = 1

a = np.linspace(-2, 1, 16)

@njit(parallel=True)
def compute_traces(index, l, angle, alpha, beta1, beta3, k, hvf):
    """
    """
    T = np.empty(N_orbit)
    return_angle = np.empty(N_orbit)

    for n in prange(N_orbit//1000):
        for i in range(1000):
            traj_id = n*1000 + i
            if traj_id >= N_orbit:
                break
            begin, end = index[traj_id]
            T_a = find_trace(l[begin:end], angle[begin:end],
                             alpha, beta3, beta1, k, hvf)
            T[traj_id] = T_a
            return_angle[traj_id] = angle[end-1]

    return T, return_angle

tic = time.time()
for i in a:
    alpha = beta1 = (10 ** i) * pi / 2
    print("i = ", i)
    tic = time.time()
    T, return_angle = compute_traces(index, l, angle, alpha, beta1, beta3, k, hvf)
    print('Computation', time.time() - tic)

    tic = time.time()
    dset1 = f2.create_dataset(f"theta={(10 **i) / 2}pi, trace", (len(T),))
    dset1[...] = T

    dset2 = f2.create_dataset(f"theta={(10 **i) / 2}pi, return angle", (len(return_angle),))
    dset2[...] = return_angle
    print('Saving', time.time() - tic)
