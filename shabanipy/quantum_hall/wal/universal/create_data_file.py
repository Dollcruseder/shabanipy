import pandas as pd
import h5py
from shabanipy.quantum_hall.wal.universal.trajectories import generate_trajectory, identify_trajectory
from shabanipy.quantum_hall.wal.universal.trajectory_parameter import *
from shabanipy.quantum_hall.wal.universal.find_trace import compute_traces
import os
import random
from math import pi


def create_all_data():
    f1 = h5py.File("cal_data.hdf5", "w")
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


def create_data_for_trace_cal():
    f1 = h5py.File("data_for_trace_cal.hdf5", "w")
    f2 = open('paper_data.txt','r')
    dt = pd.read_csv(f2, delim_whitespace = True)

    number = dt['n']
    seed = dt['seed']
    n_scat = dt['n_scat']

    n_scat_max = 5000
    d = 2.5e-5
    n_scat_cal = np.empty(len(number), dtype=np.int)

    L = []
    A = []
    index = []
    j = 0

    for i in range(0, len(number)):

        n_scat_cal[i] = identify_trajectory(seed[i], n_scat_max, d)
        if n_scat_cal[i] == n_scat[i]:

            tj = generate_trajectory(seed[i], n_scat[i])
            x = tj[0]
            y = tj[1]
            x[-1] = 0
            y[-1] = 0

            l = find_each_length(x, y)
            angle = find_each_angle(x, y) + random.uniform(0, 2 * pi)

            L.extend(l)
            A.extend(angle)
            index.append((i, i +len(l)))
            i += len(l)

    dset1 = f1.create_dataset("l", (len(L),))
    dset1[...] = L

    dset2 = f1.create_dataset("angle", (len(A),))
    dset2[...] = A

    dset3 = f1.create_dataset("index", (len(index),2), dtype = "int64")
    dset3[...] = index


def create_data_for_MC_cal():
    f2 = h5py.File("data_for_MC_cal.hdf5", "w")
    f1 = open('paper_data.txt','r')
    dt = pd.read_csv(f1, delim_whitespace = True)

    number = dt['n']
    seed = dt['seed']
    n_scat = dt['n_scat']
    L = dt['L']
    S = dt['S']

    S_new = []
    L_new = []

    n_scat_max = 5000
    d = 2.5e-5

    n_scat_cal = np.empty(len(number), dtype=np.int)

    for i in range(0, len(number)):

        n_scat_cal[i] = identify_trajectory(seed[i], n_scat_max, d)
        if n_scat_cal[i] == n_scat[i]:
            S_new.append(S[i])
            L_new.append(L[i])

    dset1 = f2.create_dataset("Surface", (len(S_new),))
    dset1[...] = S_new

    dset2 = f2.create_dataset("Length", (len(L_new),))
    dset2[...] = L_new



def get_data(data_name):
    if data_name == "all data":
        try:
            open("cal_data.hdf5")
        except IOError:
            create_all_data()
        finally:
            f = h5py.File("cal_data.hdf5", "r")
    elif data_name == "data_for_trace_cal":
        try:
            open("data_for_trace_cal.hdf5")
        except IOError:
            create_data_for_trace_cal()
        finally:
            f = h5py.File("data_for_trace_cal.hdf5", "r")
    elif data_name == "data_for_MC_cal":
        try:
            open("data_for_MC_cal.hdf5")
        except IOError:
            create_data_for_MC_cal()
        finally:
            f = h5py.File("data_for_MC_cal.hdf5", "r")

    return f



def create_trace_data(alpha, beta1, beta3, N_orbit, k, hvf):
    f1 = get_data("data_for_trace_cal")
    f2 = h5py.File("trace_data.hdf5", "a")
    l = f1["l"][:]
    angle = f1["angle"][:]
    index = f1["index"][:]

    T, return_angle = compute_traces(index, l, angle, alpha, beta1, beta3, k, hvf, N_orbit)
    dset1 = f2.create_dataset(f"alpha={alpha},beta1={beta1},beta3={beta3},N_orbit={N_orbit},k={k},hvf={hvf}, trace", (len(T),))
    dset1[...] = T
    dset2 = f2.create_dataset(f"alpha={alpha},beta1={beta1},beta3={beta3},N_orbit={N_orbit},k={k},hvf={hvf}, return angle", (len(return_angle),))
    dset2[...] = return_angle


def get_trace_data(alpha, beta1, beta3, N_orbit, k, hvf):
    f = h5py.File("trace_data.hdf5", "a")

    try:
        f[f"alpha={alpha},beta1={beta1},beta3={beta3},N_orbit={N_orbit},k={k},hvf={hvf}, trace"][:]
    except KeyError:
        create_trace_data(alpha, beta1, beta3, N_orbit, k, hvf)
    finally:
        T = f[f"alpha={alpha},beta1={beta1},beta3={beta3},N_orbit={N_orbit},k={k},hvf={hvf}, trace"][:]

    try:
        f[f"alpha={alpha},beta1={beta1},beta3={beta3},N_orbit={N_orbit},k={k},hvf={hvf}, return angle"][:]
    except KeyError:
        create_trace_data(alpha, beta1, beta3, N_orbit, k, hvf)
    finally:
        return_angle = f[f"alpha={alpha},beta1={beta1},beta3={beta3},N_orbit={N_orbit},k={k},hvf={hvf}, return angle"][:]

    return T, return_angle
