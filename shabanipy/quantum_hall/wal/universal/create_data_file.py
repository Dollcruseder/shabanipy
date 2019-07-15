# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2019 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Routines to create and access the data required for different calculations.

"""
import os
import random
from math import pi

import h5py
import pandas as pd
import numpy as np

from .find_trace import compute_traces
from .trajectories import generate_trajectory, identify_trajectory
from .trajectory_parameter import find_each_length, find_each_angle


def create_all_data():
    """Function to create a data file to store all the trajectory data

    """
    f1 = h5py.File(os.path.join(os.path.dirname(__file__),
                                "cal_data.hdf5"),
                   "w")
    f2 = open(os.path.join(os.path.dirname(__file__),
                                'paper_data.txt'),
              'r')

    dt = pd.read_csv(f2, delim_whitespace=True)

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
    """Function to create a data file to store the data for the trace calculation

    """
    f1 = h5py.File(os.path.join(os.path.dirname(__file__),
                                "data_for_trace_cal.hdf5"),
                   "w")
    f2 = open(os.path.join(os.path.dirname(__file__),
                                'paper_data.txt'),
              'r')
    dt = pd.read_csv(f2, delim_whitespace = True)

    number = dt['n']
    seed = dt['seed']
    n_scat = dt['n_scat']

    n_scat_max = 5000
    d = 2.5e-5
    n_scat_cal = np.empty(len(number), dtype=np.int)

    L = []
    C_phi = []
    C_3phi = []
    S_phi = []
    S_3phi = []
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
            c_phi = np.cos(angle)
            c_3phi = np.cos(3 * angle)
            s_phi = np.sin(angle)
            s_3phi = np.sin (3* angle)

            L.extend(l)
            C_phi.extend(c_phi)
            C_3phi.extend(c_3phi)
            S_phi.extend(s_phi)
            S_3phi.extend(s_3phi)
            index.append((j, j +len(l)))
            j += len(l)

    dset = f1.create_dataset("l", (len(L),))
    dset[...] = L

    dset = f1.create_dataset("c_phi", (len(C_phi),))
    dset[...] = C_phi

    dset = f1.create_dataset("c_3phi", (len(C_3phi),))
    dset[...] = C_3phi

    dset = f1.create_dataset("s_phi", (len(S_phi),))
    dset[...] = S_phi

    dset = f1.create_dataset("s_3phi", (len(S_3phi),))
    dset[...] = S_3phi

    dset = f1.create_dataset("index", (len(index),2), dtype = "int64")
    dset[...] = index


def create_data_for_MC_cal():
    """Function to create a data file to store the data for the magnetoconductivity calculation

    """
    f2 = h5py.File(os.path.join(os.path.dirname(__file__),
                                "data_for_MC_cal.hdf5"),
                   "w")
    f21 = open(os.path.join(os.path.dirname(__file__),
                                'paper_data.txt'),
              'r')
    dt = pd.read_csv(f1, delim_whitespace = True)

    number = dt['n']
    seed = dt['seed']
    n_scat = dt['n_scat']
    L = dt['L']
    S = dt['S']
    cosj = dt["cosj'"]

    S_new = []
    L_new = []
    cosj_new = []

    n_scat_max = 5000
    d = 2.5e-5

    n_scat_cal = np.empty(len(number), dtype=np.int)

    for i in range(0, len(number)):

        n_scat_cal[i] = identify_trajectory(seed[i], n_scat_max, d)
        if n_scat_cal[i] == n_scat[i]:
            S_new.append(S[i])
            L_new.append(L[i])
            cosj_new.append(cosj[i])

    dset1 = f2.create_dataset("Surface", (len(S_new),))
    dset1[...] = S_new

    dset2 = f2.create_dataset("Length", (len(L_new),))
    dset2[...] = L_new

    dset3 = f2.create_dataset("cosj", (len(cosj_new),))
    dset3[...] = cosj_new



def get_data(data_name: str):
    """Check if the data file exists. If it exists, get the data. If not, create the data file and get the data.

    Parameters
    ----------
    data_name : str
        the name of the data

    Returns
    -------
    [type]
        the data
    """
    if data_name == "all data":
        try:
            open(os.path.join(os.path.dirname(__file__),
                              "cal_data.hdf5"))
        except IOError:
            create_all_data()
        finally:
            f = h5py.File(os.path.join(os.path.dirname(__file__),
                                       "cal_data.hdf5"),
                          "r")
    elif data_name == "data_for_trace_cal":
        try:
            open(os.path.join(os.path.dirname(__file__),
                              "data_for_trace_cal.hdf5"))
        except IOError:
            create_data_for_trace_cal()
        finally:
            f = h5py.File(os.path.join(os.path.dirname(__file__),
                                       "data_for_trace_cal.hdf5"),
                          "r")
    elif data_name == "data_for_MC_cal":
        try:
            open(os.path.join(os.path.dirname(__file__),
                              "data_for_MC_cal.hdf5"))
        except IOError:
            create_data_for_MC_cal()
        finally:
            f = h5py.File(os.path.join(os.path.dirname(__file__),
                                       "data_for_MC_cal.hdf5"),
                          "r")

    return f


def create_trace_data(alpha, beta1, beta3, N_orbit, k, hvf):
    """Function to create a trace dataset with the input parameters

    Parameters
    ----------
    alpha : float
        the Rashba SOI coefficient
    beta1 : float
        the linear Dresselhaus coefficient
    beta3 : float
        the cubic Dresselhaus coefficient
    N_orbit : int
        orbit number
    k : float
        [description]
    hvf : float
        [description]


    """
    f1 = get_data("data_for_trace_cal")
    f2 = h5py.File(os.path.join(os.path.dirname(__file__),
                                "trace_data.hdf5"),
                   "a")
    l = f1["l"][:]
    c_phi = f1["c_phi"][:]
    c_3phi = f1["c_3phi"][:]
    s_phi = f1["s_phi"][:]
    s_3phi = f1["s_3phi"][:]
    index = f1["index"][:]

    T = compute_traces(index, l, c_phi, c_3phi, s_phi, s_3phi, alpha, beta1, beta3, k, hvf, N_orbit)
    dset1 = f2.create_dataset(f"alpha={alpha},beta1={beta1},beta3={beta3},N_orbit={N_orbit},k={k},hvf={hvf}, trace", (len(T),))
    dset1[...] = T


def get_trace_data(alpha, beta1, beta3, N_orbit, k, hvf):
    """Get the trace dataset with the input parameters. If it doesn't exist, create and get it

    Parameters
    ----------
    alpha : float
        the Rashba SOI coefficient
    beta1 : float
        the linear Dresselhaus coefficient
    beta3 : float
        the cubic Dresselhaus coefficient
    N_orbit : int
        orbit number
    k : [type]
        [description]
    hvf : [type]
        [description]

    Returns
    -------
    array
        the trace data
    """
    f = h5py.File(os.path.join(os.path.dirname(__file__),
                               "trace_data.hdf5"),
                  "a")

    try:
        f[f"alpha={alpha},beta1={beta1},beta3={beta3},N_orbit={N_orbit},k={k},hvf={hvf}, trace"][:]
    except KeyError:
        create_trace_data(alpha, beta1, beta3, N_orbit, k, hvf)
    finally:
        T = f[f"alpha={alpha},beta1={beta1},beta3={beta3},N_orbit={N_orbit},k={k},hvf={hvf}, trace"][:]


    return T
