import h5py
from math import exp, cos, pi
import numpy as np

def MC(x: float, L_phi: float, i: float, N_orbit: int) -> float:
    f1 = h5py.File("useful_data.hdf5", "r")
    f2 = h5py.File(f"trace_data_n_{N_orbit}.hdf5", "r")


    T = f2[f"theta={(10 **i) / 2}pi, trace"][:]
    angle = f2[f"theta={(10 **i) / 2}pi, return angle"][:]
    S = f1["Surface"][:len(T)]
    L = f1["Length"][:len(T)]
    cosj = np.cos(angle)
    #T = f2[f"theta={theta_R * 0.25}pi"][:]

    xj = np.exp(- L / L_phi) * 0.5 * T * (1 + cosj)
    a = xj * np.cos(x * S)

    return np.sum(a) / len(T)
