import h5py
from math import exp, cos, pi
import numpy as np

def MC(x: float, L_phi: float, theta_R: int) -> float:
    f1 = h5py.File("useful_data.hdf5", "r")
    f2 = h5py.File("trace_data.hdf5", "r")

    Sum = 0

    S = f1["Surface"][:]
    L = f1["Length"][:]
    cosj = f1["cosj'"][:]
    #T = f2[f"theta={theta_R * 0.25}pi"][:]
    T = f2[f"theta={theta_R * 0.25}pi"][:]

    xj = np.exp(- L / L_phi) * 0.5 * T * (1 + cosj)
    a = xj * np.cos(x * S)

    return np.sum(a) / len(S)
