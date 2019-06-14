import h5py
from math import exp, cos, pi
import numpy as np

def MC(x: float, L_phi: float, theta_R: int, N_orbit: int) -> float:
    f1 = h5py.File("useful_data.hdf5", "r")
    f2 = h5py.File(f"trace_data_{N_orbit}.hdf5", "r")


    T = f2[f"theta={theta_R * 0.25}pi"][:]
    S = f1["Surface"][:len(T)]
    L = f1["Length"][:len(T)]
    cosj = f1["cosj'"][:len(T)]
    #T = f2[f"theta={theta_R * 0.25}pi"][:]

    xj = np.exp(- L / L_phi) * 0.5 * T * (1 + cosj)
    a = xj * np.cos(x * S)

    return np.sum(a) / len(T)

def MC_random(x: float, L_phi: float, theta_R: int, random_N_orbit: np.ndarray) -> float:
    f1 = h5py.File("useful_data.hdf5", "r")
    f2 = h5py.File(f"trace_data_147885.hdf5", "r")

    T = f2[f"theta={theta_R * 0.25}pi"][:]
    S = f1["Surface"][:]
    L = f1["Length"][:]
    cosj = f1["cosj'"][:]

    a = []
    for i in random_N_orbit:
        xj = exp(- L[i] / L_phi) * 0.5 * T[i] * (1 + cosj[i])
        a.append(xj * cos(x * S[i]))

    return np.sum(a) / len(random_N_orbit)
