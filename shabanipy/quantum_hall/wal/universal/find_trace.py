from numba import njit, prange
import numpy as np
from math import exp, cos, pi


@njit(fastmath=True)
def find_trace(l: np.ndarray, angle: np.ndarray, alpha: float, beta3: float,
               beta1: float, k: float, hvf: float) -> float:
    """Find the trace of the matrix R_tot^2

    Parameters
    ----------
    l: np.ndarray
        (n_scat) array, length of each segment
    angle: np.ndarray
        (n_scat) array, angle of each segment
    alpha: float
        Rashba SOI coeffecient
    beta3: float
        Cubic Dresselhaus coeffecient
    beta1: float
        Linear Dresselhaus coeffecient
    k: float

    hvf: float


    Returns
    -----------
    trace: float
        The trace of the matrix R_tot^2

    """
    rotations = np.empty((len(l), 2, 2), dtype=np.complex128)

    c_phi = np.cos(angle)
    s_phi = np.sin(angle)
    c_3phi = np.cos(3 * angle)
    s_3phi = np.sin(3 * angle)

    B_x = alpha * k * s_phi + beta3 * (k ** 3) * c_3phi + beta1 * k * c_phi
    B_y = -alpha * k * c_phi + beta3 * (k ** 3) * s_3phi - beta1 * k * s_phi
    B = np.sqrt(B_x ** 2 + B_y ** 2)
    theta = 2 * B * l / hvf
    c_theta = np.cos(0.5*theta)
    s_theta = np.sin(0.5*theta)

    psi1 = np.empty(len(l), dtype=np.complex128)
    psi2 = np.empty(len(l), dtype=np.complex128)
    for i, (b, bx, by) in enumerate(zip(B, B_x, B_y)):
        if b != 0:
            psi1[i] = -1j * (bx / b + 1j * by /b)
            psi2[i] = -1j * (bx / b - 1j * by /b)
        else:
            psi1[i] = psi2[i] = 0

    rotations[:, 0, 0] = c_theta
    rotations[:, 0, 1] = psi1 * s_theta
    rotations[:, 1, 0] = psi2 * s_theta
    rotations[:, 1, 1] = c_theta

    cw_rot = np.array([[1, 0], [0, 1]], dtype=np.complex128)
    for i in range(0, len(l)):
        cw_rot = rotations[i] @ cw_rot

    return np.trace(cw_rot @ cw_rot).real



@njit(parallel=True)
def compute_traces(index, l, angle, alpha, beta1, beta3, k, hvf, N_orbit):
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
