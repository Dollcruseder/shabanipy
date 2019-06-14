from numba import njit
import numpy as np
from math import sqrt, atan2, cos, sin

@njit
def find_each_length(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Find the length of each segment.

    Parameters
    ----------
    x: np.ndarray
        (1, n_scat) array, the x position of each point
    y: np.ndarray
        (1, n_scat) array, the y position of each point

    Returns
    ----------
    length
        (1, n_scat) array, length of each segment

    """
    return np.sqrt((y[1:] - y[:-1])**2 + (x[1:] - x[:-1])**2)

@njit
def find_each_angle(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Find the angle of each segment.

    Parameters
    ----------
    x: np.ndarray
        (1, n_scat) array, the x position of each point
    y: np.ndarray
        (1, n_scat) array, the y position of each point

    Returns
    ----------
    angle
        (1, n_scat) array, angle of each segment

    """
    return np.atan2((y[1:] - y[:-1]), (x[1:] - x[:-1]))

@njit
def find_trace(l: np.ndarray, angle: np.ndarray, theta_R: float) -> float:
    """Find the trace of the matrix R_tot^2

    Parameters
    ----------
    l: np.ndarray
        (n_scat) array, length of each segment
    angle: np.ndarray
        (n_scat) array, angle of each segment
    theta_R: float
        the spin rotation angle per the MFP length

    Returns
    -----------
    trace: float
        The trace of the matrix R_tot^2


    """
    rotations = np.empty((len(l), 2, 2), dtype=np.complex128)
    theta = theta_R*l
    c_theta = np.cos(0.5*theta)
    s_theta = np.sin(0.5*theta)
    c_phi = np.cos(angle)
    s_phi = np.sin(angle)
    rotations[:, 0, 0] = c_theta
    rotations[:, 0, 1] = -1j * (c_phi - 1j*s_phi)*s_theta
    rotations[:, 1, 0] = -1j * (c_phi + 1j*s_phi)*s_theta
    rotations[:, 1, 1] = c_theta

    cw_rot = np.array([[1, 0], [0, 1]], dtype=np.complex128)
    for i in range(0, len(l)):
        cw_rot = rotations[i] @ cw_rot


    return np.trace(cw_rot @ cw_rot).real
