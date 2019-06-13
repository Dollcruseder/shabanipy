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

    length = []

    for i in range(1, len(x)):
        d = sqrt((x[i] - x[i - 1])**2 + (y[i] - y[i - 1])**2)
        length.append(d)

    return length

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


    angle = []

    for i in range(1, len(x)):
        angle_a = atan2((y[i] - y[i - 1]), (x[i] - x[i - 1]))
        angle.append(angle_a)

    return angle

def find_trace(l: np.ndarray, angle: np.ndarray, theta_R: float) -> float:
    """Find the trace of the matrix R_tot^2

    Parameters
    ----------
    l: np.ndarray
        (1, n_scat) array, length of each segment
    angle: np.ndarray
        (1, n_scat) array, angle of each segment
    theta_R: float
        the spin rotation angle per the MFP length

    Returns
    -----------
    trace: float
        The trace of the matrix R_tot^2

        
    """

    A = np.array([[1, 0], [0, 1]], dtype = complex)

    for i in range(0, len(l)):
        theta = l[i] * theta_R
        k1 = cos(angle[i]) + 1j * sin(angle[i])
        k2 = cos(angle[i]) - 1j * sin(angle[i])
        a1 = np.array([[cos(theta/2), k2 * sin(theta/2)], [k1 * sin(theta/2), cos(theta/2)]], dtype = complex)
        A = np.dot(a1, A)

    T = np.trace(np.dot(A, A))
    return T.real
