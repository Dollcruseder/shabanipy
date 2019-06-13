from numba import njit
import numpy as np
from math import sqrt, atan2, cos, sin

@njit
def find_each_length(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Find the length of each segment.

    Parameters
    ----------

    Returns
    -------

    """

    length = []

    for i in range(1, len(x)):
        d = sqrt((x[i] - x[i - 1])**2 + (y[i] - y[i - 1])**2)
        length.append(d)

    return length

@njit
def find_each_angle(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    #find the angle of each segment to the x-axis

    angle = []

    # np.atan2(y, x)
    for i in range(1, len(x)):
        angle_a = atan2((y[i] - y[i - 1]), (x[i] - x[i - 1]))
        angle.append(angle_a)

    return angle

def find_trace(l: np.ndarray, angle: np.ndarray, theta_R: float) -> complex:

    A = np.array([[1, 0], [0, 1]], dtype = complex)

    for i in range(0, len(l)):
        theta = l[i] * theta_R
        k1 = cos(angle[i]) + 1j * sin(angle[i])
        k2 = cos(angle[i]) - 1j * sin(angle[i])
        a1 = np.array([[cos(theta/2), k2 * sin(theta/2)], [k1 * sin(theta/2), cos(theta/2)]], dtype = complex)
        A = np.dot(a1, A)

    T = np.trace(np.dot(A, A))
    return T.real
