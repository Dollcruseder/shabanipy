from trajectories import generate_trajectory, identify_trajectory
from numba import njit
import numpy as np
import pandas as pd
from math import sqrt, pow, acos

def find_each_length(seed: int, n_scat: int) -> np.ndarray:

    tj1 = generate_trajectory(seed, n_scat)
    tj2 = pd.DataFrame(tj1, columns = ['x', 'y'])

    x = tj2['x']
    y = tj2['y']
    length = []

    for i in range(1, len(x)-1):
        d = sqrt(pow((x[i] - x[i - 1]), 2) + pow((y[i] - y[i - 1]), 2))
        length.append(d)

    d = sqrt(pow(x[len(x)-2],2) + pow(y[len(x)-2],2))
    length.append(d)

    return length


def find_each_angle(seed: int, n_scat: int) -> np.ndarray:
    tj1 = generate_trajectory(seed, n_scat)
    tj2 = pd.DataFrame(tj1, columns = ['x', 'y'])

    x = tj2['x']
    y = tj2['y']
    angle = []

    for i in range(1, len(x)-1):
        d = sqrt(pow((x[i] - x[i - 1]), 2) + pow((y[i] - y[i - 1]), 2))
        t1 = (x[i] - x[i - 1])/d
        t2 = acos(t1)
        angle.append(t2)

    d = sqrt(pow(x[len(x)-2],2) + pow(y[len(x)-2],2))
    angle.append(acos((-x[len(x)-2])/d))

    return angle

def find_total_length(seed: int, n_scat: int) -> float:

    tj1 = generate_trajectory(seed, n_scat)
    tj2 = pd.DataFrame(tj1, columns = ['x', 'y'])

    x = tj2['x']
    y = tj2['y']

    L = 0.0

    for i in range(1, len(x)-1):
        d = sqrt(pow((x[i] - x[i - 1]), 2) + pow((y[i] - y[i - 1]), 2))
        L = L + d

    d = sqrt(pow(x[len(x)-2],2) + pow(y[len(x)-2],2))
    L = L + d

    return L


def find_each_length_x(seed: int, n_scat: int) -> np.ndarray:

    tj1 = generate_trajectory(seed, n_scat)
    tj2 = pd.DataFrame(tj1, columns = ['x', 'y'])

    x = tj2['x']
    y = tj2['y']
    length_x = []

    for i in range(1, len(x)-1):
        d = x[i] - x[i - 1]
        length_x.append(d)

    d = x[len(x)-2]
    length_x.append(d)

    return length_x

def find_each_length_y(seed: int, n_scat: int) -> np.ndarray:

    tj1 = generate_trajectory(seed, n_scat)
    tj2 = pd.DataFrame(tj1, columns = ['x', 'y'])

    x = tj2['x']
    y = tj2['y']
    length_y = []

    for i in range(1, len(x)-1):
        d = y[i] - y[i - 1]
        length_y.append(d)

    d = y[len(x)-2]
    length_y.append(d)

    return length_y
