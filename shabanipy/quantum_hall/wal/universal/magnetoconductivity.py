# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2019 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Routines to compute the correction to the magneto-conductivity.

"""
from math import cos, exp, pi

import numpy as np
from numba import njit, prange
from shabanipy.quantum_hall.wal.universal.find_trace import compute_traces


def MC(x: float, L_phi: float, T:np.ndarray, S: np.ndarray, L: np.ndarray, cosj: np.ndarray) -> float:

    xj = np.exp(- L / L_phi) * 0.5 * T * (1 + cosj)
    a = xj * np.cos(x * S)

    return np.sum(a) / len(T)

"""
def get_MC(field: np.ndarray, field_tr: float, L_phi: float, S: np.ndarray, L: np.ndarray, cosj: np.ndarray,
           index: np.ndarray, l, c_phi: np.ndarray, c_3phi: np.ndarray, s_phi: np.ndarray, s_3phi: np.ndarray,
           alpha: float, beta1: float, beta3: float, k: float, hvf: float, N_orbit: int) -> np.ndarray:
    x = field / field_tr
    T = compute_traces(index, l, c_phi, c_3phi, s_phi, s_3phi, alpha, beta1, beta3, k, hvf, N_orbit)
    y = np.empty(len(x))
    for i in range(0, len(x)):
        y[i] = MC(x[i], L_phi, T, S, L, cosj)
    return y
"""
