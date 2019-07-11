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

import h5py
from .create_data_file import get_data, get_trace_data


def MC(x: float, L_phi: float, alpha: float, beta1: float, beta3: float, N_orbit: int, k: float, hvf: float) -> float:

    T, angle = get_trace_data(alpha, beta1, beta3, N_orbit, k, hvf)
    f = get_data("data_for_MC_cal")
    S = f["Surface"][:len(T)]
    L = f["Length"][:len(T)]
    cosj = np.cos(angle)

    xj = np.exp(- L / L_phi) * 0.5 * T * (1 + cosj)
    a = xj * np.cos(x * S)

    return np.sum(a) / len(T)
