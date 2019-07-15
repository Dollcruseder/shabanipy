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


def MC(x: float, L_phi: float, T:np.ndarray, S: np.ndarray, L: np.ndarray, cosj: np.ndarray) -> float:

    xj = np.exp(- L / L_phi) * 0.5 * T * (1 + cosj)
    a = xj * np.cos(x * S)

    return np.sum(a) / len(T)
