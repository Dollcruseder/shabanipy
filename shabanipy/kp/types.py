# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2017-2018 by Mertel Authors, see AUTHORS for more details.
#
# -----------------------------------------------------------------------------
"""Structures used to store the parameters of the bulk material.

"""
import numpy as np
from numba import float32, float64, complex64, complex128

PRECISION = 'single'

if PRECISION == 'single':
    nb_float = float32
    nb_complex = complex64
    np_float = np.float32
    np_complex = np.complex64
else:
    nb_float = float64
    nb_complex = complex128
    np_float = np.float64
    np_complex = np.complex128
