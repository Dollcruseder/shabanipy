# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2019 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Script to benchmark the calculation of the trace over the trajectories.

"""
from math import pi
import os
os.environ['MKL_NUM_THREADS'] = '1'
import time
import sys
from shabanipy.quantum_hall.wal.universal.create_data_file import get_trace_data

# --- Parameters --------------------------------------------------------------

#: Number of trajectories to consider.
TRAJECTORY_NUMBER = 40000

#: Parameters to use in the calculation: alpha, beta1, beta3, k, hvf
PARAMETERS = (pi, pi, 0, 1, 1)

# --- Imports -----------------------------------------------------------------
N_orbit = TRAJECTORY_NUMBER
alpha, beta1, beta3, k, hvf = PARAMETERS
tic = time.time()
T = get_trace_data(alpha, beta1, beta3, N_orbit, k, hvf)
print('Computation', time.time() - tic)
