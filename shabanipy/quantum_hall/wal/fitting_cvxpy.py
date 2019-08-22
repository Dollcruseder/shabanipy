# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2019 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Weak anti-localisation analysis main fitting routine.

"""
import os
import pickle
import numpy as np
import scipy.constants as cs
import matplotlib.pyplot as plt
from lmfit.model import Model

from ..conversion import (diffusion_constant_from_mobility_density,
                          mean_free_time_from_mobility)
from .wal_no_dresselhaus\
    import compute_wal_conductance_difference as simple_wal
from .wal_full_diagonalization\
    import compute_wal_conductance_difference as full_wal
from .utils import weight_wal_data
from shabanipy.quantum_hall.wal.universal.magnetoconductivity import MC
from shabanipy.quantum_hall.wal.universal.create_data_file import get_data, get_data_for_MC_cal
from shabanipy.quantum_hall.wal.universal.find_trace import compute_traces_theta_cvxpy

import cvxpy as cp

def fitting_cvxpy(field, r ,reference_field, max_field,
                  htr=None, density=None, kf=None, tf=None):

    if len(field.shape) >= 2:
        original_shape = field.shape[:-1]
        trace_number = np.prod(original_shape)
        field = field.reshape((trace_number, -1))
        r = r.reshape((trace_number, -1))
    else:
        trace_number = 1
        field = np.array((field,))
        r = np.array((r,))

    sigma = (1/r) / (cs.e**2/(2*np.pi*cs.Planck))
    mask = np.where(np.logical_and(np.greater(field[0], 0),
                                np.less(field[0], max_field)))
    m = len(field[0][mask])
    print(m)
    N_orbit = 40000
    n_s = np.arange(3, 5001)
    F = np.sum(1 / (n_s - 2))
    index, l, c_phi, c_3phi, s_phi, s_3phi = get_data("data_for_trace_cal")
    S, L, cosj = get_data_for_MC_cal(N_orbit)

    def get_MC(field, theta, field_tr):
        theta_alpha, theta_beta1, theta_beta3, L_phi = theta
        x = field / field_tr
        T = compute_traces_theta_cvxpy(index, l, c_phi, c_3phi, s_phi, s_3phi, theta_alpha, theta_beta1, theta_beta3, N_orbit)
        y0 =  -2 * F * MC(0, L_phi, T, S, L, cosj)
        y = np.empty(len(x))
        for i in range(0, len(x)):
            y[i] = -2 * F * MC(x[i], L_phi, T, S, L, cosj) - y0
        return y

    def loss_function(field, field_tr, theta, dsigma):
        y = get_MC(field, theta, field_tr)
        d = y - dsigma
        loss = np.sum(d ** 2)
        return loss

    theta = cp.Variable((4, 1))
    f_t = cp.Parameter((m, 1))
    f_tr = cp.Parameter()
    dsigma = cp.Parameter((m, 1))
    obj = cp.Minimize(loss_function(f_t, f_tr, theta, dsigma)/m)
    constraints = [theta[0] > 0,
                   theta[1] > 0,
                   theta[2] > 0,
                   theta[3] > 0]
    prob = cp.Problem(obj, constraints)

    #for i in range(trace_number):
    for i in range(0, 1):

        print(f'Treating WAL trace {i+1}/{trace_number}')

        mask = np.where(np.logical_and(np.greater(field[i], 0),
                                       np.less(field[i], max_field)))
        f, s = field[i][mask], sigma[i][mask]

        ref_ind = np.argmin(np.abs(f - reference_field))

        """
        # don't change the reference field
        dsigma.value = s - s[ref_ind]

        f_t.value = f
        f_tr.value = htr[i]

        prob.solve()

        print(theta.value)
        """
