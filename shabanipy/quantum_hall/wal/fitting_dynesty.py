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
from shabanipy.quantum_hall.wal.universal.find_trace import compute_traces_theta

import dynesty
from dynesty import plotting as dyplot

def fitting_dynesty(field, r, reference_field, max_field,
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

    number = 1
    # the number of the wal trace 
    # I think 1, 6, 8, 10, 12, 15, 50, 75, 77, 80, 85, 90, these number of the wal trace would make sense.

    print(f'Treating WAL trace {number}/{trace_number}')

    mask = np.where(np.logical_and(np.greater(field[number - 1], 0),
                                   np.less(field[number - 1], max_field)))
    f, s = field[number - 1][mask], sigma[number - 1][mask]

    ref_ind = np.argmin(np.abs(f - reference_field))
        # don't change the reference field
    dsigma = s - s[ref_ind]

    N_orbit = 40000
    n_s = np.arange(3, 5001)
    F = np.sum(1 / (n_s - 2))
    index, l, c_phi, c_3phi, s_phi, s_3phi = get_data("data_for_trace_cal")
    S, L, cosj = get_data_for_MC_cal(N_orbit)

    def get_MC(field, theta_alpha, theta_beta1, theta_beta3, L_phi, field_tr):
        x = field / field_tr
        T = compute_traces_theta(index, l, c_phi, c_3phi, s_phi, s_3phi, theta_alpha, theta_beta1, theta_beta3, N_orbit)
        y0 =  -2 * F * MC(0, L_phi, T, S, L, cosj)
        y = np.empty(len(x))
        for i in range(0, len(x)):
            y[i] = -2 * F * MC(x[i], L_phi, T, S, L, cosj) - y0
        return y

    ndim = 4

    def prior_transform(u):
        v = np.empty(ndim)
        v[0] = u[0] * 3 - 2
        v[1] = u[1] * 3 - 2
        v[2] = u[2] * 2
        v[3] = u[3] - 2
        return v


    def loglike(v):
        logtheta_alpha, logtheta_beta3, logLphi, logsigma = v
        theta_alpha, theta_beta3, L_phi, sigma = (10**logtheta_alpha, 10**logtheta_beta3, 10**logLphi, 10**logsigma)
        yr = get_MC(f, theta_alpha, 0, theta_beta3, L_phi, htr[number - 1])
        residsq = (yr - dsigma)**2 / sigma**2
        loglike = -0.5 * np.sum(residsq + np.log(2 * np.pi * sigma**2))
        return loglike

    sampler = dynesty.NestedSampler(loglike, prior_transform, ndim)

    sampler.run_nested(dlogz=0.01, print_progress=True)
    res = sampler.results

    dyplot.runplot(res)
    plt.tight_layout()
    plt.show()
