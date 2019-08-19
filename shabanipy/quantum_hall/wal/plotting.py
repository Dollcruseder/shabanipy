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


def plotting(field, r, reference_field, max_field,
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


    model_obj = Model(get_MC)


    model_obj.set_param_hint("theta_alpha", min=0, value = 1)
    model_obj.set_param_hint("theta_beta1", min=0, value = 0)
    #model_obj.set_param_hint("theta_beta1",value = 0, vary = False)
    model_obj.set_param_hint("theta_beta3", min=0, value = 1)
    model_obj.set_param_hint("L_phi", min=1, value = 35)
    model_obj.set_param_hint("field_tr", value = htr[0], vary = False)


    params = model_obj.make_params()
    #print(params)

    alpha = np.empty(trace_number)
    beta3 = np.empty(trace_number)
    beta1 = np.empty(trace_number)

    results = np.zeros((7, 2, trace_number))
    names = ("theta_alpha", "theta_beta1", "theta_beta3", "L_phi")


    for i in range(trace_number):
    #for i in range(80, 85):

        print(f'Treating WAL trace {i+1}/{trace_number}')

        mask = np.where(np.logical_and(np.greater(field[i], 0),
                                       np.less(field[i], max_field)))
        f, s = field[i][mask], sigma[i][mask]

        ref_ind = np.argmin(np.abs(f - reference_field))
        reference_field = f[ref_ind]
        dsigma = s - s[ref_ind]

        #if i != 34:
            #params = res.params
        params["field_tr"].value = htr[i]

        res = model_obj.fit(dsigma, params, field = f, method='nelder')
        #res = model_obj.fit(dsigma, params, field = f)
        print(res.best_values)

        alpha[i] = res.best_values["theta_alpha"] / (2 * kf[i] * tf[i] / cs.hbar)
        beta3[i] = res.best_values["theta_beta3"] / (2 * (kf[i] ** 3) * tf[i] / cs.hbar)
        beta1[i] = res.best_values["theta_beta1"] / (2 * kf[i] * tf[i] / cs.hbar)
        print("alpha:", alpha[i],"beta1:", beta1[i], "beta3:", beta3[i])


        for j, n in enumerate(names):
            if not n:
                continue
            results[2 * j, 0, i] = res.best_values[n]
            results[2 * j, 1, i] = res.params[n].stderr

        results[1, 0, i] = alpha[i]
        results[1, 1, i] = results[0, 1, i]
        results[3, 0, i] = beta1[i]
        results[3, 1, i] = results[2, 1, i]
        results[5, 0, i] = beta3[i]
        results[5, 1, i] = results[4, 1, i]

        print(results[:,1,i])

        fig, ax = plt.subplots(constrained_layout=True)
        if density is not None:
            fig.suptitle(f'Density {density[i]/1e4:.1e} (cm$^2$)')
        ax.plot(field[i]*1e3, sigma[i] - s[ref_ind], '+')
        ax.plot(np.concatenate((-f[::-1], f))*1e3,
                np.concatenate((res.best_fit[::-1], res.best_fit)))
        ax.set_xlabel('Magnetic field B (mT)')
        ax.set_ylabel(r'Δσ(B) - Δσ(0) ($\frac{e^2}{2\,π\,\hbar})$')
        amp = abs(np.max(s - s[ref_ind]) - np.min(s - s[ref_ind]))
        ax.set_ylim((None, np.max(s - s[ref_ind]) + 0.1*amp))
        ax.set_xlim((-max_field*1e3, max_field*1e3))
        ax.text((max_field*1e3)/4, np.max(s - s[ref_ind]) + 0.1*amp - 0.75,
                rf"$\theta_\alpha={round(res.best_values['theta_alpha'], 3)}$")
        ax.text((max_field*1e3)/4, np.max(s - s[ref_ind]) + 0.1*amp - 0.9,
                r"$\theta_{\beta1}=$"f"{round(res.best_values['theta_beta1'], 3)}")
        ax.text((max_field*1e3)/4, np.max(s - s[ref_ind]) + 0.1*amp - 1.05,
                r"$\theta_{\beta3}="f"{round(res.best_values['theta_beta3'], 3)}$")
        ax.text((max_field*1e3)/4, np.max(s - s[ref_ind]) + 0.1*amp - 1.2,
                rf"$L_\phi={round(res.best_values['L_phi'], 3)}$")
        #if htr is None:
            #ax.set_xlim((-max_field*1e3, max_field*1e3))
        #else:
            # ax.set_xlim((-5*htr[i]*1e3, 5*htr[i]*1e3))
            #ax.set_xlim((-50, 50))
        if htr is not None:
            ax.axvline(htr[i]*1e3, color='k', label='H$_{tr}$')
        ax.legend()

    return results

    """
        fig, ax = plt.subplots(constrained_layout=True)
        if density is not None:
            fig.suptitle(f'Density {density[i]/1e4:.1e} (cm$^2$)')
        ax.plot(field[i]*1e3, sigma[i] - s[ref_ind], '+')
        ax.plot(field[i]*1e3, get_MC(field[i], 0.3, 0, 0.3, 100), '*')
        ax.set_xlabel('Magnetic field B (mT)')
        ax.set_ylabel(r'Δσ(B) - Δσ(0) ($\frac{e^2}{2\,π\,\hbar})$')
        amp = abs(np.max(s - s[ref_ind]) - np.min(s - s[ref_ind]))
        ax.set_ylim((None, np.max(s - s[ref_ind]) + 0.1*amp))
        if htr is None:
            ax.set_xlim((-max_field*1e3, max_field*1e3))
        else:
            ax.set_xlim((-50, 50))
    """
    """
        x = field[i]*1e3
        y = sigma[i] - s[ref_ind]
        x1 = []
        x2 = []
        y1 = []
        y2 = []
        for j in range(len(x)):
            if x[j] < 0:
                x1.append(x[j])
                y1.append(y[j])
            else:
                x2.append(x[j])
                y2.append(y[j])

        y1m = min(y1)
        x1m = x1[y1.index(y1m)]
        y2m = min(y2)
        x2m = x2[y2.index(y2m)]
        ax.plot(x1m, y1m, '*')
        ax.plot(x2m, y2m, '*')
    """
