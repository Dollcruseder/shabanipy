# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2018 by Mertel Authors, see AUTHORS for more details.
#
# -----------------------------------------------------------------------------
"""Charge density computation from k.p data.

"""
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from .dos import average_dos


def _arange(start, stop, step):
    """Fancy arange using np.linspace under the hood.

    """
    n_points = int(round((stop - start)/step)) + 1
    return np.linspace(start, stop, n_points)


def compute_dn(energy, dos, temperature, disorder=None,
               conduction_band_edge=0., high_energy_cutoff=0.15,
               high_resolution=1e-5, low_resolution=1e-3):
    """Compute the charge carriers density variation, dn, which is the integral
    of the DoS through an energy range.

    Parameters
    ----------
    energy: np.ndarray
        1D array with the energies at which the dos has been computed.

    dos : np.ndarray
        1D array with the dos computed from k.p.

    temperature: float
        Temperature in unit of energy on which to average the dos.

    disorder: float, optional
        Strength of Gaussian disorder in unit of energy. This average is
        performed after the temperature average.

    conduction_band_edge : float, optional
        Energy value at which is located the conduction band edge.

    high_energy_cutoff : float, optional
        High energy limit for the computation.

    high_resolution : float, optional
        Resolutuion in energy to use in fast varying dos areas.

    low_resolution : float, optional
        Resolution in energy to use in slowly varying dos areas.

    Returns
    -------
    refined_energies : np.ndarray
        Energies at which the carrier density has been computed.

    refined_dos : np.ndarray
        DOS at the energies at which the carrier density has been computed.

    dn : np.ndarray
        Carrier density.


    """
    # Always average the dos to avoid too sharp variations
    avg_dos = average_dos(energy, dos, method='fermi', temperature=temperature)
    if disorder:
        avg_dos = average_dos(energy, avg_dos,
                              method='gaussian', width=disorder)
    # Spline interpolation of the averaged dos with respect to energy
    int_dos = InterpolatedUnivariateSpline(energy, avg_dos)

    # Finding the valence band edge in energy
    valence_band_edge = energy[np.sort(np.where(np.less(dos, 1e-15))[0])[0]]

    # Infra red cut-off, i.e., the excursion in the VB
    ir_cut_off = valence_band_edge - 2*temperature

    # If the cutoff we chose leads to a too small dos we go deeper in the
    # valence band.
    while int_dos(ir_cut_off) < 0.2:
        ir_cut_off -= temperature

    refined_energies = np.concatenate(
        [_arange(ir_cut_off, valence_band_edge + 2*temperature,
                 high_resolution),
         _arange(valence_band_edge + 2*temperature,
                 conduction_band_edge - 2*temperature, low_resolution),
         _arange(conduction_band_edge - 2*temperature,
                 conduction_band_edge + 2*temperature, high_resolution),
         _arange(conduction_band_edge + 2*temperature, high_energy_cutoff,
                 low_resolution)]
         )

    # Refining the dos grid wrt the refined enrgy grid
    refined_dos = np.array([int_dos(e) for e in refined_energies])

    # Computation of dn (given in /µm^2 unity) wrt the refined enrgy grid
    dn = np.array([int_dos.integral(conduction_band_edge, e)
                   for e in refined_energies])/(1.6*1e-19)*1e-12

    return refined_energies, refined_dos, dn
