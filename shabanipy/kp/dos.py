# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2017-2018 by Mertel Authors, see AUTHORS for more details.
#
# -----------------------------------------------------------------------------
"""Density of state computation from k.p data.

"""
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.signal import savgol_filter
from scipy.ndimage import convolve1d


def compute_dos(kx, ky, dispersion, e_extrema=(), k_refinment=10001,
                e_bin=2e-4, filter_size=11):
    """Compute the density of state of a band.

    Parameters
    ----------
    kx : array
        Values of kx at which the energy has been computed. This should be a
        1d array whose values are sorted in ascending order.

    ky : array
        Values of ky at which the energy has been computed. This should be a
        1d array whose values are sorted in ascending order.

    dispersion : array
        2d array of the evaluated energy. Shape should be (len(kx), len(ky))

    k_refinment : int
        Number of points to use to refine the k grid.

    e_bin : float
        Energy bin to consider when binning the dispersion
        (in unit of dispersion).

    filter_size : int
        Number of points to use when filtering the dos data to eliminate the
        numerical noise from binning.

    Returns
    -------
    energies : array
        Energies at which we compute the dos (ie lower bound of the binning
        intervals)

    dos : array
        Density of state per unit of energy and per unit surface.
        The energy unit is the one from the dispersion, the uniy surface is
        constructed from kx and ky.

    """
    interp = RectBivariateSpline(kx, ky, dispersion)
    kx = np.linspace(kx[0], kx[-1], k_refinment)
    ky = np.linspace(ky[0], ky[-1], k_refinment)
    fine_dispersion = interp(kx, ky, grid=True)

    if not e_extrema:
        e_max = np.max(fine_dispersion)
        e_min = np.min(fine_dispersion)
    else:
        e_min, e_max = e_extrema
    energies = np.linspace(e_min + 0.5*e_bin, e_max - 0.5*e_bin,
                           int(round((e_max - e_min) / e_bin)))
    hist, _ = np.histogram(fine_dispersion, len(energies), (e_min, e_max))

    k_step_x = abs(kx[1] - kx[0])
    k_step_y = abs(ky[1] - ky[0])
    # The factor (2*pi )^2 is chosen to get the expected result for a parabolic
    # band
    dos = hist / e_bin * k_step_x * k_step_y / (2*np.pi)**2
    return energies, savgol_filter(dos, filter_size, filter_size-1)


def gaussian(energy, width):
    """Gaussian of width disorder to be used as convolution kernel.

    """
    return 1/(width*(2*np.pi)**0.5)*np.exp(-0.5*(energy/width)**2)


def derivative_fermi_function(energy, temperature):
    """Derivative of the Fermi function to be used as convolution kernel.

    """
    ratio = energy/temperature
    index = np.where(np.greater(ratio, 100))
    ratio[index] = 100
    exp = np.exp(ratio)
    return 1/temperature*exp/(exp + 1)**2


def average_dos(energies, dos, method='fermi', **kwargs):
    """Average the provided DOS using the appropriate method.

    Parameters
    ----------
    energies : np.array
        Energies at which the DOS has been computed.

    dos : np.array
        Density of state to compute.

    method : {'fermi', 'gaussian'}
        Method used to average the dos. Fermi uses the derivative of the fermi
        function as kernel while gaussian uses a simple gaussian.

    kwargs :
        Additional parameters to pass to compute the kernel.
        fermi :
            - temperature: Temperature in energy unit
        gaussian :
            - width: Disorder strength in energy unit

    Returns
    -------
    np.array
        Averaged dos.

    """
    kernel_fn = (derivative_fermi_function if method == 'fermi' else
                 gaussian)
    width = kwargs['temperature'] if method == 'fermi' else kwargs['width']
    energy_step = energies[1] - energies[0]
    en = np.linspace(-5*width, 5*width, int(round(2*width/energy_step))+1)
    kernel = kernel_fn(en, **kwargs)
    return convolve1d(dos, kernel/kernel.sum())
