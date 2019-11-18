# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2017-2018 by Mertel Authors, see AUTHORS for more details.
#
# -----------------------------------------------------------------------------
"""Utility functions.

"""
import numpy as np


def find_conduction_band(energies, ks, band_index,
                         argcmp=np.argmin, index_increment=0):
    """Look for the index of the conduction band.

    The search relies on the sign change of the curvature at large k (positive
    for the conduction band and negative for the valence band). This assumes
    that the band structure was computed at large enough k.

    """
    band_extrem_indexes = np.unravel_index(argcmp(energies[..., band_index]),
                                           energies.shape[:2])

    # This is the first band we analyse so first determine if we are above or
    # below the gap
    if index_increment == 0:
        index_k0 = np.argmin(np.abs(ks))
        if (band_extrem_indexes == (index_k0,)*len(band_extrem_indexes)):
            index_increment = -1
            argcmp = np.argmin
        elif np.argmax(np.abs(ks)) in band_extrem_indexes:
            index_increment = 1
            argcmp = np.argmax
        else:
            index_increment = -1
            argcmp = np.argmin

    # This not the first band so we look for a change in large k curvature
    # characterized by the fact the extremum of interest will move to large k.
    # This assumes k is large enough
    else:
        if np.argmax(np.abs(ks)) in band_extrem_indexes:
            if index_increment == -1:
                band_index += 1
            return band_index

    return find_conduction_band(energies, ks,
                                band_index + index_increment,
                                argcmp, index_increment)


def kelvin_to_eV(value):
    """Convert an energy expressed in Kelvin to eV.

    """
    return value*1.38e-23/1.6e-19
