# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2017-2018 by Mertel Authors, see AUTHORS for more details.
#
# -----------------------------------------------------------------------------
"""Routines to build the hamiltonian of the quantum well.

"""
from math import sqrt

import numpy as np
from numba import njit

from .types import np_complex


@njit
def local_hermit(h, i, j):
    """Provided two indices i, j set h[j, i] = h[i, j].conjugate().

    Hopefully this function is short enough to be inlined by LLVM.

    """
    h[j, i] = h[i, j].conjugate()


@njit
def build_hamiltonian(kx, ky, parameters, band_number):
    """Build the hamiltonian for a given material/stack at a given kx, ky.

    Parameters
    ----------
    kx : float
        Value of kx at which to compute the Hamiltonian

    ky : float
        Value of ky at which to compute the Hamiltonian

    parameters : kp.parameters.hamiltonian.HamiltonianParameters1D
        Physical parameters of the system.

    band_number : {6, 4}
        Number of bands to take into account into the calculation.

    """
    # Direct access to structure member for faster accesses
    bn = band_number

    b_arr = parameters.kinetic_prefactor
    ec_arr = parameters.ec
    ev_arr = parameters.ev
    if bn<=6:
        f_arr = parameters.f_6bands
    else :
        f_arr = parameters.f
    p_arr = parameters.p
    gamma1 = parameters.gamma_1
    gamma2 = parameters.gamma_2
    gamma3 = parameters.gamma_3
    gammabar = parameters.gamma_bar
    mu_arr = parameters.mu
    kappa = parameters.kappa
    tbp_arr = parameters.tbp
    ubp_arr = parameters.ubp
    vbp_arr = parameters.vbp
    bia_arr = parameters.bia
    delta_arr = parameters.delta

    # Constants pre-computed for speed
    i_3 = 1/3
    sqrt2 = sqrt(2)
    sqrt3 = sqrt(3)
    sqrt3_2 = sqrt(1.5)
    sqrt2_3 = 1/sqrt(1.5)
    i_sqrt2 = 1/sqrt(2)
    i_sqrt3 = 1/sqrt(3)
    i_sqrt6 = 1/sqrt(6)
    i_sqrt12 = 1/sqrt(12)
    i_step = 1/parameters.step_size

    h = np.zeros((band_number * parameters.site_number,
                  band_number * parameters.site_number),
                 dtype=np_complex)

    # This code follows the FORTRAN 90 implementation by D. Carpentier,
    # but incorporate some fixes from V. Jouffrey Python implementation. One
    # key difference between both implementation is the labelling of the
    # bands:
    #  - the FORTRAN code: gamma6 +1/2, -1/2, gamma8 +3/2, +1/2, -1/2, -3/2
    #  - V. Jouffrey code: gamma6 +1/2, -1/2, gamma8 +1/2, -1/2, +3/2, -3/2
    # In order to be able to switch between 4 and 6 bands implementation we
    # use Victor indexing.
    # In addition, following Pfeuffer and Novik, we add an eight bands model
    # including the gamma7 band (+1/2 and -1/2) added as last columns

    # Set the main block diagonal
    for i in range(parameters.site_number):

        # Access to the parameters of the layer we are considering
        b = b_arr[i]
        p = p_arr[i]
        g1 = gamma1[i]
        g2 = gamma2[i]
        gb = gammabar[i]
        mu = mu_arr[i]
        bia = bia_arr[i]
        delta = delta_arr[i]

        im = i-1 if i > 0 else 0

        u0 = (ev_arr[i] - b*(kx**2 + ky**2)*g1 -
              b*i_step**2*(gamma1[im] + gamma1[i]) + ubp_arr[i])
        v0 = (-b*(kx**2 + ky**2)*g2 + 2*b*i_step**2*(gamma2[im] + gamma2[i]) +
              vbp_arr[i])

        # Index of the beginning of the layer in the hamiltonian
        ind = band_number*i

        # Energy for the electronic bands
        e_diag = (ec_arr[i] + b*(2*f_arr[i] + 1)*(kx**2 + ky**2) +
                  b*(2*f_arr[i] + 2*f_arr[im] + 2.0)*i_step**2 + tbp_arr[i])

        # First line: gamma 6 +1/2
        h[ind, ind] = e_diag
        h[ind, ind+3] = i_sqrt6*p*(kx - 1j*ky)
        # Symmetrize
        local_hermit(h, ind, ind+3)

        h[ind, ind+2] = 1j * i_step * sqrt2_3 * p
        local_hermit(h, ind, ind+2)

        # Second line: gamma 6 -1/2
        h[ind+1, ind+1] = e_diag
        h[ind+1, ind+2] = - i_sqrt6*p*(kx + 1j*ky)
        # Symmetrize
        local_hermit(h, ind+1, ind+2)

        h[ind+1, ind+3] = 1j * i_step * sqrt2_3 * p
        local_hermit(h, ind+1, ind+3)

        # Energy of the spin 1/2 holes
        h12_diag = u0 - v0

        # Third line: gamma 8 +1/2
        h[ind+2, ind+2] = h12_diag
        h[ind+2, ind+3] = sqrt3*0.5*bia*(kx + 1j*ky)  # BIA term
        # Symmetrize
        local_hermit(h, ind+2, ind+3)  # BIA term

        # Fourth line: gamma 8 -1/2
        h[ind+3, ind+3] = h12_diag

        if bn >= 6:

            # Additional constants used in 6 bands model
            R = - b*sqrt3*((mu - gb)*(kx**2 - ky**2) + 2j*kx*ky*(mu + gb))
            sb_0_p = - 2*b*sqrt3*(kx + 1j*ky)*1j*i_step*gamma3[i]
            sb_0_m = - 2*b*sqrt3*(kx - 1j*ky)*1j*i_step*gamma3[i]
            st_0_p = - 2*b*sqrt3*(kx + 1j*ky)*1j*i_step*gamma3[i]
            st_0_m = - 2*b*sqrt3*(kx - 1j*ky)*1j*i_step*gamma3[i]
            # Additional terms needed in the 6 bands model on the previously
            # filled lines
            h[ind, ind+4] = -i_sqrt2*p*(kx + 1j*ky)
            h[ind+1, ind+5] = i_sqrt2*p*(kx - 1j*ky)
            local_hermit(h, ind, ind+4)
            local_hermit(h, ind+1, ind+5)

            h[ind+2, ind+4] = -0.5*bia*(kx - 1j*ky) - sb_0_m.conjugate()
            h[ind+2, ind+5] = R
            h[ind+3, ind+4] = R.conjugate()
            h[ind+3, ind+5] = -0.5*bia*(kx + 1j*ky) + sb_0_p.conjugate()
            local_hermit(h, ind+2, ind+4)
            local_hermit(h, ind+2, ind+5)
            local_hermit(h, ind+3, ind+4)
            local_hermit(h, ind+3, ind+5)

            # Energy for hole 3/2
            h32_diag = u0 + v0

            # Fifth line: gamma 8 +3/2
            h[ind+4, ind+4] = h32_diag
            h[ind+4, ind+5] = -0.5*sqrt3*bia*(kx - 1j*ky)
            # Symmetrize
            local_hermit(h, ind+4, ind+5)

            # Sixth line: gamma 8 -3/2
            h[ind+5, ind+5] = h32_diag

            if bn == 8:

                # Additional terms needed in the 8 bands model on the
                # previously filled lines
                h[ind, ind+6] = -1j*i_step*i_sqrt3*p
                h[ind, ind+7] = -i_sqrt3*p*(kx - 1j*ky)
                h[ind+1, ind+6] = -i_sqrt3*p*(kx + 1j*ky)
                h[ind+1, ind+7] = 1j*i_step*i_sqrt3*p
                h[ind+2, ind+6] = sqrt2 * v0
                h[ind+2, ind+7] = -sqrt3_2*st_0_m
                h[ind+3, ind+6] = -sqrt3_2*st_0_p
                h[ind+3, ind+7] = - sqrt2 * v0
                h[ind+4, ind+6] = i_sqrt2*sb_0_m
                h[ind+4, ind+7] = - sqrt2 * R
                h[ind+5, ind+6] = sqrt2 * R.conjugate()
                h[ind+5, ind+7] = sqrt2*sb_0_p
                local_hermit(h, ind, ind+6)
                local_hermit(h, ind, ind+7)
                local_hermit(h, ind+1, ind+6)
                local_hermit(h, ind+1, ind+7)
                local_hermit(h, ind+2, ind+6)
                local_hermit(h, ind+2, ind+7)
                local_hermit(h, ind+3, ind+6)
                local_hermit(h, ind+3, ind+7)
                local_hermit(h, ind+4, ind+6)
                local_hermit(h, ind+4, ind+7)
                local_hermit(h, ind+5, ind+6)
                local_hermit(h, ind+5, ind+7)

                # Sixth line: gamma 7 +1/2
                h[ind+6, ind+6] = u0 - delta

                # Seventh line: gamma 7 -1/2
                h[ind+7, ind+7] = u0 - delta

    # Set the first upper block diagonal
    for i in range(parameters.site_number-1):

        # Index of the beginning of the layer in the hamiltonian
        ind = band_number*i

        # Access to the parameters of the layer we are considering
        b = b_arr[i]
        p = p_arr[i]
        g3 = gamma3[i]
        kap = kappa[i]
        bia = bia_arr[i]

        # Common terms
        t1 = -b*i_step**2*(1 + 2*f_arr[i])
        c = 1j*b*i_step*(kappa[i+1] - kap)*(kx - 1j*ky)
        u1 = b*i_step**2*gamma1[i]
        v1 = - b*i_step**2*2*gamma2[i]

        # First line: gamma 6 +1/2
        h[ind, ind+bn] = t1
        h[ind, ind+bn+2] = -1j*i_step*(p + p_arr[i+1])*i_sqrt6
        local_hermit(h, ind, ind+bn)
        local_hermit(h, ind, ind+bn+2)

        # Second line: gamma 6 -1/2
        h[ind+1, ind+bn+1] = t1
        h[ind+1, ind+bn+3] = -1j*i_step*(p + p_arr[i+1])*i_sqrt6
        local_hermit(h, ind+1, ind+bn+1)
        local_hermit(h, ind+1, ind+bn+3)

        # Third line: gamma 8 +1/2
        h[ind+2, ind+bn+2] = u1 - v1
        h[ind+2, ind+bn+3] = c
        # Symmetrize
        local_hermit(h, ind+2, ind+bn+2)
        local_hermit(h, ind+2, ind+bn+3)

        # Fourth line: gamma 8 -1/2
        h[ind+3, ind+bn+3] = u1 - v1
        # Symmetrize
        local_hermit(h, ind+3, ind+bn+3)

        if bn >= 6:

            # Additional common terms
            # HINT Notice that terms in \bar{S} and \bar{S}^\dag are not
            # complex conjugated
            sb_1_p = -1j*i_step*b*sqrt3*(kx + 1j*ky)*(g3+gamma3[i+1]-kappa[i+1]+kap)
            sb_1_m = -1j*i_step*b*sqrt3*(kx - 1j*ky)*(g3+gamma3[i+1]-kappa[i+1]+kap)
            # Additional common terms
            # HINT Notice that terms in \tilde{S} and \tilde{S}^\dag are
            st_1_p = -1j*i_step*b*sqrt3*(kx + 1j*ky)*(g3+gamma3[i+1]+i_3*kappa[i+1]-i_3*kap)
            st_1_m = -1j*i_step*b*sqrt3*(kx - 1j*ky)*(g3+gamma3[i+1]+i_3*kappa[i+1]-i_3*kap)
            # Additional terms needed in the 6 bands model on the previously
            # filled lines
            h[ind+2, ind+bn+5] = 1j*bia*i_step*0.5
            h[ind+3, ind+bn+4] = - 1j*bia*i_step*0.5
            local_hermit(h, ind+2, ind+bn+5)
            local_hermit(h, ind+3, ind+bn+4)

            # Fifth line: gamma 8 +3/2
            h[ind+4, ind+bn+2] = - sb_1_m  # Match docs and fortran
            h[ind+4, ind+bn+3] = -1j*bia*i_step*0.5
            h[ind+4, ind+bn+4] = u1 + v1
            # Symmetrize
            local_hermit(h, ind+4, ind+bn+2)
            local_hermit(h, ind+4, ind+bn+3)
            local_hermit(h, ind+4, ind+bn+4)

            # Sixth line: gamma 8 -3/2
            h[ind+5, ind+bn+2] = 1j*bia*i_step
            h[ind+5, ind+bn+3] = sb_1_p  # Match docs and fortran
            h[ind+5, ind+bn+5] = u1 + v1
            # Symmetrize
            local_hermit(h, ind+5, ind+bn+2)
            local_hermit(h, ind+5, ind+bn+3)
            local_hermit(h, ind+5, ind+bn+5)

            if bn == 8:

                # Additional common terms
                # HINT Notice that terms in \tilde{S} and \tilde{S}^\dag are


                # Additional terms needed in the 8 bands model on the
                # previously filled lines
                h[ind, ind+bn+6] = 1j*i_step*i_sqrt12*(p + p_arr[i+1])
                h[ind+1, ind+bn+7] = -1j*i_step*i_sqrt12*(p + p_arr[i+1])
                h[ind+2, ind+bn+6] = sqrt2 * v1
                h[ind+2, ind+bn+7] = - sqrt3_2 * st_1_m
                h[ind+3, ind+bn+6] = - sqrt3_2 * st_1_p
                h[ind+3, ind+bn+7] = - sqrt2 * v1
                h[ind+4, ind+bn+6] = i_sqrt2 * sb_1_m
                h[ind+5, ind+bn+7] = i_sqrt2 * sb_1_p
                local_hermit(h, ind, ind+bn+6)
                local_hermit(h, ind+1, ind+bn+7)
                local_hermit(h, ind+2, ind+bn+6)
                local_hermit(h, ind+2, ind+bn+7)
                local_hermit(h, ind+3, ind+bn+6)
                local_hermit(h, ind+3, ind+bn+7)
                local_hermit(h, ind+4, ind+bn+6)
                local_hermit(h, ind+5, ind+bn+7)

                # Sixth line: gamma 7 +1/2
                h[ind+6, ind+bn+2] = sqrt2*v1
                h[ind+6, ind+bn+6] = u1
                h[ind+6, ind+bn+7] = c

                # Seventh line: gamma 7 -1/2
                h[ind+7, ind+bn+3] = - sqrt2*v1
                h[ind+7, ind+bn+7] = u1

                # Symmetrize
                local_hermit(h, ind+6, ind+bn+2)
                local_hermit(h, ind+6, ind+bn+6)
                local_hermit(h, ind+6, ind+bn+7)
                local_hermit(h, ind+7, ind+bn+3)
                local_hermit(h, ind+7, ind+bn+7)

    return h
