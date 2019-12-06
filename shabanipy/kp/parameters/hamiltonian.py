# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2017-2018 by Mertel Authors, see AUTHORS for more details.
#
# -----------------------------------------------------------------------------
"""Classes used to encapsulate the Hamiltonian parameters .

"""
import numpy as np
import scipy.constants as cs
from numba import jitclass
from numba.types import int64

from ..types import np_float, nb_float


HAMILTONIAN_PARAMETERS = ('kinetic_prefactor', 'p', 'ec', 'ev', 'delta',
                          'f', 'f_6bands', 'gamma_1', 'gamma_2', 'gamma_3', 'kappa',
                          'gamma_bar', 'mu', 'tbp', 'ubp', 'vbp', 'bia')


class _HamiltonianParameters:
    """Class holding the Hamiltonian parameters computed on the lattice.

    The lattice is indexed from the top (0) to the bottom, and from left to
    right in the 2D case.

    """
    def __init__(self, site_number, step_size, discretized_materials,
                 strain_calc):
        self.site_number = site_number
        self.step_size = step_size
        self.p = self.compute_p(discretized_materials)
        self.delta = discretized_materials.spin_orbit_splitting
        self.kinetic_prefactor = \
            self.compute_kinetic_prefactor(discretized_materials)
        self.ec = discretized_materials.eg + discretized_materials.ev
        self.ev = discretized_materials.ev
        self.f, self.f_6bands = self.compute_f(discretized_materials)
        self.gamma_1 = discretized_materials.gamma_1
        self.gamma_2 = discretized_materials.gamma_2
        self.gamma_3 = discretized_materials.gamma_3
        self.kappa = discretized_materials.kappa
        self.gamma_bar = self.compute_gamma_bar(discretized_materials)
        self.mu = self.compute_mu(discretized_materials, False)
        self.tbp, self.ubp, self.vbp =\
            self.compute_strain_effect(discretized_materials, strain_calc)
        self.bia = np.zeros_like(self.ec, dtype=np_float)

    def compute_kinetic_prefactor(self, material):
        """Compute the kinetic prefactor.

        """
        reduced_mass = np_float(13.105)
        return np_float(0.5)/reduced_mass*np.ones_like(material.ep, np_float)

    def compute_p(self, material):
        """Compute P.

        """
        reduced_mass = np_float(13.105)
        return np.sqrt(np_float(0.5)*material.ep/reduced_mass)

    def compute_gamma_bar(self, material):
        """Compute gamma_bar.

        """
        return np_float(0.5)*(material.gamma_2 + material.gamma_3)

    def compute_mu(self, material, axial_approximation):
        """Compute mu.

        """
        if axial_approximation:
            return np_float(0.0)*(self.gamma_3 - self.gamma_2)
        else:
            return np_float(0.5)*(self.gamma_3 - self.gamma_2)

    def compute_f(self, material):
        """Compute F for 8 and 6 bands.
        According to Winlkler, the prefactor should be 1/6, but 6/100 works...
        """
        return (material.f,
                material.f +
                np_float(6/100) * material.ep / (material.eg +
                                                 material.spin_orbit_splitting)
                )

    def compute_strain_effect(self, material, strain_calc):
        """Compute the impact of the strain.

        """
        # HINT we assume to be always in the 001 growth direction and we
        # neglect shear strain (leading to internal electric fields)
        eps = strain_calc.compute_strain(material.lattice_constant)
        eps_zz = - 2*material.elasticity_12/material.elasticity_11*eps
        return ((material.def_c*(2*eps + eps_zz)).astype(np_float),
                (material.def_a*(2*eps + eps_zz)).astype(np_float),
                (material.def_b*(eps - eps_zz)).astype(np_float))


_1d_jitclass = jitclass([('site_number', int64), ('step_size', nb_float)] +
                        [(p, nb_float[:]) for p in HAMILTONIAN_PARAMETERS])

HamiltonianParameters1D = _1d_jitclass(_HamiltonianParameters)

# HINT not sure we would gain much using this.
# _2d_jitclass = jitclass([('site_number', int64[:]),
#                          ('step_size', nb_float[:])] +
#                         [(p, nb_float[:, :]) for p in HAMILTONIAN_PARAMETERS])

#HamiltonianParameters2D = _HamiltonianParameters
