# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2017-2018 by Mertel Authors, see AUTHORS for more details.
#
# -----------------------------------------------------------------------------
"""Classes used to describe a material.

"""
import enum

import numpy as np
from numba import njit, jitclass

from ..types import nb_float, np_float

#: Names of the material parameters
MATERIAL_PARAMETERS = ('electron_mass', 'ep',
                       'eg', 'ev', 'alpha', 'beta', 'spin_orbit_splitting',
                       'f', 'gamma_1', 'gamma_2', 'gamma_3', 'kappa',
                       'def_a', 'def_b', 'def_c', 'def_d',
                       'lattice_constant', 'dielectric_permitivity',
                       'elasticity_11', 'elasticity_12')


@jitclass([(p, nb_float) for p in MATERIAL_PARAMETERS])
class MaterialParameters:
    """Class describing a binary compound

    Parameters
    ----------
    electron_mass : float
        Effective electron mass.

    ep : float
        Kane kinetic energy term in eV.

    eg : float
        Energy band gap in eV, defined as the difference in energy between
        the valence and conduction band.

    ev : float
        Top of the conduction band in eV.

    alpha: float
        adjustable parameter for Eg in eV/K.

    beta : float
        adjustable parameter for Eg in K

    spin_orbit_splitting : float
        Spin orbit splitting energy in eV. This term is responsible for the
        splitting of the gamma 7 sub band.

    f, gamma_1, gamma_2, gamma_3, kappa : float
        Coupling to the far off bands.

    def_a : float
        Hydrostatic deformation potential.

    def_b : float
        Uniaxial deformation potential.

    def_c : float
        Hydrostatic deformation potential.

    def_d : float
        Uniaxial deformation potential.

    lattice_constant : float
        Lattice constant in nm.

    dielectric_permitivity : float
        Relative dielectric permitivity of the material

    elasticity_11, elasticity_12: float
        Elasticity parameters in GPa

    """
    def __init__(self, electron_mass, ep, eg, ev, alpha, beta, spin_orbit_splitting,
                 f, gamma_1, gamma_2, gamma_3, kappa,
                 def_a, def_b, def_c, def_d,
                 lattice_constant, dielectric_permitivity,
                 elasticity_11, elasticity_12):
        self.electron_mass = electron_mass
        self.ep = ep
        self.eg = eg
        self.ev = ev
        self.alpha = alpha
        self.beta = beta
        self.spin_orbit_splitting = spin_orbit_splitting
        self.f = f
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.gamma_3 = gamma_3
        self.kappa = kappa
        self.def_a = def_a
        self.def_b = def_b
        self.def_c = def_c
        self.def_d = def_d
        self.lattice_constant = lattice_constant
        self.dielectric_permitivity = dielectric_permitivity
        self.elasticity_11 = elasticity_11
        self.elasticity_12 = elasticity_12

    def as_tuple(self):
        """Build a tuple from the class attributes.

        """
        return (self.electron_mass, self.ep,
                self.eg, self.ev, self.spin_orbit_splitting,
                self.f, self.gamma_1, self.gamma_2, self.gamma_3, self.kappa,
                self.def_a, self.def_b, self.def_c, self.def_d,
                self.lattice_constant, self.dielectric_permitivity,
                self.elasticity_11, self.elasticity_12)


mat_class = (MaterialParameters.class_type.class_def
             if hasattr(MaterialParameters, 'class_type') else
             MaterialParameters)
DiscretizedMaterialParameters1D =\
    jitclass([(p, nb_float[:]) for p in MATERIAL_PARAMETERS])(mat_class)
DiscretizedMaterialParameters2D =\
    jitclass([(p, nb_float[:, :]) for p in MATERIAL_PARAMETERS])(mat_class)


EG_INDEX = 2
EV_INDEX = 3
GAMMA1_INDEX = 6
GAMMA2_INDEX = 7
GAMMA3_INDEX = 8
KAPPA_INDEX = 9
LATTICE_CONSTANT_INDEX = -4


class AlloyMethod(enum.IntEnum):
    """Enum used a bit flag describing the method used to compute an alloy
    parameters.

    """

    TERNARY_ALLOY = 2**0


    QUATERNARY_ALLOY = 2**1



@njit
def make_alloy(method, fractions, materials, bowing_parameter, temperature, preuffer):
    """Compute the parameters of an alloy using the specified method.

    Parameters
    ----------
    method : AlloyMethod
        Method to use to perform the computation.

    fractions : tuple
        Tuple indicating the fraction of each material in the alloy.

    materials : tuple[kp.parameters.MaterialParameters]
        Tuple containing the parameters

    bowing_parameter: float
        Bowing parameter accounts for the deviation from a linear Interpolation
        between two  binaries

    temperature : float
        Temperature in Kelvin at which we consider the alloy.

    preuffer:
        Whether use preuffer method

    Returns
    -------
    alloy : np.ndarray
        Array of the alloy parameters. Can be used to instantiate a
        MaterialParameters instance.

    """
    if len(fractions) > 2:
        raise RuntimeError('Only binary alloys are supported.')

    # For HgCdTe, m1 is HgTe and m2 is CdTe
    m1, m2 = materials
    alloy = [fractions[0]*p1 + fractions[1]*p2
             for p1, p2 in zip(m1.as_tuple(), m2.as_tuple())]


    Eg1 = m1.eg - m1.alpha * temperature**2 / (temperature + m1.beta)
    Eg2 = m2.eg - m2.alpha * temperature**2 / (temperature + m2.beta)
    if method & AlloyMethod.TERNARY_ALLOY:
        Eg = Eg1 * fractions[0] + Eg2 * fractions[1] - fractions[0] * fractions[1] * bowing_parameter
        alloy[EG_INDEX] = Eg

    if method & AlloyMethod.QUATERNARY_ALLOY:
        Eg = Eg1 * fractions[0] + Eg2 * fractions[1]
        alloy[EG_INDEX] = Eg

    if preuffer:
        kin = np_float(0.5)/materials[0].electron_mass  # hb^2/(2*m_0)

        # Compute the parameters A, G, H1 and H2 of the docs
        system = np.array(((2, 4, 4, 4),
                           (1, 2, -1, -1),
                           (1, -1, 1, -1),
                           (1, -1, -1, 1)), dtype=np_float)

        f_f = np.linalg.solve(system,
                              -np_float(6*kin)*np.array([m1.gamma_1 + 1,
                                                         m1.gamma_2,
                                                         m1.gamma_3,
                                                         m1.kappa + 1/3],
                                                        dtype=np_float))
        f_s = np.linalg.solve(system,
                              -np_float(6*kin)*np.array([m2.gamma_1 + 1,
                                                         m2.gamma_2,
                                                         m2.gamma_3,
                                                         m2.kappa + 1/3],
                                                        dtype=np_float))

        # Use linear interpolation save for h1
        a_mix = fractions[0]*f_f[0] + fractions[1]*f_s[0]
        g_mix = fractions[0]*f_f[1] + fractions[1]*f_s[1]
        h2_mix = fractions[0]*f_f[3] + fractions[1]*f_s[3]
        h1_mix = (f_f[2]*f_s[2] / (fractions[1]*f_f[2] + fractions[0]*f_s[2]))

        # gamma 1
        alloy[GAMMA1_INDEX] =\
            -1 - 1/(3*kin)*(a_mix + 2*g_mix + 2*h1_mix + 2*h2_mix)
        # XXX Hack to get the same results as Wouter
#        alloy[GAMMA1_INDEX] =\
#            4.1 - 2.8801*fractions[1] + 0.3159*fractions[1]**2 - 0.0658*fractions[1]**3
        # gamma 2
        alloy[GAMMA2_INDEX] = - 1/(6*kin)*(a_mix + 2*g_mix - h1_mix - h2_mix)
        # gamma 3
        alloy[GAMMA3_INDEX] = - 1/(6*kin)*(a_mix - g_mix + h1_mix - h2_mix)
        # kappa
        alloy[KAPPA_INDEX] = -1/3 - 1/(6*kin)*(a_mix - g_mix - h1_mix + h2_mix)

    return np.array(alloy, dtype=np_float)
