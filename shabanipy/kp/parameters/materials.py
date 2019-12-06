# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2017-2018 by Mertel Authors, see AUTHORS for more details.
#
# -----------------------------------------------------------------------------
"""Classes used to describe a material.

"""
import enum
import os
import json
from collections import namedtuple

import numpy as np
from numba import njit, jitclass

from ..types import nb_float, np_float

# #: Names of the material parameters
MATERIAL_PARAMETERS = ('electron_mass', 'ep',
                       'eg', 'ev', 'alpha', 'beta', 'spin_orbit_splitting',
                       'f', 'gamma_1', 'gamma_2', 'gamma_3', 'kappa',
                       'def_a', 'def_b', 'def_c', 'def_d',
                       'lattice_constant', 'dielectric_permitivity',
                       'elasticity_11', 'elasticity_12')


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
    def __init__(self, compounds,
                 electron_mass, ep, eg, ev, alpha, beta, spin_orbit_splitting,
                 f, gamma_1, gamma_2, gamma_3, kappa,
                 def_a, def_b, def_c, def_d,
                 lattice_constant, dielectric_permitivity,
                 elasticity_11, elasticity_12):
        self.compounds = compounds # dict[str: float]
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
                self.eg, self.ev, self.alpha, self.beta,
                self.spin_orbit_splitting,
                self.f, self.gamma_1, self.gamma_2, self.gamma_3, self.kappa,
                self.def_a, self.def_b, self.def_c, self.def_d,
                self.lattice_constant, self.dielectric_permitivity,
                self.elasticity_11, self.elasticity_12)


DiscretizedMaterialParameters1D = namedtuple("DiscretizedMaterialParameters1D",
                                             MATERIAL_PARAMETERS)

EG_INDEX = 2
EV_INDEX = 3
GAMMA1_INDEX = 8
GAMMA2_INDEX = 9
GAMMA3_INDEX = 10
KAPPA_INDEX = 11
LATTICE_CONSTANT_INDEX = -4

_CACHE_MATERIAL = {}
def load_material_parameters(name):
    """Load the parameters for a material based on its name.

    """
    if name not in _CACHE_MATERIAL:
        with open(os.path.join(os.path.dirname(__file__),
                               name.lower() + '.mat.json')) as f:
            parameters = json.load(f)

        if parameters["kappa"] == "Unknow":
            parameters["kappa"] = parameters["gamma_3"] + np_float(2/3)*parameters["gamma_2"] - np_float(1/3)*parameters["gamma_1"] - np_float(2/3)
            # Here we calculate the value of kappa if it is unknow

        _CACHE_MATERIAL[name] = MaterialParameters({name: 1.0}, **parameters)

    return _CACHE_MATERIAL[name]


def get_alloy_name(name1, name2):
    """From the name of the material name get the name of the alloy

    """
    alloy_name = "None"
    if ((name1 == "GaAs") & (name2 == "InAs"))|((name1 == "InAs") & (name2 == "GaAs")):
        alloy_name = "GaInAs"
    if ((name1 == "GaAs") & (name2 == "AlAs"))|((name1 == "AlAs") & (name2 == "GaAs")):
        alloy_name = "AlGaAs"
    if ((name1 == "InAs") & (name2 == "AlAs"))|((name1 == "AlAs") & (name2 == "InAs")):
        alloy_name = "AlInAs"
    return alloy_name


#:
_CACHE_BOWING = {}


def load_bowing_parameters(names):
    """Load the bowing parameters based on its compounds' names

    Parameters
    ----------
    names: list[str]

    """
    name1, name2 = names
    alloy_name = get_alloy_name(name1, name2)
    if alloy_name not in _CACHE_BOWING:
        with open(os.path.join(os.path.dirname(__file__),
                               alloy_name.lower() + '_bowing_parameters.mat.json')) as f:
            parameters = json.load(f)

        _CACHE_BOWING[alloy_name] = MaterialParameters({name1: 0.0, name2: 0.0},
                                                       **parameters)

    return _CACHE_BOWING[alloy_name]


def make_alloy(fractions, materials, temperature=0, pfeuffer=False):
    """Compute the parameters of an alloy using the specified method.

    Parameters
    ----------
    method : AlloyMethod
        Method to use to perform the computation.

    fractions : tuple
        Tuple indicating the fraction of each material in the alloy.

    materials : tuple[kp.parameters.MaterialParameters]
        Tuple containing the parameters

    bowing_parameter: kp.parameters.MaterialParameters
        Bowing parameters accounting for the deviation from a linear interpolation
        between two binaries

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



    m1, m2 = materials
    all_compounds = list(set(m1.compounds) | set(m2.compounds))

    A = np.empty((len(all_compounds), 2))
    for i in range(len(all_compounds)):
        A[i, 0] = m1.compounds[all_compounds[i]] if all_compounds[i] in m1.compounds.keys() else 0
        A[i, 1] = m2.compounds[all_compounds[i]] if all_compounds[i] in m2.compounds.keys() else 0

    compounds_fractions = np.dot(A, fractions)

    # Ternary alloy
    if len(all_compounds) == 2:
        bowing_parameter = load_bowing_parameters(all_compounds)

        m1 = load_material_parameters(all_compounds[0])
        m2 = load_material_parameters(all_compounds[1])

        m1.eg = m1.eg - m1.alpha * temperature**2 / (temperature + m1.beta)
        m2.eg = m2.eg - m2.alpha * temperature**2 / (temperature + m2.beta)

        alloy = {n : compounds_fractions[0]*p1 + compounds_fractions[1]*p2 - compounds_fractions[0] * compounds_fractions[1] * C
                 for n, p1, p2, C in zip(MATERIAL_PARAMETERS, m1.as_tuple(), m2.as_tuple(), bowing_parameter.as_tuple())}
        compounds = {all_compounds[0] : compounds_fractions[0], all_compounds[1] : compounds_fractions[1]}

    # Quaternary alloy
    elif len(all_compounds) == 3:
        m1.eg = m1.eg - m1.alpha * temperature**2 / (temperature + m1.beta)
        m2.eg = m2.eg - m2.alpha * temperature**2 / (temperature + m2.beta)

        alloy = {n : fractions[0]*p1 + fractions[1]*p2
                 for n,p1, p2 in zip(MATERIAL_PARAMETERS, m1.as_tuple(), m2.as_tuple())}
        compounds = {all_compounds[0] : compounds_fractions[0], all_compounds[1] : compounds_fractions[1], all_compounds[2] : compounds_fractions[2]}

    else:
        raise ValueError()

    if pfeuffer:
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
        alloy['gamma_1'] =\
            -1 - 1/(3*kin)*(a_mix + 2*g_mix + 2*h1_mix + 2*h2_mix)
        # XXX Hack to get the same results as Wouter
#        alloy[GAMMA1_INDEX] =\
#            4.1 - 2.8801*fractions[1] + 0.3159*fractions[1]**2 - 0.0658*fractions[1]**3
        # gamma 2
        alloy['gamma_2'] = - 1/(6*kin)*(a_mix + 2*g_mix - h1_mix - h2_mix)
        # gamma 3
        alloy['gamma_3'] = - 1/(6*kin)*(a_mix - g_mix + h1_mix - h2_mix)
        # kappa
        alloy['kappa'] = -1/3 - 1/(6*kin)*(a_mix - g_mix - h1_mix + h2_mix)

    return MaterialParameters(compounds, **alloy)
