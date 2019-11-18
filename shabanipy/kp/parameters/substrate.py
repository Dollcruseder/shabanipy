# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2017-2018 by Mertel Authors, see AUTHORS for more details.
#
# -----------------------------------------------------------------------------
"""Structures used to store the parameters of the quantum well.

"""
import numpy as np
from numba import jitclass
from numba.types import bool_

from ..types import nb_float, np_float


@jitclass([('lattice_constant', nb_float),
           ('use_strain', bool_),
           ('strain', nb_float)])
class StrainCalculator:
    """Class computing the strain induced by the lattice.

    """
    def __init__(self, lattice_constant, use_strain, strain):
        self.lattice_constant = lattice_constant
        self.use_strain = use_strain
        self.strain = strain

    def compute_strain(self, lattice_constant):
        """Compute the strain for a given lattice constant.

        If the substrate has been initialized with a non-default strain, this
        value is used instead.

        Parameters
        ----------
        lattice_constant: np.ndarray
            Lattice constant in the different discretized layers of the well.

        """
        if self.use_strain:
            return self.strain*np.ones_like(lattice_constant).astype(np_float)
        return ((self.lattice_constant - lattice_constant) /
                lattice_constant).astype(np_float)


class SubstrateParameters:
    """Class describing the substrate on which the quantum well is grown.

    Parameters
    ----------
    composition : dict[str, float]
        Composition of the layer expressed as fraction of the materials used
        in the layer.

    lattice_constant : float
        Lattice constant of the substrate.

    strain : float, optional
        Strain imposed on the quantum. Specifying this value will bypass the
        usual strain computation based on lattice constant mismatch.

    """
    def __init__(self, composition, lattice_constant, strain=None):
        self.composition = composition
        self.lattice_constant = lattice_constant
        self.strain = strain

    def create_strain_calculator(self):
        """Create a jitted strain calculator.

        The calculator relies either on the lattice constant of the substrate
        or on a provided pre-computed value .

        """
        return StrainCalculator(self.lattice_constant, self.strain is not None,
                                self.strain or 0.0)
