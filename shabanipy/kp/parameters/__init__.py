# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2017-2018 by Mertel Authors, see AUTHORS for more details.
#
# -----------------------------------------------------------------------------
"""Classes used to describe the structure of th studied quantum wells.

"""
import os
import json

from .substrate import SubstrateParameters
from .quantum_well import (LayerParameters, WellParameters,
                           WallParameters, QWSlabParameters)
from .materials import MaterialParameters, AlloyMethod
from ..types import nb_float, np_float

__all__ = ['SubstrateParameters', 'LayerParameters', 'WellParameters',
           'MaterialParameters', 'AlloyMethod', 'WallParameters',
           'QWSlabParameters']


def load_material_parameters(name):
    """Load the parameters for a material based on its name.

    """
    with open(os.path.join(os.path.dirname(__file__),
                           name.lower() + '.mat.json')) as f:
        parameters = json.load(f)

    if parameters["kappa"] == "Unknow":
        parameters["kappa"] = parameters["gamma_3"] + np_float(2/3)*parameters["gamma_2"] - np_float(1/3)*parameters["gamma_1"] - np_float(2/3)
    #Here we calculate the value of kappa if it is unknow

    return MaterialParameters(**parameters)


def load_substrate_parameters(name, strain=None):
    """Load the parameters for a material based on its name.

    """
    with open(os.path.join(os.path.dirname(__file__),
                           name.lower() + '.sub.json')) as f:
        parameters = json.load(f)

    return SubstrateParameters(strain=strain, **parameters)
