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
from .quantum_well import (LayerParameters, WellParameters)
from .materials import MaterialParameters
from ..types import nb_float, np_float

__all__ = ['SubstrateParameters', 'LayerParameters', 'WellParameters',
           'MaterialParameters']





def load_substrate_parameters(name, strain=None):
    """Load the parameters for a material based on its name.

    """
    with open(os.path.join(os.path.dirname(__file__),
                           name.lower() + '.sub.json')) as f:
        parameters = json.load(f)

    return SubstrateParameters(strain=strain, **parameters)
