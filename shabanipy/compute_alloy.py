import os
import sys
import time
import json
import warnings
from collections import OrderedDict
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit, prange


from kp.parameters import (load_material_parameters,
                               load_substrate_parameters,
                               SubstrateParameters,
                               LayerParameters,
                               WellParameters,
                               AlloyMethod,
                               MaterialParameters)
from kp.parameters.materials import make_alloy
m1 = load_material_parameters("GaAs")
m2 = load_material_parameters("InAs")

fractions = np.empty(2)
fractions[0] = 0.5
fractions[1] = 1 - fractions[0]
bowing_parameter = 0.5

method = getattr(AlloyMethod, "TERNARY_ALLOY")
temperature = 300
alloy = make_alloy(method, fractions, [m1, m2], bowing_parameter, temperature, True)
print(alloy)
