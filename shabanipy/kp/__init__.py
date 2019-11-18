# -----------------------------------------------------------------------------
# Copyright 2017-2018 by Mertel Authors, see AUTHORS for more details.
#
# -----------------------------------------------------------------------------
"""Tools to compute the band structure of HgTe quantum wells.

The calculations are based on k.p theory.

"""
from . import parameters
from .hamiltonian import build_hamiltonian
from .dos import compute_dos, average_dos
from .utils import find_conduction_band

__all__ = ['build_hamiltonian', 'parameters', 'compute_dos', 'average_dos',
           'find_conduction_band']
