"""Extract the soi from low field measurements.

The density and mobility are extracted at the same time.

"""

# =============================================================================
# --- Parameters --------------------------------------------------------------
# =============================================================================

#: Path towards the hdf5 file holding the data
PATH = r'd:/lab/scripts/quantum_hall/JS138_124HB_BM003_004.hdf5'

#: Index or name of the column containing the gate voltage values.
GATE_COLUMN = 1

#: Index or name of the column containing the applied magnetic field.
FIELD_COLUMN = 0

# WARNING Labber uses 2 columns per lock-in value so you should be careful when
# using indexes. The index is always assumed to refer to the first column of
# the lock-in (ie real values)

#: Index or name of the column contaning the longitudinal voltage drop
#: measurement along x.
XX_VOLTAGE_COLUMN = 2

#: Index or name of the column contaning the longitudinal voltage drop
#: measurement along y. This data will only be used its XX counterpart is not
#: provided.
YY_VOLTAGE_COLUMN = None

#: Index or name of the column contaning the transverse voltage drop
#: measurement.
XY_VOLTAGE_COLUMN = 4

#: Component of the measured voltage to use for analysis.
#: Recognized values are 'real', 'imag', 'magnitude'
LOCK_IN_QUANTITY = 'real'

#: Value of the excitation current used by the lock-in amplifier in A.
PROBE_CURRENT = 1e-6

#: Sample geometry used to compute the mobility.
#: Accepted values are 'Van der Pauw', 'Standard Hall bar'
GEOMETRY = 'Standard Hall bar'

#: Magnetic field bounds to use when extracting the density.
FIELD_BOUNDS = (40e-3, 2)

#: Parameters to use to filter the xx and yy data. The first number if the
#: number of points to consider IT MUST BE ODD, the second the order of the
#: polynomial used to smooth the data
FILTER_PARAMS = (31, 3)

#: Should we plot the fit used to extract the density at each gate.
PLOT_DENSITY_FIT = False

#: Method used to symmetrize the wal data.
#: Possible values are: 'average', 'positive', 'negative'
SYM_METHOD = 'average'

#: Reference field to use in the WAL calculation.
WAL_REFERENCE_FIELD = 0

#: Maximal field to consider in WAL fitting procedure
WAL_MAX_FIELD = 90e-3

#: Should we plot the WAL fits.
PLOT_WAL = True

#: Effective mass of the carriers in unit of the electron mass.
EFFECTIVE_MASS = 0.03

# =============================================================================
# --- Execution ---------------------------------------------------------------
# =============================================================================
import os
import sys
sys.setrecursionlimit(1000000)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.constants as cs
from scipy.signal import savgol_filter

from shabanipy.quantum_hall.conversion\
    import (convert_lock_in_meas_to_diff_res, GEOMETRIC_FACTORS,
            htr_from_mobility_density, kf_from_density, mean_free_time_from_mobility,
            fermi_velocity_from_kf)
from shabanipy.quantum_hall.density import extract_density
from shabanipy.quantum_hall.mobility import extract_mobility
from shabanipy.quantum_hall.wal.utils import (flip_field_axis,
                                              recenter_wal_data,
                                              symmetrize_wal_data)
from shabanipy.utils.labber_io import LabberData
from shabanipy.quantum_hall.wal.plotting import plotting
import h5py

plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'pdf.fonttype': 42})


with LabberData(PATH) as data:

    names = data.list_channels()
    shape = data.compute_shape((GATE_COLUMN, FIELD_COLUMN))

    #gate = data.get_data(GATE_COLUMN).reshape(shape).T
    field = data.get_data(FIELD_COLUMN).reshape(shape).T
    res = dict.fromkeys(('xx', 'yy', 'xy'))
    for k in list(res):
        name = globals()[f'{k.upper()}_VOLTAGE_COLUMN']
        if name is None:
            continue
        index = data.name_or_index_to_index(name)
        if LOCK_IN_QUANTITY == 'real':
            val = data.get_data(index)
        elif LOCK_IN_QUANTITY == 'imag':
            val = data.get_data(index+1)
        else:
            val = data.get_data(index)**2 + data.get_data(index+1)**2
        val = val.reshape(shape).T
        res[k] = convert_lock_in_meas_to_diff_res(val, PROBE_CURRENT)

    if res['xx'] is None:
        res['xx'] = res['yy']
    if res['yy'] is None:
        res['yy'] = res['xx']

flip_field_axis(field, res['xx'], res['xy'])
res['original'] = res['xx'].copy()
if FILTER_PARAMS:
    res['xx'] = savgol_filter(res['xx'], *FILTER_PARAMS)

field, _ = recenter_wal_data(field,
                             savgol_filter(res['xx'], *FILTER_PARAMS),
                             0.1, 10)

density = extract_density(field, res['xy'], FIELD_BOUNDS, PLOT_DENSITY_FIT)

mobility, std_mob = extract_mobility(field, res['xx'], res['yy'], density,
                                     GEOMETRIC_FACTORS[GEOMETRY])
density, std_density = density

mass = EFFECTIVE_MASS*cs.electron_mass
htr = htr_from_mobility_density(mobility, density, mass)

kf = kf_from_density(density)
tf = mean_free_time_from_mobility(mobility, mass)

field, res['xx'] = symmetrize_wal_data(field, res['xx'], SYM_METHOD)


results = plotting(field[:], res['xx'][:], WAL_REFERENCE_FIELD, WAL_MAX_FIELD,
                   htr=htr, density=density, kf = kf, tf = tf)
trace_number = results.shape[-1]

f = h5py.File("fitting_results.hdf5", "w")
names = ("theta_alpha", "alpha", "theta_beta1","beta1", "theta_beta3", "beta3", "L_phi")
for i, n in enumerate(names):
    if not n:
        continue
    dset = f.create_dataset(n, (trace_number,), dtype = "float64")
    dset[...] = results[i, 0]
dset = f.create_dataset("density", (len(density),))
dset[...] = density
if PLOT_WAL:
    plt.show()
