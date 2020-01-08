#: Range (min, max, point_number) in which to vary the wave-vector (nm^-1)
RANGE_K = (-0.8, 0.8, 401)

#: Orthogonal axis wave-vector
FIXED_K = 0

#: Number of bands to consider in the calculation
BAND_NUMBER = 8

#: Interface thickness (nm)
INTERFACE_THICKNESS = 0.2

#: Discretization step (nm)
DISCRETIZATION_STEP = 0.4

import numpy as np
from shabanipy.kp.parameters import LayerParameters, WellParameters, MaterialParameters, SubstrateParameters, load_substrate_parameters
from shabanipy.kp.parameters.materials import MATERIAL_PARAMETERS, make_alloy, load_material_parameters
from shabanipy.kp.parameters.hamiltonian import HamiltonianParameters1D
from shabanipy.kp import find_conduction_band
from shabanipy.kp.hamiltonian_one_side_difference import build_hamiltonian
# from shabanipy.kp.hamiltonian import build_hamiltonian
from shabanipy.kp.types import np_float
import matplotlib.pyplot as plt
import math


def compute_spectrum(kx, ky, parameters, band_number):

    energies = np.empty((len(kx), parameters.site_number*band_number))
    # energies = np.empty((len(kx), 8))

    for i in range(len(kx)):
        h = build_hamiltonian(kx[i], ky, parameters, band_number)
        # eigvalues = np.linalg.eigvalsh(h[:8,:8])
        eigvalues = np.linalg.eigvalsh(h)
        energies[i] = np.sort(eigvalues)

    return energies

kx = np.linspace(*RANGE_K)
ky = FIXED_K

layer_1 = LayerParameters(5, ['InAs', 'GaAs'], [0.81, 0.19])

layer_2 = LayerParameters(4, ['InAs'], [1.0])

layer_3 = LayerParameters(4, ['InAs', 'GaAs'], [0.81, 0.19])

layer_4 = LayerParameters(20, ['InAs', 'AlAs'], [0.81, 0.19])

layers = [layer_1, layer_2, layer_3, layer_4]

# layer_1 = LayerParameters(4, ['hgte', 'cdte'], [0.32, 0.68])
#
# layer_2 = LayerParameters(8, ['hgte'], [1.0])
#
# layer_3 = LayerParameters(4, ['hgte', 'cdte'], [0.32, 0.68])
#
# layers = [layer_1, layer_2, layer_3]

# layers = [LayerParameters(8, ['InAs'], [1.0])]


substrate = load_substrate_parameters("InAs")
# substrate = load_substrate_parameters("cdte")


interface = INTERFACE_THICKNESS
step = DISCRETIZATION_STEP
well = WellParameters(layers, substrate, True, interface)


h_par = well.generate_hamiltonian_parameters(step)
# site_number = 2
# discretization_step = 0.1
# parameters = np.empty((site_number, len(MATERIAL_PARAMETERS)), dtype=np_float)
# parameters[0] = m1.as_tuple
# parameters[1] = m2.as_tuple
# h_par = HamiltonianParameters1D(site_number, np_float(discretization_step),
#                                 DiscretizedMaterialParameters1D(*parameters.T),)

energies = compute_spectrum(kx, ky, h_par, BAND_NUMBER)

energies_k0 = energies[20]
band_index = 8 * int(math.floor(np.argmin(np.abs(energies_k0))/8))
fig, ax = plt.subplots(constrained_layout=True)
ax.set(xlabel=r'$k_x$', ylabel='energy(eV)')

# for i in range(16):
#     ax.plot(kx, energies.T[band_index-i])
#     ax.plot(kx, energies.T[band_index+i])
for e in energies.T:
    ax.plot(kx, e)

#plt.legend()
plt.show()
