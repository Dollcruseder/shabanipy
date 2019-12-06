#: Range (min, max, point_number) in which to vary the wave-vector (nm^-1)
RANGE_K = (-0.8, 0.8, 21)

#: Orthogonal axis wave-vector
FIXED_K = 0.1

#: Number of bands to consider in the calculation
BAND_NUMBER = 8

import numpy as np
from shabanipy.kp.parameters import LayerParameters, WellParameters, MaterialParameters, SubstrateParameters, load_substrate_parameters
from shabanipy.kp.parameters.materials import make_alloy, load_material_parameters
from shabanipy.kp import build_hamiltonian, find_conduction_band
import matplotlib.pyplot as plt
import math


def compute_spectrum(kx, ky, parameters, band_number):

    energies = np.empty((len(kx), parameters.site_number*band_number))

    for i in range(len(kx)):
        h = build_hamiltonian(kx[i], ky, parameters, band_number)
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
#m1 = load_material_parameters("AlAs")
#m2 = load_material_parameters("InAs")

#fractions = np.empty(2)
#fractions[0] = 0.19
#fractions[1] = 1 - fractions[0]

#sub = make_alloy(fractions, [m1, m2])
#print(sub.lattice_constant)
#substrate = SubstrateParameters({"AlAs": 0.19, "InAs": 0.81}, sub.lattice_constant)

substrate = load_substrate_parameters("InAlAs")

well = WellParameters(layers, substrate, True)


h_par = well.generate_hamiltonian_parameters(0.1)

energies = compute_spectrum(kx, ky, h_par, BAND_NUMBER)

energies_k0 = energies[0]
band_index = 8 * int(math.floor(np.argmin(np.abs(energies_k0))/8))
print(band_index)
fig, ax = plt.subplots(constrained_layout=True)
ax.set(xlabel=r'$k_x$', ylabel='energy(eV)')
for i in range(40):
    ax.plot(kx, energies.T[band_index-i])
    ax.plot(kx, energies.T[band_index+i])
# for e in energies.T:
#     ax.plot(kx, e)
#plt.legend()
plt.show()
