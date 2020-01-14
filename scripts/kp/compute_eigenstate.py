#: Orthogonal axis wave-vector
FIXED_K_X = 0.05
FIXED_K_Y = 0

#: Number of bands to consider in the calculation
BAND_NUMBER = 4

#: Interface thickness (nm)
INTERFACE_THICKNESS = 0.005

#: Discretization step (nm)
DISCRETIZATION_STEP = 0.1

ONE_SIDE_DIFFERENCE = True

import numpy as np
from shabanipy.kp.parameters import LayerParameters, WellParameters, MaterialParameters, SubstrateParameters, load_substrate_parameters
from shabanipy.kp.parameters.materials import MATERIAL_PARAMETERS, make_alloy, load_material_parameters
from shabanipy.kp.parameters.hamiltonian import HamiltonianParameters1D
from shabanipy.kp import find_conduction_band
from shabanipy.kp.hamiltonian_one_side_difference import build_hamiltonian
from shabanipy.kp.hamiltonian_three_bands import build_hamiltonian_three
from shabanipy.kp.hamiltonian import build_hamiltonian as build_hamiltonian_old
from shabanipy.kp.types import np_float
import matplotlib.pyplot as plt
import math

def compute_spectrum(kx, ky, parameters, band_number, old=False):

    energies = np.empty(parameters.site_number*band_number)
    # energies = np.empty((len(kx), 8))


    if old:
        h = build_hamiltonian_old(kx, ky, parameters, band_number)
    else:
        h = build_hamiltonian(kx, ky, parameters, band_number)
        # eigvalues = np.linalg.eigvalsh(h[:8,:8])
    eigvalues, v = np.linalg.eig(h)
    index_s = np.argsort(eigvalues)
    # print("index")
    # print(index_s)

    energy = eigvalues[index_s]
    vector = (v.T[index_s]).T
    # print("value")
    # print(energy)
    # print(eigvalues)

    return energy, vector

# def compute_spectrum(kx, ky, parameters, band_number, old=False):
#
#     energies = np.empty(parameters.site_number*band_number)
#     # energies = np.empty((len(kx), 8))
#
#
#     h = build_hamiltonian_three(kx, ky, parameters, band_number)
#         # eigvalues = np.linalg.eigvalsh(h[:8,:8])
#     eigvalues, v = np.linalg.eig(h)
#
#     # print(eigvalues)
#     return eigvalues, v

# kx = np.linspace(*RANGE_K)
kx = FIXED_K_X
ky = FIXED_K_Y

# layer_1 = LayerParameters(5, ['InAs', 'GaAs'], [0.81, 0.19])
#
# layer_2 = LayerParameters(4, ['InAs'], [1.0])
#
# layer_3 = LayerParameters(4, ['InAs', 'GaAs'], [0.81, 0.19])
#
# layer_4 = LayerParameters(20, ['InAs', 'AlAs'], [0.81, 0.19])
#
# layers = [layer_1, layer_2, layer_3, layer_4]

layer_1 = LayerParameters(15, ['hgte', 'cdte'], [0.32, 0.68])

layer_2 = LayerParameters(4, ['hgte'], [1.0])

layer_3 = LayerParameters(15, ['hgte', 'cdte'], [0.32, 0.68])

layers = [layer_1, layer_2, layer_3]

# layers = [LayerParameters(8, ['InAs'], [1.0])]
# layers = [LayerParameters(8, ['InAs', 'GaAs'], [0.53, 0.47])]
# substrate = load_substrate_parameters("InAs")
substrate = load_substrate_parameters("cdte")

interface = INTERFACE_THICKNESS
step = DISCRETIZATION_STEP

well = WellParameters(layers, substrate, True, interface)

h_par = well.generate_hamiltonian_parameters(step, ONE_SIDE_DIFFERENCE)

# print(h_par.ec - h_par.ev)
# site_number = 2
# discretization_step = 0.1
# parameters = np.empty((site_number, len(MATERIAL_PARAMETERS)), dtype=np_float)
# parameters[0] = m1.as_tuple
# parameters[1] = m2.as_tuple
# h_par = HamiltonianParameters1D(site_number, np_float(discretization_step),
#                                 DiscretizedMaterialParameters1D(*parameters.T),)

energies, v = compute_spectrum(kx, ky, h_par, BAND_NUMBER)
energies_old, v_old = compute_spectrum(kx, ky, h_par, BAND_NUMBER, True)



x = np.linspace(0, len(v.T)/BAND_NUMBER-1, len(v.T)//BAND_NUMBER) * step

m_index = np.argmax(1/energies)
energies[m_index] = -energies[m_index]
m_index_second = np.argmax(1/energies)
energies[m_index] = -energies[m_index]

m_index_old = np.argmax(1/energies_old)
energies_old[m_index_old] = -energies_old[m_index_old]
m_index_old_second = np.argmax(1/energies_old)
energies_old[m_index_old] = -energies_old[m_index_old]
print(m_index, m_index_old, m_index_second, m_index_old_second)

# fig, ax = plt.subplots(constrained_layout=True)
# ax.set(xlabel='position(nm)', ylabel=r'$eigenstate(\psi)$')
# for i in range(BAND_NUMBER):
#
#     ax.plot(x, np.abs(v.T[m_index][i::BAND_NUMBER]), '+', color=f"C{i}", label=f"Component {i}")
#     ax.plot(x, np.abs(v_old.T[m_index_old][i::BAND_NUMBER]), '*', color=f"C{i}", label=f"Component {i}")
#
# # ax.plot(x, np.abs(v.T[np.argmin(np.abs(energies))])**2, label='one side difference')
# # ax.plot(x, np.abs(v_old.T[np.argmin(np.abs(energies_old))])**2, label='old')
# # print(energies[np.argmax(np.abs(energies))])
# # print(energies_old[np.argmin(np.abs(energies_old))])
# plt.legend()
fig2, ax2 = plt.subplots(constrained_layout=True)
ax2.set(xlabel='position(nm)', ylabel=r'$eigenstate(\psi)$')

ax2.plot(x, np.sqrt(np.abs(v.T[m_index][::BAND_NUMBER])**2 +
                    np.abs(v.T[m_index][1::BAND_NUMBER])**2 +
                    np.abs(v.T[m_index][2::BAND_NUMBER])**2 +
                    np.abs(v.T[m_index][3::BAND_NUMBER])**2), '+', label = "new, first")
ax2.plot(x, np.sqrt(np.abs(v_old.T[m_index_old][::BAND_NUMBER])**2 +
                    np.abs(v_old.T[m_index_old][1::BAND_NUMBER])**2 +
                    np.abs(v_old.T[m_index_old][2::BAND_NUMBER])**2 +
                    np.abs(v_old.T[m_index_old][3::BAND_NUMBER])**2), '*', label = "old, first")

ax2.plot(x, np.sqrt(np.abs(v.T[m_index_second][::BAND_NUMBER])**2 +
                    np.abs(v.T[m_index_second][1::BAND_NUMBER])**2 +
                    np.abs(v.T[m_index_second][2::BAND_NUMBER])**2 +
                    np.abs(v.T[m_index_second][3::BAND_NUMBER])**2), '+', label = "new, second")
ax2.plot(x, np.sqrt(np.abs(v_old.T[m_index_old_second][::BAND_NUMBER])**2 +
                    np.abs(v_old.T[m_index_old_second][1::BAND_NUMBER])**2 +
                    np.abs(v_old.T[m_index_old_second][2::BAND_NUMBER])**2 +
                    np.abs(v_old.T[m_index_old_second][3::BAND_NUMBER])**2), '*', label = "old, second")
plt.legend()
ax2.twinx().plot(x, h_par.ec - h_par.ev)

plt.show()
