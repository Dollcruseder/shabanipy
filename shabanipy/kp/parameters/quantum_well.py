# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2017-2018 by Mertel Authors, see AUTHORS for more details.
#
# -----------------------------------------------------------------------------
"""Structures used to store the parameters of the quantum well.

"""
from collections import Iterable
from itertools import chain

import numpy as np
import math

from ..types import np_float
from .hamiltonian import HamiltonianParameters1D
from .materials import (MATERIAL_PARAMETERS,
                        DiscretizedMaterialParameters1D,
                        load_material_parameters,
                        load_bowing_parameters,
                        make_alloy)


def compute_thickness(layer_like_list):
    """Compute the total thickness of a stack.

    """
    total_thickness = 0
    for layer in layer_like_list:
        if layer is None:
            continue
        total_thickness += layer.thickness
    return total_thickness


def compute_stack_thickness_and_interfaces(layer_like_list,
                                           discretization_step):
    """Compute the total thickness of a stack and the interfaces positions.

    Parameters
    ----------
    layer_like_list : list
        List of LayerParameters like object (need to have a thickness
        attribute).

    discretisation_step : float
        Discretization step in nm used to determine the required number of
        sites.

    Returns
    -------
    total_thickness : float
        Total thickness of the stack in nm.

    site_number : int
        Equivalent number of sites given the discretization step.

    interface_indexes : list
        List of the index of the interfaces between two materials.

    """
    # Computed values
    total_thickness = 0
    interface_indexes = []
    for layer in layer_like_list:
        if layer is None:
            continue
        total_thickness += layer.thickness
        interface_indexes.append(
                int(round(total_thickness/discretization_step)))
    interface_indexes.pop()
    site_number = int(round(total_thickness/discretization_step)) + 1

    return total_thickness, site_number, interface_indexes


def build_layer_weights(site_number, interface_indexes,
                        discretization_step, interface_lengths, one_side_difference=False):
    """Build the weights of each layer (array varying between 0 and 1).

    Parameters
    ----------
    site_number : int
        Number of site along the discretization axis.
    interface_indexes : tuple[int]
        Indexes at which the interfaces are located.
    discretization_step : float
        Discretization step in nm
    interface_lengths : float | tuple[float]
        Smoothness of the interfaces  expressed as the length normalizing the
        argument of tanh (a single value is interpreted as an equal value for
        all interfaces).
    one_side_difference
        whether use one_side_difference

    Returns
    -------
    layer_weights : list[np.array[float]]
        Weights of each layer as a function of the position in the stack. For
        smooth interfaces the weight of one layer is non-zero in the adjacent
        layers.

    """
    # In the absence of any interface simply return a constant profile.
    if not interface_indexes:
        return np.ones(site_number)

    sites = np.linspace(0, site_number-1, site_number)
    if isinstance(interface_lengths, Iterable):
        assert len(interface_indexes) == len(interface_lengths)
        interface_lengths = np.array(interface_lengths)
    else:
        interface_lengths =\
            interface_lengths*np.ones(len(interface_indexes))
    interface_lengths /= discretization_step

    # Build profiles for each interface.

    # The profiles are close to one before the interface and close to zero
    # after. For sharp interfaces use a small value of the interface length.
    #
    if one_side_difference:
        profiles = [(1 - np.tanh((sites - index - 0.5)/length))*0.5
                    for index, length in zip(interface_indexes,
                                             interface_lengths)]
    #the heterojunction must be coincident with the point at n−1 for one side difference

    else:
        profiles = [(1 - np.tanh((sites - index)/length))*0.5
                    for index, length in zip(interface_indexes,
                                             interface_lengths)]



    layer_weights = []
    for i in range(len(profiles) + 1):
        if i == 0:
            layer_weights.append(profiles[0])
        else:
            try:
                layer_weights.append(profiles[i] - profiles[i-1])
            except IndexError:
                layer_weights.append(1-profiles[i-1])

    return layer_weights


class LayerParameters:
    """Class describing one layer of a MBE grown sample.

    Parameters
    ----------
    thickness : float
        Thickness of the layer expressed in nm.

    materials : list[str]
        Materials names of the compounds

    composition : list[float]
        Composition of the layer expressed as fraction of the materials used
        in the layer.

    """
    def __init__(self, thickness, materials, composition):
        self.thickness = thickness
        self.materials = [load_material_parameters(m) for m in materials]
        self.composition = composition
        if len(materials) == 1:
            self.alloy = self.materials[0]
        elif len(materials) == 2:
            self.alloy = make_alloy(self.composition, self.materials)

class WellParameters:
    """Class holding the dimensions and parameters of the quantum well.

    When dicretized the quantum well is indexed from top to bottom, with the
    cap layer hence starting at index 0.

    All physical sizes are expressed in nm.

    Parameters
    ----------
    layers : list[kp.parameters.LayerParameters]
        List of the layers from which the quantum well is built from. Layers
        should be given from top to bottom so usually as:
        [cap_layer, well_layer, bottom_layer]

    substrate : kp.parameters.SubstrateParameters
        Parameters of the substrate which determine in particular the strain of
        the whole structure.

    smooth_interfaces : bool
        Should the interfaces be smoothed over `interface_length` or
        considered to be infinitely  sharp.

    interface_length : float or list, optional
        Distance over which to smooth the interfaces. If a list is passed the
        number of values should match the number of interfaces.

    """
    def __init__(self, layers, substrate, smooth_interfaces,
                 interface_lengths=0.1):
        self.layers = layers
        self.substrate = substrate
        self.smooth_interfaces = smooth_interfaces
        self.interface_lengths = (interface_lengths if smooth_interfaces else
                                  1e-9)


    @property
    def physical_dimensions(self):
        """Physical dimension of the system in nm.

        """
        return {'z': compute_thickness(self.layers)}

    def get_layers_concentration(self, site_number, interface_indexes, discretization_step, one_side_difference=False):
        """Compute the layers concentration for each site point

        Parameters
        ----------
        discretization_step : float
            Step of discretization used in the lattice model.
        site_number : int
            Number of sites along the discretization axis considered.
        interface_indexes : tuple[int]
            Indexes of the of the interfaces between the layers.
        one_side_difference
            whether use one_side_difference

        Returns
        -------
        layer_fractions : list[tuple[float]]

        layer_materials : list[list[kp.parameters.MaterialsParameters]]

        """
        layer_weights = build_layer_weights(site_number,
                                            interface_indexes,
                                            discretization_step,
                                            self.interface_lengths,
                                            one_side_difference)
        layer_materials=[]
        layer_fractions=[]
        for i in range(site_number):
            fractions = []
            materials = []
            if len(self.layers) == 1:
                fractions.append(1.0)
                materials.append(self.layers[0].alloy)
            else:
                for j in range(len(self.layers)):
                    if layer_weights[j][i] > 0.001:
                        fractions.append(layer_weights[j][i])
                        materials.append(self.layers[j].alloy)
            layer_fractions.append(fractions)
            layer_materials.append(materials)
        return layer_fractions, layer_materials

    def generate_hamiltonian_parameters(self, discretization_step,one_side_difference=False,temperature=0):
        """Compute the material parameters on the dicrete lattice.

        Parameters
        ----------
        discretization_step: float
            Step used to discretize the continuous Hamiltonian. Expressed in
            nm.

        one_side_difference
            whether use one_side_difference

        temperature: float
            Temperature at which to compute the Hamiltonian parameters.


        Returns
        -------
        parameters: list[kp.parameters.hamiltonian._HamiltonianParameters]
            Discretized parameters that can be used to build a Hamiltonian

        """

        total_thickness, site_number, interface_indexes =\
            compute_stack_thickness_and_interfaces(self.layers,
                                                   discretization_step)
        layer_fractions, layer_materials =\
            self.get_layers_concentration(site_number, interface_indexes,
                                          discretization_step, one_side_difference)

        parameters = np.empty((site_number, len(MATERIAL_PARAMETERS)),
                              dtype=np_float)
        for i in range(site_number):
            if len(layer_fractions[i]) == 2:
                parameters[i] = make_alloy(layer_fractions[i],
                                           layer_materials[i]).as_tuple()
            else:
                parameters[i] = layer_materials[i][0].as_tuple()

        return HamiltonianParameters1D(
                site_number, np_float(discretization_step),
                DiscretizedMaterialParameters1D(*parameters.T),
                self.substrate.create_strain_calculator())
