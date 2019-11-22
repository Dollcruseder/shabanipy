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

from ..types import np_float
from .hamiltonian import HamiltonianParameters1D, HamiltonianParameters2D
from .materials import (MATERIAL_PARAMETERS, AlloyMethod,
                        DiscretizedMaterialParameters1D,
                        DiscretizedMaterialParameters2D,
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
                        discretization_step, interface_lengths):
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
        Materials names of the two compounds

    composition : dict[str, float]
        Composition of the layer expressed as fraction of the materials used
        in the layer.

    """
    def __init__(self, thickness, materials, composition):
        self.thickness = thickness
        self.materials = {m: load_material_parameters for m in materials}
        self.bowing = load_bowing_parameters(materials)
        self.composition = composition


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

        self.get_and_validate_materials()

    @property
    def physical_dimensions(self):
        """Physical dimension of the system in nm.

        """
        return {'z': compute_thickness(self.layers)}

    def generate_hamiltonian_parameters(self, discretization_step,
                                        alloy_method, materials_order,
                                        bowing_parameter, temperature=0):
        """Compute the material parameters on the dicrete lattice.

        Parameters
        ----------
        discretization_step: float
            Step used to discretize the continuous Hamiltonian. Expressed in
            nm.

        alloy_method: kp.parameters.AlloyMethod
            Enum value indicating the method to use to compute the parameters
            of an alloy.

        bowing_parameter:

        materials_order: list
            The material order if the layer

        temperature: float
            Temperature at which to compute the Hamiltonian parameters.


        Returns
        -------
        parameters: kp.parameters.hamiltonian.HamiltonianParameters1D
            Discretized parameters that can be used to build a Hamiltonian

        """
        materials = self.get_and_validate_materials()

        # Computed values
        total_thickness, site_number, interface_indexes =\
            compute_stack_thickness_and_interfaces(self.layers,
                                                   discretization_step)

        c_profiles = \
            self.compute_concentration_profiles(materials, discretization_step,
                                                site_number, interface_indexes)

        parameters = np.empty((site_number, len(MATERIAL_PARAMETERS)),
                              dtype=np_float)

        m_pars = tuple([materials[m] for m in materials_order])
        for i in range(site_number):
            parameters[i] = make_alloy(
                alloy_method, [c_profiles[m][i] for m in materials_order],
                m_pars, bowing_parameter[i], temperature)

        return HamiltonianParameters1D(
                site_number, discretization_step,
                DiscretizedMaterialParameters1D(*parameters.T),
                self.substrate.create_strain_calculator())

    def get_and_validate_materials(self):
        """Get all used materials and check they always have the same params.

        Returns
        -------
        materials : dict
            Dictionary of the materials parameters.

        """
        # Aggregate layers parameters
        materials = dict()
        for layer in self.layers:
            materials.update(layer.materials)

        # Check we got the same materials parameters in all layers
        for layer in self.layers:
            for m_name, m_pars in materials.items():
                if m_name in layer.materials:
                    l_pars = layer.materials[m_name].as_tuple()
                    if not l_pars == m_pars.as_tuple():
                        msg = 'Incoherent parameters in the stack for %s'
                        raise ValueError(msg % m_name)

        return materials

    def compute_concentration_profiles(self, materials, discretization_step,
                                       site_number, interface_indexes):
        """Compute the concentration profile of the different materials.

        Parameters
        ----------
        materials : dict[MaterialParameters]
            Dictionary containing the parameters to use for each materials.
        discretization_step : float
            Step of discretization used in the lattice model.
        site_number : int
            Number of sites along the discretization axis considered.
        interface_indexes : tuple[int]
            Indexes of the of the interfaces between the layers.

        Returns
        -------
        concentration_profiles : dict
            Dict mapping the material names to their profile of concentration.

        """
        layer_weights = build_layer_weights(site_number,
                                            interface_indexes,
                                            discretization_step,
                                            self.interface_lengths)

        c_profiles = {}
        for m in materials:
            profile = sum([layer_weights[i]*layer.composition.get(m, 0)
                           for i, layer in enumerate(self.layers)])
            c_profiles[m] = profile
        return c_profiles


class WallParameters:
    """Class describing a wall on the side of a quantum well..

    Parameters
    ----------
    thickness : float
        Thickness of the wall expressed in nm.

    materials : dict[str, kp.parameters.MaterialParameters]
        Mapping between materials names and the associated parameters.

    composition : dict[str, float]
        Composition of the layer expressed as fraction of the materials used
        in the layer.

    """
    def __init__(self, thickness, materials, composition):
        self.thickness = thickness
        self.materials = materials
        self.composition = composition


# TODO add a way by which to modulate the well parameters, maybe through making
# it callable and passing a way to compute some parameters function of x.
class QWSlabParameters:
    """Container describing a slab of a quantum well in between walls.

    The quantum wells is a grown along a given axis and the walls give a finite
    size to teh system along an orthogonal axis. This is hence parametrizing
    a 2D or 3D system depending on the number of walls.

    Parameters
    ----------
    well_parameters : WellParameters
        Parameters of the quantum well.

    well_dimension : float
        Dimensions of the well in between the walls along the axis defined by
        the `axis` parameter.

    walls : tuple[kp.parametWellParametersers.WallParameters]
        List of parameters for the walls. We expect a tuple which can contain
        either a WallParameters instance or None indicating that there is no
        wall on this side. The walls induce a quantization along the direction
        defined by the `axis` parameter.

    wall_interfaces : tuple[float]
        Distances over which to smooth the interfaces. The order has the same
        meaning it has in `walls`. 0 can be used to indicate perfectly sharp
        interfaces.

    axis : float
        Angle between the natural 'x' axis of teh model and the x axis to use
        to express the Hamiltonian

    """
    def __init__(self, well_parameters, well_dimension, walls,
                 wall_interfaces, axis):
        # TODO check that walls and wall_interfaces match
        self.well_parameters = well_parameters
        self.well_dimension = well_dimension
        self.walls = walls
        self.wall_interfaces = tuple(wi if wi != 0 else 1e-9
                                     for wi in filter(None, wall_interfaces))
        self.axis = axis

    @property
    def physical_dimensions(self):
        """Dimensions of the system in nm.

        """
        dims = self.well_parameters.physical_dimensions
        dims['x'] = compute_thickness(self.walls) + self.well_dimension
        return dims

    def get_and_validate_materials(self):
        """Get all used materials and check they always have the same params.

        Returns
        -------
        materials : dict
            Dictionary of the materials parameters.

        """
        qw_materials = self.well_parameters.get_and_validate_materials()

        # Aggregate walls parameters
        wall_materials = dict()
        for wall in chain(self.walls):
            if wall is None:
                continue
            wall_materials.update(wall.materials)

        # Check we got the same materials parameters in all layers
        for wall in chain(self.walls):
            if wall is None:
                continue
            for m_name, m_pars in wall_materials.items():
                if m_name in wall.materials:
                    w_pars = wall.materials[m_name].as_tuple()
                    if not w_pars == m_pars.as_tuple():
                        msg = 'Incoherent parameters in the walls for %s'
                        raise ValueError(msg % m_name)

        for wm, wm_pars in wall_materials.items():
            if wm in qw_materials:
                qw_pars = qw_materials[wm].as_tuple()
                if not qw_pars == wm_pars.as_tuple():
                    msg = ('Incoherent parameters between walls and '
                           'quantum well for %s')
                    raise ValueError(msg % m_name)

        return qw_materials, wall_materials

    def generate_hamiltonian_parameters(self, discretization_steps,
                                        alloy_methods, temperature=0):
        """Compute the material parameters on the dicrete lattice.

        Parameters
        ----------
        discretization_steps: tuple[float]
            Step used to discretize the continuous Hamiltonian in each
            direction (x and then z). Expressed in nm.

        alloy_methods: tuple[kp.parameters.AlloyMethod]
            Enum value indicating the method to use to compute the parameters
            of an alloy. The first value is used for computing the parameters
            of the well, the second to connect the well to the walls.

        temperature: float
            Temperature at which to compute the Hamiltonian parameters.

        Returns
        -------
        parameters: kp.parameters.hamiltonian.HamiltonianParameters1D
            Discretized parameters that can be used to build a Hamiltonian

        """
        slab_step, qw_step = discretization_steps

        # Get the materials
        qw_materials, wall_materials = self.get_and_validate_materials()
        materials = qw_materials.copy()
        materials.update(wall_materials)

        # Compute QW values
        qw_thickness, qw_site_number, qw_interface_indexes =\
            compute_stack_thickness_and_interfaces(self.well_parameters.layers,
                                                   qw_step)

        # To compute the slab dimensions we add a fictitious wall representing
        # the QW.
        slabs = self.walls[:]
        slabs.insert(1, WallParameters(self.well_dimension, None, None))
        slab_thickness, slab_site_number, slab_interface_indexes =\
            compute_stack_thickness_and_interfaces(slabs, slab_step)

        c_profiles = self.compute_concentration_profiles(
            qw_materials, qw_step, qw_site_number, qw_interface_indexes,
            wall_materials, slab_step, slab_site_number, slab_interface_indexes
            )

        # HINT for the time being focus on HgCdTe only structure and hence on
        # the case in which the walls are treated on the same footing as the
        # well.
        if alloy_methods[0] != alloy_methods[1]:
            raise NotImplementedError('Using different methods for alloy '
                                      'properties for the well and walls is '
                                      'not yet supported.')

        parameters = np.empty((slab_site_number,
                               qw_site_number,
                               len(MATERIAL_PARAMETERS)),
                              dtype=np_float)
        if alloy_methods[0] != AlloyMethod.LINEAR:
            materials_order = ('HgTe', 'CdTe')
        else:
            materials_order = list(materials[m] for m in materials_order)

        m_pars = tuple([materials[m] for m in materials_order])
        for i in range(slab_site_number):
            for j in range(qw_site_number):
                parameters[i, j] = make_alloy(
                    alloy_methods[0],
                    [c_profiles[m][i, j] for m in materials_order],
                    m_pars, temperature)

        # Reorder the axis so that the first axis allow to select a material
        parameters = np.moveaxis(parameters, 2, 0)

        return HamiltonianParameters2D(
                np.array((slab_site_number, qw_site_number)),
                np.array(discretization_steps),
                DiscretizedMaterialParameters2D(*parameters),
                self.well_parameters.substrate.create_strain_calculator())

    def compute_concentration_profiles(
            self,
            qw_mats, qw_step, qw_site_number, qw_interface_indexes,
            sl_mats, sl_step, sl_site_number, sl_interface_indexes):
        """Compute the concentration profile of the different materials.

        Parameters
        ----------
        qw_mats : dict[MaterialsParameters]
            Materials used in the central quantum well.
        qw_step : float
            Discretization in the direction of the well (z usually)
        qw_site_number : int
            Number of sites in the direction of the well.
        qw_interface_indexes : tuple[int]
            Indexes of the interface position in the quantum well.
        sl_mats : dict[MaterialsParameters]
            Materials used in the 'lateral' slabs.
        sl_step : float
            Discretization in the direction of the slabs (x usually)
        sl_site_number : int
            Number of sites in the direction of the slabs.
        sl_interface_indexes : tuple[int]
            Indexes of the interface position between the slabs.

        Returns
        -------
        concentration_profiles : dict
            Dict mapping the material names to their profile of concentration.

        """
        qw_c_profiles = \
            self.well_parameters.compute_concentration_profiles(
                qw_mats, qw_step, qw_site_number, qw_interface_indexes)

        slab_weights = build_layer_weights(sl_site_number,
                                           sl_interface_indexes,
                                           sl_step,
                                           self.wall_interfaces)
        # Make the weight 2 dimensional to handle the modulations in the well
        slab_weights = [(np.ones((sl_site_number, qw_site_number)).T*sl_w).T
                        for sl_w in slab_weights]

        # Add the quantum well between the walls
        slabs = self.walls[:]
        slabs.insert(1, WallParameters(self.well_dimension, None,
                                       qw_c_profiles))

        # Create concentration for all materials
        c_profiles = {m: np.zeros_like(slab_weights[0])
                      for m in chain(qw_mats, sl_mats)}
        for m in set(chain(qw_mats, sl_mats)):
            for i, layer in enumerate(slabs):
                if layer is None:
                    continue
                c_profiles[m] += slab_weights[i]*layer.composition.get(m, 0)

        return c_profiles
