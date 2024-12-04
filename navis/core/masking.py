#    This script is part of navis (http://www.github.com/navis-org/navis).
#    Copyright (C) 2018 Philipp Schlegel
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

import numpy as np
import pandas as pd

from .neuronlist import NeuronList
from .skeleton import TreeNeuron
from .dotprop import Dotprops
from .voxel import VoxelNeuron
from .mesh import MeshNeuron


__all__ = ["NeuronMask"]

# mode = "r"\w"?
class NeuronMask:
    """Mask neuron(s) by a specific property.

    Parameters
    ----------
    x :         Neuron/List
                Neuron(s) to mask.
    mask :      str | array | callable | list | dict
                The mask to apply:
                - str: The name of the property to mask by
                - array: boolean mask
                - callable: A function that takes a neuron as input
                and returns a boolean mask
                - list: a list of the above
                - dict: a dictionary mapping neuron IDs to one of the
                above
    copy_data : bool
                Whether to copy the neuron data (e.g. node table for skeletons)
                when masking. Set this to `True` if you know your code will modify
                the masked data and you want to prevent changes to the original.
    reset_neurons : bool
                If True, reset the neurons to their original state after the
                context manager exits. If False, will try to incorporate any
                changes made to the masked neurons. Note that this may not
                work for destructive operations.
    validate_mask : bool
                If True, validate `mask` against the neurons before setting it.
                This is recommended but can come with an overhead (in particular
                if `mask` is a callable).

    Examples
    --------
    >>> import navis
    >>> # Grab a few skeletons
    >>> nl = navis.example_neurons(3)
    >>> # Label axon and dendrites
    >>> _ = navis.split_axon_dendrite(nl, label_only=True)
    >>> # Mask by axon
    >>> with navis.NeuronMask(nl, lambda x: x.nodes.compartment == 'axon'):
    ...    print("Axon cable length:", nl.cable_length * nl[0].units)
    Axon cable length: [363469.75 411147.1875 390231.8125] nanometer
    >>> # Mask by dendrite
    >>> with navis.NeuronMask(nl, lambda x: x.nodes.compartment == 'dendrite'):
    ...    print("Dendrite cable length:", nl.cable_length * nl[0].units)
    Dendrite cable length: [1410770.0 1612187.25 1510453.875] nanometer

    See Also
    --------
    [`navis.BaseNeuron.is_masked`][]
            Check if a neuron is masked. Property exists for all neuron types.
    [`navis.BaseNeuron.mask`][]
            Mask a neuron. Implementation details depend on the neuron type.
    [`navis.BaseNeuron.unmask`][]
            Unmask a neuron. Implementation details depend on the neuron type.

    """

    def __init__(self, x, mask, reset_neurons=True, copy_data=False, validate_mask=True):
        self.neurons = x

        if validate_mask:
            self.mask = mask
        else:
            self._mask = mask

        self.reset = reset_neurons
        self.copy = copy_data

    @property
    def neurons(self):
        return self._neurons

    @neurons.setter
    def neurons(self, value):
        self._neurons = NeuronList(value)

        if any(n.is_masked for n in self._neurons):
            raise MaskingError("At least some neuron(s) are already masked")

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask):
        # Validate the mask
        if isinstance(mask, str):
            for n in self.neurons:
                if isinstance(n, TreeNeuron):
                    if mask not in n.nodes.columns:
                        raise MaskingError(f"Neuron does not have '{mask}' column")
                elif not hasattr(n, mask):
                    raise MaskingError(f"Neuron does not have '{mask}' attribute")
        elif isinstance(mask, (list, np.ndarray, pd.Series)):
            if len(self.neurons) == 1 and len(mask) != 1:
                # If we only have one neuron, we can accept a single mask
                # but we still want to wrap it in a list for consistency
                mask = [np.asarray(mask)]

            if len(mask) != len(self.neurons):
                raise MaskingError("Number of masks does not match number of neurons")

            # Validate each mask
            for m, n in zip(mask, self.neurons):
                validate_mask_length(m, n)
        elif isinstance(mask, dict):
            for n in self.neurons:
                if n.id not in mask:
                    raise MaskingError(f"Neuron {n.id} not in mask dictionary")
                validate_mask_length(mask[n.id], n)
        elif callable(mask):
            # If this is a function, try calling it on the first neuron
            test = mask(self.neurons[0])
            if not isinstance(test, (pd.Series, np.ndarray)) or test.dtype != bool:
                raise MaskingError("Callable mask must return a boolean array")
            validate_mask_length(test, self.neurons[0])
        else:
            raise MaskingError(f"Unexpected mask type: {type(mask)}")

        self._mask = mask

    def __enter__(self):
        for i, n in enumerate(self.neurons):
            if callable(self.mask):
                mask = self.mask(n)
            elif isinstance(self.mask, dict):
                mask = self.mask[n.id]
            elif isinstance(self.mask, str):
                mask = self.mask
            else:
                mask = self.mask[i]

            n.mask(mask, copy=self.copy, inplace=True)

        return self

    def __exit__(self, *args):
        for i, n in enumerate(self.neurons):
            n.unmask(reset=self.reset)


def validate_mask_length(mask, neuron):
    """Validate mask length for a given neuron."""
    if callable(mask):
        mask = mask(neuron)
    elif isinstance(mask, str):
        if isinstance(neuron, TreeNeuron):
            mask = neuron.nodes[mask]
        else:
            mask = getattr(neuron, mask)

    if isinstance(mask, list):
        mask = np.asarray(mask)

    if not isinstance(mask, (np.ndarray, pd.Series)) or mask.dtype != bool:
        raise MaskingError("Mask must be a boolean array")

    if isinstance(neuron, TreeNeuron):
        if len(mask) != len(neuron.nodes):
            raise MaskingError("Mask length does not match number of nodes")
    elif isinstance(neuron, VoxelNeuron):
        if neuron._base_data_type == "grid" and mask.shape != neuron.shape:
            raise MaskingError("Mask shape does not match voxel shape")
        elif len(neuron.voxels) != len(mask):
            raise MaskingError("Mask length does not match number of voxels")
    elif isinstance(neuron, Dotprops):
        if len(mask) != len(neuron.points):
            raise MaskingError("Mask length does not match number of points")
    elif isinstance(neuron, MeshNeuron):
        if len(mask) != len(neuron.vertices):
            raise MaskingError("Mask length does not match number of vertices")


class MaskingError(Exception):
    pass