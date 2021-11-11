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

from .volumes import Volume
from .base import Neuron, BaseNeuron
from .skeleton import TreeNeuron
from .mesh import MeshNeuron
from .dotprop import Dotprops
from .voxel import VoxelNeuron
from .neuronlist import NeuronList
from .core_utils import make_dotprops, to_neuron_space, NeuronProcessor

from typing import Union

NeuronObject = Union[NeuronList, TreeNeuron, BaseNeuron, MeshNeuron]

__all__ = ['Volume', 'Neuron', 'BaseNeuron', 'TreeNeuron', 'MeshNeuron',
           'Dotprops', 'VoxelNeuron', 'NeuronList', 'make_dotprops']
