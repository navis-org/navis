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
from .skeleton import Skeleton
from .mesh import Mesh
from .dotprop import Dotprops
from .voxel import Voxels
from .neuronlist import NeuronList
from .core_utils import make_dotprops, to_neuron_space, cast_neuron, NeuronProcessor
from .pipeline import Pipeline, PipelineStepError
from .masking import masked

from .skeleton import TreeNeuron  # noqa: F401  pre-2.0 name, see navis/_deprecated.py
from .mesh import MeshNeuron  # noqa: F401
from .voxel import VoxelNeuron  # noqa: F401

from typing import Union

NeuronObject = Union[NeuronList, Skeleton, BaseNeuron, Mesh]

__all__ = ['Volume', 'Neuron', 'BaseNeuron', 'Skeleton', 'Mesh',
           'Dotprops', 'Voxels', 'NeuronList', 'make_dotprops',
           'cast_neuron', 'Pipeline', 'PipelineStepError', 'masked']
