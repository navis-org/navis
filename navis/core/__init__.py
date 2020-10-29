from .volumes import Volume
from .neurons import Neuron, BaseNeuron, TreeNeuron, MeshNeuron, Dotprops
from .neuronlist import NeuronList
from .core_utils import make_dotprops

from typing import Union

NeuronObject = Union[NeuronList, TreeNeuron, BaseNeuron, MeshNeuron]
