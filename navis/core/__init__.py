from .volumes import Volume
from .dotprops import Dotprops
from .neurons import Neuron, TreeNeuron
from .neuronlist import NeuronList

from typing import Union

NeuronObject = Union[NeuronList, TreeNeuron]
