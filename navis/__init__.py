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

from .__version__ import __version__, __version_vector__

from .compute import *
from .connectivity import *
from .conversion import *
from .core import *
from .data import *
from .graph import *
from .intersection import *
from .io import *
from .meshes import *
from .morpho import *
from .nbl import *
from .plotting import *
from .sampling import *
from .transforms import *
from .utils import *

# `navis.ml` groups the machine-learning helpers under their own namespace
# (`navis.ml.chunk_neuron`, ...) rather than lifting them to the top level.
# Imported last so its dependencies (graph, sampling, core) are already loaded.
from . import ml

# `navis.TreeNeuron` & co: the pre-2.0 class names, served lazily so that using
# one warns. This is the only namespace that warns - see `navis/_deprecated.py`.
from ._deprecated import deprecated_getattr as _deprecated_getattr

__getattr__ = _deprecated_getattr(__name__)
