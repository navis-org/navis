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

from .downsampling import downsample_neuron
from .resampling import resample_skeleton, resample_along_axis
from .points import sample_skeleton

# `sample_cable` / `sample_surface` also live in `.points` but are deliberately
# NOT lifted to the top-level `navis.*` namespace - they are exposed only via
# `navis.ml` (like `sample_points_uniform`), see `navis/ml/__init__.py`.

__all__ = ['downsample_neuron', 'resample_skeleton', 'resample_along_axis',
           'sample_skeleton']
