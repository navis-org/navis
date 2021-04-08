#    This script is part of navis (http://www.github.com/schlegelp/navis).
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

from .templates import registry, xform_brain, mirror_brain, symmetrize_brain
from .xfm_funcs import xform, mirror
from .base import AliasTransform
from .affine import AffineTransform
from .thinplate import TPStransform
from .h5reg import H5transform
from .cmtk import CMTKtransform
from .moving_least_squares import MovingLeastSquaresTransform

# Make sure that only these functions are avaialable at top level
__all__ = ['xform_brain', 'mirror_brain', 'xform', 'mirror', 'symmetrize_brain']
