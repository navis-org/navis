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

from .d import *
from .dd import *
from .ddd import *
from .vispy import *

from .colors import vary_colors

__all__ = ['plot1d', 'plot2d', 'plot3d', 'vary_colors', 'Viewer',
           'get_viewer', 'clear3d', 'close3d', 'screenshot']
