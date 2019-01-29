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

__version__ = "0.0.1"

from . import config

logger = config.logger

from .core import *
from .plotting import *
from .data import *

"""


# Flatten namespace by importing contents of all modules of navis
try:
    from .cluster import *
except Exception as error:
    logger.warning(str(error))
    logger.warning('Error importing pymaid.cluster:\n' + str(error))

try:
    from .morpho import *
except Exception as error:
    logger.warning(str(error))
    logger.warning('Error importing pymaid.morpho:\n' + str(error))



# This needs to be AFTER plotting b/c in plotting vispy is imported first
# and we set the backend!
try:
    from .scene3d import *
except Exception as error:
    logger.warning(str(error))
    logger.warning('Error importing pymaid.scene3d:\n' + str(error))

try:
    from .user_stats import *
except Exception as error:
    logger.warning(str(error))
    logger.warning('Error importing pymaid.user_stats:\n' + str(error))

try:
    from .graph import *
except Exception as error:
    logger.warning(str(error))
    logger.warning('Error importing pymaid.graph:\n' + str(error))

try:
    from .graph_utils import *
except Exception as error:
    logger.warning(str(error))
    logger.warning('Error importing pymaid.graph_utils:\n' + str(error))

try:
    from .resample import *
except Exception as error:
    logger.warning(str(error))
    logger.warning('Error importing pymaid.resample:\n' + str(error))

try:
    from .intersect import *
except Exception as error:
    logger.warning(str(error))
    logger.warning('Error importing pymaid.intersect:\n' + str(error))

try:
    from .connectivity import *
except Exception as error:
    logger.warning(str(error))
    logger.warning('Error importing pymaid.connectivity:\n' + str(error))

try:
    from .utils import *
except Exception as error:
    logger.warning(str(error))
    logger.warning('Error importing pymaid.utils:\n' + str(error))

try:
    from .snapshot import *
except Exception as error:
    logger.warning(str(error))
    logger.warning('Error importing pymaid.snapshot:\n' + str(error))

"""
