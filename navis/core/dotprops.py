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

from .. import config

import pandas as pd

__all__ = ['Dotprops']

# Set up logging
logger = config.logger


class Dotprops(pd.DataFrame):
    """ Class to hold dotprops. This is essentially a pandas DataFrame - we
    just use it to tell dotprops from other objects.

    See Also
    --------
    :func:`navis.interfaces.r.dotprops2py`
        Converts R dotprops to :class:`~navis.Dotprops`.

    Notes
    -----
    This class is still in the making but the idea is to write methods for it
    like .plot3d(), .to_X().
    """
