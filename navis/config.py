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

import logging
import pint
import os

import matplotlib as mpl

logger = logging.getLogger('navis')


def default_logging():
    """Add a formatted stream handler to the ``navis`` logger.

    Called by default when navis is imported for the first time.
    To prevent this behaviour, set an environment variable:
    ``NAVIS_SKIP_LOG_SETUP=True``.
    """
    logger.setLevel(logging.INFO)
    if len(logger.handlers) == 0:
        sh = logging.StreamHandler()
        sh.setLevel(logging.DEBUG)
        # Create formatter and add it to the handlers
        formatter = logging.Formatter(
            '%(levelname)-5s : %(message)s (%(name)s)')
        sh.setFormatter(formatter)
        logger.addHandler(sh)


def remove_log_handlers():
    """Remove all handlers from the ``navis`` logger.

    It may be preferable to skip navis' default log handler being added in the
    first place.
    Do this by setting an environment variable before the first import:
    ``NAVIS_SKIP_LOG_SETUP=True``.
    """
    logger.handlers.clear()


skip_log_setup = os.environ.get('NAVIS_SKIP_LOG_SETUP', '').lower() == 'true'
if not skip_log_setup:
    default_logging()


def get_logger(name: str):
    if skip_log_setup:
        return logging.getLogger(name)
    return logger


# Default settings for progress bars
pbar_hide = False
pbar_leave = False

# Default settings for caching
warn_caching = True

# Default setting for igraph:
#   If True, will use iGraph if possible
#   If False, will ignore iGraph even if present
# Primarily used for debugging
use_igraph = True

# Default color for neurons
default_color = (.95, .65, .04)

# Unit registry
ureg = pint.UnitRegistry()

# Set to true to prevent Viewer from ever showing
headless = os.environ.get('NAVIS_HEADLESS', 'False').lower() == 'true'
if headless:
    logger.info('Running in headless mode.')
    mpl.use('template')
    pbar_hide = True

# Default connector color palette
default_connector_colors = {
    0: {'name': 'Presynapses',
        'color': (1, 0, 0)},
    1: {'name': 'Postsynapses',
        'color': (0, .75, .75)},
    2: {'name': 'Gap junctions',
        'color': (0, 1, 0)},
    'display': 'lines',  # can also be 'circles'
    'size': 2  # for "circles" only
                        }
# Set some synonyms
default_connector_colors['pre'] = default_connector_colors['Pre'] = default_connector_colors[0]
default_connector_colors['post'] = default_connector_colors['Post'] = default_connector_colors[1]
default_connector_colors['gap'] = default_connector_colors['Gap'] = default_connector_colors[0]
default_connector_colors['gap_junction'] = default_connector_colors['Gap_junction'] = default_connector_colors[0]
default_connector_colors['gap_junctions'] = default_connector_colors['Gap_junctions'] = default_connector_colors[0]


def _type_of_script():
    """Returns context in which navis is run. """
    try:
        ipy_str = str(type(get_ipython()))
        if 'zmqshell' in ipy_str:
            return 'jupyter'
        if 'terminal' in ipy_str:
            return 'ipython'
    except BaseException:
        return 'terminal'


def is_jupyter():
    """Test if navis is run in a Jupyter notebook."""
    return _type_of_script() == 'jupyter'


# Here, we import tqdm and determine whether we use classic notebook tbars
from tqdm.notebook import tqdm as tqdm_notebook
from tqdm.notebook import trange as trange_notebook
from tqdm import tqdm as tqdm_classic
from tqdm import trange as trange_classic

# Keep this because `tqdm_notebook` is only a wrapper (type "function")
tqdm_class = tqdm_classic

if is_jupyter():
    tqdm = tqdm_notebook
    trange = trange_notebook
else:
    tqdm = tqdm_classic
    trange = trange_classic
