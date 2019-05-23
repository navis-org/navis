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
#
#    You should have received a copy of the GNU General Public License
#    along

import pandas as pd

from .. import config

from .iterables import *

# Set up logging
logger = config.logger


def validate_options(x, options, kwargs, raise_on_error=True):
    """ Checks if neuron contains the relevant data e.g. for
    ``plot3d(plot_connectors=True)``.

    Parameters
    ----------
    x :                 TreeNeuron
                        Neuron to check for data.
    options :           str | list of str
                        Options to check, e.g. "plot".
    kwargs :            dict
                        Keyword arguments to check for options.
    raise_on_error :    bool, optional
                        If True, will raise error if data not found.

    Returns
    -------
    None
    """

    options = _make_iterable(options)

    for o in options:
        for k in kwargs:
            if isinstance(k, str) and k.startswith(o):
                d = k[k.index('_'):]
                if not hasattr(x, d):
                    msg = f'Option "{k}" but {type(x)} has no "{d}"'
                    if raise_on_error:
                        raise ValueError(msg)
                    else:
                        logger.warning(msg)


def validate_table(x, required, restrict=False, optional={}):
    """ Validates DataFrame

    Parameters
    ----------
    x :         pd.DataFrame
                DataFrame to validate.
    required :  iterable
                Columns to check for. If column is given as tuple (e.g.
                ``('type', 'relation', 'label')`` one of these columns
                has to exist)
    restrict :  bool, optional
                If True, will return only ``required`` columns.
    optional :  dict, optional
                Optional columns. If column not present will be generated.
                Dictionary must map column name to default value.
    """

    if not isinstance(x, pd.DataFrame):
        raise TypeError(f'Need DataFrame, got "{type(x)}"')

    for r in required:
        if isinstance(r, (tuple, list)):
            if not any([c in x.columns for c in r]):
                raise ValueError('Table must contain either of these columns'
                                 f' {", ".join(r)}')
        else:
            if r not in x.columns:
                raise ValueError(f'Table missing required column: {r}')

    if restrict:
        flat_req = [r for r in required if not isinstance(r, (list, tuple))]
        flat_req += [r for l in required if isinstance(l, (list, tuple)) for r in l]

        x = x[[r for r in flat_req if r in x.columns]]

    for c, v in optional.items():
        if c not in x.columns:
            x[c] = v

    return x
