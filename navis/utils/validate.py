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

import pandas as pd

from typing import Union, List, Tuple

from .. import config, core

from .iterables import *

# Set up logging
logger = config.logger


def validate_options(x: 'core.TreeNeuron',
                     options: Union[List[str], str],
                     kwargs: dict,
                     raise_on_error: bool = True) -> None:
    """Check if neuron contains all required data.

    E.g. for ``plot3d(plot_connectors=True)``.

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
    options = make_iterable(options)

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


def validate_table(x: pd.DataFrame,
                   required: List[Union[str, Tuple[str]]],
                   rename: bool = False,
                   restrict: bool = False,
                   optional: dict = {}) -> pd.DataFrame:
    """Validate DataFrame.

    Parameters
    ----------
    x :         pd.DataFrame
                DataFrame to validate.
    required :  iterable
                Columns to check for. If column is given as tuple (e.g.
                ``('type', 'relation', 'label')`` one of these columns
                has to exist)
    rename :    bool, optional
                If True and a required column is given as tuple, will rename
                that column to the first entry in tuple.
    restrict :  bool, optional
                If True, will return only ``required`` columns.
    optional :  dict, optional
                Optional columns. If column not present will be generated.
                Dictionary must map column name to default value. Keys can also
                be tuples - like ``required``.

    Returns
    -------
    pandas.DataFrame
                If ``restrict=True`` will return DataFrame subset to only the
                required columns. Columns defined in ``optional`` will be
                added if they don't already exist.

    Examples
    --------
    >>> from navis.utils import validate_table
    >>> from navis.data import example_neurons
    >>> n = example_neurons(1)
    >>> tbl = validate_table(n.nodes, ['x', 'y', 'z', 'node_id'])
    >>> tbl = validate_table(n.nodes, ['does_not_exist'])       # doctest: +SKIP
    ValueError: Table missing required column: "does_not_exist"

    Raises
    ------
    ValueError
            If any of the required columns are not in the table.

    """
    if not isinstance(x, pd.DataFrame):
        raise TypeError(f'Need DataFrame, got "{type(x)}"')

    for r in required:
        if isinstance(r, (tuple, list)):
            if not any(set(r) & set(x.columns)):
                raise ValueError('Table must contain either of these columns'
                                 f' {", ".join(r)}')
        else:
            if r not in x.columns:
                raise ValueError(f'Table missing required column: "{r}"')

    # Rename columns if necessary
    if rename:
        # Generate mapping. Order makes sure that required columns take
        # precedence in case of a name clash
        new_name = {c: t[0] for t in optional if isinstance(t, (tuple, list)) for c in t[1:]}
        new_name.update({c: t[0] for t in required if isinstance(t, (tuple, list)) for c in t[1:]})

        # Apply mapping
        x.columns = [new_name.get(c, c) for c in x.columns]

    if restrict:
        flat_req = [r for r in required if not isinstance(r, (list, tuple))]
        flat_req += [r for l in required if isinstance(l, (list, tuple)) for r in l]

        x = x[[r for r in flat_req if r in x.columns]]

    for c, v in optional.items():
        # Convert to tuples
        if not isinstance(c, (tuple, list)):
            c = (c, )

        if not any(set(c) & set(x.columns)):
            x[c] = v

    return x
