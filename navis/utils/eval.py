
#    This script is part of navis (http://www.github.com/schlegelp/navis).
#    Copyright (C) 2017 Philipp Schlegel
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

import uuid

import pandas as pd
import numpy as np

from typing import Tuple, Iterable, List, Union, Any, Optional, Sequence

from .. import config, core
from .iterables import is_iterable

from .iterables import *

# Set up logging
logger = config.logger

# Boolean, unsigned integer, signed integer, float, complex.
_NUMERIC_KINDS = set('buifc')


def eval_param(value: Any,
               name: str,
               allowed_values: Optional[tuple] = None,
               allowed_types: Optional[tuple] = None,
               on_error: str = 'raise'):
    """Check if parameter has expected type and/or value.

    Parameters
    ----------
    value :             any
                        Value to be checked.
    name :              str
                        Name of the parameter. Used for warnings/exceptions.
    allowed_values :    tuple
                        Iterable containing the allowed values.
    allowed_types  :    tuple
                        Iterable containing the allowed types.
    on_error :          "raise" | "warn"
                        What to do if ``value`` is not in ``allowed_values``.

    Returns
    -------
    None

    """
    assert on_error in ('raise', 'warn')
    assert isinstance(allowed_values, (tuple, type(None)))
    assert isinstance(allowed_types, (tuple, type(None)))

    if allowed_types:
        if not isinstance(value, allowed_types):
            msg = (f'Unexpected type for "{name}": {type(value)}. ',
                   f'Allowed type(s): {", ".join([str(t) for t in allowed_types])}')
            if on_error == 'raise':
                raise ValueError(msg)
            elif on_error == 'warn':
                logger.warning(msg)

    if allowed_values:
        if value not in allowed_values:
            msg = (f'Unexpected value for "{name}": {value}. ',
                   f'Allowed value(s): {", ".join([str(t) for t in allowed_values])}')
            if on_error == 'raise':
                raise ValueError(msg)
            elif on_error == 'warn':
                logger.warning(msg)


def is_numeric(array: np.ndarray, bool_numeric: bool = True) -> bool:
    """Determine whether the argument has a numeric datatype.

    Booleans, unsigned integers, signed integers, floats and complex
    numbers are the kinds of numeric datatype.

    Arrays with "dtype=object" will return True if data can be cast to floats.

    Parameters
    ----------
    array :         array-like
                    The array to check.
    bool_numeric :  bool
                    If True (default), we count booleans as numeric data types.

    Returns
    -------
    is_numeric :    `bool`
                    True if the array has a numeric datatype, False if not.

    """
    array = np.asarray(array)

    # If array
    if array.dtype.kind == 'O':
        try:
            array = array.astype(float)
        except ValueError:
            pass

    if not bool_numeric:
        _NUMERIC_KINDS_NO_BOOL = _NUMERIC_KINDS.copy()
        _NUMERIC_KINDS_NO_BOOL.remove('b')
        return array.dtype.kind in _NUMERIC_KINDS_NO_BOOL

    return array.dtype.kind in _NUMERIC_KINDS


def is_mesh(x) -> Tuple[List[bool], List[bool]]:
    """Check if object is mesh (i.e. contains vertices and faces).

    Examples
    --------
    >>> import navis
    >>> is_mesh(navis.example_neurons(1))
    False
    >>> is_mesh(navis.example_volume('LH'))
    True

    """
    if hasattr(x, 'vertices') and hasattr(x, 'faces'):
        return True

    return False


def eval_conditions(x) -> Tuple[List[bool], List[bool]]:
    """Split list of strings into positive (no "~") and negative ("~").

    Examples
    --------
    >>> eval_conditions('~negative condition')
    ([], ['negative condition'])
    >>> eval_conditions(['positive cond1', '~negative cond1', 'positive cond2'])
    (['positive cond1', 'positive cond2'], ['negative cond1'])

    """
    x = make_iterable(x, force_type=str)

    return [i for i in x if not i.startswith('~')], [i[1:] for i in x if i.startswith('~')]


def eval_id(x: Union[uuid.UUID, str, 'core.NeuronObject', pd.DataFrame],
              warn_duplicates: bool = True) -> List[uuid.UUID]:
    """Evaluate neuron ID(s).

    Parameters
    ----------
    x :                str | uuid.UUID | Tree/MeshNeuron | NeuronList | DataFrame
                       For Neuron/List or pandas.DataFrames/Series will
                       look for ``id`` attribute/column.
    warn_duplicates :  bool, optional
                       If True, will warn if duplicate IDs are found.
                       Only applies to NeuronLists.

    Returns
    -------
    list
                    List containing IDs.

    """
    if isinstance(x, (uuid.UUID, str, np.str, int, np.int64, np.int)):
        return [x]
    elif isinstance(x, (list, np.ndarray, set)):
        uu: List[uuid.UUID] = []
        for e in x:
            temp = eval_id(e, warn_duplicates=warn_duplicates)
            if isinstance(temp, (list, np.ndarray)):
                uu += temp
            else:
                uu.append(temp)  # type: ignore
        return sorted(set(uu), key=uu.index)
    elif isinstance(x, core.BaseNeuron):
        return [x.id]
    elif isinstance(x, core.NeuronList):
        if len(x.id) != len(set(x.id)) and warn_duplicates:
            logger.warning('Duplicate IDs found in NeuronList.'
                           'The function you are using might not respect '
                           'fragments of the same neuron. For explanation see '
                           'http://navis.readthedocs.io/en/latest/source/conn'
                           'ectivity_analysis.html.')
        return list(x.id)
    elif isinstance(x, pd.DataFrame):
        if 'id' not in x.columns:
            raise ValueError('Expect "id" column in pandas DataFrames')
        return x.id.tolist()
    elif isinstance(x, pd.Series):
        if x.name == 'id':
            return x.tolist()
        elif 'id' in x:
            return [x.id]
        else:
            raise ValueError(f'Unable to extract ID from pandas series {x}')
    elif isinstance(x, type(None)):
        return None
    else:
        msg = f'Unable to extract ID(s) from data of type "{type(x)}"'
        logger.error(msg)
        raise TypeError(msg)


def eval_neurons(x: Any,
                 warn_duplicates: bool = True,
                 raise_other: bool = True) -> Optional[List['core.TreeNeuron']]:
    """Extract neurons.

    Parameters
    ----------
    x
                       Data to be checked for neurons.
    warn_duplicates :  bool, optional
                       If True, will warn if duplicate neurons are found.
                       Only applies to NeuronLists.
    raise_other :      bool, optional
                       If True, will raise error if non- neurons are found.

    Returns
    -------
    list
                    List containing neurons.
    None
                    If no neurons found.

    """
    if isinstance(x, core.BaseNeuron):
        return [x]
    elif isinstance(x, (list, np.ndarray, set)):
        neurons: List['core.BaseNeuron'] = []
        for e in x:
            temp = eval_neurons(e, warn_duplicates=warn_duplicates,
                                raise_other=raise_other)
            if isinstance(temp, (list, np.ndarray)):
                neurons += temp
            elif temp:
                neurons.append(temp)
        return sorted(set(neurons), key=neurons.index)
    elif isinstance(x, core.NeuronList):
        if len(x.id) != len(set(x.id)) and warn_duplicates:
            logger.warning('Duplicate IDs found in NeuronList.'
                           'The function you are using might not respect '
                           'fragments of the same neuron. For explanation see '
                           'http://navis.readthedocs.io/en/latest/source/conn'
                           'ectivity_analysis.html.')
        return x.neurons
    elif isinstance(x, type(None)):
        return None
    elif raise_other:
        msg = f'Unable to extract neurons from data of type "{type(x)}"'
        logger.error(msg)
        raise TypeError(msg)
    return None


def eval_node_ids(x: Union[int, str,
                           Sequence[Union[str, int]],
                           'core.NeuronObject',
                           pd.DataFrame]
                  ) -> List[int]:
    """Extract node IDs from data.

    Parameters
    ----------
    x :             int | str | TreeNeuron | NeuronList | DataFrame
                    Your options are either::
                    1. int or list of ints will be assumed to be node IDs
                    2. str or list of str will be checked if convertible to int
                    3. For TreeNeuron/List or pandas.DataFrames will try
                       to extract node IDs

    Returns
    -------
    list
                    List containing node IDs (integer)

    """
    if isinstance(x, (int, np.int64, np.int32, np.int)):
        return [x]
    elif isinstance(x, (str, np.str)):
        try:
            return [int(x)]
        except BaseException:
            raise TypeError(f'Unable to extract node ID from string "{x}"')
    elif isinstance(x, (set, list, np.ndarray)):
        # Check non-integer entries
        ids: List[int] = []
        for e in x:
            temp = eval_node_ids(e)
            if isinstance(temp, (list, np.ndarray)):
                ids += temp
            else:
                ids.append(temp)  # type: ignore
        # Preserving the order after making a set is super costly
        # return sorted(set(ids), key=ids.index)
        return list(set(ids))
    elif isinstance(x, core.TreeNeuron):
        return x.nodes.node_id.astype(int).tolist()
    elif isinstance(x, core.NeuronList):
        to_return: List[int] = []
        for n in x:
            to_return += n.nodes.node_id.astype(int).tolist()
        return to_return
    elif isinstance(x, (pd.DataFrame, pd.Series)):
        to_return = []
        if 'node_id' in x:
            to_return += x.node_id.astype(int).tolist()
        return to_return
    else:
        raise TypeError(f'Unable to extract node IDs from type {type(x)}')
