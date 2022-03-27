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

import six
import pint

import numpy as np
import pandas as pd

from typing import Optional, Any
from collections.abc import Iterable

from .. import config


def make_iterable(x,
                  force_type: Optional[type] = None
                  ) -> np.ndarray:
    """Force input into a numpy array.

    For dicts, keys will be turned into array.

    Examples
    --------
    >>> from navis.utils import make_iterable
    >>> make_iterable(1)
    array([1])
    >>> make_iterable([1])
    array([1])
    >>> make_iterable({'a': 1})
    array(['a'], dtype='<U1')

    """
    # Quantities are a special case
    if isinstance(x, pint.Quantity) and not isinstance(x.magnitude, np.ndarray):
        return config.ureg.Quantity(np.array([x.magnitude]), x.units)

    if not isinstance(x, Iterable) or isinstance(x, six.string_types):
        x = [x]

    if isinstance(x, (dict, set)):
        x = list(x)

    return np.asarray(x, dtype=force_type)


def make_non_iterable(x):
    """Turn input into non-iterable, if it isn't already.

    Will raise error if ``len(x) > 1``.

    Examples
    --------
    >>> from navis.utils import make_non_iterable
    >>> make_non_iterable([1])
    1
    >>> make_non_iterable(1)
    1
    >>> make_non_iterable([1, 2])
    Traceback (most recent call last):
    ValueError: Iterable must not contain more than one entry.

    """
    if not is_iterable(x):
        return x
    elif len(x) == 1:
        return x[0]
    else:
        raise ValueError('Iterable must not contain more than one entry.')


def is_iterable(x: Any) -> bool:
    """Test if input is iterable (but not str).

    Examples
    --------
    >>> from navis.utils import is_iterable
    >>> is_iterable(['a'])
    True
    >>> is_iterable('a')
    False
    >>> is_iterable({'a': 1})
    True

    """
    if isinstance(x, pint.Quantity):
        x = x.magnitude

    if isinstance(x, Iterable) and not isinstance(x, (six.string_types, pd.DataFrame)):
        return True
    else:
        return False

def multi_split(s, sep, strip=False):
    """Split string at any of the given separators.

    Returns
    -------
    list

    Examples
    --------
    >>> from navis.utils import multi_split
    >>> s = '123;456,789:'
    >>> multi_split(s, ';')
    ['123', '456,789']
    >>> multi_split(s, [';', ','])
    ['123', '456', '789']

    """
    assert isinstance(s, str)

    if isinstance(sep, str):
        sep = [sep]

    splits = []
    prev_sp = 0
    for i in range(len(s)):
        if s[i] in sep or i == (len(s) - 1):
            splits.append(s[prev_sp:i])
            prev_sp = i + 1

    if strip:
        splits = [sp.strip() for sp in splits]

    return splits
