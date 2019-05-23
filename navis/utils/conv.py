#    This script is part of navis (http://www.github.com/schlegelp/navis).
#    Copyright (C) 2018 Philipp Schlegel
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along

import collections
import six

import numpy as np


def make_iterable(x, force_type=None):
    """ Helper function. Turns x into a np.ndarray, if it isn't already. For
    dicts, keys will be turned into array.
    """
    if not isinstance(x, collections.Iterable) or isinstance(x, six.string_types):
        x = [x]

    if isinstance(x, dict):
        x = list(x)

    if force_type:
        return np.array(x).astype(force_type)
    else:
        return np.array(x)


def make_non_iterable(x):
    """ Helper function. Turns x into non-iterable, if it isn't already. Will
    raise error if len(x) > 1.
    """
    if not is_iterable(x):
        return x
    elif len(x) == 1:
        return x[0]
    else:
        raise ValueError('Iterable must not contain more than one entry.')


def is_iterable(x):
    """ Helper function. Returns True if x is an iterable but not str or
    dictionary.
    """
    if isinstance(x, collections.Iterable) and not isinstance(x, six.string_types):
        return True
    else:
        return False
