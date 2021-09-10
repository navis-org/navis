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
"""Module containing utility functions for `NEURON`."""


def is_NEURON_object(x):
    """Best guess whether object comes from NEURON."""
    # Note:
    # NEURON objects also have a `.hname()` method that
    # might be useful
    if not hasattr(x, '__module__'):
        return False
    if x.__module__ == 'nrn' or x.__module__ == 'hoc':
        return True
    return False


def is_segment(x):
    """Check if object is a segment."""
    if 'nrn.Segment' in str(type(x)):
        return True
    return False


def is_section(x):
    """Check if object is a section."""
    if 'nrn.Section' in str(type(x)):
        return True
    return False
