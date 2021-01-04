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

import copy
import functools

import numpy as np

from inspect import signature

from .. import utils


def trigger_init(func):
    """Trigger delayed initialization."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]
        # Check if has already been initialized
        if not self.initialized:
            self.__delayed_init__()
        return func(*args, **kwargs)
    return wrapper


class BaseTransform:
    """Abstract base class for transforms."""

    def append(self, other: 'BaseTransform'):
        """Append another transform to this one."""
        raise NotImplementedError(f'Unable to append {type(other)} to {type(self)}')

    def check_if_possible(self, on_error: str = 'raise'):
        """Test if running the transform is possible."""
        return

    def copy(self) -> 'BaseTransform':
        """Return copy."""
        # Attributes not to copy
        no_copy = []
        # Generate new empty transform
        x = self.__class__()
        # Override with this neuron's data
        x.__dict__.update({k: copy.copy(v) for k, v in self.__dict__.items() if k not in no_copy})

        return x


class AliasTransform(BaseTransform):
    """Helper transform that simply passes points through.

    Useful for defining aliases.
    """

    def __init__(self):
        """Initialize."""
        pass

    def xform(self, points: np.ndarray) -> np.ndarray:
        """Pass through.

        Be aware that the returned points are NOT a copy but the originals.
        """
        return points

    def __neg__(self) -> 'AliasTransform':
        """Invert transform."""
        return self.copy()


class TransformSequence:
    """A sequence of transforms.

    Use this to apply multiple (different types of) transforms in sequence.

    """

    def __init__(self, *args):
        """Initialize."""
        self.transforms = []
        for tr in args:
            if not isinstance(tr, BaseTransform):
                raise TypeError(f'Expected transform, got "{type(tr)}"')
            self.append(tr)

    def __len__(self) -> int:
        """Count number of transforms in this sequence."""
        return len(self.transforms)

    def __neg__(self) -> 'TransformSequence':
        """Invert transform sequence."""
        return TransformSequence(*[-t for t in self.transforms[::-1]])

    def append(self, transform: 'BaseTransform'):
        """Add transform to list."""
        if isinstance(transform, TransformSequence):
            # Unpack if other is sequence of transforms
            transform = transform.transforms

        for tr in utils.make_iterable(transform):
            if not isinstance(tr, BaseTransform):
                raise TypeError(f'Unable append "{type(tr)}"')

            if not hasattr(transform, 'xform') or not callable(transform.xform):
                raise TypeError('Transform does not appear to have a `xform` method')

            # Try to merge with the last method
            if len(self):
                try:
                    self.transforms[-1].append(tr)
                except NotImplementedError:
                    self.transforms.append(tr)
                except BaseException:
                    raise
            else:
                self.transforms.append(tr)

    def xform(self, points: np.ndarray,
              affine_fallback: bool = False,
              **kwargs) -> np.ndarray:
        """Perform transforms in sequence."""
        # First check if any of the transforms raise problems.
        for tr in self.transforms:
            tr.check_if_possible(on_error='raise')

        # Now transform points in sequence
        xf = np.asarray(points).copy()  # copy is important here!
        for tr in self.transforms:
            # Check this transforms signature for accepted Parameters
            params = signature(tr.xform).parameters

            # We must not pass None value from one transform to the next
            is_nan = np.any(np.isnan(xf), axis=1)

            if affine_fallback and 'affine_fallback' in params:
                xf[~is_nan] = tr.xform(xf[~is_nan],
                                       affine_fallback=True, **kwargs)
            else:
                xf[~is_nan] = tr.xform(xf[~is_nan], **kwargs)

        return xf
