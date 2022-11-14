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

import functools

import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from inspect import signature

from .. import utils, config

logger = config.get_logger(__name__)


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


class BaseTransform(ABC):
    """Abstract base class for transforms.

    If the transform is invertible, implement via __neg__ method.
    """

    def append(self, other: 'BaseTransform'):
        """Append another transform to this one.

        This is used to try to concatenate transforms of the same type into
        a single step to speed things up (e.g. for CMTK transforms). If that's
        not possible or not useful, must raise a ``NotImplementedError``.
        """
        raise NotImplementedError(f'Unable to append {type(other)} to {type(self)}')

    def check_if_possible(self, on_error: str = 'raise'):
        """Test if running the transform is possible."""
        return

    @abstractmethod
    def copy(self) -> 'BaseTransform':
        """Return copy."""
        pass

    @abstractmethod
    def xform(self, points: np.ndarray) -> np.ndarray:
        """Return copy.

        Must accept a (N, 3) numpy array as first input and return the
        transformed (N, 3) points as sole output.s
        """
        pass


class AliasTransform(BaseTransform):
    """Helper transform that simply passes points through.

    Useful for defining aliases.
    """

    def __init__(self):
        """Initialize."""
        pass

    def __neg__(self) -> 'AliasTransform':
        """Invert transform."""
        return self.copy()

    def __eq__(self, other):
        """Check if the same."""
        if isinstance(other, AliasTransform):
            True
        return False

    def copy(self):
        """Return copy."""
        x = AliasTransform()
        x.__dict__.update(self.__dict__)
        return x

    def xform(self, points: np.ndarray) -> np.ndarray:
        """Pass through.

        Note that the returned points are NOT a copy but the originals.
        """
        return points


class FunctionTransform(BaseTransform):
    """Apply custom function as transform.

    Parameters
    ----------
    func :      callable
                Function that accepts and returns an (N, 3) array.

    """

    def __init__(self, func):
        """Initialize."""
        if not callable(func):
            raise TypeError('`func` must be callable')
        self.func = func

    def __eq__(self, other):
        """Check if the same."""
        if not isinstance(other, FunctionTransform):
            return False
        if self.func != other.func:
            return False
        return True

    def copy(self):
        """Return copy."""
        x = self.__class__(self.func)
        x.__dict__.update(self.__dict__)
        return x

    def xform(self, points: np.ndarray) -> np.ndarray:
        """Xform data.

        Parameters
        ----------
        points :        (N, 3) numpy array | pandas.DataFrame
                        Points to xform. DataFrame must have x/y/z columns.

        Returns
        -------
        pointsxf :      (N, 3) numpy array
                        Transformed points.

        """
        if isinstance(points, pd.DataFrame):
            # Make sure x/y/z columns are present
            if np.any([c not in points for c in ['x', 'y', 'z']]):
                raise ValueError('points DataFrame must have x/y/z columns.')
            points = points[['x', 'y', 'z']].values
        elif not (isinstance(points, np.ndarray) and points.ndim == 2 and points.shape[1] == 3):
            raise TypeError('`points` must be numpy array of shape (N, 3) or '
                            'pandas DataFrame with x/y/z columns')
        return self.func(points.copy())

class TransformSequence:
    """A sequence of transforms.

    Use this to apply multiple (different types of) transforms in sequence.

    Parameters
    ----------
    *transforms :   Transform/Sequences.
                    The transforms to bundle in this sequence.
    copy :          bool
                    Whether to make a copy of the transform on initialization.
                    This is highly recommended because otherwise we might alter
                    the original as we add more transforms (e.g. for CMTK
                    transforms).

    """

    def __init__(self, *transforms, copy=True):
        """Initialize."""
        self.transforms = []
        for tr in transforms:
            if not isinstance(tr, (BaseTransform, TransformSequence)):
                raise TypeError(f'Expected transform, got "{type(tr)}"')
            if copy:
                tr = tr.copy()
            self.append(tr)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f'TransformSequence with {len(self)} transform(s)'

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

            # Try to merge with the last transform in the sequence
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
              affine_fallback: bool = True,
              **kwargs) -> np.ndarray:
        """Perform transforms in sequence."""
        # First check if any of the transforms raise any issues ahead of time
        # This can e.g. be missing binaries like CMTK's streamxform
        for tr in self.transforms:
            tr.check_if_possible(on_error='raise')

        # Now transform points in sequence
        # Make a copy of the points to avoid changing the originals
        # Note dtype float64 in case our precision in case precisio must go up
        # -> e.g. when converting from nm to micron space
        xf = np.asarray(points).astype(np.float64)
        for tr in self.transforms:
            # Check this transforms signature for accepted Parameters
            params = signature(tr.xform).parameters

            # We must not pass NaN value from one transform to the next
            is_nan = np.any(np.isnan(xf), axis=1)

            # Skip if all points are NaN
            if all(is_nan):
                continue

            if 'affine_fallback' in params:
                xf[~is_nan] = tr.xform(xf[~is_nan],
                                       affine_fallback=affine_fallback,
                                       **kwargs)
            else:
                xf[~is_nan] = tr.xform(xf[~is_nan], **kwargs)

        return xf


class TransOptimizer:
    """Optimizes a Transform or TransformSequence.

    The purpose of this class is to change a bunch of settings (depending on the
    type of transforms) before running the transformation and do a clean up
    after it has finished.

    Currently, the optimizations are very hands-on but in the future, this might
    be delegated to the individuals `Transform` classes - e.g. by implementing
    an `.optimize()` method which is then called by `TransOptimizer`.

    Currently, it really only manages caching for H5 transforms.

    Parameters
    ----------
    tr :        Transform | TransformSequence
                The transform or sequence thereof to be optimized.
    mode :      None | "medium" | "aggressive"
                Mode for optimization:
                  - ``None``: no optimization
                  - "medium": some optimization but keep upfront cost low
                  - "aggressive": high upfront cost but should be faster in the long run

    Examples
    --------
    >>> from navis.transforms import h5reg
    >>> from navis.transforms.base import TransOptimizer
    >>> tr = h5reg.H5transform('path/to/reg.h5', direction='inverse') # doctest: +SKIP
    >>> with TransOptimizer(tr, mode='aggressive'):                   # doctest: +SKIP
    >>>     xf = tr.xform(pts)                                        # doctest: +SKIP

    """

    def __init__(self, tr, bbox, caching: bool):
        """Initialize Optimizer."""
        assert isinstance(caching, bool)

        self.caching = caching
        self.bbox = np.asarray(bbox)

        assert self.bbox.ndim == 2 and self.bbox.shape == (3, 2)

        if isinstance(tr, BaseTransform):
            self.transforms = [tr]
        elif isinstance(tr, TransformSequence):
            self.transforms = tr.transforms
        else:
            raise TypeError(f'Expected Transform/Sequence, got "{type(tr)}"')

    def __enter__(self):
        """Apply optimizations."""
        if not self.caching:
            return

        # Check if there are any transforms we can optimize
        if not any(['H5transform' in str(type(tr)) for tr in self.transforms]):
            return

        if not config.pbar_hide:
            logger.info('Pre-caching deformation field(s) for transforms...')

        bbox_xf = self.bbox
        for tr in self.transforms:
            # We are monkey patching here to avoid circular imports
            # not pretty but works
            if 'H5transform' in str(type(tr)):
                # Precache values in the bounding box
                tr.precache(bbox_xf, padding=True)
            # To pre-cache sequential transforms we need to xform the bounding
            # box as we move along
            bbox_xf = tr.xform(bbox_xf.T).T

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Revert optimizations."""
        if not self.caching:
            return

        for tr in self.transforms:
            # We are monkey patching here to avoid circular imports
            # not pretty but works
            if 'H5transform' in str(type(tr)):
                # Clears the cache
                tr.use_cache = False
