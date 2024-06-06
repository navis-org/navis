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

import copy
import numbers
import pint
import types
import uuid
import warnings

import numpy as np
import pandas as pd

from typing import Union, Callable, List, Optional, Tuple
from typing_extensions import Literal

from .. import utils, config, core, sampling, graph

from .base import BaseNeuron

try:
    import xxhash
except ImportError:
    xxhash = None

try:
    from pykdtree.kdtree import KDTree
except ImportError:
    from scipy.spatial import cKDTree as KDTree

__all__ = ['Dotprops']

# Set up logging
logger = config.get_logger(__name__)

# This is to prevent pint to throw a warning about numpy integration
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pint.Quantity([])


class Dotprops(BaseNeuron):
    """Neuron represented as points + local vectors.

    Dotprops consist of points with x/y/z coordinates, a tangent vector and an
    alpha value describing the immediate neighbourhood (see also references).

    Typically constructed using :func:`navis.make_dotprops`.

    References
    ----------
    Masse N.Y., Cachero S., Ostrovsky A., and Jefferis G.S.X.E. (2012). A mutual
    information approach to automate identification of neuronal clusters in
    Drosophila brain images. Frontiers in Neuroinformatics 6 (00021).
    doi: 10.3389/fninf.2012.00021

    Parameters
    ----------
    points :        numpy array
                    (N, 3) array of x/y/z coordinates.
    k :             int, optional
                    Number of nearest neighbors for tangent vector calculation.
                    This can be ``None`` or ``0`` but then vectors must be
                    provided on initialization and can subsequently not be
                    re-calculated. Typical values here are ``k=20`` for dense
                    (e.g. from light level data) and ``k=5`` for sparse
                    (e.g. from skeletons) point clouds.
    vect :          numpy array, optional
                    (N, 3) array of vectors. If not provided will
                    recalculate both ``vect`` and ``alpha`` using ``k``.
    alpha :         numpy array, optional
                    (N, ) array of alpha values. If not provided will
                    recalculate both ``alpha`` and ``vect`` using ``k``.
    units :         str | pint.Units | pint.Quantity
                    Units for coordinates. Defaults to ``None`` (dimensionless).
                    Strings must be parsable by pint: e.g. "nm", "um",
                    "micrometer" or "8 nanometers".
    **metadata
                    Any additional data to attach to neuron.

    """

    connectors: Optional[pd.DataFrame]

    points: np.ndarray
    alpha: np.ndarray
    vect:  np.ndarray
    k: Optional[int]

    soma: Optional[Union[list, np.ndarray]]

    #: Attributes used for neuron summary
    SUMMARY_PROPS = ['type', 'name', 'k', 'units', 'n_points']

    #: Attributes to be used when comparing two neurons.
    EQ_ATTRIBUTES = ['name', 'n_points', 'k']

    #: Temporary attributes that need clearing when neuron data changes
    TEMP_ATTR = ['_memory_usage']

    #: Core data table(s) used to calculate hash
    _CORE_DATA = ['points', 'vect']

    def __init__(self,
                 points: np.ndarray,
                 k: int,
                 vect: Optional[np.ndarray] = None,
                 alpha: Optional[np.ndarray] = None,
                 units: Union[pint.Unit, str] = None,
                 **metadata
                 ):
        """Initialize Dotprops Neuron."""
        super().__init__()

        self.k = k
        self.points = points
        self.alpha = alpha
        self.vect = vect

        self.soma = None

        for k, v in metadata.items():
            try:
                setattr(self, k, v)
            except AttributeError:
                raise AttributeError(f"Unable to set neuron's `{k}` attribute.")

        self.units = units

    def __truediv__(self, other, copy=True):
        """Implement division for coordinates."""
        if isinstance(other, numbers.Number) or utils.is_iterable(other):
            # If a number, consider this an offset for coordinates
            n = self.copy() if copy else self
            _ = np.divide(n.points, other, out=n.points, casting='unsafe')
            if n.has_connectors:
                n.connectors.loc[:, ['x', 'y', 'z']] /= other

            # Force recomputing of KDTree
            if hasattr(n, '_tree'):
                delattr(n, '_tree')

            # Convert units
            # Note: .to_compact() throws a RuntimeWarning and returns unchanged
            # values  when `units` is a iterable
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                n.units = (n.units * other).to_compact()

            return n
        return NotImplemented

    def __mul__(self, other, copy=True):
        """Implement multiplication for coordinates."""
        if isinstance(other, numbers.Number) or utils.is_iterable(other):
            # If a number, consider this an offset for coordinates
            n = self.copy() if copy else self
            _ = np.multiply(n.points, other, out=n.points, casting='unsafe')
            if n.has_connectors:
                n.connectors.loc[:, ['x', 'y', 'z']] *= other

            # Force recomputing of KDTree
            if hasattr(n, '_tree'):
                delattr(n, '_tree')

            # Convert units
            # Note: .to_compact() throws a RuntimeWarning and returns unchanged
            # values  when `units` is a iterable
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                n.units = (n.units / other).to_compact()

            return n
        return NotImplemented

    def __getstate__(self):
        """Get state (used e.g. for pickling)."""
        state = {k: v for k, v in self.__dict__.items() if not callable(v)}

        # The KDTree from pykdtree does not like being pickled
        # We will have to remove it which will force it to be regenerated
        # after unpickling
        if '_tree' in state:
            if 'pykdtree' in str(type(state['_tree'])):
                _ = state.pop('_tree')

        return state

    @property
    def alpha(self):
        """Alpha value for tangent vectors (optional)."""
        if isinstance(self._alpha, type(None)):
            if isinstance(self.k, type(None)) or (self.k <= 0):
                raise ValueError('Unable to calculate `alpha` for Dotprops not '
                                 'generated using k-nearest-neighbors.')

            self.recalculate_tangents(self.k, inplace=True)
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if not isinstance(value, type(None)):
            value = np.asarray(value)
            if value.ndim != 1:
                raise ValueError(f'alpha must be (N, ) array, got {value.shape}')
        self._alpha = value

    @property
    def bbox(self) -> np.ndarray:
        """Bounding box (includes connectors)."""
        mn = np.min(self.points, axis=0)
        mx = np.max(self.points, axis=0)

        if self.has_connectors:
            cn_mn = np.min(self.connectors[['x', 'y', 'z']].values, axis=0)
            cn_mx = np.max(self.connectors[['x', 'y', 'z']].values, axis=0)

            mn = np.min(np.vstack((mn, cn_mn)), axis=0)
            mx = np.max(np.vstack((mx, cn_mx)), axis=0)

        return np.vstack((mn, mx)).T

    @property
    def datatables(self) -> List[str]:
        """Names of all DataFrames attached to this neuron."""
        return [k for k, v in self.__dict__.items() if isinstance(v, pd.DataFrame, np.ndarray)]

    @property
    def kdtree(self):
        """KDTree for points."""
        if not getattr(self, '_tree', None):
            self._tree = KDTree(self.points)
        return self._tree

    @property
    def points(self):
        """Center of tangent vectors."""
        return self._points

    @points.setter
    def points(self, value):
        if isinstance(value, type(None)):
            value = np.zeros((0, 3))
        value = np.asarray(value)
        if value.ndim != 2 or value.shape[1] != 3:
            raise ValueError(f'points must be (N, 3) array, got {value.shape}')
        self._points = value
        # Also reset KDtree
        self._tree = None

    @property
    def vect(self):
        """Tangent vectors."""
        if isinstance(self._vect, type(None)):
            self.recalculate_tangents(self.k, inplace=True)
        return self._vect

    @vect.setter
    def vect(self, value):
        if not isinstance(value, type(None)):
            value = np.asarray(value)
            if value.ndim != 2 or value.shape[1] != 3:
                raise ValueError(f'vectors must be (N, 3) array, got {value.shape}')
        self._vect = value

    @property
    def sampling_resolution(self):
        """Mean distance between points."""
        dist, _ = self.kdtree.query(self.points, k=2)
        return np.mean(dist[:, 1])

    @property
    def soma(self) -> Optional[int]:
        """Index of soma point.

        ``None`` if no soma. You can assign either a function that accepts a
        Dotprops as input or a fix value. Default is None.
        """
        if callable(self._soma):
            soma = self._soma.__call__()  # type: ignore  # say int not callable
        else:
            soma = self._soma

        # Sanity check to make sure that the soma node actually exists
        if isinstance(soma, type(None)):
            # Return immmediately without expensive checks
            return soma
        elif utils.is_iterable(soma):
            if not any(soma):
                soma = None
            elif any(np.array(soma) < 0) or any(np.array(soma) > self.points.shape[0]):
                logger.warning(f'Soma(s) {soma} not found in points.')
                soma = None
        else:
            if 0 < soma < self.points.shape[0]:
                logger.warning(f'Soma {soma} not found in node table.')
                soma = None

        return soma

    @soma.setter
    def soma(self, value: Union[Callable, int, None]) -> None:
        """Set soma."""
        if hasattr(value, '__call__'):
            self._soma = types.MethodType(value, self)
        elif isinstance(value, type(None)):
            self._soma = None
        elif isinstance(value, bool) and not value:
            self._soma = None
        else:
            if 0 < value < self.points.shape[0]:
                self._soma = value
            else:
                raise ValueError('Soma must be function, None or a valid node index.')

    @property
    def type(self) -> str:
        """Neuron type."""
        return 'navis.Dotprops'

    def dist_dots(self,
                  other: 'Dotprops',
                  alpha: bool = False,
                  distance_upper_bound: Optional[float] = None,
                  **kwargs) -> Union[
                      Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]
                    ]:
        """Query this Dotprops against another.

        This function is mainly for ``navis.nblast``.

        Parameters
        ----------
        other :                 Dotprops
        alpha :                 bool
                                If True, will also return the product of the
                                product of the alpha values of matched points.
        distance_upper_bound :  non-negative float, optional
                                If provided, we will stop the nearest neighbor
                                search at this distance which can vastly speed
                                up the query. For points with no hit within this
                                distance, `dist` will be set to
                                `distance_upper_bound`, and `dotprods` and
                                `alpha_prod` will be set to 0.
        kwargs
                                Keyword arguments are passed to the KDTree's
                                ``query()`` method. Note that we are using
                                ``pykdtree.kdtree.KDTree`` if available and fall
                                back to ``scipy.spatial.cKDTree`` if pykdtree is
                                not installed.

        Returns
        -------
        dist :          np.ndarray
                        For each point in ``self``, the distance to the closest
                        point in ``other``.
        dotprods :      np.ndarray
                        Dotproduct of each pair of closest points between
                        ``self`` and ``other``.
        alpha_prod :    np.ndarray
                        Dotproduct of each pair of closest points between
                        ``self`` and ``other``. Only returned if ``alpha=True``.

        """
        if not isinstance(other, Dotprops):
            raise TypeError(f'Expected Dotprops, got "{type(other)}"')

        # If we are using pykdtree we need to make sure that self.points is
        # of the same dtype as other.points - not a problem with scipy but
        # the overhead is typically only a few micro seconds anyway
        points = self.points.astype(other.points.dtype, copy=False)

        # Scipy's KDTree does not like the distance to be None
        diub = distance_upper_bound if distance_upper_bound else np.inf
        fast_dists, fast_idxs = other.kdtree.query(points,
                                                   distance_upper_bound=diub,
                                                   **kwargs)

        # If upper distance we have to worry about infinite distances
        if distance_upper_bound:
            no_nn = fast_dists == np.inf
            fast_dists[no_nn] = distance_upper_bound

            # Temporarily give those nodes a match
            fast_idxs[no_nn] = 0

        fast_dotprods = np.abs((self.vect * other.vect[fast_idxs]).sum(axis=1))

        if distance_upper_bound:
            fast_dotprods[no_nn] = 0

        if not alpha:
            return fast_dists, fast_dotprods

        fast_alpha = self.alpha * other.alpha[fast_idxs]

        if distance_upper_bound:
            fast_alpha[no_nn] = 0

        return fast_dists, fast_dotprods, fast_alpha

    def downsample(self, factor=5, inplace=False, **kwargs):
        """Downsample the neuron by given factor.

        Parameters
        ----------
        factor :                int, optional
                                Factor by which to downsample the neurons.
                                Default = 5.
        inplace :               bool, optional
                                If True, operation will be performed on
                                itself. If False, operation is performed on
                                copy which is then returned.
        **kwargs
                                Additional arguments passed to
                                :func:`~navis.downsample_neuron`.

        See Also
        --------
        :func:`~navis.downsample_neuron`
            Base function. See for details and examples.

        """
        if inplace:
            x = self
        else:
            x = self.copy()

        sampling.downsample_neuron(x, factor, inplace=True, **kwargs)

        if not inplace:
            return x
        return None

    def copy(self) -> 'Dotprops':
        """Return a copy of the dotprops.

        Returns
        -------
        Dotprops

        """
        # Don't copy the KDtree - when using pykdtree, copy.copy throws an
        # error and the construction is super fast anyway
        no_copy = ['_lock', '_tree']
        # Generate new empty neuron - note we pass vect and alpha to
        # prevent calculation on initialization
        x = self.__class__(points=np.zeros((0, 3)), k=1,
                           vect=np.zeros((0, 3)), alpha=np.zeros(0))
        # Populate with this neuron's data
        x.__dict__.update({k: copy.copy(v) for k, v in self.__dict__.items() if k not in no_copy})

        return x

    def recalculate_tangents(self, k: int, inplace=False) -> None:
        """Recalculate tangent vectors and alpha with a new ``k``.

        Parameters
        ----------
        k :         int
                    Number of nearest neighbours to use for tangent vector
                    calculation.
        inplace :   bool
                    If False, will return a copy and leave the original data
                    unmodified.

        Returns
        -------
        Dotprops
                    Only if ``inplace=False``.

        """
        if not inplace:
            x = self.copy()
        else:
            x = self

        if isinstance(k, type(None)) or k < 1:
            raise ValueError(f'`k` must be integer >= 1, got "{k}"')

        # Checks and balances
        n_points = x.points.shape[0]
        if n_points < k:
            raise ValueError(f"Too few points ({n_points}) to calculate {k} "
                             "nearest-neighbors")

        # Create the KDTree and get the k-nearest neighbors for each point
        dist, ix = self.kdtree.query(x.points, k=k)

        # Get points: array of (N, k, 3)
        pt = x.points[ix]

        # Generate centers for each cloud of k nearest neighbors
        centers = np.mean(pt, axis=1)

        # Generate vector from center
        cpt = pt - centers.reshape((pt.shape[0], 1, 3))

        # Get inertia (N, 3, 3)
        inertia = cpt.transpose((0, 2, 1)) @ cpt

        # Extract vector and alpha
        u, s, vh = np.linalg.svd(inertia)
        x.vect = vh[:, 0, :]
        x.alpha = (s[:, 0] - s[:, 1]) / np.sum(s, axis=1)

        # Keep track of k
        x.k = k

        if not inplace:
            return x

    def snap(self, locs, to='points'):
        """Snap xyz location(s) to closest point or synapse.

        Parameters
        ----------
        locs :      (N, 3) array | (3, ) array
                    Either single or multiple XYZ locations.
        to :        "points" | "connectors"
                    Whether to snap to points or connectors.

        Returns
        -------
        ix :        int | list of int
                    Index/Indices of the closest point/connector.
        dist :      float | list of float
                    Distance(s) to the closest point/connector.

        Examples
        --------
        >>> import navis
        >>> n = navis.example_neurons(1)
        >>> dp = navis.make_dotprops(n, k=5)
        >>> ix, dist = dp.snap([0, 0, 0])
        >>> ix
        1123

        """
        locs = np.asarray(locs).astype(np.float64)

        is_single = (locs.ndim == 1 and len(locs) == 3)
        is_multi = (locs.ndim == 2 and locs.shape[1] == 3)
        if not is_single and not is_multi:
            raise ValueError('Expected a single (x, y, z) location or a '
                             '(N, 3) array of multiple locations')

        if to not in ['points', 'connectors']:
            raise ValueError('`to` must be "points" or "connectors", '
                             f'got {to}')

        # Generate tree
        tree = graph.neuron2KDTree(self, data=to)

        # Find the closest node
        dist, ix = tree.query(locs)

        return ix, dist

    def to_skeleton(self,
                    scale_vec: Union[float, Literal['auto']] = 'auto'
                    ) -> core.TreeNeuron:
        """Turn dotprops into a skeleton.

        This is mainly for things like plotting as it does not produce
        meaningful edges. Also note that only minimal meta data is carried over.

        Parameters
        ----------
        scale_vec :     "auto" | float
                        Factor by which to scale each tangent vector when
                        generating the line segments. If "auto" (default for
                        plotting) will use the sampling resolution (median
                        distance between points) to determine a suitable
                        values.

        Returns
        -------
        TreeNeuron

        """
        if not isinstance(scale_vec, numbers.Number) and scale_vec != 'auto':
            raise ValueError('`scale_vect` must be "auto" or a number, '
                             f'got {scale_vec}')

        if scale_vec == 'auto':
            scale_vec = self.sampling_resolution * .8

        # Prepare segments - this is based on nat:::plot3d.dotprops
        halfvect = self.vect / 2 * scale_vec
        starts = self.points - halfvect
        ends = self.points + halfvect

        # Interweave starts and ends
        segs = np.zeros((starts.shape[0] * 2, 3))
        segs[::2] = starts
        segs[1::2] = ends

        # Generate node table
        nodes = pd.DataFrame(segs, columns=['x', 'y', 'z'])
        nodes['node_id'] = nodes.index
        nodes['parent_id'] = -1
        nodes.loc[1::2, 'parent_id'] = nodes.index.values[::2]

        # Produce a minimal TreeNeuron
        tn = core.TreeNeuron(nodes, units=self.units, id=self.id)

        # Carry over the label
        if getattr(self, '_label', None):
            tn._label = self._label

        # Add some other relevant attributes directly
        if self.has_connectors:
            tn._connectors = self._connectors
        tn._soma = self._soma

        return tn

    def __len__(self):
        return len(self.points)
