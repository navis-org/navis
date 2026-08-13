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
import os
import pint
import warnings
import scipy

import networkx as nx
import numpy as np
import pandas as pd
import skeletor as sk
import sparsecubes
import trimesh as tm

from typing import Union, Optional

from .. import utils, config, meshes, conversion, graph, morpho
from ..utils.subclasses import TrimeshPlus, validate_extra_edges
from .base import BaseNeuron
from . import schema
from .schema import (CONNECTOR_AXIS, Axis, Link, Ref, axes, connector_link,
                     links)
from .neuronlist import NeuronList
from .skeleton import Skeleton
from .core_utils import temp_property, add_units


try:
    import xxhash
except ModuleNotFoundError:
    xxhash = None


__all__ = ['Mesh']

# Set up logging
logger = config.get_logger(__name__)

# This is to prevent pint to throw a warning about numpy integration
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pint.Quantity([])


class Mesh(BaseNeuron):
    """Neuron represented as mesh with vertices and faces.

    Parameters
    ----------
    x :             mesh-like | tuple | dictionary | filepath | None
                    Data to construct neuron from:
                     - any object that has `.vertices` and `.faces`
                       properties (e.g. a trimesh.Trimesh)
                     - a tuple `(vertices, faces)`
                     - a dictionary `{"vertices": (N, 3), "faces": (M, 3)}`
                     - filepath to a file that can be read by `trimesh.load`
                     - `None` will initialize an empty Mesh
                     - `skeletor.Skeleton` will use the mesh and the skeleton
                       (including the vertex to node map)

    units :         str | pint.Units | pint.Quantity
                    Units for coordinates. Defaults to `None` (dimensionless).
                    Strings must be parsable by pint: e.g. "nm", "um",
                    "micrometer" or "8 nanometers".
    process :       bool
                    If True (default and highly recommended), will remove NaN
                    and infinite values, and merge duplicate vertices.
    validate :      bool
                    If True, will try to fix some common problems with
                    meshes. See `navis.fix_mesh` for details.
    **metadata
                    Any additional data to attach to neuron.

    """

    connectors: Optional[pd.DataFrame]

    vertices: np.ndarray
    faces: np.ndarray

    soma: Optional[Union[list, np.ndarray]]

    #: Attributes used for neuron summary
    SUMMARY_PROPS = ['type', 'name', 'units', 'n_vertices', 'n_faces']

    #: Attributes to be used when comparing two neurons.
    EQ_ATTRIBUTES = ['name', 'n_vertices', 'n_faces', 'n_extra_edges']

    #: Temporary attributes that need clearing when neuron data changes.
    #: N.B. `_skeleton` is deliberately *not* in here: it is governed by the
    #: `skeleton` link below, which can tell a change that was carried through
    #: to it from one that happened behind its back. `TEMP_ATTR` cannot, and
    #: would throw away a skeleton we had just gone to the trouble of keeping.
    TEMP_ATTR = ['_memory_usage', '_trimesh', '_igraph', '_graph_nx']

    #: Core data table(s) used to calculate hash
    CORE_DATA = ['vertices', 'faces', 'extra_edges']

    #: Element axes: what is aligned to the vertices, and what references them.
    #: See `navis/core/schema.py` - this drives `subset_neuron`. The axis is
    #: positional, so references store indices and have to be remapped, not just
    #: filtered.
    AXES = axes(
        Axis(
            name='vertices',
            data=('_vertices',),
            refs=(
                # A face survives only if all three of its corners do. A face
                # *is* three vertex indices, so anything that rebuilds the
                # vertices has necessarily rebuilt these too.
                Ref('_faces', kind='index_array', on_rebuild='rebuilt'),
                # An extra edge is a bridge between two vertices, i.e. a place
                # on the surface rather than a piece of it. A rebuild does not
                # take that place away, so the edge follows its endpoints to
                # wherever they were re-made - the same reading `connector_link`
                # takes. An edge whose two ends land on one vertex says nothing
                # and is dropped by `validate_extra_edges`.
                Ref('_extra_edges', kind='index_array',
                    write_attr='extra_edges', on_rebuild='snap'),
            ),
        ),
        CONNECTOR_AXIS,
    )

    #: Links to this neuron's other representations. Each is one array wearing
    #: two hats - aligned to one axis, and naming elements of another - which is
    #: what lets a single selection carry all of them.
    LINKS = links(
        # The vertex map: keeping it is what lets a selection carry the skeleton
        # along instead of throwing it away and re-skeletonizing the remainder
        # into a different set of nodes.
        Link(
            name='skeleton',
            source='vertices',
            mapping='_skeleton._vertex_map',
            target='_skeleton',
            target_axis='nodes',
            # A vertex whose node vanished is still a vertex
            dangling='blank',
            # Re-making the vertices does not re-make the skeleton: the arbour
            # is where it was, so a rebuild that can say which vertices became
            # which keeps the map rather than throwing it away - same reason as
            # above, for the other half of what can happen to an axis.
            on_rebuild_aligned='carry',
        ),
        # Connectors sit on a vertex, and compose through the above onto the
        # skeleton's nodes without anyone declaring that mapping.
        connector_link('vertices', 'vertex_id'),
    )

    #: The soma position is a coordinate, not a vertex index - it survives a
    #: subset unchanged (and may end up outside the remaining geometry).
    AXIS_INDEPENDENT = ('_soma_pos',)

    def __init__(self,
                 x,
                 units: Union[pint.Unit, str] = None,
                 process: bool = True,
                 validate: bool = False,
                 **metadata
                 ):
        """Initialize Mesh Neuron."""
        super().__init__()

        # Lock neuron during initialization
        self._lock = 1
        self._trimesh = None  # this is required to avoid recursion during init
        skeleton = None  # attached at the very end, see below

        if isinstance(x, Mesh):
            self.__dict__.update(x.copy().__dict__)
            self.vertices, self.faces = x.vertices, x.faces
        elif hasattr(x, 'faces') and hasattr(x, 'vertices'):
            self.vertices, self.faces = x.vertices, x.faces
            # Pick up extra edges from e.g. a TrimeshPlus
            if len(getattr(x, 'extra_edges', ())):
                self.extra_edges = x.extra_edges
        elif isinstance(x, dict):
            if 'faces' not in x or 'vertices' not in x:
                raise ValueError('Dictionary must contain "vertices" and "faces"')
            self.vertices, self.faces = x['vertices'], x['faces']
        elif isinstance(x, str) and os.path.isfile(x):
            m = tm.load(x)
            self.vertices, self.faces = m.vertices, m.faces
        elif isinstance(x, type(None)):
            # Empty neuron
            self.vertices, self.faces = np.zeros((0, 3)), np.zeros((0, 3))
        elif isinstance(x, sk.Skeleton):
            self.vertices, self.faces = x.mesh.vertices, x.mesh.faces
            skeleton = x
        elif isinstance(x, tuple):
            if len(x) != 2 or any([not isinstance(v, np.ndarray) for v in x]):
                raise TypeError('Expect tuple to be two arrays: (vertices, faces)')
            self.vertices, self.faces = x[0], x[1]
        else:
            raise utils.ConstructionError(f'Unable to construct Mesh from "{type(x)}"')

        for k, v in metadata.items():
            try:
                setattr(self, k, v)
            except AttributeError:
                raise AttributeError(f"Unable to set neuron's `{k}` attribute.")

        if process and self.vertices.shape[0]:
            # For some reason we can't do self._trimesh at this stage
            _trimesh = tm.Trimesh(self.vertices, self.faces,
                                  process=process,
                                  validate=validate)
            self.vertices = _trimesh.vertices
            self.faces = _trimesh.faces

        self._lock = 0

        if validate:
            # N.B. must be in-place: `validate()` otherwise fixes a copy and
            # hands it back, leaving this neuron untouched
            self.validate(inplace=True)

        self.units = units

        if skeleton is not None:
            # Last, and through the setter, so that there is only ever one way a
            # skeleton gets attached. It has to be last because `process=True`
            # merges duplicate vertices, so the mesh the skeleton was built
            # against is not necessarily the one we ended up with - and because
            # only now do we have the final `id`/`name` to hand it.
            self.skeleton = skeleton

    def __getstate__(self):
        """Get state (used e.g. for pickling)."""
        state = {k: v for k, v in self.__dict__.items() if not callable(v)}

        # We don't need the trimesh object
        if '_trimesh' in state:
            _ = state.pop('_trimesh')

        return state

    def __setstate__(self, d):
        """Update state (used e.g. for pickling)."""
        self.__dict__.update(d)

    def __truediv__(self, other, copy=True):
        """Implement division for coordinates (vertices, connectors)."""
        if isinstance(other, numbers.Number) or utils.is_iterable(other):
            # If a number, consider this an offset for coordinates
            n = self.copy() if copy else self
            _ = np.divide(n.vertices, other, out=n.vertices, casting='unsafe')
            if n.has_connectors:
                # Note: reassign (instead of in-place /=) so that integer
                # connector coordinates can be cast to float if necessary
                n.connectors[['x', 'y', 'z']] = n.connectors[['x', 'y', 'z']] / other

            # Convert units
            # Note: .to_compact() throws a RuntimeWarning and returns unchanged
            # values  when `units` is a iterable
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                n.units = (n.units * other).to_compact()

            n._clear_temp_attr()

            return n
        return NotImplemented

    def __mul__(self, other, copy=True):
        """Implement multiplication for coordinates (vertices, connectors)."""
        if isinstance(other, numbers.Number) or utils.is_iterable(other):
            # If a number, consider this an offset for coordinates
            n = self.copy() if copy else self
            _ = np.multiply(n.vertices, other, out=n.vertices, casting='unsafe')
            if n.has_connectors:
                # Note: reassign (instead of in-place *=) so that integer
                # connector coordinates can be cast to float if necessary
                n.connectors[['x', 'y', 'z']] = n.connectors[['x', 'y', 'z']] * other

            # Convert units
            # Note: .to_compact() throws a RuntimeWarning and returns unchanged
            # values  when `units` is a iterable
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                n.units = (n.units / other).to_compact()

            n._clear_temp_attr()

            return n
        return NotImplemented

    def __add__(self, other, copy=True):
        """Implement addition for coordinates (vertices, connectors)."""
        if isinstance(other, numbers.Number) or utils.is_iterable(other):
            n = self.copy() if copy else self
            _ = np.add(n.vertices, other, out=n.vertices, casting='unsafe')
            if n.has_connectors:
                # Note: reassign (instead of in-place +=) so that integer
                # connector coordinates can be cast to float if necessary
                n.connectors[['x', 'y', 'z']] = n.connectors[['x', 'y', 'z']] + other

            n._clear_temp_attr()

            return n
        # If another neuron, return a list of neurons
        elif isinstance(other, BaseNeuron):
            return NeuronList([self, other])
        return NotImplemented

    def __sub__(self, other, copy=True):
        """Implement subtraction for coordinates (vertices, connectors)."""
        if isinstance(other, numbers.Number) or utils.is_iterable(other):
            n = self.copy() if copy else self
            _ = np.subtract(n.vertices, other, out=n.vertices, casting='unsafe')
            if n.has_connectors:
                # Note: reassign (instead of in-place -=) so that integer
                # connector coordinates can be cast to float if necessary
                n.connectors[['x', 'y', 'z']] = n.connectors[['x', 'y', 'z']] - other

            n._clear_temp_attr()

            return n
        return NotImplemented

    @property
    def bbox(self) -> np.ndarray:
        """Bounding box (includes connectors)."""
        mn = np.min(self.vertices, axis=0)
        mx = np.max(self.vertices, axis=0)

        if self.has_connectors:
            cn_mn = np.min(self.connectors[['x', 'y', 'z']].values, axis=0)
            cn_mx = np.max(self.connectors[['x', 'y', 'z']].values, axis=0)

            mn = np.min(np.vstack((mn, cn_mn)), axis=0)
            mx = np.max(np.vstack((mx, cn_mx)), axis=0)

        return np.vstack((mn, mx)).T

    @property
    def vertices(self):
        """Vertices making up the neuron."""
        return self._vertices

    @vertices.setter
    def vertices(self, verts):
        if not isinstance(verts, np.ndarray):
            raise TypeError(f'Vertices must be numpy array, got "{type(verts)}"')
        if verts.ndim != 2:
            raise ValueError('Vertices must be 2-dimensional array')

        # Extra edges are vertex *indices* and hence only meaningful for a given
        # set of vertices. We can't tell whether vertices moved but a change in
        # their number is a reliable sign that they did (e.g. after merging
        # duplicates or subsetting) - so we drop the edges instead of silently
        # rewiring the neuron. Callers that know better (e.g. `subset_neuron`)
        # re-set them explicitly.
        old = getattr(self, '_vertices', None)
        if old is not None and len(old) != len(verts) and self.n_extra_edges:
            logger.warning(
                f'Number of vertices changed ({len(old)} -> {len(verts)}): '
                f'dropping this neuron\'s {self.n_extra_edges} extra edges.'
            )
            self.extra_edges = None

        # Replacing the elements, not selecting from them - see `_replacing`
        self._replacing('vertices', verts)
        self._vertices = verts
        self._clear_temp_attr()

    @property
    def faces(self):
        """Faces making up the neuron."""
        return self._faces

    @faces.setter
    def faces(self, faces):
        if not isinstance(faces, np.ndarray):
            raise TypeError(f'Faces must be numpy array, got "{type(faces)}"')
        if faces.ndim != 2:
            raise ValueError('Faces must be 2-dimensional array')
        self._faces = faces
        self._clear_temp_attr()

    @property
    def extra_edges(self):
        """Edges that are not part of any face.

        These express connectivity the surface itself does not have - e.g.
        bridges between disconnected fragments. Always a `(M, 2)` array of
        vertex indices; empty if there are none.

        Note that extra edges are dropped whenever the number of vertices
        changes (see `.vertices`).

        """
        edges = getattr(self, '_extra_edges', None)
        if edges is None:
            return np.zeros((0, 2), dtype=np.int64)
        return edges

    @extra_edges.setter
    def extra_edges(self, edges):
        self._extra_edges = validate_extra_edges(edges, n_vertices=len(self.vertices))

        # Only clutter the summary if there actually are extra edges
        if len(self._extra_edges):
            if 'n_extra_edges' not in self.SUMMARY_PROPS:
                self.SUMMARY_PROPS.append('n_extra_edges')
        elif 'n_extra_edges' in self.SUMMARY_PROPS:
            self.SUMMARY_PROPS.remove('n_extra_edges')

        self._clear_temp_attr()

    @property
    def n_extra_edges(self) -> int:
        """Number of edges that are not part of any face."""
        return len(self.extra_edges)

    @property
    @temp_property
    def igraph(self) -> 'igraph.Graph':
        """iGraph representation of the vertex connectivity."""
        # If igraph does not exist, create and return
        if not hasattr(self, '_igraph'):
            # This also sets the attribute
            self._igraph = graph.neuron2igraph(self)
        return self._igraph

    @property
    @temp_property
    def graph(self) -> nx.DiGraph:
        """Networkx Graph representation of the vertex connectivity."""
        # If graph does not exist, create and return
        if not hasattr(self, '_graph_nx'):
            # This also sets the attribute
            self._graph_nx = graph.neuron2nx(self)
        return self._graph_nx

    @property
    def sampling_resolution(self) -> float:
        """Average distance between vertices."""
        # N.B. `extra_edges=False`: this describes how finely the *surface* is
        # sampled, and bridges are typically far longer than any face edge
        return float(
            utils.mesh_unique_edges(
                self.trimesh, return_lengths=True, extra_edges=False
            )[1].mean()
        )

    @property
    @add_units(compact=True, power=3)
    def volume(self) -> float:
        """Volume of the neuron.

        Calculated from the surface integral. Garbage if neuron is not
        watertight.

        """
        return float(self.trimesh.volume)

    @property
    def skeleton(self) -> 'Skeleton':
        """Skeleton representation of this neuron.

        Generated with [`navis.conversion.mesh2skeleton`][] the first time it is
        asked for, and kept as long as it still describes this mesh.

        A selection - `subset_neuron`, `mask` - carries it along through the
        vertex map, so node IDs survive and anything computed on the skeleton
        can still be traced back. Any other change to the vertices (assigning
        them, transforming the neuron) is one we cannot follow, so the skeleton
        is regenerated on next access, as it always was.

        """
        # While the neuron is locked we take what is there, exactly as
        # `temp_property` does: the lock is held across operations that
        # legitimately leave the data inconsistent halfway through, and
        # re-deriving from it is both wrong and (this being a hash of every
        # vertex) not cheap.
        if self.is_locked and '_skeleton' in self.__dict__:
            return self._skeleton

        if not schema.target_is_current(self, self._skeleton_link):
            self.skeleton = self.skeletonize()
        return self._skeleton

    @skeleton.setter
    def skeleton(self, s):
        """Attach skeleton respresentation for this neuron."""
        if isinstance(s, (sk.Skeleton, sparsecubes.Skeleton)):
            s = Skeleton(s, id=self.id, name=self.name)
        elif not isinstance(s, Skeleton):
            raise TypeError(f'`.skeleton` must be a Skeleton, got "{type(s)}"')
        self._skeleton = s
        # Stamp the link, i.e. record that this skeleton describes the mesh as
        # it is now. A skeleton without a vertex map gets one too: there is no
        # correspondence to carry, but it is still *this* mesh's skeleton until
        # the mesh changes.
        schema.stamp_link(self, self._skeleton_link)

    @property
    def _skeleton_link(self) -> Link:
        """The declaration tying our vertices to the skeleton's nodes."""
        return schema.get_link(self, 'skeleton', source='vertices')

    @property
    def soma(self):
        """Not implemented for Meshes - use `.soma_pos`."""
        raise AttributeError("Meshes have a soma position (`.soma_pos`), not a soma.")

    @property
    def soma_pos(self):
        """X/Y/Z position of the soma.

        Returns `None` if no soma.
        """
        return getattr(self, '_soma_pos', None)

    @soma_pos.setter
    def soma_pos(self, value):
        """Set soma by position."""
        if value is None:
            self._soma_pos = None
            return

        try:
            value = np.asarray(value).astype(np.float64).reshape(3)
        except BaseException:
            raise ValueError(f'Unable to convert soma position "{value}" '
                             f'to numeric (3, ) numpy array.')

        self._soma_pos = value

    @property
    def type(self) -> str:
        """Neuron type."""
        return 'navis.Mesh'

    @property
    @temp_property
    def trimesh(self):
        """Trimesh representation of the neuron.

        Note that this is a `navis.utils.TrimeshPlus` - a `trimesh.Trimesh`
        that also carries this neuron's `.extra_edges`.

        """
        if not getattr(self, '_trimesh', None):
            self._trimesh = TrimeshPlus(vertices=self._vertices,
                                        faces=self._faces,
                                        process=False)
            if self.n_extra_edges:
                self._trimesh._extra_edges = self.extra_edges
        return self._trimesh

    def copy(self) -> 'Mesh':
        """Return a copy of the neuron."""
        no_copy = ['_lock']

        # Generate new neuron
        x = self.__class__(None)
        # Override with this neuron's data
        x.__dict__.update({k: copy.copy(v) for k, v in self.__dict__.items() if k not in no_copy})

        return x

    def snap(self, locs, to='vertices'):
        """Snap xyz location(s) to closest vertex or synapse.

        Parameters
        ----------
        locs :      (N, 3) array | (3, ) array
                    Either single or multiple XYZ locations.
        to :        "vertices" | "connectors"
                    Whether to snap to vertex or connector.

        Returns
        -------
        ix :        int | list of int
                    Index/indices of the closest vertex/connector.
        dist :      float | list of float
                    Distance(s) to the closest vertex/connector.

        Examples
        --------
        >>> import navis
        >>> n = navis.example_neurons(1, kind='mesh')
        >>> ix, dist = n.snap([0, 0, 0])
        >>> ix
        300

        """
        locs = np.asarray(locs).astype(self.vertices.dtype)

        is_single = (locs.ndim == 1 and len(locs) == 3)
        is_multi = (locs.ndim == 2 and locs.shape[1] == 3)
        if not is_single and not is_multi:
            raise ValueError('Expected a single (x, y, z) location or a '
                             '(N, 3) array of multiple locations')

        if to in ('vertices', 'vertex'):
            data = self.vertices
        elif to in ('connectors', 'connector'):
            if not self.has_connectors:
                raise ValueError('Neuron does not have connectors to snap to.')
            data = self.connectors[['x', 'y', 'z']].values
        else:
            raise ValueError('`to` must be "vertices" or "connectors", '
                             f'got {to}')

        # Generate tree
        tree = scipy.spatial.cKDTree(data=data)

        # Find the closest node
        dist, ix = tree.query(locs)

        return ix, dist

    def skeletonize(self, method='wavefront', heal=True, inv_dist=None, **kwargs) -> 'Skeleton':
        """Skeletonize mesh.

        See [`navis.conversion.mesh2skeleton`][] for details.

        Parameters
        ----------
        method :    "wavefront" | "teasar"
                    Method to use for skeletonization.
        heal :      bool
                    Whether to heal a fragmented skeleton after skeletonization.
        inv_dist :  int | float
                    Only required for method "teasar": invalidation distance for
                    the traversal. Smaller `inv_dist` captures smaller features
                    but is slower and vice versa. A good starting value is around
                    2-5 microns.
        **kwargs
                    Additional keyword are passed through to
                    [`navis.conversion.mesh2skeleton`][].

        Returns
        -------
        skeleton :  navis.Skeleton

        """
        return conversion.mesh2skeleton(self, method=method, heal=heal,
                                        inv_dist=inv_dist, **kwargs)

    def fill_holes(self, inplace=False):
        """Triangulate the holes in this mesh.

        See [`navis.fill_holes`][] for details.

        """
        return morpho.fill_holes(self, inplace=inplace)

    def validate(self, inplace=False):
        """Use trimesh to try and fix some common mesh issues.

        See [`navis.fix_mesh`][] for details.

        """
        return meshes.fix_mesh(self, inplace=inplace)

    def heal(self, inplace=False, **kwargs):
        """Heal fragmentation by bridging this mesh's components.

        Thin wrapper around [`navis.heal_mesh`][] - see there for `max_dist`,
        `min_size`, `keep_largest` and `mask`. The repair is topological: the
        bridges land in [`extra_edges`][navis.Mesh.extra_edges] and the surface
        is untouched.

        Returns
        -------
        Mesh
                    Only if `inplace=False`.

        """
        x = morpho.heal_mesh(self, inplace=inplace, **kwargs)

        if not inplace:
            return x

    def smooth(self, inplace=False, **kwargs):
        """Smooth this mesh.

        Thin wrapper around [`navis.smooth_mesh`][] - see there for
        `iterations`, `L`, `method` and `backend`.

        Returns
        -------
        Mesh
                    Only if `inplace=False`.

        """
        x = meshes.smooth_mesh(self, inplace=inplace, **kwargs)

        if not inplace:
            return x


# Pre-2.0 name. Must be a plain alias: `pickle` resolves classes by their
# defining module and has to find this one without a warning.
MeshNeuron = Mesh
