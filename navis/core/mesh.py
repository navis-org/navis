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
import trimesh as tm

from typing import Union, Optional

from .. import utils, config, meshes, conversion, graph, morpho
from .base import BaseNeuron
from .neuronlist import NeuronList
from .skeleton import TreeNeuron
from .core_utils import temp_property, add_units


try:
    import xxhash
except ModuleNotFoundError:
    xxhash = None


__all__ = ["MeshNeuron"]

# Set up logging
logger = config.get_logger(__name__)

# This is to prevent pint to throw a warning about numpy integration
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pint.Quantity([])


class MeshNeuron(BaseNeuron):
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
                     - `None` will initialize an empty MeshNeuron
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
    SUMMARY_PROPS = ["type", "name", "units", "n_vertices", "n_faces"]

    #: Attributes to be used when comparing two neurons.
    EQ_ATTRIBUTES = ["name", "n_vertices", "n_faces"]

    #: Temporary attributes that need clearing when neuron data changes
    TEMP_ATTR = ["_memory_usage", "_trimesh", "_skeleton", "_igraph", "_graph_nx"]

    #: Core data table(s) used to calculate hash
    CORE_DATA = ["vertices", "faces"]

    #: Property used to calculate length of neuron
    _LENGTH_DATA = "vertices"

    def __init__(
        self,
        x,
        units: Union[pint.Unit, str] = None,
        process: bool = True,
        validate: bool = False,
        **metadata,
    ):
        """Initialize Mesh Neuron."""
        super().__init__()

        # Lock neuron during initialization
        self._lock = 1
        self._trimesh = None  # this is required to avoid recursion during init

        if isinstance(x, MeshNeuron):
            self.__dict__.update(x.copy().__dict__)
            self.vertices, self.faces = x.vertices, x.faces
        elif hasattr(x, "faces") and hasattr(x, "vertices"):
            self.vertices, self.faces = x.vertices, x.faces
        elif isinstance(x, dict):
            if "faces" not in x or "vertices" not in x:
                raise ValueError('Dictionary must contain "vertices" and "faces"')
            self.vertices, self.faces = x["vertices"], x["faces"]
        elif isinstance(x, str) and os.path.isfile(x):
            m = tm.load(x)
            self.vertices, self.faces = m.vertices, m.faces
        elif isinstance(x, type(None)):
            # Empty neuron
            self.vertices, self.faces = np.zeros((0, 3)), np.zeros((0, 3))
        elif isinstance(x, sk.Skeleton):
            self.vertices, self.faces = x.mesh.vertices, x.mesh.faces
            self._skeleton = TreeNeuron(x)
        elif isinstance(x, tuple):
            if len(x) != 2 or any([not isinstance(v, np.ndarray) for v in x]):
                raise TypeError("Expect tuple to be two arrays: (vertices, faces)")
            self.vertices, self.faces = x[0], x[1]
        else:
            raise utils.ConstructionError(
                f'Unable to construct MeshNeuron from "{type(x)}"'
            )

        for k, v in metadata.items():
            try:
                setattr(self, k, v)
            except AttributeError:
                raise AttributeError(f"Unable to set neuron's `{k}` attribute.")

        if process and self.vertices.shape[0]:
            # For some reason we can't do self._trimesh at this stage
            _trimesh = tm.Trimesh(
                self.vertices, self.faces, process=process, validate=validate
            )
            self.vertices = _trimesh.vertices
            self.faces = _trimesh.faces

        self._lock = 0

        if validate:
            self.validate()

        self.units = units

    def __getattr__(self, key):
        """We will use this magic method to calculate some attributes on-demand."""
        # Note that we're mixing @property and __getattr__ which causes problems:
        # if a @property raises an Exception, Python falls back to __getattr__
        # and traceback is lost!

        # Last ditch effort - maybe the base class knows the key?
        return super().__getattr__(key)

    def __getstate__(self):
        """Get state (used e.g. for pickling)."""
        state = {k: v for k, v in self.__dict__.items() if not callable(v)}

        # We don't need the trimesh object
        if "_trimesh" in state:
            _ = state.pop("_trimesh")

        return state

    def __setstate__(self, d):
        """Update state (used e.g. for pickling)."""
        self.__dict__.update(d)

    def __truediv__(self, other, copy=True):
        """Implement division for coordinates (vertices, connectors)."""
        if isinstance(other, numbers.Number) or utils.is_iterable(other):
            # If a number, consider this an offset for coordinates
            n = self.copy() if copy else self
            _ = np.divide(n.vertices, other, out=n.vertices, casting="unsafe")
            if n.has_connectors:
                n.connectors.loc[:, ["x", "y", "z"]] /= other

            # Convert units
            # Note: .to_compact() throws a RuntimeWarning and returns unchanged
            # values  when `units` is a iterable
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                n.units = (n.units * other).to_compact()

            self._clear_temp_attr()

            return n
        return NotImplemented

    def __mul__(self, other, copy=True):
        """Implement multiplication for coordinates (vertices, connectors)."""
        if isinstance(other, numbers.Number) or utils.is_iterable(other):
            # If a number, consider this an offset for coordinates
            n = self.copy() if copy else self
            _ = np.multiply(n.vertices, other, out=n.vertices, casting="unsafe")
            if n.has_connectors:
                n.connectors.loc[:, ["x", "y", "z"]] *= other

            # Convert units
            # Note: .to_compact() throws a RuntimeWarning and returns unchanged
            # values  when `units` is a iterable
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                n.units = (n.units / other).to_compact()

            self._clear_temp_attr()

            return n
        return NotImplemented

    def __add__(self, other, copy=True):
        """Implement addition for coordinates (vertices, connectors)."""
        if isinstance(other, numbers.Number) or utils.is_iterable(other):
            n = self.copy() if copy else self
            _ = np.add(n.vertices, other, out=n.vertices, casting="unsafe")
            if n.has_connectors:
                n.connectors.loc[:, ["x", "y", "z"]] += other

            self._clear_temp_attr()

            return n
        # If another neuron, return a list of neurons
        elif isinstance(other, BaseNeuron):
            return NeuronList([self, other])
        return NotImplemented

    def __sub__(self, other, copy=True):
        """Implement subtraction for coordinates (vertices, connectors)."""
        if isinstance(other, numbers.Number) or utils.is_iterable(other):
            n = self.copy() if copy else self
            _ = np.subtract(n.vertices, other, out=n.vertices, casting="unsafe")
            if n.has_connectors:
                n.connectors.loc[:, ["x", "y", "z"]] -= other

            self._clear_temp_attr()

            return n
        return NotImplemented

    @property
    def bbox(self) -> np.ndarray:
        """Bounding box (includes connectors)."""
        mn = np.min(self.vertices, axis=0)
        mx = np.max(self.vertices, axis=0)

        if self.has_connectors:
            cn_mn = np.min(self.connectors[["x", "y", "z"]].values, axis=0)
            cn_mx = np.max(self.connectors[["x", "y", "z"]].values, axis=0)

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
            raise ValueError("Vertices must be 2-dimensional array")
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
            raise ValueError("Faces must be 2-dimensional array")
        self._faces = faces
        self._clear_temp_attr()

    @property
    @temp_property
    def igraph(self) -> "igraph.Graph":
        """iGraph representation of the vertex connectivity."""
        # If igraph does not exist, create and return
        if not hasattr(self, "_igraph"):
            # This also sets the attribute
            self._igraph = graph.neuron2igraph(self, raise_not_installed=False)
        return self._igraph

    @property
    @temp_property
    def graph(self) -> nx.DiGraph:
        """Networkx Graph representation of the vertex connectivity."""
        # If graph does not exist, create and return
        if not hasattr(self, "_graph_nx"):
            # This also sets the attribute
            self._graph_nx = graph.neuron2nx(self)
        return self._graph_nx

    @property
    def sampling_resolution(self) -> float:
        """Average distance between vertices."""
        return float(self.trimesh.edges_unique_length.mean())

    @property
    @add_units(compact=True, power=3)
    def volume(self) -> float:
        """Volume of the neuron.

        Calculated from the surface integral. Garbage if neuron is not
        watertight.

        """
        return float(self.trimesh.volume)

    @property
    @temp_property
    def skeleton(self) -> "TreeNeuron":
        """Skeleton representation of this neuron.

        Uses [`navis.conversion.mesh2skeleton`][].

        """
        if not hasattr(self, "_skeleton"):
            self._skeleton = self.skeletonize()
        return self._skeleton

    @skeleton.setter
    def skeleton(self, s):
        """Attach skeleton respresentation for this neuron."""
        if isinstance(s, sk.Skeleton):
            s = TreeNeuron(s, id=self.id, name=self.name)
        elif not isinstance(s, TreeNeuron):
            raise TypeError(f'`.skeleton` must be a TreeNeuron, got "{type(s)}"')
        self._skeleton = s

    @property
    def soma(self):
        """Not implemented for MeshNeurons - use `.soma_pos`."""
        raise AttributeError(
            "MeshNeurons have a soma position (`.soma_pos`), not a soma."
        )

    @property
    def soma_pos(self):
        """X/Y/Z position of the soma.

        Returns `None` if no soma.
        """
        return getattr(self, "_soma_pos", None)

    @soma_pos.setter
    def soma_pos(self, value):
        """Set soma by position."""
        if value is None:
            self._soma_pos = None
            return

        try:
            value = np.asarray(value).astype(np.float64).reshape(3)
        except BaseException:
            raise ValueError(
                f'Unable to convert soma position "{value}" '
                f"to numeric (3, ) numpy array."
            )

        self._soma_pos = value

    @property
    def type(self) -> str:
        """Neuron type."""
        return "navis.MeshNeuron"

    @property
    @temp_property
    def trimesh(self):
        """Trimesh representation of the neuron."""
        if not getattr(self, "_trimesh", None):
            if hasattr(self, "extra_edges"):
                # Only use TrimeshPlus if we actually need it
                # to avoid unnecessarily breaking stuff elsewhere
                self._trimesh = tm.Trimesh(
                    vertices=self._vertices, faces=self._faces, process=False
                )
                self._trimesh.extra_edges = self.extra_edges
            else:
                self._trimesh = tm.Trimesh(
                    vertices=self._vertices, faces=self._faces, process=False
                )
        return self._trimesh

    def copy(self) -> "MeshNeuron":
        """Return a copy of the neuron."""
        no_copy = ["_lock"]

        # Generate new neuron
        x = self.__class__(None)
        # Override with this neuron's data
        x.__dict__.update(
            {k: copy.copy(v) for k, v in self.__dict__.items() if k not in no_copy}
        )

        return x

    def view(self) -> "MeshNeuron":
        """Create a view of the neuron without copying data.

        Be aware that changes to the view may affect the original neuron!

        """
        no_copy = ["_lock"]

        # Generate new empty neuron
        x = self.__class__(None)

        # Override with this neuron's data
        x.__dict__.update({k: v for k, v in self.__dict__.items() if k not in no_copy})

        return x

    def mask(self, mask, inplace=False, copy=False):
        """Mask neuron with given mask.

        Parameters
        ----------
        mask :      np.ndarray
                    Mask to apply. Can be:
                     - 1D array with boolean values
                     - string with property name
                     - callable that accepts a neuron and returns a valid mask
                    The mask can be either for vertices or faces but will ultimately be
                    used to mask out faces. Vertices not participating in any face
                    will be removed regardless of the mask.
        inplace :   bool, optional
                    Whether to mask the neuron inplace.
        copy :      bool, optional
                    Whether to copy data (faces, vertices, etc.) after masking. This
                    is useful if you want to avoid accidentally modifying
                    the original nodes table.

        Returns
        -------
        self

        See Also
        --------
        [`MeshNeuron.is_masked`][navis.MeshNeuron.is_masked]
                    Returns True if neuron is masked.
        [`MeshNeuron.unmask`][navis.MeshNeuron.unmask]
                    Remove mask from neuron.
        [`navis.NeuronMask`][]
                    Context manager for masking neurons.

        """
        if self.is_masked:
            raise ValueError(
                "Neuron already masked! Layering multiple masks is currently not supported. "
                "Please either apply the existing mask or unmask first."
            )

        n = self
        if not inplace:
            n = self.view()

        if callable(mask):
            mask = mask(n)
        elif isinstance(mask, str):
            mask = getattr(n, mask)

        mask = np.asarray(mask)

        # Some checks
        if mask.dtype != bool:
            raise ValueError("Mask must be boolean array.")
        elif len(mask) not in (n.vertices.shape[0], n.faces.shape[0]):
            raise ValueError("Mask length does not match number of vertices or faces.")

        # Transate vertex mask to face mask
        if mask.shape[0] == n.vertices.shape[0]:
            vert_mask = mask
            face_mask = np.all(mask[n.faces], axis=1)

            # Apply mask
            verts_new, faces_new, vert_map, face_map = morpho.subset.submesh(
                n, vertex_index=np.where(vert_mask)[0], return_map=True
            )
        else:
            face_mask = mask
            vert_mask = np.zeros(n.vertices.shape[0], dtype=bool)
            vert_mask[np.unique(n.faces[face_mask])] = True

            # Apply mask
            verts_new, faces_new, vert_map, face_map = morpho.subset.submesh(
                n, faces_index=np.where(face_mask)[0], return_map=True
            )

        # The above will have likely dropped some vertices - we need to update the vertex mask
        vert_mask = np.zeros(n.vertices.shape[0], dtype=bool)
        vert_mask[np.where(vert_map != -1)[0]] = True

        # Track mask, vertices and faces before masking
        n._mask = face_mask  # mask is always the face mask
        n._masked_data = {}
        n._masked_data["_vertices"] = n._vertices
        n._masked_data["_faces"] = n._faces

        # Update vertices and faces
        n._vertices = verts_new
        n._faces = faces_new

        # See if we can mask the mesh's skeleton as well
        if hasattr(n, "_skeleton"):
            # If the skeleton has a vertex map, we can use it to mask the skeleton
            if hasattr(n._skeleton, "vertex_map"):
                # Generate a mask for the skeleton
                # (keep in mind vertex_map are node IDs, not indices)
                sk_mask = n._skeleton.nodes.node_id.isin(
                    n._skeleton.vertex_map[vert_mask]
                )

                # Apply mask
                n._skeleton.mask(sk_mask)

                # Last but not least: we need to update the vertex map
                # Track the old map. N.B. we're not adding this to
                # skeleton._masked_data since the remapping is done by
                # the MeshNeuron itself!
                n._skeleton._vertex_map_unmasked = n._skeleton.vertex_map

                # Subset the vertex map to the surviving mesh vertices
                # N.B. that the node IDs don't change when masking skeletons!
                n._skeleton.vertex_map = n._skeleton.vertex_map[vert_mask]
            # If the skeleton has no vertex map, we have to ditch it and
            # let it be regenerated when needed
            else:
                n._masked_data["_skeleton"] = n._skeleton
                n._skeleton = None  # Clear the skeleton

        # See if we need to mask any connectors as well
        if hasattr(n, "_connectors"):
            # Only mask if there is an actual "vertex_ind" or "face_ind" column
            if "vertex_ind" in n._connectors.columns:
                cn_mask = n._connectors.vertex_id.isin(np.where(vert_mask)[0])
            elif "face_ind" in n._connectors.columns:
                cn_mask = n._connectors.face_id.isin(np.where(face_mask)[0])
            else:
                cn_mask = None

            if cn_mask is not None:
                n._masked_data["_connectors"] = n._connectors
                n._connectors = n._connectors.loc[mask]
                if copy:
                    n._connectors = n._connectors.copy()

        # Check if we need to drop the soma position
        if hasattr(n, "soma_pos"):
            vid = n.snap(self.soma_pos, to="vertices")[0]
            if not vert_mask[vid]:
                n._masked_data["_soma_pos"] = n.soma_pos
                n.soma_pos = None

        # Clear temporary attributes but keep the skeleton since we already fixed that manually
        n._clear_temp_attr(exclude=["_skeleton"])

        return n

    def unmask(self, reset=True):
        """Unmask neuron.

        Returns the neuron to its original state before masking.

        Parameters
        ----------
        reset :     bool
                    Whether to reset the neuron to its original state before masking.
                    If False, edits made to the neuron after masking will be kept.

        Returns
        -------
        self

        See Also
        --------
        [`MeshNeuron.is_masked`][navis.MeshNeuron.is_masked]
                    Returns True if neuron is masked.
        [`MeshNeuron.mask`][navis.MeshNeuron.mask]
                    Mask neuron.
        [`navis.NeuronMask`][]
                    Context manager for masking neurons.

        """
        if not self.is_masked:
            raise ValueError("Neuron is not masked.")

        # First fix the skeleton (if it exists)
        skeleton = getattr(self, "_skeleton", None)
        if skeleton is not None:
            # If the skeleton is not masked, it was created after the masking
            # - in which case we have to throw it away because we can't recover
            # the full neuron state.
            if not skeleton.is_masked:
                skeleton = None
            else:
                # Unmask the skeleton as well
                skeleton.unmask(reset=reset)

                # Manually restore the vertex map
                # N.B. that any destructive action (e.g. twig pruning) may have
                # removed nodes from the skeleton. If that's the case, we can't
                # restore the vertex map and have to ditch it.
                if hasattr(skeleton, "_vertex_map_unmasked"):
                    skeleton.vertex_map = skeleton._vertex_map_unmasked

                # Important note: currently the skeleton gets ditched whenever the MeshNeuron
                # is stale. That's mainly because (a) functions modify the mesh but not
                # the vertex map and (b) re-generating the skeleton is usually cheap.
                # In the long run, we need to make sure the skeleton is always in sync
                # and not cleared unless that's explicitly requested.
                # I'm thinking something like a MeshNeuron.sync_skeleton() method that
                # can either sync the skeleton with the mesh or vice versa.

        if reset:
            # Unmask and reset (this will clear temporary attributes including the skeleton)
            super().unmask()
            if skeleton is not None:
                self.skeleton = skeleton
            return self

        # Regenerate the vertex mask from the stored face mask
        face_mask = self._mask
        vert_mask = np.zeros(len(self._masked_data["_vertices"]), dtype=bool)
        vert_mask[np.unique(self._masked_data["_faces"][face_mask])] = True

        # Generate a mesh for the masked-out data:
        # The mesh prior to masking
        pre_mesh = tm.Trimesh(
            self._masked_data["_vertices"], self._masked_data["_faces"]
        )
        # The vertices and faces that were masked out
        pre_vertices, pre_faces, vert_map, face_map = morpho.subset.submesh(
            pre_mesh, faces_index=np.where(~face_mask)[0], return_map=True
        )

        # Combine the two
        comb = tm.util.concatenate(
            [tm.Trimesh(self.vertices, self.faces), tm.Trimesh(pre_vertices, pre_faces)]
        )

        # Drop duplicate faces
        comb.update_faces(comb.unique_faces())

        # Merge vertices that are exactly the same
        comb.merge_vertices(digis=6)

        # Update the neuron
        self._vertices, self._faces = np.asarray(comb.vertices), np.asarray(comb.faces)

        del self._mask
        del self._masked_data

        self._clear_temp_attr()

        # Check if we can re-use the skeleton
        if skeleton is not None:
            # Check if the vertex map is still valid
            # Note to self: we could do some elaborate checks here to map old to
            # most likely new vertex / nodes but that's a bit overkill for now.
            if hasattr(skeleton, "vertex_map"):
                if skeleton.vertex_map.shape[0] != self._vertices.shape[0]:
                    skeleton = None
                elif skeleton.vertex_map.max() >= self._faces.shape[0]:
                    skeleton = None

        # If we still have a skeleton at this point, we can re-use it
        if skeleton is not None:
            self._skeleton = skeleton

        return self

    def snap(self, locs, to="vertices"):
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
        4134

        """
        locs = np.asarray(locs).astype(self.vertices.dtype)

        is_single = locs.ndim == 1 and len(locs) == 3
        is_multi = locs.ndim == 2 and locs.shape[1] == 3
        if not is_single and not is_multi:
            raise ValueError(
                "Expected a single (x, y, z) location or a "
                "(N, 3) array of multiple locations"
            )

        if to not in ("vertices", "vertex", "connectors", "connectors"):
            raise ValueError('`to` must be "vertices" or "connectors", ' f"got {to}")

        # Generate tree
        tree = scipy.spatial.cKDTree(data=self.vertices)

        # Find the closest node
        dist, ix = tree.query(locs)

        return ix, dist

    def skeletonize(
        self, method="wavefront", heal=True, inv_dist=None, **kwargs
    ) -> "TreeNeuron":
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
        skeleton :  navis.TreeNeuron
                    Has a `.vertex_map` attribute that maps each vertex in the
                    input mesh to a skeleton node ID.

        """
        return conversion.mesh2skeleton(
            self, method=method, heal=heal, inv_dist=inv_dist, **kwargs
        )

    def validate(self, inplace=False):
        """Use trimesh to try and fix some common mesh issues.

        See [`navis.fix_mesh`][] for details.

        """
        return meshes.fix_mesh(self, inplace=inplace)
