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


import pandas as pd
import numpy as np
import networkx as nx

from typing import Union, Sequence, Callable

from .. import utils, config, core, graph

# Set up logging
logger = config.get_logger(__name__)

__all__ = sorted(["subset_neuron"])


@utils.map_neuronlist(desc="Subsetting", allow_parallel=True)
@utils.lock_neuron
def subset_neuron(
    x: Union["core.TreeNeuron", "core.MeshNeuron"],
    subset: Union[Sequence[Union[int, str]], nx.DiGraph, pd.DataFrame, Callable],
    inplace: bool = False,
    keep_disc_cn: bool = False,
    prevent_fragments: bool = False,
) -> "core.NeuronObject":
    """Subset a neuron to a given set of nodes/vertices.

    Note that for `MeshNeurons` it is not guaranteed that all vertices in
    `subset` survive because we will also drop degenerate vertices that do
    not participate in any faces.

    Parameters
    ----------
    x :                   TreeNeuron | MeshNeuron | Dotprops | NeuronList
                          Neuron to subset. When passing a NeuronList, it's advised
                          to use a function for `subset` (see below).
    subset :              list-like | set | NetworkX.Graph | pandas.DataFrame | Callable
                          Subset of the neuron to keep. Depending on the neuron:
                            For TreeNeurons:
                             - node IDs
                             - a boolean mask matching the number of nodes
                             - DataFrame with `node_id` column
                            For MeshNeurons:
                             - vertex indices
                             - a boolean mask matching either the number of
                               vertices or faces
                            For Dotprops:
                             - point indices
                             - a boolean mask matching the number of points
                          Alternatively, you can pass a function that accepts
                          a neuron and returns a suitable `subset` as described
                          above. This is useful e.g. when wanting to subset a
                          list of neurons.
    keep_disc_cn :        bool, optional
                          If False, will remove disconnected connectors that
                          have "lost" their parent node/vertex.
    prevent_fragments :   bool, optional
                          If True, will add nodes/vertices to `subset`
                          required to keep neuron from fragmenting. Ignored for
                          `Dotprops`.
    inplace :             bool, optional
                          If False, a copy of the neuron is returned.

    Returns
    -------
    TreeNeuron | MeshNeuron | Dotprops | NeuronList

    Examples
    --------
    Subset skeleton to all branches with less than 10 nodes

    >>> import navis
    >>> # Get neuron
    >>> n = navis.example_neurons(1)
    >>> # Get all linear segments
    >>> segs = n.segments
    >>> # Get short segments
    >>> short_segs = [s for s in segs if len(s) <= 10]
    >>> # Flatten segments into list of nodes
    >>> nodes_to_keep = [n for s in short_segs for n in s]
    >>> # Subset neuron
    >>> n_short = navis.subset_neuron(n, subset=nodes_to_keep)

    Subset multiple neurons using a callable

    >>> import navis
    >>> nl = navis.example_neurons(2)
    >>> # Subset neurons to all leaf nodes
    >>> nl_end = navis.subset_neuron(
    ...     nl,
    ...     subset=lambda x: x.leafs.node_id
    ... )

    See Also
    --------
    [`navis.cut_skeleton`][]
            Cut neuron at specific points.
    [`navis.in_volume`][]
            To intersect a neuron with a volume (mesh).

    """
    if isinstance(x, core.NeuronList) and len(x) == 1:
        x = x[0]

    utils.eval_param(
        x, name="x", allowed_types=(core.TreeNeuron, core.MeshNeuron, core.Dotprops)
    )

    if callable(subset):
        subset = subset(x)

    # Make a copy of the neuron
    if not inplace:
        x = x.copy()
        # We have to run this in a separate function so that the lock is applied
        # to the copy
        subset_neuron(
            x,
            subset=subset,
            inplace=True,
            keep_disc_cn=keep_disc_cn,
            prevent_fragments=prevent_fragments,
        )
        return x

    if isinstance(x, core.TreeNeuron):
        x = _subset_treeneuron(
            x,
            subset=subset,
            keep_disc_cn=keep_disc_cn,
            prevent_fragments=prevent_fragments,
        )
    elif isinstance(x, core.MeshNeuron):
        x = _subset_meshneuron(
            x,
            subset=subset,
            keep_disc_cn=keep_disc_cn,
            prevent_fragments=prevent_fragments,
        )
    elif isinstance(x, core.Dotprops):
        x = _subset_dotprops(x, subset=subset, keep_disc_cn=keep_disc_cn)

    return x


def _subset_dotprops(x, subset, keep_disc_cn):
    """Subset Dotprops."""
    if not utils.is_iterable(subset):
        raise TypeError(
            "Can only subset Dotprops to list, set or "
            f'numpy.ndarray, not "{type(subset)}"'
        )

    subset = utils.make_iterable(subset)

    # Convert indices to mask
    if subset.dtype == bool:
        if subset.shape != (x.points.shape[0],):
            raise ValueError("Boolean mask must be of same length as points.")
        mask = subset
    else:
        mask = np.isin(np.arange(0, len(x.points)), subset)

    # Filter connectors
    if not keep_disc_cn and x.has_connectors:
        if "point" not in x.connectors.columns:
            x.connectors["point"] = x.snap(x.connectors[["x", "y", "z"]].values)[0]

        if subset.dtype == bool:
            subset = np.arange(0, len(x.points))[subset]

        x._connectors = x.connectors[x.connectors.point.isin(subset)].copy()
        x._connectors.reset_index(inplace=True, drop=True)

        # Make old -> new indices map
        new_ix = dict(zip(subset, np.arange(0, len(subset))))

        x.connectors["point"] = x.connectors.point.map(new_ix)

    # Mask vectors
    # This will also trigger re-calculation which is necessary for two reasons:
    # 1. Vectors will change if they have to be recalculated from
    #    the downsampled dotprops.
    # 2. There might not be enough points left after downsampling given the
    #    original k.
    if isinstance(x._vect, type(None)) and x.k:
        if x.n_points >= x.k:
            x.recalculate_tangents(k=x.k, inplace=True)
    x._vect = x._vect[mask]

    # Mask alphas if exists
    if not isinstance(x._alpha, type(None)):
        x._alpha = x._alpha[mask]

    # Finally mask points
    x._points = x._points[mask]

    return x


def _subset_meshneuron(x, subset, keep_disc_cn, prevent_fragments):
    """Subset MeshNeuron."""
    if not utils.is_iterable(subset):
        raise TypeError(
            "Can only subset MeshNeuron to list, set or "
            f'numpy.ndarray, not "{type(subset)}"'
        )

    subset = utils.make_iterable(subset)

    # Convert mask to vertex indices
    if subset.dtype == bool:
        if subset.shape[0] == x.vertices.shape[0]:
            subset = np.arange(len(x.vertices))[subset]
        elif subset.shape[0] == x.faces.shape[0]:
            # Translate face mask to vertex indices
            subset = np.unique(x.faces[subset])
        else:
            raise ValueError(
                "Boolean mask must be of same length as vertices or faces."
            )

    if prevent_fragments:
        # Generate skeleton
        sk = x.skeleton
        # Convert vertex IDs to node IDs
        subset_nodes = np.unique(x.skeleton.vertex_map[subset])
        # Find connected subgraph
        subset, _ = graph.connected_subgraph(x.skeleton, subset_nodes)
        # Convert node IDs back to vertex IDs
        subset = np.arange(0, len(x.vertices))[np.isin(sk.vertex_map, subset)]

    # Filter connectors
    # (connectors are associated with vertices, not faces which is why
    # our `subset` is always a list of vertex indices)
    if not keep_disc_cn and x.has_connectors:
        if "vertex_id" not in x.connectors.columns:
            x.connectors["vertex"] = x.snap(x.connectors[["x", "y", "z"]].values)[0]

        x._connectors = x.connectors[x.connectors.vertex.isin(subset)].copy()
        x._connectors.reset_index(inplace=True, drop=True)

        # Make old -> new indices map
        new_ix = dict(zip(subset, np.arange(0, len(subset))))

        x.connectors["vertex"] = x.connectors.vertex.map(new_ix)

    if len(subset):
        x.vertices, x.faces = submesh(x, vertex_index=subset)
    else:
        x.vertices, x.faces = np.empty((0, 3)), np.empty((0, 3))

    return x


def _subset_treeneuron(x, subset, keep_disc_cn, prevent_fragments):
    """Subset skeleton."""
    if isinstance(subset, (nx.DiGraph, nx.Graph)):
        subset = subset.nodes
    elif isinstance(subset, pd.DataFrame):
        subset = subset.node_id.values
    elif utils.is_iterable(subset):
        # This forces subset into numpy array (important for e.g. sets)
        subset = utils.make_iterable(subset)
    else:
        raise TypeError(
            "Can only subset to list, set, numpy.ndarray or"
            f'networkx.Graph, not "{type(subset)}"'
        )

    if prevent_fragments:
        subset, new_root = graph.connected_subgraph(x, subset)
    else:
        new_root = None  # type: ignore # new_root has already type from before

    # Filter nodes
    # Note that we are setting the nodes directly (here and later) thereby
    # circumventing (or rather postponing) checks and safeguards.
    if isinstance(subset, np.ndarray) and subset.dtype == bool:
        # For boolean mask
        x._nodes = x._nodes.loc[subset]
    else:
        # For sets of nodes
        x._nodes = x.nodes[x.nodes.node_id.isin(subset)]

    # Make sure that there are root nodes
    # This is the fastest "pandorable" way: instead of overwriting the column,
    # concatenate a new column to this DataFrame
    x._nodes = pd.concat(
        [
            x.nodes.drop("parent_id", inplace=False, axis=1),  # type: ignore  # no stubs for concat
            x.nodes[["parent_id"]].where(
                x.nodes.parent_id.isin(x.nodes.node_id.values), other=-1, inplace=False
            ),
        ],
        axis=1,
    )

    # Make sure any new roots or leafs are properly typed
    # We won't produce new slabs but roots, branches and leaves might change
    graph.classify_nodes(x, inplace=True)

    # Filter connectors
    if not keep_disc_cn and x.has_connectors:
        x._connectors = x.connectors[x.connectors.node_id.isin(x.nodes.node_id)]
        x._connectors.reset_index(inplace=True, drop=True)

    if getattr(x, "tags", None) is not None:
        # Filter tags
        x.tags = {
            t: [tn for tn in x.tags[t] if tn in x.nodes.node_id.values] for t in x.tags
        }  # type: ignore  # TreeNeuron has no tags

        # Remove empty tags
        x.tags = {t: x.tags[t] for t in x.tags if x.tags[t]}  # type: ignore  # TreeNeuron has no tags

    # Fix graph representations (avoids having to recompute them)
    if "_graph_nx" in x.__dict__:
        x._graph_nx = x.graph.subgraph(x.nodes.node_id.values)
    if "_igraph" in x.__dict__:
        if x.igraph and config.use_igraph:
            id2ix = {
                n: ix
                for ix, n in zip(
                    x.igraph.vs.indices, x.igraph.vs.get_attribute_values("node_id")
                )
            }
            indices = [id2ix[n] for n in x.nodes.node_id.values]
            vs = x.igraph.vs[indices]
            x._igraph = x.igraph.subgraph(vs)

    if hasattr(x, "_soma") and x._soma is not None:
        # Check if soma is still in the neuron
        if x._soma not in x.nodes.node_id.values:
            x._soma = None

    # Reset indices of data tables
    x.nodes.reset_index(inplace=True, drop=True)

    if new_root:
        x.reroot(new_root, inplace=True)

    return x


def submesh(mesh, *, faces_index=None, vertex_index=None):
    """Re-imlementation of trimesh.submesh that is faster for our use case.

    Notably we:
     - ignore normals (possibly needed) and visuals (definitely not needed)
     - allow only one set of faces to be passed
     - return vertices and faces instead of a new mesh
     - make as few copies as possible
     - allow passing vertex indices instead of faces

    This function is 5-10x faster than trimesh.submesh for our use case.
    Note that the speed of this function was never the bottleneck though,
    it's about the memory footprint.
    See https://github.com/navis-org/navis/issues/154.

    Parameters
    ----------
    mesh :          trimesh.Trimesh
                    Mesh to submesh.
    faces_index :   array-like
                    Indices of faces to keep.
    vertex_index :  array-like
                    Indices of vertices to keep.

    Returns
    -------
    vertices :  np.ndarray
                Vertices of submesh.
    faces :     np.ndarray
                Faces of submesh.

    """
    if faces_index is None and vertex_index is None:
        raise ValueError("Either `faces_index` or `vertex_index` must be provided.")
    elif faces_index is not None and vertex_index is not None:
        raise ValueError("Only one of `faces_index` or `vertex_index` can be provided.")

    # First check if we can return either an empty mesh or the original mesh right away
    if faces_index is not None:
        if len(faces_index) == 0:
            return np.array([]), np.array([])
        elif len(faces_index) == len(mesh.faces):
            if len(np.unique(faces_index)) == len(mesh.faces):
                return mesh.vertices.copy(), mesh.faces.copy()
    else:
        if len(vertex_index) == 0:
            return np.array([]), np.array([])
        elif len(vertex_index) == len(mesh.vertices):
            if len(np.unique(vertex_index)) == len(mesh.vertices):
                return mesh.vertices.copy(), mesh.faces.copy()

    # Use a view of the original data
    original_faces = mesh.faces.view(np.ndarray)
    original_vertices = mesh.vertices.view(np.ndarray)

    # If we're starting with vertices, find faces that contain at least one of our vertices
    # This way we will also make sure to drop unreferenced vertices
    if vertex_index is not None:
        faces_index = np.arange(len(original_faces))[
            np.isin(original_faces, vertex_index).all(axis=1)
        ]

    # Get unique vertices in the to-be-kept faces
    faces = original_faces[faces_index]
    unique = np.unique(faces.reshape(-1))

    # Generate a mask for the vertices
    # (using int32 here since we're unlikey to have more than 2B vertices)
    mask = np.arange(len(original_vertices), dtype=np.int32)

    # Remap the vertices to the new indices
    mask[unique] = np.arange(len(unique))

    # Grab the vertices in the order they are referenced
    vertices = original_vertices[unique].copy()

    # Remap the faces to the new vertex indices
    # (making a copy to allow `mask` to be garbage collected)
    faces = mask[faces].copy()

    return vertices, faces
