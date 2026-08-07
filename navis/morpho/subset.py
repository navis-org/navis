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
from ..core import schema
from . import caps

# Set up logging
logger = config.get_logger(__name__)

__all__ = sorted(["subset_neuron", "merge_subset"])


def _skip_connectors(keep_disc_cn):
    """Data to leave alone when the caller asked to keep disconnected connectors."""
    return ("_connectors",) if keep_disc_cn else ()


def _snap_connectors(x, axis, keep_disc_cn):
    """Make sure connectors say which element they sit on, before we renumber.

    Meshes and dotprops locate connectors by index, and that index is only
    computed on demand - so a neuron can reach here with connectors that have no
    column tying them to a vertex/point at all. Snapping now is the last moment
    the indices still mean what they say.
    """
    if keep_disc_cn or not x.has_connectors:
        return

    # `connector_link` names its far end after the axis it points at, so the
    # link into this axis is the one to look for. Through `get_link` so that a
    # second link a user attached under the same name is an error rather than a
    # coin toss over which one we snap.
    try:
        link = schema.get_link(x, axis.name, source="connectors")
    except KeyError:
        return
    if link.column in x.connectors.columns:
        return

    x.connectors[link.column] = x.snap(x.connectors[["x", "y", "z"]].values)[0]
    # The mapping exists as of now, so say so - otherwise nothing downstream
    # would trust it enough to compose across it.
    schema.stamp_link(x, link)


@utils.map_neuronlist(desc="Subsetting", allow_parallel=True)
@utils.lock_neuron
def subset_neuron(
    x: Union["core.Skeleton", "core.Mesh"],
    subset: Union[Sequence[Union[int, str]], nx.DiGraph, pd.DataFrame, Callable],
    inplace: bool = False,
    keep_disc_cn: bool = False,
    prevent_fragments: bool = False,
    cap_holes: bool = False,
    track: bool = False,
) -> "core.NeuronObject":
    """Subset a neuron to a given set of nodes/vertices.

    Note that for `Meshes` it is not guaranteed that all vertices in
    `subset` survive because we will also drop degenerate vertices that do
    not participate in any faces.

    Parameters
    ----------
    x :                   Skeleton | Mesh | Dotprops | NeuronList
                          Neuron to subset. When passing a NeuronList, it's advised
                          to use a function for `subset` (see below).
    subset :              list-like | set | NetworkX.Graph | pandas.DataFrame | Callable
                          Subset of the neuron to keep. Depending on the neuron:
                            For Skeletons:
                             - node IDs
                             - a boolean mask matching the number of nodes
                             - DataFrame with `node_id` column
                            For Meshes:
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
    cap_holes :           bool, optional
                          `Meshes` only: if True, triangulate the openings the
                          cut leaves behind instead of returning a mesh with
                          holes in it. Only openings this call created are
                          closed - any the mesh already had are left alone. No
                          vertices are added, so all existing indices stand.
    inplace :             bool, optional
                          If False, a copy of the neuron is returned.
    track :               bool, optional
                          If True, record where each surviving element came
                          from so that edits made to the subset can later be
                          folded back with [`navis.merge_subset`][]. Costs an
                          extra array the size of the subset.

    Returns
    -------
    Skeleton | Mesh | Dotprops | NeuronList

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
        x, name="x", allowed_types=(core.Skeleton, core.Mesh, core.Dotprops)
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
            cap_holes=cap_holes,
            track=track,
        )
        return x

    # Capture the parent's identity *before* we mutate it - `x` is the neuron
    # being subsetted, so after this there is nothing left to describe
    parent = (x.id, x.core_md5) if track else None

    if isinstance(x, core.Skeleton):
        x, axis, survivors = _subset_treeneuron(
            x,
            subset=subset,
            keep_disc_cn=keep_disc_cn,
            prevent_fragments=prevent_fragments,
        )
    elif isinstance(x, core.Mesh):
        x, axis, survivors = _subset_meshneuron(
            x,
            subset=subset,
            keep_disc_cn=keep_disc_cn,
            prevent_fragments=prevent_fragments,
            cap_holes=cap_holes,
        )
    elif isinstance(x, core.Dotprops):
        x, axis, survivors = _subset_dotprops(
            x, subset=subset, keep_disc_cn=keep_disc_cn
        )

    if track:
        schema.record_provenance(x, parent[0], parent[1], axis, survivors)

    # N.B. links carried above are stamped by `lock_neuron` on the way out, once
    # everything that moves has stopped - `_subset_treeneuron` reroots *after*
    # selecting, which moves the data an epoch is taken from.
    return x


@utils.lock_neuron
def merge_subset(
    x: "core.NeuronObject",
    subset: "core.NeuronObject",
    inplace: bool = False,
) -> "core.NeuronObject":
    """Fold a tracked subset back into the neuron it was taken from.

    Elements that were not part of the subset are kept as they are; elements
    that were come back in whatever state the subset left them - edited, or
    gone. This is what makes "work on part of a neuron, then put it back" a
    single operation rather than a bespoke stitching job each time.

    Parameters
    ----------
    x :         Skeleton | Mesh | Dotprops
                The neuron `subset` was taken from.
    subset :    Skeleton | Mesh | Dotprops
                A subset produced by `subset_neuron(..., track=True)`, possibly
                since edited.
    inplace :   bool, optional
                If False, a copy of `x` is returned.

    Returns
    -------
    Skeleton | Mesh | Dotprops

    Raises
    ------
    navis.core.schema.MergeError
                If `subset` carries no provenance, came from a different neuron,
                or `x` has been modified since the subset was taken. Refusing is
                deliberate: there is no way to tell a safe merge from a wrong one
                once the neuron it was mapped against has moved.

    Examples
    --------
    Prune twigs on the axon only, then put the axon back

    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> _ = navis.split_axon_dendrite(n, label_only=True)
    >>> axon = navis.subset_neuron(n, n.nodes.compartment == 'axon', track=True)
    >>> axon = navis.prune_twigs(axon, 5000)
    >>> merged = navis.merge_subset(n, axon)
    >>> merged.n_nodes < n.n_nodes
    True

    See Also
    --------
    [`navis.subset_neuron`][]
            Produces the subset in the first place.

    """
    utils.eval_param(
        x, name="x", allowed_types=(core.Skeleton, core.Mesh, core.Dotprops)
    )
    utils.eval_param(
        subset,
        name="subset",
        allowed_types=(core.Skeleton, core.Mesh, core.Dotprops),
    )

    prov = schema.check_provenance(x, subset)

    if not inplace:
        x = x.copy()

    for axis_name, origin in prov.origin.items():
        schema.merge_selection(
            x,
            subset,
            schema.get_axis(x, axis_name),
            np.asarray(origin),
            np.asarray(prov.covered[axis_name]),
        )

    if isinstance(x, core.Skeleton):
        # Roots, branch points and leaves have all potentially moved
        graph.classify_nodes(x, inplace=True)

    x._clear_temp_attr()

    return x


def _subset_dotprops(x, subset, keep_disc_cn):
    """Subset Dotprops."""
    if not utils.is_iterable(subset):
        raise TypeError(
            "Can only subset Dotprops to list, set or "
            f'numpy.ndarray, not "{type(subset)}"'
        )

    axis = schema.get_axis(x, "points")
    keep = schema.resolve_selection(x, axis, subset)
    _snap_connectors(x, axis, keep_disc_cn)

    # Vectors have to be materialised *before* we subset, for two reasons:
    # 1. Recalculating them afterwards would use the subsetted points and give
    #    different answers.
    # 2. There might not be enough points left for the original `k`.
    if isinstance(x._vect, type(None)) and x.k:
        if x.n_points >= x.k:
            x.recalculate_tangents(k=x.k, inplace=True)

    survivors = schema.apply_selection(
        x, axis, keep, skip=_skip_connectors(keep_disc_cn)
    )

    return x, axis, survivors


def _subset_meshneuron(x, subset, keep_disc_cn, prevent_fragments, cap_holes=False):
    """Subset Mesh."""
    if not utils.is_iterable(subset):
        raise TypeError(
            "Can only subset Mesh to list, set or "
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
        # Generate skeleton. Note we hold on to it rather than going back
        # through `x.skeleton`, which re-checks the link (and so re-hashes every
        # vertex) on each access.
        sk = x.skeleton
        # Convert vertex IDs to node IDs
        subset_nodes = np.unique(sk.vertex_map[subset])
        # Find connected subgraph
        subset, _ = graph.connected_subgraph(sk, subset_nodes)
        # Convert node IDs back to vertex IDs
        subset = np.flatnonzero(np.isin(sk.vertex_map, subset))

    axis = schema.get_axis(x, "vertices")
    _snap_connectors(x, axis, keep_disc_cn)
    n_old = len(x.vertices)

    # Take this before `submesh` moves the vertices: whether a link can still be
    # followed is a question about the neuron we are about to stop having.
    live_links = schema.snapshot_links(x, axis)

    # Drop the extra edges *before* touching the vertices: the `.vertices`
    # setter would otherwise (rightly) warn about dropping them itself. They go
    # back in un-remapped below and are repaired along with everything else.
    extra_edges = getattr(x, "_extra_edges", None)
    x._extra_edges = None

    # Where the cut is about to open the mesh up has to be worked out while the
    # original faces are still here. It is only the collar around the cut that
    # gets looked at, so this is cheap - see `morpho.caps`.
    exposed = None
    if cap_holes and len(subset):
        dropped = np.ones(n_old, dtype=bool)
        dropped[subset] = False
        exposed = utils.fastcore.exposed_halfedges(np.asarray(x.faces), dropped)

    # `submesh` does the vertex/face subsetting itself - it resolves which faces
    # survive and drops degenerate vertices, so only it knows which vertices
    # *actually* made it through.
    # ... which means assigning through the public setter, whose orphan check
    # cannot tell this from a caller replacing the vertices outright. It is
    # `apply_selection` below that carries what is aligned to them.
    with schema.replacing(x, "vertices"):
        if len(subset):
            x.vertices, x.faces, kept = submesh(x, vertex_index=subset, return_map=True)
        else:
            x.vertices, x.faces = np.empty((0, 3)), np.empty((0, 3))
            kept = np.array([], dtype=int)

    x._extra_edges = extra_edges

    # `submesh` did the data subsetting and knows which vertices really made it
    # through; `_faces` it already remapped itself, hence skipped here.
    survivors = schema.apply_selection(
        x,
        axis,
        survivors=schema.Survivors.from_kept(n_old, kept),
        # `submesh` remapped the faces itself; the vertices go without saying,
        # since passing `survivors` is what says them.
        skip=("_faces",) + _skip_connectors(keep_disc_cn),
        links=live_links,
    )

    # Capping only ever *adds* faces, so it can happen last: every vertex index
    # handed out above - including the provenance `survivors` - still stands.
    if exposed is not None and len(exposed):
        renumber = np.full(n_old, -1, dtype=np.int64)
        renumber[kept] = np.arange(len(kept))
        exposed = renumber[exposed]
        exposed = exposed[(exposed >= 0).all(axis=1)]
        if len(exposed):
            new_faces = caps.cap_boundary(x.vertices, exposed)
            if len(new_faces):
                x.faces = np.vstack((x.faces, new_faces.astype(x.faces.dtype)))

    return x, axis, survivors


def _subset_treeneuron(x, subset, keep_disc_cn, prevent_fragments):
    """Subset skeleton."""
    # Everything else - node IDs, boolean masks, sets, DataFrames - is handled
    # by `resolve_selection` off the axis' declared ID column
    if isinstance(subset, (nx.DiGraph, nx.Graph)):
        subset = np.array(list(subset.nodes))
    elif not isinstance(subset, pd.DataFrame) and not utils.is_iterable(subset):
        raise TypeError(
            "Can only subset to list, set, numpy.ndarray or"
            f'networkx.Graph, not "{type(subset)}"'
        )

    axis = schema.get_axis(x, "nodes")
    keep = schema.resolve_selection(x, axis, subset)

    if prevent_fragments:
        # `connected_subgraph` speaks node IDs, so resolve, expand, re-resolve
        subset, new_root = graph.connected_subgraph(
            x, schema.axis_ids(x, axis)[keep]
        )
        keep = schema.resolve_selection(x, axis, subset)
    else:
        new_root = None  # type: ignore # new_root has already type from before

    # Subset the node table, then repair everything pointing into it: parent IDs
    # of nodes whose parent was dropped (they become roots), connectors, tags
    # and the soma. Note this sets `_nodes` directly, circumventing (or rather
    # postponing) the checks and safeguards of the `.nodes` setter.
    survivors = schema.apply_selection(
        x, axis, keep, skip=_skip_connectors(keep_disc_cn)
    )

    # Make sure any new roots or leafs are properly typed
    # We won't produce new slabs but roots, branches and leaves might change
    graph.classify_nodes(x, inplace=True)

    # Fix graph representations (avoids having to recompute them)
    if "_graph_nx" in x.__dict__:
        x._graph_nx = x.graph.subgraph(x.nodes.node_id.values)
    if "_igraph" in x.__dict__:
        id2ix = {
            n: ix
            for ix, n in zip(
                x.igraph.vs.indices, x.igraph.vs.get_attribute_values("node_id")
            )
        }
        indices = [id2ix[n] for n in x.nodes.node_id.values]
        vs = x.igraph.vs[indices]
        x._igraph = x.igraph.subgraph(vs)

    if new_root:
        x.reroot(new_root, inplace=True)

    return x, axis, survivors


def submesh(mesh, *, faces_index=None, vertex_index=None, return_map=False):
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
    return_map :    bool
                    If True, also return the indices of the original vertices
                    that survived, in the order they appear in the new mesh.

    Returns
    -------
    vertices :  np.ndarray
                Vertices of submesh.
    faces :     np.ndarray
                Faces of submesh.
    kept :      np.ndarray
                Only if `return_map=True`: index into the *original* vertices
                for each vertex of the submesh.

    """
    if faces_index is None and vertex_index is None:
        raise ValueError("Either `faces_index` or `vertex_index` must be provided.")
    elif faces_index is not None and vertex_index is not None:
        raise ValueError("Only one of `faces_index` or `vertex_index` can be provided.")

    def _return(vertices, faces, kept=None):
        if not return_map:
            return vertices, faces
        if kept is None:
            kept = np.arange(len(vertices))
        return vertices, faces, kept

    # First check if we can return either an empty mesh or the original mesh right away
    if faces_index is not None:
        if len(faces_index) == 0:
            return _return(np.array([]), np.array([]), np.array([], dtype=int))
        elif len(faces_index) == len(mesh.faces):
            if len(np.unique(faces_index)) == len(mesh.faces):
                return _return(mesh.vertices.copy(), mesh.faces.copy())
    else:
        if len(vertex_index) == 0:
            return _return(np.array([]), np.array([]), np.array([], dtype=int))
        elif len(vertex_index) == len(mesh.vertices):
            if len(np.unique(vertex_index)) == len(mesh.vertices):
                return _return(mesh.vertices.copy(), mesh.faces.copy())

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

    return _return(vertices, faces, unique)
