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

import numbers

from collections import defaultdict

import igraph
import numpy as np
import pandas as pd
import sparsecubes
import trimesh as tm
import networkx as nx

from typing_extensions import Literal
from typing import Union, Optional, List, Tuple, Sequence, Dict, Set, overload, Iterable

from scipy.special import softmax
from scipy.sparse import csgraph, coo_matrix, csr_matrix, diags

from .. import graph, utils, config, core, morpho

# Set up logging
logger = config.get_logger(__name__)

# The node types `classify_nodes` assigns, in the order of their categorical codes
NODE_TYPES = ["end", "branch", "root", "slab"]

# Fastcore classifies into 0=root, 1=leaf, 2=branch, 3=slab - use this to translate
# its output into the codes of `NODE_TYPES`
_FASTCORE_NODE_TYPES = np.array(
    [NODE_TYPES.index(t) for t in ("root", "end", "branch", "slab")], dtype=np.int8
)

__all__ = sorted(
    [
        "classify_nodes",
        "connected_components",
        "connecting_nodes",
        "cut_skeleton",
        "longest_neurite",
        "split_neurites",
        "reroot_skeleton",
        "distal_to",
        "dist_between",
        "find_main_branchpoint",
        "generate_list_of_childs",
        "geodesic_matrix",
        "node_label_sorting",
        "segment_length",
        "rewire_skeleton",
        "insert_nodes",
        "remove_nodes",
        "collapse_nodes",
        "dist_to_root",
        "skeleton_adjacency_matrix",
        "propagate_labels",
    ]
)


@utils.map_neuronlist(desc="Gen. segments", allow_parallel=True)
def _generate_segments(
    x: "core.NeuronObject", weight: Optional[str] = None, return_lengths: bool = False
) -> Union[list, Tuple[list, list]]:
    """Generate segments maximizing segment lengths.

    Isolated nodes will be included as segments of length 0.

    Parameters
    ----------
    x :         Skeleton | NeuronList
                May contain multiple neurons.
    weight :    'weight' | None, optional
                If `"weight"` use physical, geodesic length to determine
                segment length. If `None` use number of nodes (faster).
    return_lengths : bool
                If True, also return lengths of segments according to `weight`.

    Returns
    -------
    segments :  list
                Segments as list of lists containing node IDs. List is
                sorted by segment lengths.
    lengths :   list
                Length for each segment according to `weight`. Only provided
                if `return_lengths` is True.

    Examples
    --------
    This is primarily for doctests:

    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> unweighted = navis.graph_utils._generate_segments(n)
    >>> weighted = navis.graph_utils._generate_segments(n, weight='weight')

    """
    if not isinstance(x, core.Skeleton):
        raise ValueError(f'Expected Skeleton, got "{type(x)}"')

    # At this point x is Skeleton
    x: core.Skeleton

    assert weight in ("weight", None), f'Unable to use weight "{weight}"'

    if weight == "weight":
        weight = morpho.mmetrics.parent_dist(x, root_dist=0)

    segs, lengths = utils.fastcore.generate_segments(
        x.nodes.node_id.values, x.nodes.parent_id.values, weights=weight
    )

    if return_lengths:
        return segs, lengths
    else:
        return segs


def _compress(
    labels: np.ndarray, where: Optional[np.ndarray], n: int
) -> Tuple[np.ndarray, int]:
    """Renumber `labels` to a contiguous `0 .. k - 1` and place them in an array.

    The union-find primitives all label a component by its smallest member's
    index, which is neither contiguous nor - once a mask has taken elements out
    of the running - free of gaps. This is the one way back.

    Parameters
    ----------
    labels :    (M, ) array
                Labels of the elements that are *in* a component.
    where :     (M, ) index or bool array, optional
                Where those elements sit in the full array. `None` means they
                already are the full array.
    n :         int
                Length of the full array. Ignored if `where` is `None`.

    Returns
    -------
    labels :    (N, ) int64 array
                `-1` wherever `where` left a gap.
    k :         int
                Number of distinct labels.

    """
    uniq, out = np.unique(labels, return_inverse=True)
    out = out.reshape(-1).astype(np.int64, copy=False)

    if where is None:
        return out, len(uniq)

    full = np.full(n, -1, dtype=np.int64)
    full[where] = out
    return full, len(uniq)


def _merge_labels(labels: np.ndarray, edges: Optional[np.ndarray]) -> np.ndarray:
    """Merge component labels that `edges` connect.

    Used to fold a mesh's extra (non-face) edges into a component labelling that
    was derived from the faces alone. We contract each component into a single
    node first, so the graph we actually run this on has one node per component
    (typically a handful) rather than one per vertex.

    Parameters
    ----------
    labels :    (N, ) array
                Component label for each node. Labels need not be contiguous.
    edges :     (M, 2) array, optional
                Additional edges as node indices. `None` or empty is a no-op.

    Returns
    -------
    labels :    (N, ) array
                Updated labels. Same as the input where nothing merged.

    """
    if edges is None or not len(edges):
        return labels

    # Contract to one node per component
    uniq, comp = np.unique(labels, return_inverse=True)
    comp = comp.reshape(-1)

    if len(uniq) == 1:
        return labels

    # N.B. scipy rather than fastcore here, deliberately: the relabelling below
    # needs *contiguous* `0..n-1` labels, which `csgraph` gives and fastcore's
    # "smallest member index" convention does not. This runs on the contracted
    # graph (one node per component, typically a handful), so there is nothing to
    # gain from switching and an extra relabel pass to lose.
    comp_edges = comp[np.asarray(edges)]
    adj = coo_matrix(
        (
            np.ones(len(comp_edges), dtype=np.int8),
            (comp_edges[:, 0], comp_edges[:, 1]),
        ),
        shape=(len(uniq), len(uniq)),
    ).tocsr()

    _, merged = csgraph.connected_components(adj, directed=False)

    # Relabel using the original labels so the output stays interchangeable with
    # the input (e.g. component labels remain "the smallest member's index" for
    # fastcore's mesh components)
    order = np.argsort(merged, kind="stable")
    _, first = np.unique(merged[order], return_index=True)
    representative = uniq[order[first]]

    return representative[merged[comp]]


# The readings of "connected" that label a mesh's faces rather than its vertices
# - see `_mesh_component_labels`
_FACE_CONNECTIVITIES = ("face", "manifold")


def _extra_edges_as_faces(
    faces: np.ndarray, edges: Optional[np.ndarray]
) -> np.ndarray:
    """Re-express extra (non-face) edges as edges between faces.

    Face components are made of faces, so a bridge between two *vertices* only
    means something to them once it is a bridge between the faces using those
    vertices: every face at one end joins every face at the other.

    Note this also welds together the face components that merely meet at an
    endpoint - at a pinch vertex, or (under `"manifold"`) along a seam. There is
    no labelling in which they stay apart while both connect to the far end, so
    this is the one case where face components come out coarser than the
    connectivity asked for.

    Parameters
    ----------
    faces :     (F, 3) array
    edges :     (M, 2) array, optional
                Extra edges as vertex indices. `None` or empty gives an empty
                result.

    Returns
    -------
    edges :     (K, 2) int array
                The same connections as face indices, ready for `_merge_labels`.
                Endpoints that no face uses drop out: under face connectivity a
                vertex on its own is not a component to begin with.

    """
    empty = np.zeros((0, 2), dtype=np.int64)

    if edges is None or not len(edges):
        return empty

    edges = np.asarray(edges)

    # The faces using each endpoint vertex, as (vertex, face) pairs sorted by vertex
    corner_vert = np.asarray(faces).reshape(-1)
    corner_face = np.repeat(np.arange(len(faces)), 3)
    used = np.isin(corner_vert, edges)
    order = np.argsort(corner_vert[used], kind="stable")
    vert, face = corner_vert[used][order], corner_face[used][order]

    if not len(vert):
        return empty

    # Pick one representative face per endpoint vertex, and tie that vertex's
    # other faces to it (the star) so that reaching the representative is enough
    uniq, start, counts = np.unique(vert, return_index=True, return_counts=True)
    rep = face[start]
    star = np.stack([np.repeat(rep, counts), face], axis=1)
    star = star[star[:, 0] != star[:, 1]]

    # ... which leaves one edge per bridge, between the two representatives
    pos = np.searchsorted(uniq, edges)
    known = uniq[np.minimum(pos, len(uniq) - 1)] == edges
    bridges = rep[pos[known.all(axis=1)]]

    return np.vstack((star, bridges)).astype(np.int64, copy=False)


def _mesh_component_labels(
    x: Union["core.Mesh", "tm.Trimesh"],
    connectivity: str = "vertex",
    keep: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int]:
    """Label each vertex (or face) with the connected component it belongs to.

    Unlike [`navis.connected_components`][] the labels here are contiguous but
    in no particular order - sorting them by size is that function's job.

    Note this labels the components of the *graph*: a mesh's extra edges (edges
    that are not part of any face) count as connections.

    Parameters
    ----------
    x :             Mesh | Trimesh
    connectivity :  "vertex" | "face" | "manifold" , optional
                    What two faces must share to count as connected: a corner
                    (`"vertex"`, default), any edge (`"face"`) or an edge
                    carrying exactly two faces (`"manifold"`). Each is strictly
                    finer than the one before it, dropping a kind of junction:
                    `"face"` drops the pinch points (two triangles meeting at a
                    corner are one component under `"vertex"` and two under
                    `"face"`), `"manifold"` also drops the seams (three sheets
                    along one edge are one component under `"face"` and three
                    under `"manifold"`). `"manifold"` is `trimesh.split`.

                    The latter two also change what the labels are *of*: a pinch
                    vertex belongs to several face components at once, so those
                    can only be reported per face.
    keep :          (N, ) bool array, optional
                    Boolean mask over *vertices* - always vertices, whatever the
                    labels end up being of, because vertices are what a mesh is
                    subset by. Restricts this to the induced sub-mesh: a face
                    survives only if all three of its corners do, and an extra
                    edge only if both its ends do. Elements left out come back
                    as `-1`.

    Returns
    -------
    labels :    (N, ) or (F, ) int array
                Component label for each vertex (`"vertex"`) or for each face
                (`"face"`/`"manifold"`), in vertex/face order. Labels are
                contiguous (`0 .. n_components - 1`) but otherwise arbitrary;
                `-1` marks an element `keep` left out.
    n :         int
                Number of connected components.

    """
    if connectivity not in ("vertex",) + _FACE_CONNECTIVITIES:
        raise ValueError(
            '`connectivity` must be "vertex", "face" or "manifold", '
            f'got "{connectivity}"'
        )

    # N.B. the dtype: an empty `Mesh` hands back a float64 `(0, 3)`, which would
    # make the boolean indexing below an IndexError rather than a no-op.
    faces = np.asarray(x.faces, dtype=np.int64)
    extra_edges = getattr(x, "extra_edges", None)

    # Inducing the sub-mesh is just dropping whatever crosses the mask: no
    # re-indexing, so the labels below still line up with the original vertices.
    face_ix = None
    if keep is not None:
        face_ix = np.flatnonzero(keep[faces].all(axis=1))
        faces = faces[face_ix]
        if extra_edges is not None and len(extra_edges):
            extra_edges = np.asarray(extra_edges)
            extra_edges = extra_edges[keep[extra_edges].all(axis=1)]

    if connectivity == "vertex":
        n_verts = len(x.vertices)

        if not n_verts:
            return np.zeros(0, dtype=np.int64), 0

        # This is a plain union-find over the faces - no adjacency is built
        labels = utils.fastcore.mesh_connected_components(faces, n_verts)  # type: ignore
    else:
        if not len(faces):
            # Nothing left to label. Without a mask this is an empty array (the
            # mesh has no faces); with one it is a face per -1 (all masked out).
            return np.full(len(x.faces), -1, dtype=np.int64), 0

        # Face adjacency does have to group the faces' edges first, which is
        # where all of its (still modest) extra time goes. The two face readings
        # differ only in what they then do with an edge more than two faces deep
        labels = utils.fastcore.mesh_connected_components(  # type: ignore
            faces, connectivity=connectivity
        )
        # The extra edges are given as vertices and mean nothing to face labels
        # until they are translated into the faces at either end
        extra_edges = _extra_edges_as_faces(faces, extra_edges)

    # Edges that are not part of any face (e.g. bridges added by
    # `navis.heal_mesh`) are invisible to the above and have to be merged in
    labels = _merge_labels(labels, extra_edges)

    if keep is None:
        return _compress(labels, None, len(labels))

    if connectivity == "vertex":
        # Masked-out vertices are in no face and so came back as singletons of
        # their own; they have to go before the labels are compressed
        return _compress(labels[keep], keep, len(labels))

    # Face labels are over the surviving faces only, so scatter them back
    return _compress(labels, face_ix, len(x.faces))


def skeleton_edges(x: "core.Skeleton"):
    """A skeleton's child -> parent edges as 0-based node *indices*.

    The fastcore graph primitives all work in index space (`0 .. n_nodes - 1`)
    off a plain edge list, whereas navis works in node IDs - so anything wiring
    one to the other needs this.

    Returns
    -------
    edges :     (E, 2) int64 array
                Roots have no parent and simply contribute no edge.
    node_ids :  (N, ) array
                The node ID for each index, i.e. the inverse mapping.

    """
    node_ids = x.nodes.node_id.values
    parent_ids = x.nodes.parent_id.values
    n_nodes = len(node_ids)

    if not n_nodes:
        return np.zeros((0, 2), dtype=np.int64), node_ids

    id2ix = pd.Series(np.arange(n_nodes), index=node_ids)
    par_ix = id2ix.reindex(parent_ids).values
    has_parent = ~np.isnan(par_ix)

    edges = np.stack(
        [np.arange(n_nodes)[has_parent], par_ix[has_parent].astype(np.int64)], axis=1
    )
    return edges, node_ids


def _resolve_connectivity(
    x: Union["core.BaseNeuron", "tm.Trimesh"],
    connectivity: Optional[Union[int, str]],
) -> Optional[Union[int, str]]:
    """Fill in and check the `connectivity` argument for a given neuron.

    What counts as "connected" is a different question for each type of neuron,
    so `connectivity` takes different values (and has a different default)
    depending on what it is handed - and none at all for skeletons and dotprops,
    which come with their edges already decided.

    Parameters
    ----------
    x :             Neuron
    connectivity :  int | str, optional
                    `None` picks the default for this type of neuron.

    Returns
    -------
    connectivity :  int | str | None
                    `None` for the neurons that have no use for it.

    """
    if isinstance(x, core.Voxels):
        allowed, default = (6, 18, 26), 26
    elif isinstance(x, (core.Mesh, tm.Trimesh)):
        allowed, default = ("vertex",) + _FACE_CONNECTIVITIES, "vertex"
    else:
        if connectivity is not None:
            logger.warning(
                f"`connectivity` does not apply to {type(x).__name__} and is "
                "ignored: the edges are given by the neuron itself."
            )
        return None

    if connectivity is None:
        return default

    if connectivity not in allowed:
        raise ValueError(
            f"`connectivity` for a {type(x).__name__} must be one of "
            f"{', '.join(repr(a) for a in allowed)}, got {connectivity!r}"
        )

    return connectivity


#: Per type of neuron: the elements a component labelling can be *of*, and the
#: `subset_neuron` axis that a `mask` selects. The first element is the one the
#: neuron is made of and the one the axis holds - only meshes have a second, and
#: `faces` is not an axis (a mesh is subset by vertices).
_ELEMENTS = {
    "Skeleton": (("node",), "nodes"),
    "Mesh": (("vertex", "face"), "vertices"),
    "Dotprops": (("point",), "points"),
    "Voxels": (("voxel",), "voxels"),
}


def _element_kind(x: Union["core.BaseNeuron", "tm.Trimesh"]) -> str:
    """The `_ELEMENTS` key for a neuron, i.e. its type modulo Trimesh."""
    if isinstance(x, (core.Mesh, tm.Trimesh)):
        return "Mesh"
    for kind in ("Skeleton", "Dotprops", "Voxels"):
        if isinstance(x, getattr(core, kind)):
            return kind
    raise TypeError(f"Neuron type {type(x).__name__} has no connected components")


def _resolve_mask(
    x: Union["core.BaseNeuron", "tm.Trimesh"], mask, axis_name: str
) -> Optional[np.ndarray]:
    """Normalise `mask` into a boolean array over the axis a neuron is subset by.

    `None` means "everything", and is returned as `None` rather than an
    all-`True` array so callers can skip the induced-subgraph work entirely -
    and so that an empty neuron, whose elements cannot even be counted, never
    has to answer.
    """
    if mask is None:
        return None

    # `Voxels` declares no axes (its `.voxels` are derived from a grid half the
    # time, so there is no attribute for an `Axis` to own) and a bare `Trimesh`
    # declares nothing at all - but `resolve_selection` only ever reads the axis'
    # length and its ID column, so a positional stand-in serves them both and
    # keeps one set of accepted mask forms for every type.
    axis = (
        core.schema.get_axis(x, axis_name)
        if isinstance(x, core.BaseNeuron) and axis_name in core.schema.declared_axes(x)
        else core.schema.Axis(name=axis_name, data=(axis_name,))
    )

    return np.asarray(core.schema.resolve_selection(x, axis, mask), dtype=bool)


def _sort_labels_by_size(
    labels: np.ndarray, n: int, masked: bool = True
) -> np.ndarray:
    """Relabel components largest-first.

    `labels` must be contiguous `0 .. n - 1`; the `-1` standing for "no
    component" passes through untouched. Ties are broken by the index of the
    component's first element, which makes the labelling a function of the
    neuron rather than of union-find's internals.

    `masked=False` promises there are no `-1`s - the common case, and worth a
    promise because it lets the counting skip a compaction of the whole array.
    """
    if n <= 1:
        return labels.astype(np.int64, copy=False)

    valid = np.flatnonzero(labels >= 0) if masked else None
    counts = np.bincount(labels if valid is None else labels[valid], minlength=n)

    # First occurrence of each label: scatter the indices in reverse so that the
    # earliest one is what remains. Only ties need it, and ties are rare, so the
    # `n log n` check over the (few) components buys skipping two passes over N.
    if len(np.unique(counts)) == n:
        order = np.argsort(-counts, kind="stable")
    else:
        first = np.full(n, len(labels), dtype=np.int64)
        if valid is None:
            first[labels[::-1]] = np.arange(len(labels) - 1, -1, -1)
        else:
            first[labels[valid[::-1]]] = valid[::-1]
        order = np.lexsort((first, -counts))

    # One entry past the end holds the `-1`s: they index it by wrapping round,
    # which is what lets this be a single gather rather than a gather plus a
    # `np.where` over the whole array.
    remap = np.empty(n + 1, dtype=np.int64)
    remap[order] = np.arange(n)
    remap[n] = -1

    return remap[labels]


def _voxel_component_labels(voxels: np.ndarray, connectivity: int) -> np.ndarray:
    """Label a set of sparse voxels, straight through `sparse-cubes`.

    Takes the voxels rather than the neuron so that a masked call can label a
    subset without building a neuron around it. Labels come back contiguous, so
    unlike the other types these need no compressing.

    Requires sparse-cubes >= 0.4.0.
    """
    _, labels = sparsecubes.measure.connected_components(
        voxels, connectivity=connectivity
    )
    return labels


def _natural_component_labels(
    x, connectivity, epsilon, keep: Optional[np.ndarray]
) -> Tuple[np.ndarray, int]:
    """Label a neuron's components over whichever element is natural to it.

    "Natural" means nodes for a skeleton, points for dotprops, voxels for a
    voxel neuron, and - because that is what `_mesh_component_labels` decides -
    vertices or faces for a mesh, depending on `connectivity`.

    `keep`, if given, restricts this to the *induced* sub-neuron: elements
    outside it come back as `-1` and, crucially, do not conduct. Each type gets
    there by dropping whatever carries connectivity (edges, faces, voxels)
    rather than by building a subset neuron, so nothing needs re-indexing.

    Returns
    -------
    labels :    (N, ) int array
                Contiguous `0 .. n - 1`, in no particular order, or `-1`.
    n :         int

    """
    if isinstance(x, core.Voxels):
        # `sparse-cubes` labels the components straight off the sparse voxels,
        # so this never builds a graph (or the dense grid). Its labels are
        # already contiguous, so they skip `_compress` and its sort.
        voxels = x.voxels if keep is None else x.voxels[keep]
        sub = _voxel_component_labels(voxels, connectivity)
        n = int(sub.max()) + 1 if len(sub) else 0

        if keep is None:
            return sub, n

        labels = np.full(len(x.voxels), -1, dtype=np.int64)
        labels[keep] = sub
        return labels, n

    if isinstance(x, (core.Mesh, tm.Trimesh)):
        return _mesh_component_labels(x, connectivity=connectivity, keep=keep)

    if isinstance(x, core.Skeleton):
        node_ids = x.nodes.node_id.values

        if keep is None:
            # This returns for each node the ID of its root, which is as good a
            # component label as any - just not a contiguous one
            roots = utils.fastcore.connected_components(
                node_ids, x.nodes.parent_id.values
            )
            return _compress(roots, None, len(node_ids))

        # Inducing the subgraph is just dropping every edge with an endpoint
        # outside the mask - no graph object needs building, which is what makes
        # this ~60x quicker than going via igraph.
        edges, _ = skeleton_edges(x)
        n_nodes = len(node_ids)
    else:
        # Dotprops: the edges are whatever `epsilon` says they are, so the graph
        # has to be built before anything can be dropped from it. Read them out
        # as an array rather than deleting edges on the graph - that would mean a
        # Python-level pass over every edge of a neighbourhood graph.
        G: igraph.Graph = graph.neuron2igraph(x, epsilon=epsilon)
        edges = np.asarray(G.get_edgelist(), dtype=np.int64).reshape(-1, 2)
        n_nodes = G.vcount()

    if keep is not None and len(edges):
        edges = edges[keep[edges[:, 0]] & keep[edges[:, 1]]]

    raw = utils.fastcore.connected_components_graph(edges, n_nodes)

    if keep is None:
        return _compress(raw, None, n_nodes)

    # Masked-out elements are isolated in the induced subgraph but still carry a
    # label of their own, so they have to go before we compress
    return _compress(raw[keep], keep, n_nodes)


def connected_components(
    x: Union[
        "core.Skeleton",
        "core.Mesh",
        "core.Dotprops",
        "core.Voxels",
        "tm.Trimesh",
    ],
    *,
    connectivity: Optional[Union[int, str]] = None,
    epsilon: Optional[float] = None,
    element: Optional[str] = None,
    mask: Optional[Sequence] = None,
) -> np.ndarray:
    """Label the connected components of a neuron.

    Every node/vertex/point/voxel gets the label of the component it belongs to,
    numbered `0 .. n_components - 1` **largest component first** - so `== 0`
    always selects the biggest piece, and `np.bincount` of the result gives the
    component sizes in descending order.

    Uses `navis-fastcore` for skeletons and meshes, and `sparse-cubes` for
    voxels.

    Parameters
    ----------
    x :             Skeleton | Mesh | Dotprops | Voxels | Trimesh
                    Neuron to label.
    connectivity :  6 | 18 | 26 | "vertex" | "face" | "manifold" , optional
                    What counts as connected - which is a different question for
                    each type of neuron.
                    For Voxels: which neighbouring voxels count as connected.
                    6 = faces only, 18 = faces + edges, 26 (default) = faces +
                    edges + corners.
                    For Meshes: whether two faces must share a corner
                    (`"vertex"`, default), an edge (`"face"`) or an edge with no
                    third face on it (`"manifold"`).
                    For Skeletons/Dotprops: nothing - their edges are already
                    decided, and passing it warns.
    epsilon :       float, optional
                    For Dotprops only: distance at which two points count as
                    connected. Defaults to 5x the average distance between
                    points (`x.sampling_resolution`).
    element :       "node" | "vertex" | "face" | "point" | "voxel", optional
                    What the labels should be *of*. Only meshes have a choice
                    here, and only there does the default depend on anything:
                    `"vertex"`, except under a face-based `connectivity`, where
                    a pinch vertex belongs to several components at once and
                    only `"face"` is a partition. Asking for `"vertex"` anyway
                    is allowed - each such vertex then takes the label of the
                    largest component it touches.
    mask :          list-like, optional
                    Restrict to a part of the neuron: components are those of
                    the *induced* sub-neuron, i.e. anything outside the mask
                    neither belongs to a component nor connects two. Accepts
                    what `navis.subset_neuron` accepts (IDs, indices or a
                    boolean array).

    Returns
    -------
    np.ndarray
                (N, ) array of labels, one per element and aligned with it.
                `-1` marks an element that is in no component at all: masked
                out, or - under a face-based `connectivity` with
                `element="vertex"` - a vertex that no face uses.

    See Also
    --------
    [`navis.split_components`][]
                Turn the components into separate neurons.
    [`navis.drop_fluff`][]
                Drop all but the largest component(s).
    [`navis.heal_skeleton`][], [`navis.heal_mesh`][]
                Reconnect the components instead of separating them.

    Examples
    --------
    >>> import navis
    >>> import numpy as np
    >>> m = navis.example_neurons(1, kind='mesh')
    >>> labels = navis.connected_components(m)
    >>> labels.max() + 1                      # number of components
    14
    >>> np.bincount(labels)[:3]               # sizes, largest first
    array([17058,   240,    12])
    >>> # Vertices of the largest component
    >>> np.flatnonzero(labels == 0).shape
    (17058,)

    Label a skeleton's nodes instead, and only within a mask:

    >>> n = navis.example_neurons(1, kind='skeleton')
    >>> labels = navis.connected_components(n)
    >>> labels.max() + 1
    1
    >>> twigs = n.nodes.node_id.values[n.nodes.type != 'slab']
    >>> masked = navis.connected_components(n, mask=twigs)
    >>> int((masked == -1).sum()) == len(n.nodes) - len(twigs)
    True

    """
    if not isinstance(x, (core.BaseNeuron, tm.Trimesh)):
        raise TypeError(f'Expected neuron or Trimesh, got "{type(x)}"')

    connectivity = _resolve_connectivity(x, connectivity)
    allowed, axis_name = _ELEMENTS[_element_kind(x)]

    # `allowed[0]` is what the neuron is made of, and so what a `mask` selects -
    # never faces, which are a mesh's *edges* as far as subsetting is concerned.
    # The natural labelling follows `connectivity`, because the face readings are
    # exactly the ones vertices cannot express: a pinch vertex belongs to several
    # face components at once, so only a labelling of faces is a partition.
    natural = "face" if connectivity in _FACE_CONNECTIVITIES else allowed[0]

    if element is None:
        element = natural
    elif element not in allowed:
        raise ValueError(
            f"`element` for a {type(x).__name__} must be one of "
            f"{', '.join(repr(a) for a in allowed)}, got {element!r}"
        )

    keep = _resolve_mask(x, mask, axis_name)

    labels, n = _natural_component_labels(x, connectivity, epsilon, keep)
    labels = _sort_labels_by_size(labels, n, masked=keep is not None)

    if element == natural:
        return labels

    faces = np.asarray(x.faces, dtype=np.int64)
    if not len(faces):
        return np.full(len(x.vertices) if element == "vertex" else 0, -1, np.int64)

    if element == "face":
        # Every corner of a face is in the same vertex component, so any of them
        # will do. Faces are what a mesh is made of, so there is no "no face"
        # case to worry about here.
        return labels[faces[:, 0]]

    # face -> vertex. A pinch vertex is in several components at once; give it
    # the largest, which - labels being size-sorted - is simply the smallest
    # label. Vertices that no (kept) face uses stay at -1.
    if keep is not None:
        # Only kept faces have a say. A masked-out one carries -1, which would
        # otherwise sort below every real label and win the scatter - including
        # at a kept vertex that a kept face also uses.
        alive = labels >= 0
        faces, labels = faces[alive], labels[alive]

    # Ordering the *faces* descending and expanding afterwards is the same
    # scatter as ordering their corners, over a third as many items: the
    # smallest label is then the write that lands last and wins. Cheaper than
    # `np.minimum.at`, which falls back to an unbuffered loop.
    out = np.full(len(x.vertices), -1, dtype=np.int64)
    order = np.argsort(labels)[::-1]
    out[faces[order].reshape(-1)] = np.repeat(labels[order], 3)
    return out


def _n_components(
    x,
    connectivity: Optional[Union[int, str]] = None,
    epsilon: Optional[float] = None,
    mask: Optional[Sequence] = None,
) -> int:
    """How many connected components a neuron has.

    The count the labelling produces anyway, without the size-sort that
    [`navis.connected_components`][] adds on top - see `BaseNeuron.n_components`.
    """
    connectivity = _resolve_connectivity(x, connectivity)
    keep = _resolve_mask(x, mask, _ELEMENTS[_element_kind(x)][1])
    return _natural_component_labels(x, connectivity, epsilon, keep)[1]


def _component_ids(
    x,
    connectivity: Optional[Union[int, str]] = None,
    epsilon: Optional[float] = None,
    mask: Optional[Sequence] = None,
) -> List[np.ndarray]:
    """Component membership as one array of IDs per component, largest first.

    The list form of [`navis.connected_components`][], for the callers that want
    to iterate over components rather than index by them. Members are node IDs
    for a Skeleton and indices otherwise.

    Mesh components are always given as *vertices*, whatever `connectivity`
    says, because that is the unit sizes are counted in - and under the face
    readings they can therefore overlap: a pinch vertex belongs to every face
    component that meets there, which is what lets `drop_fluff` keep the piece
    it belongs to and still drop the other one.
    """
    connectivity = _resolve_connectivity(x, connectivity)
    keep = _resolve_mask(x, mask, _ELEMENTS[_element_kind(x)][1])

    # Straight off the *unsorted* labels: sorting them costs several passes over
    # every element, where sorting the handful of groups afterwards is nothing.
    # The tuple key reproduces `_sort_labels_by_size` - size first, then the
    # index of the component's first member.
    labels, _ = _natural_component_labels(x, connectivity, epsilon, keep)
    groups = _groups_from_labels(labels)
    groups.sort(key=lambda g: (-len(g), g[0]))

    if isinstance(x, core.Skeleton):
        node_ids = x.nodes.node_id.values
        return [node_ids[g] for g in groups]

    if isinstance(x, (core.Mesh, tm.Trimesh)) and connectivity in _FACE_CONNECTIVITIES:
        faces = np.asarray(x.faces, dtype=np.int64)
        return [np.unique(faces[g]) for g in groups]

    return groups


def _groups_from_labels(labels: np.ndarray) -> List[np.ndarray]:
    """Indices of each component, one array per label.

    The `-1`s are skipped: they are not a component.
    """
    valid = np.flatnonzero(labels >= 0)
    if not len(valid):
        return []

    order = np.argsort(labels[valid], kind="stable")
    valid = valid[order]
    bounds = np.concatenate(([0], np.cumsum(np.bincount(labels[valid]))))
    return [valid[a:b] for a, b in zip(bounds[:-1], bounds[1:])]


def _break_segments(x: "core.NeuronObject") -> list:
    """Break neuron into small segments connecting ends, branches and root.

    Parameters
    ----------
    x :         Skeleton | NeuronList
                May contain multiple neurons.

    Returns
    -------
    list
                Segments as list of lists containing node IDs.

    Examples
    --------
    For doctest only

    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> seg = navis.graph_utils._break_segments(n)

    """
    if isinstance(x, core.NeuronList):
        return [_break_segments(x[i]) for i in range(len(x))]
    elif isinstance(x, core.Skeleton):
        pass
    else:
        logger.error("Unexpected datatype: %s" % str(type(x)))
        raise ValueError

    # At this point x is Skeleton
    x: core.Skeleton

    # Segments come back ordered by the node table position of their (distal) seed
    # node. Consumers such as `segment_analysis`, `resample_skeleton` and the NEURON
    # interface enumerate the segments, so that order ends up in their output.
    return utils.fastcore.break_segments(
        x.nodes.node_id.values, x.nodes.parent_id.values
    )


@utils.lock_neuron
def dist_to_root(
    x: "core.Skeleton", weight=None, igraph_indices: bool = False
) -> dict:
    """Calculate distance to root for each node.

    Parameters
    ----------
    x :                 Skeleton
    weight :            str, optional
                        Use "weight" if you want geodesic distance and `None`
                        if you want node count.
    igraph_indices :    bool
                        Whether to return node *indices* instead of node IDs.
                        This is mainly used for internal functions. (igraph's
                        vertex indices are the node table's row numbers, which
                        is the index space used throughout.)

    Returns
    -------
    dist :              dict
                        Dictionary with root distances.

    Examples
    --------
    For doctest only

    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> seg = navis.graph.dist_to_root(n)

    See Also
    --------
    [`navis.geodesic_matrix`][]
                        For distances between all points.

    """
    if not isinstance(x, core.Skeleton):
        raise TypeError(f"Expected Skeleton, got {type(x)}")

    ids = x.nodes.node_id.values
    parents = x.nodes.parent_id.values

    # Every node reaches exactly one root, so this stays O(N) - and unlike a
    # `geodesic_matrix(from_=roots)` it never materialises a roots x N block,
    # which is hundreds of MB on a badly fragmented neuron.
    dists = utils.fastcore.dist_to_root(
        ids,
        parents,
        weights=None
        if weight is None
        else morpho.mmetrics.parent_dist(x, root_dist=0),
    )

    # Unreachable nodes (i.e. those in another fragment) come back as -1 and are
    # simply left out - matching the networkx behaviour this replaced.
    keys = np.arange(len(ids)) if igraph_indices else ids
    reachable = np.asarray(dists) >= 0

    return dict(zip(keys[reachable].tolist(), np.asarray(dists)[reachable].tolist()))


@utils.map_neuronlist(desc="Classifying", allow_parallel=True)
@utils.lock_neuron
def classify_nodes(x: "core.NeuronObject", categorical=True, inplace: bool = True):
    """Classify neuron's nodes into end nodes, branches, slabs or root.

    Adds a `'type'` column to `x.nodes` table.

    Parameters
    ----------
    x :         Skeleton | NeuronList
                Neuron(s) whose nodes to classify.
    categorical : bool
                If True (default), will use categorical data type which takes
                up much less memory at a small run-time overhead.
    inplace :   bool, optional
                If `False`, nodes will be classified on a copy which is then
                returned leaving the original neuron unchanged.

    Returns
    -------
    Skeleton/List

    Examples
    --------
    >>> import navis
    >>> nl = navis.example_neurons(2)
    >>> _ = navis.graph.classify_nodes(nl, inplace=True)

    """
    if not inplace:
        x = x.copy()

    if not isinstance(x, core.Skeleton):
        raise TypeError(f'Expected Skeleton(s), got "{type(x)}"')

    if x.nodes.empty:
        x.nodes["type"] = None
        return x

    node_ids = x.nodes.node_id.values
    parent_ids = x.nodes.parent_id.values

    # Note: we work with the integer *codes* of `NODE_TYPES` throughout and only
    # ever turn them into labels at the very end. Going via a string array (as
    # this used to) makes `pd.Categorical` factorize N strings, which costs more
    # than the classification itself.
    # Fastcore uses its own order (0=root, 1=leaf, 2=branch, 3=slab)
    cl = _FASTCORE_NODE_TYPES[utils.fastcore.classify_nodes(node_ids, parent_ids)]

    if categorical:
        x.nodes["type"] = pd.Categorical.from_codes(
            cl, categories=NODE_TYPES, ordered=False
        )
    else:
        x.nodes["type"] = np.asarray(NODE_TYPES, dtype="<U6")[cl]

    return x


@utils.lock_neuron
def distal_to(
    x: "core.Skeleton",
    a: Optional[Union[str, int, List[Union[str, int]]]] = None,
    b: Optional[Union[str, int, List[Union[str, int]]]] = None,
) -> Union[bool, pd.DataFrame]:
    """Check if nodes A are distal to nodes B.

    Important
    ---------
    Please note that if node A is not distal to node B, this does **not**
    automatically mean it is proximal instead: if nodes are on different
    branches, they are neither distal nor proximal to one another! To test
    for this case run a->b and b->a - if both return `False`, nodes are on
    different branches.

    Also: if a and b are the same node, this function will return `True`!

    Parameters
    ----------
    x :     Skeleton
    a,b :   single node ID | list of node IDs | None, optional
            If no node IDs are provided, will consider all node. Note that for
            large sets of nodes it might be more efficient to use
            [`navis.geodesic_matrix`][] (see examples).

    Returns
    -------
    bool
            If `a` and `b` are single node IDs respectively.
    pd.DataFrame
            If `a` and/or `b` are lists of node IDs. Columns and rows
            (index) represent node IDs. Neurons `a` are rows, neurons
            `b` are columns.

    Examples
    --------
    >>> import navis
    >>> # Get a neuron
    >>> x = navis.example_neurons(1)
    >>> # Get a random node
    >>> n = x.nodes.iloc[100].node_id
    >>> # Check all nodes if they are distal or proximal to that node
    >>> df = navis.distal_to(x, n)
    >>> # Get the IDs of the nodes that are distal
    >>> dist = df.loc[n, df.loc[n]].index.values
    >>> len(dist)
    101

    For large neurons and/or large sets of `a`/`b` it can be much faster to use
    `geodesic_matrix` instead:

    >>> import navis
    >>> import numpy as np
    >>> x = navis.example_neurons(1)
    >>> # Get an all-by-all distal_to
    >>> df = navis.geodesic_matrix(x, weight=None, directed=True) < np.inf
    >>> # Get distal_to for specific nodes
    >>> df = navis.geodesic_matrix(x, weight=None, directed=True) < np.inf
    >>> # Get distal_to for specific nodes
    >>> a, b = x.nodes.node_id.values[:100], x.nodes.node_id.values[-100:]
    >>> dist = navis.geodesic_matrix(x, weight=None, directed=True, from_=a)
    >>> distal_to = dist[b] < np.inf

    See Also
    --------
    [`navis.geodesic_matrix`][]
            Depending on your neuron and how many nodes you're asking for,
            this function can be considerably faster! See examples.

    """
    if isinstance(x, core.NeuronList) and len(x) == 1:
        x = x[0]

    if not isinstance(x, core.Skeleton):
        raise ValueError(f"Please pass a single Skeleton, got {type(x)}")

    # At this point x is Skeleton
    x: core.Skeleton

    if not isinstance(a, type(None)):
        tnA = utils.make_iterable(a)
        # Make sure we're dealing with integers
        tnA = np.unique(tnA).astype(int)
    else:
        tnA = x.nodes.node_id.values

    if not isinstance(b, type(None)):
        tnB = utils.make_iterable(b)
        # Make sure we're dealing with integers
        tnB = np.unique(tnB).astype(int)
    else:
        tnB = x.nodes.node_id.values

    # `targets` is what keeps this cheap: a full all-sources search would produce a
    # len(a) x n_nodes matrix either way. Here we only ever materialise the
    # len(a) x len(b) block we actually return.
    le = utils.fastcore.geodesic_matrix(
        x.nodes.node_id.values,
        x.nodes.parent_id.values,
        sources=tnA,
        targets=tnB,
        directed=True,
        weights=None,
    )
    # Fastcore uses -1 (not inf) for unreachable pairs
    reachable = le >= 0

    df = pd.DataFrame(reachable, index=tnA, columns=tnB)

    if df.shape == (1, 1):
        return df.values[0][0]
    else:
        # Return boolean
        return df


def skeleton_adjacency_matrix(
    x: "core.NeuronObject", sort: bool = True
) -> pd.DataFrame:
    """Generate adjacency matrix for a skeleton.

    Parameters
    ----------
    x :         Skeleton
                Neuron for which to generate adjacency matrix.
    sort :      bool, optional
                If True, will sort the adjacency matrix by topology.

    Returns
    -------
    pd.DataFrame
                Adjacency matrix where rows are nodes and columns are
                their parents.

    See Also
    --------
    [`navis.geodesic_matrix`][]
        For distances between all points.
    [`navis.distal_to`][]
        Check if a node A is distal to node B.
    [`navis.dist_between`][]
        Get point-to-point geodesic ("along-the-arbor") distances.

    """
    if isinstance(x, core.NeuronList):
        if len(x) == 1:
            x = x[0]
        else:
            raise ValueError("Cannot process more than a single neuron.")
    elif not isinstance(x, (core.Skeleton,)):
        raise ValueError(f'Unable to process data of type "{type(x)}"')

    # Generate the empty adjacency matrix
    adj = pd.DataFrame(
        np.zeros((len(x.nodes), len(x.nodes)), dtype=bool),
        index=x.nodes.node_id.values,
        columns=x.nodes.node_id.values,
    )

    # Fill in the parent-child relationships
    not_root = x.nodes.parent_id.values >= 0
    node_ix = np.arange(len(x.nodes))[not_root]
    parent_ids = x.nodes.parent_id.values[not_root]
    parent_ix = np.searchsorted(x.nodes.node_id.values, parent_ids)
    adj.values[node_ix, parent_ix] = True

    if sort:
        sort = node_label_sorting(x)
        adj = adj.loc[sort, sort]

    return adj


def geodesic_matrix(
    x: "core.NeuronObject",
    from_: Optional[Iterable[int]] = None,
    to_: Optional[Iterable[int]] = None,
    directed: bool = False,
    weight: Optional[str] = "weight",
    max_dist: Union[float, int] = np.inf,
) -> pd.DataFrame:
    """Generate geodesic ("along-the-arbor") distance matrix between nodes/vertices.

    Parameters
    ----------
    x :         Skeleton | Mesh | NeuronList
                If list, must contain a SINGLE neuron.
    from_ :     list | numpy.ndarray, optional
                Node IDs (for Skeletons) or vertex indices (for Meshes).
                If provided, will compute distances only FROM this subset to
                all other nodes/vertices.
    to_ :       list | numpy.ndarray, optional
                Node IDs (for Skeletons) or vertex indices (for Meshes).
                If provided, will compute distances only TO this subset. Use
                together with `from_` to get just the block you need instead of
                slicing a full matrix afterwards - that can be the difference
                between a few MB and a few GB on a large neuron.
    directed :  bool, optional
                For Skeletons only: if True, pairs without a child->parent
                path will be returned with `distance = "inf"`.
    weight :    'weight' | None, optional
                If "weight" distances are given as physical length.
                If `None` distance is the number of nodes.
    max_dist :  int | float | str, optional
                Use to cap distance calculations. Nodes that are not within
                `max_dist` will have distance `np.inf`. If neuron has its
                `.units` set, you can also pass a string such as "10 microns".

    Returns
    -------
    pd.DataFrame
                Geodesic distance matrix. If the neuron is fragmented or
                `directed=True`, unreachable node pairs will have distance `np.inf`.

    See Also
    --------
    [`navis.distal_to`][]
        Check if a node A is distal to node B.
    [`navis.dist_between`][]
        Get point-to-point geodesic distances.
    [`navis.dist_to_root`][]
        Distances from all skeleton node to their root(s).
    [`navis.graph.skeleton_adjacency_matrix`][]
        Generate adjacency matrix for a skeleton.

    Examples
    --------
    Find average geodesic distance between all leaf nodes

    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> leafs = n.nodes[n.nodes.type=='end'].node_id.values
    >>> # Compute just the leaf-by-leaf block. Note that generating the full
    >>> # matrix and subsetting it afterwards would give the same answer but
    >>> # has to materialise every node-to-node distance to get there.
    >>> l_dist = navis.geodesic_matrix(n, from_=leafs, to_=leafs)
    >>> round(l_dist.mean().mean())
    12983

    """
    if isinstance(x, core.NeuronList):
        if len(x) != 1:
            raise ValueError("Input must be a single neuron.")
        x = x[0]

    if not isinstance(x, (core.Skeleton, core.Mesh)):
        raise ValueError(f'Unable to process data of type "{type(x)}"')

    max_dist = x.map_units(max_dist, on_error="raise")

    def _check(sel, valid):
        """Normalise a `from_`/`to_` selection and make sure it exists."""
        sel = np.unique(utils.make_iterable(sel))
        miss = sel[~np.isin(sel, valid)]
        if len(miss):
            raise ValueError(
                f"Node/vertex IDs not present: {', '.join(miss.astype(str))}"
            )
        return sel

    if isinstance(x, core.Skeleton):
        node_ids = x.nodes.node_id.values

        # Calculate node distances
        if weight == "weight":
            weight = morpho.mmetrics.parent_dist(x, root_dist=0)

        from_ = None if from_ is None else _check(from_, node_ids)
        to_ = None if to_ is None else _check(to_, node_ids)

        dmat = utils.fastcore.geodesic_matrix(
            node_ids,
            x.nodes.parent_id.values,
            weights=weight,
            directed=directed,
            sources=from_,
            targets=to_,
        )

        # Fastcore returns -1 for unreachable node pairs
        dmat[dmat < 0] = np.inf

        if max_dist is not None and max_dist is not np.inf:
            dmat[dmat > max_dist] = np.inf

        return pd.DataFrame(
            dmat,
            index=node_ids if from_ is None else from_,
            columns=node_ids if to_ is None else to_,
        )

    # Only Meshes reach here, and `directed` makes no sense for those - the edge
    # list is undirected, so it is ignored rather than honoured.
    vertex_ids = np.arange(len(x.vertices))

    from_ = None if from_ is None else _check(from_, vertex_ids)
    to_ = None if to_ is None else _check(to_, vertex_ids)

    # Fastcore takes `None` rather than infinity for "no limit"
    limit_ = None if max_dist is None or not np.isfinite(max_dist) else max_dist

    if not x.n_extra_edges:
        dmat = utils.fastcore.geodesic_matrix_mesh(
            x.faces,
            # Without coordinates fastcore weights every edge as 1 (i.e. hop count)
            vertices=x.vertices if weight == "weight" else None,
            n_vertices=len(vertex_ids),
            sources=from_,
            targets=to_,
            limit=limit_,
        )
    else:
        # `geodesic_matrix_mesh` derives the adjacency from the faces and so
        # can't see edges that aren't part of one - we have to hand it the
        # full edge list instead
        edges, lengths = utils.mesh_unique_edges(x, return_lengths=True)
        dmat = utils.fastcore.geodesic_matrix_graph(
            edges,
            n_nodes=len(vertex_ids),
            weights=lengths if weight == "weight" else None,
            directed=False,
            sources=from_,
            targets=to_,
            limit=limit_,
            # `lengths` is float64 (trimesh vertices are), and fastcore takes the
            # distances' width from the weights' - which would make this the one
            # branch of this function returning float64. The other two can't
            # follow: `geodesic_matrix_mesh` reads nothing off `vertices` and the
            # skeleton kernels are float32 only.
            dtype=np.float32,
        )

    # Fastcore returns -1 for unreachable vertex pairs
    dmat[dmat < 0] = np.inf

    return pd.DataFrame(
        dmat,
        index=vertex_ids if from_ is None else from_,
        columns=vertex_ids if to_ is None else to_,
    )


def _geodesic_nearest(
    x: "core.Skeleton",
    targets: Iterable[int],
    query: Optional[Iterable[int]] = None,
    weight: Optional[str] = None,
    directed: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find, for each query node, the geodesically nearest of the `targets` nodes.

    This is a memory-efficient alternative to building a full geodesic distance
    matrix with [`navis.geodesic_matrix`][]: it only keeps, for each query node,
    the nearest target and the distance to it. It therefore scales to several
    100k nodes (O(N) memory) where `geodesic_matrix` would materialise an
    `(n_query, n_nodes)` matrix and run out of memory.

    Note
    ----
    `query` and `targets` are expected to be disjoint (the typical "assign
    unlabeled nodes to the nearest labeled node" use case). A query node that is
    itself a target matches itself at distance 0.

    Parameters
    ----------
    x :         Skeleton
    targets :   iterable of node IDs
                Candidate nodes to snap to.
    query :     iterable of node IDs, optional
                Nodes to find a nearest target for. If `None`, uses all nodes.
    weight :    'weight' | None
                If "weight" distances are physical edge lengths, if `None`
                distances are the number of nodes (hops).
    directed :  bool
                If True, only travel child -> parent (towards the root).

    Returns
    -------
    nearest :   np.ndarray
                Node ID of the nearest target for each query node (`-1` if no
                target is reachable). Ordered to match `query` (or `x.nodes` if
                `query` is `None`).
    distances : np.ndarray
                Distance to that nearest target (`np.inf` if unreachable).

    """
    if not isinstance(x, core.Skeleton):
        raise ValueError(f'Expected Skeleton, got "{type(x)}"')

    node_ids = x.nodes.node_id.values
    parent_ids = x.nodes.parent_id.values

    targets = np.asarray(list(targets))
    query = node_ids if query is None else np.asarray(list(query))

    # Nothing to snap to (or nothing to snap) -> everything unreachable.
    if not len(targets) or not len(query):
        return (
            np.full(len(query), -1, dtype=node_ids.dtype),
            np.full(len(query), np.inf, dtype=float),
        )

    # Per-node distance to parent (root = 0). `None` -> unweighted (hop count).
    weights = morpho.mmetrics.parent_dist(x, root_dist=0) if weight == "weight" else None

    distances, nearest = utils.fastcore.geodesic_nearest(
        node_ids,
        parent_ids,
        sources=query,
        targets=targets,
        directed=directed,
        weights=weights,
    )
    distances = np.asarray(distances, dtype=float)
    nearest = np.asarray(nearest)
    # fastcore returns -1 for unreachable sources
    distances[distances < 0] = np.inf
    return nearest, distances


def geodesic_clusters(
    x: Union["core.Skeleton", "core.Mesh"],
    max_dist: float,
    weight: Optional[str] = "weight",
    seeds: Optional[Iterable[int]] = None,
    connected: bool = True,
) -> np.ndarray:
    """Partition a neuron into clusters of bounded geodesic radius.

    Repeatedly takes an unassigned node as a seed and grows a cluster outwards
    from it, absorbing every node within `max_dist` *along the arbor* that no
    earlier cluster has already claimed.

    The radius is the true geodesic distance from the seed, not the length of
    the walk that reached it - a node close to a seed is never excluded merely
    because the traversal arrived the long way round. Clusters therefore expand
    *through* nodes an earlier cluster claimed, so each raw cluster is a true
    ball around its seed **minus** whatever earlier clusters took.

    !!! warning "This is a bounded-radius partition, not a uniform downsampling"
        The subtraction above routinely disconnects a cluster: on the example
        neurons 33-50% of raw clusters come back in several pieces (up to 35),
        and a cluster scattered in pieces has a centroid that need not lie near
        any of them. `connected=True` (the default) repairs that by splitting
        each cluster into its connected components - but the pieces are then
        many and small (on `example_neurons(1)` at `max_dist=5000`: 16 raw
        clusters become 77, median size 10 nodes, 9 of them singletons).

        So while each cluster is guaranteed to be connected and to lie within
        `max_dist` of its seed, cluster *sizes* are very uneven and their
        centroids are **not** spaced by anything like `max_dist`. If you want an
        even spacing along the arbor use [`navis.resample_skeleton`][]; this is
        the right tool when you need a bounded-radius partition of the graph
        itself.

    Parameters
    ----------
    x :         Skeleton | Mesh
    max_dist :  float | str
                Maximum distance from a cluster's seed, in the neuron's units
                (or in hops if `weight=None`). Must be finite and non-negative.
                If the neuron has its `.units` set and `weight` is not `None`,
                you can also pass a string such as "10 microns".
    weight :    'weight' | None
                If "weight" (default) distances are physical edge lengths, if
                `None` they are the number of hops.
    seeds :     iterable of node IDs (vertex indices for meshes), optional
                Nodes to prefer as seeds, in order. Anything left unassigned
                afterwards seeds a cluster of its own. A seed an earlier cluster
                already claimed is skipped.
    connected : bool
                Whether to split clusters that the greedy assignment left
                disconnected into their connected components (see above). This
                only ever *increases* the number of clusters, and every piece
                still lies within `max_dist` of the original seed. Set to False
                for the raw greedy assignment.

    Returns
    -------
    labels :    np.ndarray
                Cluster index for each node, aligned with `x.nodes` (or
                `x.vertices` for a Mesh) and contiguous in
                `[0, n_clusters)`. Every node is labelled, so the number of
                clusters is `labels.max() + 1`.

    Examples
    --------
    >>> import navis
    >>> import numpy as np
    >>> n = navis.example_neurons(1, kind='skeleton')
    >>> labels = navis.graph.geodesic_clusters(n, max_dist=5000)
    >>> # Every node is assigned, and there are fewer clusters than nodes
    >>> bool((labels >= 0).all()), bool(labels.max() + 1 < n.n_nodes)
    (True, True)
    >>> # Collapse each cluster to its centroid
    >>> co = n.nodes[['x', 'y', 'z']].values
    >>> centroids = np.stack([co[labels == i].mean(axis=0)
    ...                       for i in range(labels.max() + 1)])

    See Also
    --------
    [`navis.downsample_neuron`][]
                Topological downsampling - keeps every Nth node.
    [`navis.resample_skeleton`][]
                Resamples to an even spacing along the arbor. Prefer this if
                what you want is uniform sampling - see the warning above.

    """
    # A unit string only means something when the weights are physical lengths -
    # with `weight=None` the radius is a number of hops, which has no units
    if weight is not None:
        max_dist = x.map_units(max_dist, on_error="raise")
    max_dist = float(max_dist)
    if not np.isfinite(max_dist) or max_dist < 0:
        raise ValueError(f"`max_dist` must be finite and non-negative, got {max_dist}")

    edges, weights, n_nodes, ids = _cluster_graph(x, weight)

    seed_ix = None
    if seeds is not None:
        seeds = np.fromiter(seeds, dtype=ids.dtype, count=len(list(seeds))) \
            if not isinstance(seeds, np.ndarray) else seeds
        seed_ix = pd.Index(ids).get_indexer(np.asarray(seeds))
        if (seed_ix < 0).any():
            raise ValueError("Some `seeds` are not part of this neuron.")

    labels, _ = utils.fastcore.geodesic_clusters(
        edges, n_nodes, max_dist, weights=weights, seeds=seed_ix
    )
    labels = np.asarray(labels)

    if connected:
        labels = _split_disconnected(edges, labels, n_nodes)

    return labels


def _split_disconnected(edges, labels, n_nodes):
    """Split each cluster into its connected components and relabel.

    An edge joins two nodes of the same cluster or it does not; dropping the
    ones that do not and taking connected components of what remains splits
    every cluster exactly along its own internal breaks.
    """
    if not len(edges):
        return np.arange(n_nodes, dtype=np.int32)

    intra = edges[labels[edges[:, 0]] == labels[edges[:, 1]]]

    comp = utils.fastcore.connected_components_graph(intra, n_nodes)

    # Relabel contiguously. N.B. this renumbers clusters - the raw growth order
    # does not survive a split anyway.
    _, out = np.unique(comp, return_inverse=True)
    return out.reshape(-1).astype(np.int32, copy=False)


def _cluster_graph(x, weight):
    """Edge list, edge weights, node count and node IDs for `geodesic_clusters`."""
    if isinstance(x, core.Skeleton):
        edges, ids = skeleton_edges(x)
        if weight == "weight":
            co = x.nodes[["x", "y", "z"]].values.astype(np.float64)
            w = np.linalg.norm(co[edges[:, 0]] - co[edges[:, 1]], axis=1)
        else:
            w = None
        return edges, w, len(ids), ids

    if isinstance(x, core.Mesh):
        edges = np.asarray(utils.mesh_unique_edges(x.trimesh), dtype=np.int64)
        if weight == "weight":
            co = np.asarray(x.vertices, dtype=np.float64)
            w = np.linalg.norm(co[edges[:, 0]] - co[edges[:, 1]], axis=1)
        else:
            w = None
        n = len(x.vertices)
        return edges, w, n, np.arange(n)

    raise TypeError(f"Expected Skeleton or Mesh, got {type(x)}")


@utils.lock_neuron
def segment_length(x: "core.Skeleton", segment: List[int]) -> float:
    """Get length of a linear segment.

    This function is superfast but has no checks - you must provide a
    valid segment.

    Parameters
    ----------
    x :         Skeleton
                Neuron to which this segment belongs.
    segment :   list of ints
                Linear segment as list of node IDs ordered child->parent.

    Returns
    -------
    length :    float

    See Also
    --------
    [`navis.dist_between`][]
        If you only know start and end points of the segment.

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> l = navis.segment_length(n, n.segments[0])
    >>> round(l)
    56356

    """
    if not isinstance(x, core.Skeleton):
        raise ValueError(f'Unable to process data of type "{type(x)}"')

    return float(segment_lengths(x, [segment])[0])


def segment_lengths(x: "core.Skeleton", segments: Sequence[Sequence[int]]):
    """Get the length of each of many linear segments.

    Same as calling [`navis.segment_length`][] on each segment but builds the node
    lookup once instead of once per segment.

    Returns
    -------
    np.ndarray
                Length of each segment.
    """
    if not len(segments):
        return np.zeros(0)

    # An edge's weight is just the distance between its two nodes, so we can read
    # the lengths straight off the coordinates rather than going via a graph.
    # Note the cast to float64: node coordinates are often float32, and summing
    # those would drift away from the weights networkx used to hand us.
    coords = x.nodes[["x", "y", "z"]].values.astype(float)

    # Resolve every segment's node IDs in one lookup - `get_indexer` has enough
    # per-call overhead that doing it once per segment costs more than the walk it
    # replaces.
    lengths = np.array([len(s) for s in segments])
    flat = np.concatenate([np.asarray(s) for s in segments])
    coords = coords[pd.Index(x.nodes.node_id).get_indexer(flat)]

    # Distance from each node to the one before it...
    step = np.zeros(len(flat))
    step[1:] = np.linalg.norm(np.diff(coords, axis=0), axis=1)

    # ...except the first node of each segment, which has no predecessor *in that
    # segment* - this also discards the bogus step across each segment boundary.
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    step[starts] = 0

    return np.add.reduceat(step, starts)


@utils.lock_neuron
def dist_between(x: "core.NeuronObject", a, b):
    """Get the geodesic distance between nodes in nanometers.

    Parameters
    ----------
    x :             Skeleton | Mesh | NeuronList
                    If NeuronList must contain only a single neuron.
    a,b :           int | list of int
                    Node IDs (for Skeletons) or vertex indices (Meshes)
                    to check the distance between. Can be single nodes or
                    matched arrays of nodes, in which case distances are
                    computed pairwise (`a[0]` to `b[0]`, `a[1]` to `b[1]`, ...).
                    One of them may also be a single node, which is then
                    broadcast against the other.

    Returns
    -------
    float
                    Distance in nm if `a` and `b` are single nodes.
    np.ndarray
                    Distances in nm if either `a` or `b` is a list of nodes.
                    Unreachable pairs are `np.inf`.

    See Also
    --------
    [`navis.distal_to`][]
        Check if a node A is distal to node B.
    [`navis.geodesic_matrix`][]
        Get all-by-all geodesic distance matrix. Use this if you want distances
        between *every* A and *every* B rather than between matched pairs.
    [`navis.segment_length`][]
        Much faster if you have a linear segment and know all node IDs.

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> d = navis.dist_between(n,
    ...                        n.nodes.node_id.values[0],
    ...                        n.nodes.node_id.values[1])

    Distances between many pairs at once:

    >>> d = navis.dist_between(n,
    ...                        n.nodes.node_id.values[:100],
    ...                        n.nodes.node_id.values[-100:])
    >>> d.shape
    (100,)

    """
    if isinstance(x, core.NeuronList):
        if len(x) == 1:
            x = x[0]
        else:
            raise ValueError(f"Need a single Skeleton, got {len(x)}")

    if not isinstance(x, (core.Skeleton, core.Mesh, igraph.Graph, nx.DiGraph)):
        raise ValueError(f"Unable to process data of type {type(x)}")

    # Scalar in -> scalar out. Note that a length-1 iterable counts as a scalar
    # here, which is what this function has always done.
    scalar = not utils.is_iterable(a) and not utils.is_iterable(b)

    try:
        a = np.asarray(utils.make_iterable(a)).astype(int)
        b = np.asarray(utils.make_iterable(b)).astype(int)
    except BaseException:
        raise ValueError("a, b need to be node IDs or vertex indices!")

    if a.size != b.size and a.size != 1 and b.size != 1:
        raise ValueError(
            f"Got {a.size} nodes for `a` and {b.size} for `b`. These must "
            "either match up pairwise or one of them must be a single node."
        )
    a, b = np.broadcast_arrays(a, b)

    if isinstance(x, core.Skeleton):
        node_ids = x.nodes.node_id.values
        parent_ids = x.nodes.parent_id.values

        weights = morpho.mmetrics.parent_dist(x, root_dist=0)
        dist = utils.fastcore.geodesic_pairs(
            node_ids,
            parent_ids,
            pairs=np.stack((a, b), axis=1),
            weights=weights,
        ).astype(float)

        # Fastcore returns -1 for unreachable pairs
        dist[dist < 0] = np.inf
        return float(dist[0]) if scalar else dist

    # Meshes and raw graphs - Skeletons returned above.
    G: Union[igraph.Graph, nx.DiGraph] = x.igraph if isinstance(x, core.Mesh) else x

    # If we're working with a networkx DiGraph
    if isinstance(G, nx.DiGraph):
        und = G.to_undirected(as_view=True)
        dist = np.array(
            [
                nx.algorithms.shortest_path_length(und, int(i), int(j), weight="weight")
                for i, j in zip(a, b)
            ]
        )
        return int(dist[0]) if scalar else dist

    # Ask igraph only for the unique sources/targets and fan the answers back
    # out - `distances` returns a full sources x targets matrix, so handing it
    # the raw (possibly very repetitive) pair lists would be quadratic.
    ua, a_inv = np.unique(a, return_inverse=True)
    ub, b_inv = np.unique(b, return_inverse=True)
    dmat = np.asarray(
        G.distances(ua.tolist(), ub.tolist(), weights="weight", mode="ALL")
    )
    dist = dmat[a_inv, b_inv]

    return float(dist[0]) if scalar else dist


@utils.map_neuronlist(desc="Searching", allow_parallel=True)
@utils.meshneuron_skeleton(method="node_to_vertex")
def find_main_branchpoint(
    x: "core.NeuronObject",
    method: Union[Literal["longest_neurite"], Literal["betweenness"]] = "betweenness",
    threshold: float = 0.95,
    reroot_soma: bool = False,
) -> Union[int, List[int]]:
    """Find main branch point of unipolar (e.g. insect) neurons.

    Note that this might produce garbage if the neuron is fragmented.

    Parameters
    ----------
    x :             Skeleton | NeuronList
                    May contain multiple neurons.
    method :        "longest_neurite" | "centrality"
                    The method to use:
                      - "longest_neurite" assumes that the main branch point
                        is where the two largest branches converge
                      - "betweenness" uses centrality to determine the point
                        which most shortest paths traverse
    threshold :     float [0-1]
                    Sets the cutoff for method "betweenness". Decrease threshold
                    to be more inclusive (useful if the cell body fiber has
                    little bristles), increase to be more stringent (i.e. when
                    the skeleton is very clean).
    reroot_soma :   bool, optional
                    If True, neuron will be rerooted to soma.

    Returns
    -------
    branch_point :  int | list of int
                    Node ID or list of node IDs of the main branch point(s).

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> navis.find_main_branchpoint(n, reroot_soma=True)
    110
    >>> # Cut neuron into axon, dendrites and primary neurite tract:
    >>> # for this we need to cut twice - once at the main branch point
    >>> # and once at one of its childs
    >>> child = n.nodes[n.nodes.parent_id == 2066].node_id.values[0]
    >>> split = navis.cut_skeleton(n, [2066, child])
    >>> split                                                   # doctest: +SKIP
    <class 'navis.core.neuronlist.NeuronList'> of 3 neurons
              type  n_nodes  n_connectors  n_branches  n_leafs   cable_length    soma
    0  Skeleton     2572             0         170      176  475078.177926    None
    1  Skeleton      139             0           1        3   89983.511392  [3490]
    2  Skeleton     3656             0          63       66  648285.745750    None

    """
    utils.eval_param(
        method, name="method", allowed_values=("longest_neurite", "betweenness")
    )

    if not isinstance(x, core.Skeleton):
        raise TypeError(f'Expected Skeleton(s), got "{type(x)}"')

    # At this point x is Skeleton
    x: core.Skeleton

    # If no branches
    if x.nodes[x.nodes.type == "branch"].empty:
        raise ValueError("Neuron has no branch points.")

    if reroot_soma and not isinstance(x.soma, type(None)):
        x = x.reroot(x.soma, inplace=False)

    if method == "longest_neurite":
        ids = x.nodes.node_id.values
        parents = x.nodes.parent_id.values

        # The second longest path - i.e. the longest of what remains once the
        # longest itself has been peeled off
        _, sc_longest = utils.fastcore.longest_paths(
            ids,
            parents,
            2,
            weights=morpho.mmetrics.parent_dist(x, root_dist=0),
        )

        # Paths run distal -> proximal, so the parent of the second path's last
        # node is where the two converge
        bp = parents[ids == sc_longest[-1]][0]
    else:
        # Get betweenness for each node
        x = morpho.betweenness_centrality(x, directed=True, from_="branch_points")
        # Get branch points with highest centrality
        high_between = (
            x.branch_points.betweenness >= x.branch_points.betweenness.max() * threshold
        )
        candidates = x.branch_points[high_between]

        # If only one nodes just go with it
        if candidates.shape[0] == 1:
            bp = candidates.node_id.values[0]
        else:
            # If multiple points get the farthest one from the root
            root_dists = dist_to_root(x)
            bp = sorted(candidates.node_id.values, key=lambda x: root_dists[x])[-1]

    # This makes sure we get the same data type as in the node table
    # -> Network X seems to sometimes convert integers to floats
    return x.nodes.node_id.dtype.type(bp)


@utils.meshneuron_skeleton(method="split")
def split_neurites(
    x: "core.NeuronObject",
    n: int = 2,
    min_length: Optional[Union[float, str]] = None,
    reroot_soma: bool = False,
) -> "core.NeuronList":
    """Split a neuron into its longest neurites.

    Cuts are based on longest neurites: the first cut is made where the second
    largest neurite merges onto the largest neurite, the second cut is made
    where the third largest neurite merges into either of the first pieces
    and so on.

    Note this cuts a *connected* arbor apart - it has nothing to do with the
    pieces a neuron is already in. Use [`navis.split_components`][] for those.

    Parameters
    ----------
    x :                 Skeleton | Mesh | NeuronList
                        Must be a single neuron.
    n :                 int, optional
                        Number of neurites to split into. Must be >1.
    min_length :          int | str, optional
                        Minimum size of a neurite to be cut off. If too
                        small, will stop cutting. This takes only the longest
                        path in each piece into account! If the neuron(s),
                        has its `.units` set, you can also pass this as a string
                        such as "10 microns".
    reroot_soma :        bool, optional
                        If True, neuron will be rerooted to soma.

    Returns
    -------
    NeuronList

    See Also
    --------
    [`navis.split_components`][]
                        Split a neuron into the pieces it is *already* in.

    Examples
    --------
    >>> import navis
    >>> x = navis.example_neurons(1)
    >>> # Cut into two neurites
    >>> cut1 = navis.split_neurites(x, n=2)
    >>> # Cut into neurites of >10 um size
    >>> cut2 = navis.split_neurites(x, n=float('inf'), min_length=10e3)

    """
    if isinstance(x, core.NeuronList):
        if len(x) == 1:
            x = x[0]
        else:
            raise Exception(
                f"{x.shape[0]} neurons provided. Please provide only a single neuron!"
            )

    if not isinstance(x, core.Skeleton):
        raise TypeError(f'Expected a single Skeleton, got "{type(x)}"')

    if n < 2:
        raise ValueError("Number of neurites must be at least 2.")

    # At this point x is Skeleton
    x: core.Skeleton

    min_length = x.map_units(min_length, on_error="raise")

    if reroot_soma and not isinstance(x.soma, type(None)):
        x.reroot(x.soma, inplace=True)

    ids = x.nodes.node_id.values
    parents = x.nodes.parent_id.values
    weights = morpho.mmetrics.parent_dist(x, root_dist=0)

    # Collect nodes of the n longest neurites. Each is peeled off before the next
    # is sought, which is what makes the second one the longest of the *remainder*.
    # N.B. `min_length` reproduces the quirk this check has always had: it measures
    # the path's whole catchment (every edge whose parent lies on the path, so each
    # twig hanging off it contributes its first edge too), compares with `<=`, and
    # stops the search rather than skipping the one path.
    fragments = utils.fastcore.longest_paths(
        ids,
        parents,
        len(ids) if not np.isfinite(n) else int(n),
        weights=weights,
        min_length=min_length if min_length else None,
    )

    # Next, make some virtual cuts and get the complement of nodes for each
    # fragment. The first fragment starts out as the whole neuron; every other one
    # is the sub-tree distal to its proximal-most node.
    node_sets = [set(ids.tolist())]
    node_sets += [
        set(d.tolist())
        for d in utils.fastcore.descendants(
            ids, parents, [fr[-1] for fr in fragments[1:]]
        )
    ]

    # Remove nodes that are claimed by a subsequent (i.e. more distal) fragment
    for i, s in enumerate(node_sets):
        for s2 in node_sets[i + 1 :]:
            s -= s2

    # Now make neurons - keep node-table order for a stable result
    nl = core.NeuronList(
        [morpho.subset_neuron(x, ids[np.isin(ids, list(s))]) for s in node_sets]
    )

    return nl


@utils.map_neuronlist(desc="Pruning", allow_parallel=True)
@utils.meshneuron_skeleton(method="subset")
def longest_neurite(
    x: "core.NeuronObject",
    n: int = 1,
    reroot_soma: bool = False,
    from_root: bool = True,
    inverse: bool = False,
    inplace: bool = False,
) -> "core.NeuronObject":
    """Return a neuron consisting of only the longest neurite(s).

    Based on geodesic distances.

    Parameters
    ----------
    x :                 Skeleton | NeuronList
                        Neuron(s) to prune.
    n :                 int | slice
                        Number of longest neurites to preserve. For example:
                         - `n=1` keeps the longest neurites
                         - `n=2` keeps the two longest neurites
                         - `n=slice(1, None)` removes the longest neurite
    reroot_soma :       bool
                        If True, neuron will be rerooted to soma.
    from_root :         bool
                        If True, will look for longest neurite from root.
                        If False, will look for the longest neurite between any
                        two tips.
    inverse :           bool
                        If True, will instead *remove* the longest neurite.
    inplace :           bool
                        If False, copy of the neuron will be trimmed down to
                        longest neurite and returned.

    Returns
    -------
    Skeleton/List
                        Pruned neuron.

    See Also
    --------
    [`navis.split_neurites`][]
            Split neuron into its longest neurites.

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> # Keep only the longest neurite
    >>> ln1 = navis.longest_neurite(n, n=1, reroot_soma=True)
    >>> # Keep the two longest neurites
    >>> ln2 = navis.longest_neurite(n, n=2, reroot_soma=True)
    >>> # Keep everything but the longest neurite
    >>> ln3 = navis.longest_neurite(n, n=slice(1, None), reroot_soma=True)

    """
    if not isinstance(x, core.Skeleton):
        raise TypeError(f'Expected Skeleton(s), got "{type(x)}"')

    if isinstance(n, numbers.Number) and n < 1:
        raise ValueError("Number of longest neurites to preserve must be >=1")

    # At this point x is Skeleton
    x: core.Skeleton

    if not inplace:
        x = x.copy()

    if not from_root:
        # Find the two most distal points (N.B. roots can also be "ends")
        leafs = np.unique(
            x.nodes.loc[x.nodes.type.isin(("root", "end")), "node_id"].values
        )

        # We only need each leaf's distance to its farthest leaf, so there is no
        # point in materialising the full leafs x leafs matrix just to take its
        # maximum. Note fastcore uses -1 for unreachable (i.e. fragmented).
        dists, _ = utils.fastcore.geodesic_farthest(
            x.nodes.node_id.values,
            x.nodes.parent_id.values,
            sources=leafs,
            targets=leafs,
            weights=morpho.mmetrics.parent_dist(x, root_dist=0),
        )

        # The longest neurite has two ends and either is a valid place to root it.
        # Which one we get is down to float32 rounding, so pick deterministically:
        # the first leaf (leafs are sorted) that is within rounding of the maximum.
        start = leafs[np.isclose(dists, np.max(dists), rtol=1e-5).argmax()]

        # Reroot to one of the nodes that gives the longest distance
        x.reroot(start, inplace=True)
    elif reroot_soma and not isinstance(x.soma, type(None)):
        x.reroot(x.soma, inplace=True)

    segments = _generate_segments(x, weight="weight")

    if isinstance(n, (int, np.integer)):
        tn_to_preserve: List[int] = [tn for s in segments[:n] for tn in s]
    elif isinstance(n, slice):
        tn_to_preserve = [tn for s in segments[n] for tn in s]
    else:
        raise TypeError(f'Unable to use `n` of type "{type(n)}"')

    if not inverse:
        _ = morpho.subset_neuron(x, tn_to_preserve, inplace=True)
    else:
        _ = morpho.subset_neuron(
            x, ~np.isin(x.nodes.node_id.values, tn_to_preserve), inplace=True
        )

    return x


@utils.lock_neuron
def reroot_skeleton(
    x: "core.NeuronObject", new_root: Union[int, str], inplace: bool = False
) -> "core.Skeleton":
    """Reroot neuron to new root.

    Parameters
    ----------
    x :        Skeleton | NeuronList
               List must contain only a SINGLE neuron.
    new_root : int | iterable
               Node ID(s) of node(s) to reroot to. If multiple new roots are
               provided, they will be rerooted in sequence.
    inplace :  bool, optional
               If True the input neuron will be rerooted in place. If False will
               reroot and return a copy of the original.

    Returns
    -------
    Skeleton
               Rerooted neuron.

    See Also
    --------
    [`navis.Skeleton.reroot`][]
                Quick access to reroot directly from Skeleton/List
                objects.

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1, kind='skeleton')
    >>> # Reroot neuron to its soma
    >>> n2 = navis.reroot_skeleton(n, n.soma)

    """
    if isinstance(x, core.NeuronList):
        if len(x) == 1:
            x = x[0]
        else:
            raise ValueError(f"Expected a single neuron, got {len(x)}")

    if not isinstance(x, core.Skeleton):
        raise ValueError(f'Unable to reroot object of type "{type(x)}"')

    # Make new root an iterable
    new_roots = utils.make_iterable(new_root)

    # Parse new roots
    for i, root in enumerate(new_roots):
        if root is None:
            raise ValueError("New root can not be <None>")

        # If new root is a tag, rather than a ID, try finding that node
        if isinstance(root, str):
            if x.tags is None:
                raise ValueError("Neuron does not have tags")

            if root not in x.tags:
                raise ValueError(
                    f"#{x.id}: Found no nodes with tag {root} - please double check!"
                )

            elif len(x.tags[root]) > 1:
                raise ValueError(
                    f"#{x.id}: Found multiple node with tag "
                    f"{root} - please double check!"
                )
            else:
                new_roots[i] = x.tags[root][0]

    # Every root is a node ID by now - check they exist. Without this the
    # lookup happens deep inside the igraph rerooting below and surfaces as a
    # bare `KeyError: np.int64(123)`, which mentions neither the neuron nor
    # that the number was supposed to be a node ID.
    known_ids = set(x.nodes.node_id.values)
    missing = [r for r in new_roots if r not in known_ids]
    if missing:
        raise ValueError(
            f"#{x.id}: no node with ID {missing[0]} in this neuron. `new_root` "
            "must be an existing node ID (see `x.nodes.node_id`) or a node tag. "
            "Use `navis.find_soma(x)` to root the neuron at its soma."
        )

    # At this point x is Skeleton
    x: core.Skeleton
    # At this point new_roots is list of int
    new_roots: Iterable[int]

    if not inplace:
        # Make a copy
        x = x.copy()
        # Run this in a separate function so that the lock is applied to copy
        _ = reroot_skeleton(x, new_root=new_roots, inplace=True)
        return x

    # Keep track of parent ID dtype
    parentid_dtype = x.nodes.parent_id.dtype

    # One call per root, not one call with every root: `new_root` is documented to
    # reroot "in sequence", so two roots naming the *same* component must leave the
    # second one as that component's root. Passing both at once would instead have
    # them compete. Only the edges between each new root and the root it displaces
    # are reversed; components nobody names keep the root they had.
    for new_root in new_roots:
        # Skip if new root is old root
        if any(x.root == new_root):
            continue

        x.nodes["parent_id"] = utils.fastcore.reroot(
            x.nodes.node_id.values, x.nodes.parent_id.values, [new_root]
        )

    # Make sure parent ID has the same dtype as before: `reroot` promotes to int64
    # where the ID dtype cannot hold the -1 root sentinel.
    if x.nodes.parent_id.dtype != parentid_dtype:
        x.nodes["parent_id"] = x.nodes.parent_id.astype(parentid_dtype)

    # Node types are stale for the old and new roots - let them be recomputed
    x._clear_temp_attr()

    return x


def cut_skeleton(
    x: "core.NeuronObject",
    where: Union[int, str, List[Union[int, str]]],
    ret: Union[Literal["both"], Literal["proximal"], Literal["distal"]] = "both",
) -> "core.NeuronList":
    """Split skeleton at given point and returns two new neurons.

    Split is performed between cut node and its parent node. The cut node itself
    will still be present in both resulting neurons.

    Parameters
    ----------
    x :        Skeleton | NeuronList
               Must be a single skeleton.
    where :    int | str | list
               Node ID(s) or tag(s) of the node(s) to cut. The edge that is
               cut is the one between this node and its parent. So cut node
               must not be a root node! Multiple cuts are performed in the
               order of `cut_node`. Fragments are ordered distal -> proximal.
    ret :      'proximal' | 'distal' | 'both', optional
               Define which parts of the neuron to return. Use this to speed
               up processing when you need only parts of the neuron.

    Returns
    -------
    split :    NeuronList
               Fragments of the input neuron after cutting sorted such that
               distal parts come before proximal parts. For example, with a
               single cut you can expect to return a NeuronList containing two
               neurons: the first contains the part distal and the second the
               part proximal to the cut node.

               The distal->proximal order of fragments is tried to be maintained
               for multiple cuts but this is not guaranteed.

    Examples
    --------
    Cut skeleton at a (somewhat random) branch point

    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> bp = n.nodes[n.nodes.type=='branch'].node_id.values
    >>> dist, prox = navis.cut_skeleton(n, bp[0])

    Make cuts at multiple branch points

    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> bp = n.nodes[n.nodes.type=='branch'].node_id.values
    >>> splits = navis.cut_skeleton(n, bp[:10])

    See Also
    --------
    [`navis.Skeleton.prune_distal_to`][]
    [`navis.Skeleton.prune_proximal_to`][]
            `Skeleton/List` shorthands to this function.
    [`navis.subset_neuron`][]
            Returns a neuron consisting of a subset of its nodes.

    """
    utils.eval_param(ret, name="ret", allowed_values=("proximal", "distal", "both"))

    if isinstance(x, core.NeuronList):
        if len(x) == 1:
            x = x[0]
        else:
            raise Exception(f"Expected a single Skeleton, got {len(x)}")

    if not isinstance(x, core.Skeleton):
        raise TypeError(f'Expected a single Skeleton, got "{type(x)}"')

    if x.n_components != 1:
        raise ValueError(
            f"Unable to cut: neuron {x.id} consists of multiple "
            "disconnected trees. Use navis.heal_skeleton()"
            " to fix."
        )

    # At this point x is Skeleton
    x: core.Skeleton

    # Turn cut node into iterable
    if not utils.is_iterable(where):
        where = [where]

    # Process cut nodes (i.e. if tag)
    node_ids = set(x.nodes.node_id.values)  # O(1) membership in the loop below
    cn_ids: List[int] = []
    for cn in where:
        # If cut_node is a tag (rather than an ID), try finding that node
        if isinstance(cn, str):
            if x.tags is None:
                raise ValueError(f"Neuron {x.id} has no tags")
            if cn not in x.tags:
                raise ValueError(
                    f"#{x.id}: Found no node with tag {cn} - please double check!"
                )
            cn_ids += x.tags[cn]
        elif cn not in node_ids:
            raise ValueError(f'No node with ID "{cn}" found.')
        elif cn in x.root:
            raise ValueError(f'Unable to cut at node "{cn}" - node is root')
        else:
            cn_ids.append(cn)

    # Remove duplicates while retaining order - set() would mess that up
    seen: Set[int] = set()
    cn_ids = [cn for cn in cn_ids if not (cn in seen or seen.add(cn))]

    # Warn if not all returned
    if len(cn_ids) > 1 and ret != "both":
        logger.warning('Multiple cuts should use `ret = "both"`.')

    # Go over all cut_nodes -> order matters!
    res = [x]
    for cn in cn_ids:
        # First, find out in which neuron the cut node is
        to_cut = [n for n in res if cn in n.nodes.node_id.values][0]
        to_cut_ix = res.index(to_cut)

        # Remove this neuron from results (will be cut into two)
        res.remove(to_cut)

        # Cut neuron
        cut = _cut_skeleton(to_cut, cn, ret)

        # If ret != 'both', we will get only a single neuron - therefore
        # make sure cut is iterable
        cut = utils.make_iterable(cut)

        # Add results back to results at same index, proximal first
        for c in cut[::-1]:
            res.insert(to_cut_ix, c)

    return core.NeuronList(res)


def _cut_skeleton(
    x: "core.Skeleton", cut_node: int, ret: str
) -> Union["core.Skeleton", Tuple["core.Skeleton", "core.Skeleton"]]:
    """Cut a neuron at a single node."""
    ids = x.nodes.node_id.values

    # Cutting at a node is just splitting off its sub-tree: everything distal to
    # the cut node (itself included) goes one way, everything else the other. The
    # cut node belongs to both halves - it becomes the distal fragment's root and
    # the proximal fragment's leaf.
    distal = utils.fastcore.descendants(ids, x.nodes.parent_id.values, [cut_node])[0]

    if ret == "distal" or ret == "both":
        dist = morpho.subset_neuron(x, subset=distal, inplace=False)

        # Change new root for dist
        dist.nodes.loc[dist.nodes.node_id == cut_node, "type"] = "root"

        # Clear other temporary attributes
        dist._clear_temp_attr(exclude=["type", "classify_nodes"])

    if ret == "proximal" or ret == "both":
        # Everything above the cut, cut node included - in node-table order
        ss = ids[~np.isin(ids, distal[distal != cut_node])]
        prox = morpho.subset_neuron(x, subset=ss, inplace=False)

        # Change new root for dist
        prox.nodes.loc[prox.nodes.node_id == cut_node, "type"] = "end"

        # Clear other temporary attributes
        prox._clear_temp_attr(exclude=["type", "classify_nodes"])

    if ret == "both":
        return dist, prox
    elif ret == "distal":
        return dist
    else:  # elif ret == 'proximal':
        return prox


def generate_list_of_childs(x: "core.NeuronObject") -> Dict[int, List[int]]:
    """Return list of childs.

    Parameters
    ----------
    x :     Skeleton | NeuronList
            If List, must contain a SINGLE neuron.

    Returns
    -------
    dict
        `{parent_id: [child_id, child_id, ...]}`

    """
    assert isinstance(x, core.Skeleton)

    # The node table already *is* a child->parent map, so we can invert it directly
    # instead of building a graph and asking it for `in_edges` once per node.
    nid = x.nodes.node_id.values
    pid = x.nodes.parent_id.values

    childs: Dict[int, List[int]] = {n: [] for n in nid.tolist()}
    has_parent = pid >= 0
    for c, p in zip(nid[has_parent].tolist(), pid[has_parent].tolist()):
        childs[p].append(c)

    return childs


def _simplified_childs(x: "core.Skeleton") -> Dict[int, List[int]]:
    """`{node: [children]}` on the skeleton reduced to roots, leafs and branches.

    The simplified counterpart to [`generate_list_of_childs`][navis.graph.generate_list_of_childs]:
    the interior of each segment is dropped, so a branch point's children are the
    next branch points or leafs below it rather than its immediate neighbours.

    N.B. children come out in node-table order, which is what makes a walk over
    this reproducible where two of them tie on whatever key the caller sorts by.
    Anything comparing itself against this walk has to enumerate children the same
    way, so call this rather than rebuilding it.
    """
    # N.B. only the first two of what fastcore returns are named: it also hands
    # back the replacement edge weights and, since 0.11.0, a node map, and this
    # wants neither.
    ids, parents = utils.fastcore.simplify_skeleton(
        x.nodes.node_id.values, x.nodes.parent_id.values
    )[:2]

    childs: Dict[int, List[int]] = defaultdict(list)
    for c, p in zip(ids.tolist(), parents.tolist()):
        if p >= 0:
            childs[p].append(c)

    return childs


def node_label_sorting(
    x: "core.Skeleton", weighted: bool = False
) -> List[Union[str, int]]:
    """Return nodes ordered by node label sorting according to Cuntz
    et al., PLoS Computational Biology (2010).

    Parameters
    ----------
    x :         Skeleton
    weighted :  bool
                If True will use actual distances instead of just node count.
                Depending on how evenly spaced your points are, this might not
                make much difference.

    Returns
    -------
    list
        `[root, node_id, node_id, ...]`

    """
    if isinstance(x, core.NeuronList) and len(x) == 1:
        x = x[0]

    if not isinstance(x, core.Skeleton):
        raise TypeError(f'Expected a single Skeleton, got "{type(x)}"')

    if len(x.root) > 1:
        raise ValueError("Unable to process multi-root neurons!")

    # Get relevant terminal nodes (as a set for O(1) membership in the walk below)
    term = set(x.nodes[x.nodes.type == "end"].node_id.values)

    weight = "weight" if weighted else None

    # The walk below sorts each node `n` by "distance to the farthest terminal below
    # `n`" plus "distance from `n` up to the node we are walking from". This used to
    # come out of a directed breaks-by-breaks geodesic matrix, but neither term needs
    # one: the first is `n`'s subtree height and the second - since we only ever walk
    # to an ancestor of `n` - is a difference of root distances. Both are O(N).
    # The matrix was the single largest allocation in navis (19512 x 19512, i.e.
    # 4.5GB, for a 71k node skeleton).
    # Plain dicts: the sort keys below do one scalar lookup per node and pandas
    # charges ~2.5us for each of those.
    height = morpho.manipulation._subtree_height(x, weight=weight).to_dict()
    depth = dist_to_root(x, weight=weight)

    def sort_key(parent):
        """Sort a node's children by the longest path running through them."""
        return lambda n: height[n] + (depth[n] - depth[parent])

    # Get starting points (i.e. branches off the root) and sort by longest
    # path to a terminal (note we're operating on the simplified version
    # of the skeleton - roots, leafs and branch points only)
    childs = _simplified_childs(x)

    curr_points = sorted(childs[x.root[0]], key=sort_key(x.root[0]), reverse=True)

    # Walk from root towards terminals, prioritising longer branches
    nodes_walked = []
    while curr_points:
        nodes_walked.append(curr_points.pop(0))
        # If the current point is a terminal point, stop here
        if nodes_walked[-1] in term:
            pass
        else:
            new_points = sorted(
                childs[nodes_walked[-1]],
                key=sort_key(nodes_walked[-1]),
                reverse=True,
            )
            curr_points = new_points + curr_points

    # Translate into segments
    node_list = [x.root[0:]]
    # Note that we're inverting here so that the segments are ordered
    # proximal -> distal (i.e. root to tips)
    seg_dict = {s[0]: s[::-1] for s in _break_segments(x)}

    for n in nodes_walked:
        # Note that we're skipping the first (proximal) node to avoid double
        # counting nodes
        node_list.append(seg_dict[n][1:])

    return np.concatenate(node_list, dtype=int)


def subset_igraph(x: "core.Skeleton", keep) -> "igraph.Graph":
    """Induce the sub-graph of a neuron's igraph on a set of node IDs.

    The igraph equivalent of `x.graph.subgraph(keep)`, but without paying to build
    the networkx graph. `node_id` is carried over onto the new vertices.
    """
    G: igraph.Graph = x.igraph
    ids = np.asarray(G.vs["node_id"])
    keep = np.fromiter(keep, dtype=ids.dtype, count=len(keep))
    return G.subgraph(np.where(np.isin(ids, keep))[0])


def _sparse_adjacency(x: "core.NeuronObject", directed: bool = True) -> csr_matrix:
    """Weighted adjacency matrix of a neuron, in node/vertex table order.

    Rows and columns are indices into the node (Skeleton) or vertex (Mesh) table.
    `directed=True` keeps a skeleton's child->parent orientation; a mesh has no
    orientation to keep, so both directions are always emitted for it.
    """
    if isinstance(x, core.Skeleton):
        indptr, indices, data = utils.fastcore.adjacency(
            x.nodes.node_id.values,
            x.nodes.parent_id.values,
            weights=morpho.mmetrics.parent_dist(x, root_dist=0),
        )
        n = len(x.nodes)
        A = csr_matrix((data, indices, indptr), shape=(n, n), dtype=np.float32)
        # A skeleton has no reciprocal edges, so the transpose can't double-count
        return (A + A.T).tocsr() if not directed else A

    # A Mesh's edges are undirected to begin with, so both directions go in and
    # `directed` has nothing to act on. N.B. straight off the edge list, the same
    # way `geodesic_matrix` gets a mesh's adjacency - building the igraph first
    # only to read its edges back out is what this used to do.
    edges, lengths = utils.mesh_unique_edges(x, return_lengths=True)
    n = len(x.vertices)
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.concatenate([lengths, lengths]).astype(np.float32)
    return csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)


def connecting_nodes(
    x: Union["core.Skeleton", nx.DiGraph], ss: Sequence[Union[str, int]]
) -> Tuple[np.ndarray, Union[int, str]]:
    """Return the nodes needed to connect all nodes in subset `ss`.

    That is `ss` plus whatever has to come along for the ride: for each node,
    the path back to the point where it meets the rest of the subset. Nothing to
    do with [`navis.connected_components`][], which asks what already *is*
    connected rather than what it would take.

    Parameters
    ----------
    x :         navis.Skeleton | nx.DiGraph
                Neuron (or graph thereof) to get subgraph for.
    ss :        list | array-like
                Node IDs of node to subset to.

    Returns
    -------
    np.ndarray
                Node IDs of the connecting subgraph.
    root ID
                ID of the node most proximal to the old root in the
                connecting subgraph.

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> ends = n.nodes[n.nodes.type.isin(['end', 'root'])].node_id.values
    >>> sg, root = navis.graph.connecting_nodes(n, ends)
    >>> # Since we asked for a subgraph connecting all terminals + root,
    >>> # we expect to see all nodes in the subgraph
    >>> sg.shape[0] == n.nodes.shape[0]
    True

    """
    # `src` is the Skeleton we can pull node/parent arrays from (if any). For a
    # bare graph we fall back to reading its edges.
    src = None
    g = None
    if isinstance(x, core.NeuronList):
        if len(x) == 1:
            src = x[0]
    elif isinstance(x, core.Skeleton):
        src = x
    elif isinstance(x, (nx.DiGraph, igraph.Graph)):
        g = x
    else:
        raise TypeError(f'Input must be a single Skeleton or graph, got "{type(x)}".')

    # Build a `node -> parent` map and the set of nodes in the graph.
    # `parent.get(n)` returns None for roots (which have no parent) - this is our
    # natural walk terminator (mirrors `next(g.successors(n), None)`).
    # For a Skeleton we can build this straight from the node table (faster and
    # avoids touching a graph library at all); for a bare graph (e.g. the induced
    # sub-graph passed by `split_axon_dendrite`) we do a single pass over its edges.
    if src is not None:
        nid = src.nodes.node_id.values
        pid = src.nodes.parent_id.values
        parent = {n: p for n, p in zip(nid, pid) if p >= 0}
        nodes = set(nid.tolist())
    elif isinstance(g, igraph.Graph):
        ids = np.asarray(g.vs["node_id"])
        edges = np.asarray(g.get_edgelist(), dtype=np.int64).reshape(-1, 2)
        # edge (u, v) => v is parent of u
        parent = dict(zip(ids[edges[:, 0]].tolist(), ids[edges[:, 1]].tolist()))
        nodes = set(ids.tolist())
    else:
        parent = {u: v for u, v in g.edges()}  # edge (u, v) => v is parent of u
        nodes = set(g.nodes())

    ss = set(ss)
    missing = ss - nodes
    if missing:
        missing = np.array(list(missing)).astype(str)  # do NOT remove list() here!
        raise ValueError(f"Nodes not found: {','.join(missing)}")

    # Find nodes that are leafs WITHIN the subset: an ss node is an ss-leaf iff none
    # of its children are in ss, i.e. it is not the parent of any other ss node.
    ss_parents = {parent[n] for n in ss if parent.get(n) in ss}
    leafs = ss - ss_parents

    # Memoised depth (distance to root; root = 0). Each node is resolved exactly
    # once thanks to the `n in depth` early stop -> O(N). Replaces the old
    # `longest_path.index(...)` ordering key (which was O(depth) per lookup).
    depth = {}

    def fill_depth(n):
        stack = []
        while n is not None and n not in depth:
            stack.append(n)
            n = parent.get(n)
        d = depth[n] + 1 if n is not None else 0
        for m in reversed(stack):
            depth[m] = d
            d += 1

    # Walk every ss-leaf towards its root, stopping as soon as we hit an already
    # visited node. We accumulate, per node, how many leaf-walks pass through it
    # (`pass_count`) and which component (terminal root) it belongs to (`comp_of`).
    # Components are derived implicitly: leaves ending at the same root share one.
    pass_count = {}
    comp_of = {}
    comp_leaves = defaultdict(list)
    comp_touched = defaultdict(list)
    for leaf in leafs:
        fill_depth(leaf)
        # First pass: walk to root, counting passes and finding the component root.
        n = leaf
        root = leaf
        while n is not None:
            pass_count[n] = pass_count.get(n, 0) + 1
            root = n
            n = parent.get(n)
        comp_leaves[root].append(leaf)
        # Second pass: tag every (not yet tagged) node on this path with its
        # component root and record it as touched. Early-stops where a previous
        # leaf-walk already tagged the shared upper segment.
        n = leaf
        while n is not None and n not in comp_of:
            comp_of[n] = root
            comp_touched[root].append(n)
            n = parent.get(n)

    # Group ss nodes by component once (every ss node lies on some leaf-walk and is
    # therefore tagged in `comp_of`). Avoids re-scanning all of ss per component.
    ss_by_comp = defaultdict(list)
    for n in ss:
        ss_by_comp[comp_of[n]].append(n)

    include = set()
    new_roots = []
    for root, cleaves in comp_leaves.items():
        need = len(cleaves)
        # Nodes common to ALL leaf-walks form a contiguous root->LCA chain; the LCA
        # (branch point / new root) is the deepest of them.
        common = [n for n in comp_touched[root] if pass_count[n] == need]
        lca = max(common, key=lambda n: depth[n])

        # Include, for each leaf, every node up to and including the LCA.
        for leaf in cleaves:
            n = leaf
            while n is not None and n not in include:
                include.add(n)
                if n == lca:
                    break
                n = parent.get(n)

        # Edge case: ss may contain nodes that are strict ancestors of the LCA
        # (they are never ss-leaves, so they're not in `include` yet). The new root
        # must be the most proximal of those (closest to the old root, i.e. smallest
        # depth); we then fill the *full* chain from the LCA up to it so the result
        # stays connected (the old code added only the ss nodes, leaving a
        # disconnected gap between the LCA and the new root).
        this_ss = ss_by_comp[root]
        proximal = [n for n in this_ss if n not in include]
        if proximal:
            new_root = min(proximal, key=lambda n: depth[n])
            new_roots.append(new_root)
            # All proximal ss nodes are ancestors of the LCA, so walking from the
            # LCA towards the root reaches `new_root` and passes every one of them.
            n = lca
            while True:
                include.add(n)
                if n == new_root:
                    break
                n = parent.get(n)
        else:
            new_roots.append(lca)

    return np.array(list(include)), new_roots


@utils.rebuilds("nodes")
def insert_nodes(
    x: "core.Skeleton",
    where: List[tuple],
    coords: List[tuple] = None,
    validate: bool = True,
    inplace: bool = False,
) -> Optional["core.Skeleton"]:
    """Insert new nodes between existing nodes.

    Parameters
    ----------
    x :         Skeleton
                Neuron to insert new nodes into.
    where :     list of node pairs
                Must be a list of node ID pairs. A new node will be added
                between the nodes of each pair (see examples).
    coords :    None | list of (x, y, z) coordinates | list of fractions
                Can be:
                 - `None`: new nodes will be inserted exactly between the two
                             nodes
                 - (N, 3) array of coordinates for the newly inserted nodes
                 - (N, ) array of fractional distances [0-1]: e.g. 0.25 means
                   that a new node will be inserted a quarter of the way between
                   the two nodes (from the child's perspective)
    validate :  bool
                If True, will make sure that pairs in `where` are always
                in (parent, child) order. If you know this to already be the
                case, set `validate=False` to save some time.
    inplace :   bool
                If True, will rewire the neuron inplace. If False, will return
                a rewired copy of the neuron.

    Returns
    -------
    Skeleton

    Examples
    --------
    Insert new nodes between some random points

    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> n.n_nodes
    4465
    >>> where = n.nodes[['parent_id', 'node_id']].values[100:200]
    >>> _ = navis.insert_nodes(n, where=where, inplace=True)
    >>> n.n_nodes
    4565

    """
    utils.eval_param(x, name="x", allowed_types=(core.Skeleton,))

    where = np.asarray(where)
    if where.ndim != 2 or where.shape[1] != 2:
        raise ValueError(
            f"Expected `where` to be a (N, 2) list of pairs. Got {where.shape}"
        )

    # Validate if that's desired
    if validate:
        # Setup to get parents
        parent = x.nodes.set_index("node_id").parent_id

        # Get parents of the left and the right nodes of each pair
        parent_left = parent.loc[where[:, 0]].values
        parent_right = parent.loc[where[:, 1]].values

        # Check if the right node is parent of the left or the other way around
        correct_order = where[:, 0] == parent_right
        swapped = where[:, 1] == parent_left
        not_connected = ~(correct_order | swapped)

        if np.any(not_connected):
            raise ValueError(
                f"The following pairs are not connected: {where[not_connected]}"
            )

        # Flip nodes where necessary to sure we have (parent, child) order
        if np.any(swapped):
            where[swapped, :] = where[swapped][:, [1, 0]]

    # If not provided, generate coordinates in the center between each node pair
    if isinstance(coords, type(None)):
        node_locs = x.nodes.set_index("node_id")[["x", "y", "z"]]
        left_loc = node_locs.loc[where[:, 0]].values
        right_loc = node_locs.loc[where[:, 1]].values

        # Find center between each node
        coords = left_loc + (right_loc - left_loc) / 2

    coords = np.asarray(coords)
    # Make sure we have correct coordinates
    if coords.shape[0] != where.shape[0]:
        raise ValueError(
            f"Expected {where.shape[0]} coordinates or distances, got {coords.shape[0]}"
        )

    # If array of fractional distances translate to coordinates
    if coords.ndim == 1:
        node_locs = x.nodes.set_index("node_id")[["x", "y", "z"]]
        left_loc = node_locs.loc[where[:, 0]].values
        right_loc = node_locs.loc[where[:, 1]].values

        # Find center between each node
        coords = left_loc + (right_loc - left_loc) * coords.reshape(-1, 1)

    # For the moment, we will interpolate the radius
    rad = x.nodes.set_index("node_id").radius
    new_rad = (rad.loc[where[:, 0]].values + rad.loc[where[:, 1]].values) / 2

    # Generate table for new nodes
    new_nodes = pd.DataFrame()
    max_id = x.nodes.node_id.max() + 1
    new_nodes["node_id"] = np.arange(max_id, max_id + where.shape[0]).astype(int)
    new_nodes["parent_id"] = where[:, 0]
    new_nodes["x"] = coords[:, 0]
    new_nodes["y"] = coords[:, 1]
    new_nodes["z"] = coords[:, 2]
    new_nodes["radius"] = new_rad

    # Merge tables
    nodes = pd.concat(
        [x.nodes, new_nodes], join="outer", axis=0, sort=True, ignore_index=True
    )

    # Remap nodes
    new_parents = dict(zip(where[:, 1], new_nodes.node_id.values))
    to_rewire = nodes.node_id.isin(new_parents)
    nodes.loc[to_rewire, "parent_id"] = (
        nodes.loc[to_rewire, "node_id"]
        .map(new_parents)
        .values.astype(nodes.dtypes["parent_id"], copy=False)
    )

    if not inplace:
        x = x.copy()

    x._nodes = nodes

    # Every old node came through at its own ID, and the inserted ones are
    # genuinely new - `pd.concat` put them after the old ones, in order. That
    # second half is why nothing aligned to the nodes can be carried: there is
    # no label, no radius-like value, nothing to give a node that was not there
    # before, and inventing one is worse than saying so.
    kept = nodes.node_id.values.copy()
    kept[len(kept) - len(new_nodes):] = core.schema.DROPPED
    return x, core.schema.Rebuild(kept=kept)


def remove_nodes(
    x: "core.Skeleton", which: List[int], inplace: bool = False
) -> Optional["core.Skeleton"]:
    """Drop nodes from neuron without disconnecting it.

    Dropping node 2 from 1->2->3 will lead to connectivity 1->3.

    Parameters
    ----------
    x :         Skeleton
                Neuron to remove nodes from.
    which :     list of node IDs
                IDs of nodes to remove.
    inplace :   bool
                If True, will rewire the neuron inplace. If False, will return
                a rewired copy of the neuron.

    Returns
    -------
    Skeleton

    Examples
    --------
    Drop points from a neuron

    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> n.n_nodes
    4465
    >>> # Drop a hundred nodes
    >>> n2 = navis.remove_nodes(n, n.nodes.node_id.values[100:200])
    >>> n2.n_nodes
    4365

    See Also
    --------
    [`navis.collapse_nodes`][]
            Collapse a group of nodes into a single node. This is different
            from removing nodes as it will potentially change the structure of
            the neuron.

    """
    utils.eval_param(x, name="x", allowed_types=(core.Skeleton,))

    if not utils.is_iterable(which):
        which = [which]
    which = np.asarray(which)

    miss = ~np.isin(which, x.nodes.node_id.values)
    if np.any(miss):
        raise ValueError(f"{len(miss)} node IDs not found in neuron")

    if not inplace:
        x = x.copy()

    # Generate new list of parents
    lop = dict(zip(x.nodes.node_id.values, x.nodes.parent_id.values))

    # Rewire to skip the to-be-removed nodes
    for n in which:
        lop.update({c: lop[n] for c, p in lop.items() if p == n})

    # Rewire neuron
    x.nodes["parent_id"] = x.nodes.node_id.map(lop)

    # Drop nodes
    x.nodes = x.nodes[~x.nodes.node_id.isin(which)].copy()

    # Clear temporary attributes
    x._clear_temp_attr()

    return x


def collapse_nodes(
    x: "core.Skeleton",
    which: List[int],
    new_co: Iterable[Union[float, int]] = None,
    inplace: bool = False,
) -> Optional["core.Skeleton"]:
    """Collapse group of nodes into a single node.

    Parameters
    ----------
    x :         Skeleton
                Neuron to collapse nodes in.
    which :     list of node IDs
                IDs of nodes to collapse. The first node in the list will be
                the one that the others are collapsed into.
    new_co :    (x, y, z) coordinates, optional
                Coordinates for the new node. If not given, will use the
                center of the nodes to be collapsed.
    inplace :   bool
                If True, will modify the neuron inplace. If False, will return
                a modified copy of the neuron.

    Returns
    -------
    Skeleton

    Examples
    --------
    Collapse a group of nodes into a single node

    >>> import navis
    >>> import numpy as np
    >>> n = navis.example_neurons(1)
    >>> n.n_nodes
    4465
    >>> # Collapse nodes around the soma
    >>> soma_dist = np.linalg.norm(n.vertices - n.soma_pos, axis=1)
    >>> to_collapse = n.nodes.node_id[soma_dist < 1000].values
    >>> x = navis.collapse_nodes(n, to_collapse, new_co=n.soma_pos[0])
    >>> x.n_nodes
    4415

    See Also
    --------
    [`navis.remove_nodes`][]
            Remove nodes from the neuron without changing the structure.

    """
    utils.eval_param(x, name="x", allowed_types=(core.Skeleton,))

    if not utils.is_iterable(which):
        which = [which]
    which = np.asarray(which)

    miss = ~np.isin(which, x.nodes.node_id.values)
    if np.any(miss):
        raise ValueError(f"{len(miss)} node IDs not found in neuron")

    if not inplace:
        x = x.copy()

    # We will use the lowest node ID as the node to collapse into
    center_node = np.min(which)

    # Move that new center node
    if new_co is None:
        new_co = x.nodes.loc[x.nodes.node_id.isin(which), ["x", "y", "z"]].values.mean(
            axis=0
        )
    x.nodes.loc[x.nodes.node_id == center_node, ["x", "y", "z"]] = new_co

    # `mapping` is in ID space, not index space (see the large-ID regression test)
    node_ids = x.nodes.node_id.values
    collapsed = np.isin(node_ids, which)
    mapping = node_ids.copy()
    mapping[collapsed] = center_node

    _, new_parents = utils.fastcore.contract_nodes(
        node_ids, x.nodes.parent_id.values, mapping
    )

    # Rewire and drop the collapsed nodes in one step: the survivors are every node
    # bar the ones we just folded away, which is exactly what `contract_nodes`
    # returns - in the same order.
    x.nodes = x.nodes[~collapsed | (node_ids == center_node)].copy()
    x.nodes["parent_id"] = new_parents.astype(x.nodes.parent_id.dtype)

    # Check if there is a vertex map to update. Note we build a new array rather
    # than writing into the old one: skeletor hands its `mesh_map` back as a
    # read-only view, and even where it is writeable it may be shared with a
    # copy of this neuron.
    if hasattr(x, "vertex_map"):
        vertex_map = np.asarray(x.vertex_map)
        x.vertex_map = np.where(np.isin(vertex_map, which), center_node, vertex_map)

    # Clear temporary attributes
    x._clear_temp_attr()

    return x


def rewire_skeleton(
    x: "core.Skeleton", g: nx.Graph, root: Optional[id] = None, inplace: bool = False
) -> Optional["core.Skeleton"]:
    """Rewire neuron from graph.

    This function takes a graph representation of a neuron and rewires its
    node table accordingly. This is useful if we made changes to the graph
    (i.e. adding or removing edges) and want those to propagate to the node
    table.

    Parameters
    ----------
    x :         Skeleton
                Neuron to be rewired.
    g :         networkx.Graph
                Graph to use for rewiring. Please note that directionality (if
                present) is not taken into account. Nodes not included in the
                graph will be disconnected (i.e. won't have a parent). Nodes
                in the graph but not in the table are ignored!
    root :      int
                Node ID for the new root. If not given, will try to use the
                current root.
    inplace :   bool
                If True, will rewire the neuron inplace. If False, will return
                a rewired copy of the neuron.

    Returns
    -------
    Skeleton

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> n.n_components
    1
    >>> # Drop one edge from graph
    >>> g = n.graph.copy()
    >>> g.remove_edge(310, 309)
    >>> # Rewire neuron
    >>> n2 = navis.rewire_skeleton(n, g, inplace=False)
    >>> n2.n_components
    2

    """
    assert isinstance(x, core.Skeleton), f"Expected Skeleton, got {type(x)}"
    assert isinstance(g, nx.Graph), f"Expected networkx graph, got {type(g)}"

    if not inplace:
        x = x.copy()

    # A *view*: nothing below cares about direction (the fastcore calls all treat
    # edges as undirected), so all this has to do is collapse a reciprocal pair
    # into one edge. `to_undirected()` proper deep-copies every node and edge
    # attribute dict, which cost more than the rest of this function put together.
    if g.is_directed():
        g = g.to_undirected(as_view=True)

    if not root:
        root = x.root[0] if x.root[0] in g.nodes else next(iter(g.nodes), None)

    # Work in index space from here on: `parents_from_edges` takes edges as
    # indices into 0..n-1. Nodes in the graph but not in the table are dropped
    # (as they always were); nodes in the table but not in the graph name no edge
    # and so come back as isolated roots, which is the same as before.
    ids = x.nodes.node_id.values
    ix = pd.Series(np.arange(len(ids)), index=ids)

    # Drop any edge with an endpoint outside the node table, then translate to
    # indices. N.B. keep these arrays *typed* - an object-dtype edge array turns
    # the `isin` below into a quadratic scan (and it is the hot path here).
    raw = np.asarray(list(g.edges)).reshape(-1, 2)
    keep = np.isin(raw[:, 0], ids) & np.isin(raw[:, 1], ids)
    edges = np.stack(
        [ix.reindex(raw[keep, 0]).values, ix.reindex(raw[keep, 1]).values], axis=1
    ).astype(np.int32)

    # The MST is only needed to break cycles. If the graph is already a forest
    # (which is the common case - e.g. when edges were only removed, or when
    # fragments were bridged) we can skip it: the MST of a forest is that same
    # forest, and computing it is expensive on large neurons. Note the MST is what
    # decides *which* edge of a cycle to drop (the heaviest); `parents_from_edges`
    # only decides which way the survivors point.
    n_cc = len(np.unique(utils.fastcore.connected_components_graph(edges, len(ids))))
    if len(edges) != (len(ids) - n_cc):
        weights = np.fromiter(
            (g.edges[u, v].get("weight", 1) for u, v in raw[keep]),
            dtype=np.float32,
            count=int(keep.sum()),
        )
        edges = edges[
            utils.fastcore.minimum_spanning_tree(edges, len(ids), weights=weights)
        ]

    # One search over the whole graph, rather than a DFS per component. Components
    # holding no named root fall back to their lowest node index - the docstring
    # already promises nothing better than "arbitrary" for those.
    roots = None if root is None else [int(ix[root])]
    parents = utils.fastcore.parents_from_edges(edges, len(ids), roots=roots)[0]

    # Update parent IDs (translating back out of index space; -1 stays -1)
    x.nodes["parent_id"] = np.where(parents >= 0, ids[parents], -1).astype(
        x.nodes.parent_id.dtype
    )

    x._clear_temp_attr()

    return x


def match_mesh_skeleton(mesh, skeleton):
    """Match vertices of Mesh to nodes of Skeleton.

    Parameters
    ----------
    mesh :      Mesh
                Mesh to match.
    skeleton :  Skeleton
                Skeleton to match.

    Returns
    -------
    np.ndarray
                Array of skeleton node IDs for each vertex in the mesh.

    """
    if not isinstance(mesh, core.Mesh):
        raise TypeError(f"Expected Mesh, got {type(mesh)}")

    if not isinstance(skeleton, core.Skeleton):
        raise TypeError(f"Expected Skeleton, got {type(skeleton)}")

    # Generate a KDTree for the skeleton
    tree = graph.neuron2KDTree(skeleton)

    # Find closest node for each vertex
    dist, ix = tree.query(mesh.vertices, k=1)

    return skeleton.nodes.node_id.values[ix]


@utils.map_neuronlist(desc="Propagating labels", allow_parallel=True)
def propagate_labels(
    x,
    labels,
    clamping=True,
    weights=None,
    directed=False,
    max_iter=10000,
    tol=1,
    return_probs: Union[bool, Literal["softmax", "raw"]] = False,
    verbose=False,
):
    """Propagate labels from a subset of nodes/vertices to the rest of the neuron.

    Parameters
    ----------
    x :         Skeleton | Mesh
                Neuron(s) to propagate labels in.
    labels :    array | dict | str
                Labels to propagate. Can be:
                    - array-like: a label for each node/vertex
                    - dict: mapping node IDs/vertex indices to labels
                    - str: name of a neuron property
                Note that None/NaN will be treated as unlabeled and will not
                be propagated.
    clamping :  bool | "soft"
                Whether to clamp labeled nodes during propagation:
                  - If `True` (default), labeled nodes can not change their label.
                  - If `False`, labeled nodes can change their label just like any other node.
                  - If "soft", they can change but will be biased towards their original label.
                    You can provide a bias strength by using "soft:alpha" where alpha is a float
                    between 0 and 1 (e.g. "soft:0.5"). The lower the alpha, the stronger the
                    bias towards the original label.
    weights :   dict, optional
                Optional importance weights for each label. The keys should be the
                same values as in `labels` (e.g. "pre", "post") and the values should be
                floats (higher = more influence on propagation). If `None` (default), all
                labels are treated equally.
    directed :  bool
                Whether to treat the graph as directed during propagation. Only
                applicable for Skeletons. If `True`, labels will only propagate
                from parent to child nodes. If `False` (default), labels can propagate in both
                directions.
    max_iter :  int
                Maximum number of iterations for label propagation.
    tol :       int | float
                Tolerance for convergence. If >=1 (default), we stop when not a single node's
                hard assignment has changed in `tol` iterations. That does not mean that the
                probabilities have fully converged but it's a sign that things are slowing down.
                If < 1, we stop when the maximum change in probabilities across all nodes is
                less than `tol`.
    return_probs : bool | "softmax" | "raw"
                Whether to also return the propagated probabilities. If not `False`,
                will return a tuple of `(prop, probs, labels)` (see Returns).
                The format of `probs` depends on the value of `return_probs`:
                  - `False` (default) returns only `pred` (hard labels)
                  - `True` means `probs` are row-normalized scores (sum to 1 per node)
                  - `softmax` means `probs` are softmaxed scores
                  - `raw` means `probs` are the raw propagated scores without any normalization

    Returns
    -------
    prop :      array
                Object-dtype array of propagated labels for each node/vertex in the neuron.
                Nodes/vertices that weren't visited (e.g. disconnected from any labeled
                nodes) will have NaN.
    (prop, probs, labels) : tuple, optional
                If `return_probs!=False`, returns a tuple containing:
                  - `prop`: array of propagated labels
                  - `probs`: (n_nodes, n_labels) float array of normalized scores
                  - `labels`: list of label names corresponding to `probs` columns

    Examples
    --------
    >>> import navis
    >>> import numpy as np
    >>> n = navis.example_neurons(1)

    >>> # Prepare labels to propagate:
    >>> # Here we will label nodes based on whether they are pre- and postsynaptic sites
    >>> pre_nodes = n.snap(n.presynapses[['x', 'y', 'z']].values)[0]
    >>> post_nodes = n.snap(n.postsynapses[['x', 'y', 'z']].values)[0]
    >>> labels = np.full(n.n_nodes, np.nan, dtype=object)
    >>> labels[post_nodes] = "post"
    >>> labels[pre_nodes] = "pre"
    >>> labels[:5]  # most labels will be NaN since only a subset of nodes are labeled
    array([nan, nan, nan, nan, nan], dtype=object)

    >>> # Propagate labels
    >>> # We're not clamping here which will allow the initial labels to be overridden if the
    >>> # neighborhood suggests a different label.
    >>> prop_labels = navis.graph.graph_utils.propagate_labels(n, labels, clamping=False)
    >>> prop_labels[:5]
    array(['post', 'post', 'post', 'post', 'post'], dtype=object)

    >>> # To visualize
    >>> # navis.plot3d(n, color_by=prop_labels, palette={"pre": "red", "post": "blue"})

    """
    if not isinstance(x, (core.Skeleton, core.Mesh)):
        raise TypeError(f"Expected Skeleton or Mesh, got {type(x)}")

    assert return_probs in (
        False,
        True,
        "softmax",
        "raw",
    ), f"Invalid value for return_probs: {return_probs}"
    assert max_iter > 0, "max_iter must be a positive integer"
    assert isinstance(tol, (int, float)) and tol > 0, "tol must be a positive float"

    if isinstance(labels, str):
        if isinstance(x, core.Skeleton):
            if labels not in x.nodes.columns:
                raise ValueError(f'No node property "{labels}" found in neuron.')
            elif getattr(x, labels).shape[0] != len(x.nodes):
                raise ValueError(
                    f'Length of node property "{labels}" does not match number of nodes ({len(x.nodes)})'
                )
            labels = dict(zip(x.nodes.node_id.values, x.nodes[labels].values))
        elif isinstance(x, core.Mesh):
            if not hasattr(x, labels):
                raise ValueError(f'No vertex property "{labels}" found in neuron.')
            elif getattr(x, labels).shape[0] != len(x.vertices):
                raise ValueError(
                    f'Length of vertex property "{labels}" does not match number of vertices ({len(x.vertices)})'
                )
            labels = dict(zip(range(len(x.vertices)), getattr(x, labels)))
    elif not isinstance(labels, dict):
        if isinstance(x, core.Skeleton):
            if len(labels) != len(x.nodes):
                raise ValueError(
                    f"Length of labels ({len(labels)}) does not match number of nodes ({len(x.nodes)})"
                )
            labels = dict(zip(x.nodes.node_id.values, labels))
        elif isinstance(x, core.Mesh):
            if len(labels) != len(x.vertices):
                raise ValueError(
                    f"Length of labels ({len(labels)}) does not match number of vertices ({len(x.vertices)})"
                )
            labels = dict(zip(range(len(x.vertices)), labels))

    # Drop missing labels from the dict
    labels = {k: v for k, v in labels.items() if not pd.isnull(v)}

    # Row order throughout is the node/vertex table's own order, which is what
    # `return_probs` exposes as the row order of `F`.
    is_skeleton = isinstance(x, core.Skeleton)

    nodes = (
        x.nodes.node_id.values.tolist() if is_skeleton else list(range(len(x.vertices)))
    )
    n = len(nodes)
    node_index = {node: i for i, node in enumerate(nodes)}

    label_set = sorted(set(labels.values()))
    label_index = {l: i for i, l in enumerate(label_set)}
    k = len(label_set)

    # Optional importance weights per label or per labeled node
    label_weights = np.ones(k, dtype=np.float32)
    per_node_weights = None

    if weights is not None:
        if isinstance(weights, dict):
            for l, w in weights.items():
                if l not in label_index:
                    raise ValueError(
                        f"Unknown label '{l}' in weights (expected one of: {label_set})"
                    )
                label_weights[label_index[l]] = float(w)
        elif isinstance(weights, str):
            if isinstance(x, core.Skeleton):
                if weights not in x.nodes.columns:
                    raise ValueError(f'No node property "{weights}" found in neuron.')
                elif getattr(x, weights).shape[0] != len(x.nodes):
                    raise ValueError(
                        f'Length of node property "{weights}" does not match number of nodes ({len(x.nodes)})'
                    )
                per_node_weights = getattr(x, weights).values.astype(np.float32)
            elif isinstance(x, core.Mesh):
                if not hasattr(x, weights):
                    raise ValueError(f'No vertex property "{weights}" found in neuron.')
                elif getattr(x, weights).shape[0] != len(x.vertices):
                    raise ValueError(
                        f'Length of vertex property "{weights}" does not match number of vertices ({len(x.vertices)})'
                    )
                per_node_weights = getattr(x, weights).astype(np.float32)
        else:
            weights_arr = np.asarray(weights, dtype=np.float32)
            if weights_arr.ndim != 1:
                raise ValueError(
                    f"Expected 1D array-like weights, got shape {weights_arr.shape}"
                )
            if weights_arr.shape[0] == n:
                per_node_weights = weights_arr
            elif weights_arr.shape[0] == len(labels):
                # Align per-label weights with the order of the provided labels dict
                # (insertion order is preserved in Python 3.7+)
                per_node_weights = np.zeros(n, dtype=np.float32)
                for w, node in zip(weights_arr, labels.keys()):
                    per_node_weights[node_index[node]] = w
            else:
                raise ValueError(
                    "Weights array must have length equal to the number of nodes "
                    f"({n}) or the number of labeled nodes ({len(labels)})."
                )

    A = _sparse_adjacency(x, directed=directed)

    # Row-normalize adjacency (sparse)
    row_sums = np.asarray(A.sum(axis=1)).flatten().astype(np.float32)
    row_sums[row_sums == 0] = 1
    D_inv = diags(1.0 / row_sums, dtype=np.float32)
    S = D_inv.dot(A)

    # Label matrix (float32)
    Y = np.zeros((n, k), dtype=np.float32)
    labeled_mask = np.zeros(n, dtype=bool)

    for node, label in labels.items():
        i = node_index[node]
        w = label_weights[label_index[label]]
        if per_node_weights is not None:
            w *= per_node_weights[i]
        Y[i, label_index[label]] = w
        labeled_mask[i] = True

    F = Y.copy()

    # Parse clamping parameter
    alpha = 0.5
    if isinstance(clamping, str):
        if "soft:" in clamping:
            try:
                alpha = float(clamping.split(":")[1])
                if not (0 <= alpha <= 1):
                    raise ValueError
            except (IndexError, ValueError):
                raise ValueError(
                    f'Invalid clamping parameter "{clamping}". Expected format "soft:alpha" where alpha is a float between 0 and 1.'
                )
            clamping = "soft"

    if tol >= 1:
        prev_hard = np.argmax(F, axis=1)
        n_hard = 0

    F_max = F.max() + 1e-16  # to normalize change for convergence check

    for it in range(max_iter):
        # Propagate labels
        F_new = S @ F

        # Clamp labeled nodes
        if clamping == "soft":
            F_new[labeled_mask] = (
                alpha * F_new[labeled_mask] + (1 - alpha) * Y[labeled_mask]
            )
        elif clamping:
            F_new[labeled_mask] = Y[labeled_mask]

        if tol >= 1:
            hard = np.argmax(F_new, axis=1)
            if np.array_equal(hard, prev_hard):
                n_hard += 1
            if n_hard > tol:
                change = np.abs(F_new - F).max() / F_max
                F = F_new
                break  # no change in hard labels between iterations
            prev_hard = hard
        elif change := (np.abs(F_new - F).max() / F_max) < tol:
            F = F_new
            break

        # Make sure we have a change value to report if we hit max_iter without convergence
        if it == max_iter - 1:
            change = np.abs(F_new - F).max() / F_max

        F = F_new

    if verbose:
        change_str = "0" if change == 0 else f"{change:.2e}"
        if it == max_iter - 1:
            print(
                f"Finished {max_iter:,} iterations without convergence (last largest change: {change_str})."
            )
        else:
            print(
                f"Converged after {it:,} iterations (last largest change: {change_str})."
            )

    # Convert to predicted labels
    prop = {}
    for node in nodes:
        i = node_index[node]
        # Nodes that never receive any signal during propagation will have all-zero
        # scores. In that case, return `None` rather than defaulting to the first
        # label.
        if np.all(F[i] == 0):
            prop[node] = None
        else:
            prop[node] = label_set[np.argmax(F[i])]

    # Map the labels back into node/vertex order.
    # N.B. we explicitly build an object-dtype array instead of letting pandas/numpy
    # infer it: the inferred dtype is not stable across versions. pandas >= 3 infers
    # `str` (-> ArrowStringArray) for string labels while pandas < 3 gives `object`,
    # and `np.array` on an all-labeled mesh yields a fixed-width `<U*` array that
    # cannot hold NaN.
    if isinstance(x, core.Skeleton):
        keys = x.nodes.node_id.values
    else:
        keys = range(len(x.vertices))

    prop_array = np.full(len(keys), np.nan, dtype=object)
    for i, key in enumerate(keys):
        label = prop.get(key)
        if label is not None:
            prop_array[i] = label

    if return_probs:
        if return_probs == "raw":
            probs = F.copy()
        elif return_probs == "softmax":
            probs = softmax(F, axis=1)
        else:  # return_probs is True
            probs = F.copy()
            row_sums = probs.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            probs = probs / row_sums
        return prop_array, probs, label_set

    return prop_array
