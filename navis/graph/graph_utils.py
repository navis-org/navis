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
import warnings

from collections import defaultdict

import igraph
import numpy as np
import pandas as pd
import trimesh as tm
import networkx as nx

from typing_extensions import Literal
from typing import Union, Optional, List, Tuple, Sequence, Dict, Set, overload, Iterable

from scipy.special import softmax
from scipy.sparse import csgraph, csr_matrix, diags

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
        "cut_skeleton",
        "longest_neurite",
        "split_into_fragments",
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
    x :         TreeNeuron | NeuronList
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
    if not isinstance(x, core.TreeNeuron):
        raise ValueError(f'Expected TreeNeuron, got "{type(x)}"')

    # At this point x is TreeNeuron
    x: core.TreeNeuron

    assert weight in ("weight", None), f'Unable to use weight "{weight}"'

    if utils.fastcore:
        if weight == "weight":
            weight = utils.fastcore.dag.parent_dist(
                x.nodes.node_id.values,
                x.nodes.parent_id.values,
                x.nodes[["x", "y", "z"]].values,
                root_dist=0,
            )

        # As of version >0.0.9, fastcore returns a tuple of (`segs`, `lengths`)
        segs, lengths = utils.fastcore.generate_segments(
            x.nodes.node_id.values, x.nodes.parent_id.values, weights=weight
        )

        if return_lengths:
            return segs, lengths
        else:
            return segs

    # Find leaf nodes and sort by distance to root
    d = dist_to_root(x, igraph_indices=False, weight=weight)
    endNodeIDs = x.nodes[x.nodes.type == "end"].node_id.values
    endNodeIDs = sorted(endNodeIDs, key=lambda x: d.get(x, 0), reverse=True)

    g: igraph.Graph = x.igraph
    # Convert endNodeIDs to indices
    id2ix = dict(zip(g.vs["node_id"], range(len(g.vs))))
    endNodeIDs = [id2ix[n] for n in endNodeIDs]

    seen: set = set()
    sequences = []
    for nodeID in endNodeIDs:
        sequence = [nodeID]
        parents = list(g.successors(nodeID))
        while True:
            if not parents:
                break
            parentID = parents[0]
            sequence.append(parentID)
            if parentID in seen:
                break
            seen.add(parentID)
            parents = list(g.successors(parentID))

        if len(sequence) > 1:
            sequences.append(sequence)

    # Turn indices back to node IDs
    ix2id = {v: k for k, v in id2ix.items()}
    sequences = [[ix2id[ix] for ix in s] for s in sequences]

    # Sort sequences by length
    lengths = [d[s[0]] - d[s[-1]] for s in sequences]
    sequences = [x for _, x in sorted(zip(lengths, sequences), reverse=True)]

    # Turn into list of arrays
    sequences = [np.array(s) for s in sequences]

    # Isolated nodes would not be included in the sequences(because they are treated
    # as roots, not leafs. Let's add them manually here.
    for node in nx.isolates(x.graph):
        sequences.append(np.array([node]))
        lengths.append(0)

    if return_lengths:
        return sequences, np.array(sorted(lengths, reverse=True))
    else:
        return sequences


def _group_by_label(labels: np.ndarray) -> List[np.ndarray]:
    """Group indices by their label: one array of indices per component."""
    order = np.argsort(labels, kind="mergesort")
    _, start_idx, counts = np.unique(
        labels[order], return_index=True, return_counts=True
    )
    return [order[start : start + count] for start, count in zip(start_idx, counts)]


def skeleton_edges(x: "core.TreeNeuron"):
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


def _fastcore_has(name: str) -> bool:
    """Whether the installed navis-fastcore provides `name`.

    Probed per function rather than by version: these primitives arrived in
    different releases, and the version metadata is unreliable in editable
    installs.
    """
    return utils.fastcore is not None and hasattr(utils.fastcore, name)


def _connected_components(
    x: Union[
        "core.TreeNeuron",
        "core.MeshNeuron",
        "core.Dotprops",
        "core.VoxelNeuron",
        "tm.Trimesh",
    ],
    epsilon: Optional[float] = None,
    connectivity: int = 26,
) -> List[Set[int]]:
    """Extract the connected components within a neuron.

    Will use `navis-fastcore` for skeletons and meshes, and `sparse-cubes` for
    voxels, if available.

    Parameters
    ----------
    x :             TreeNeuron | MeshNeuron | Dotprops | VoxelNeuron | Trimesh
                    Neuron for which to extract connected components.
    epsilon :       float, optional
                    For Dotprops only: distance threshold to consider two points
                    connected. If not provided, will use 5 x the average distance
                    between points.
    connectivity :  6 | 18 | 26, optional
                    For VoxelNeurons only: which neighbours count as connected.
                    6 = faces only, 18 = faces + edges, 26 (default) = faces +
                    edges + corners.

    Returns
    -------
    list
                List containing sets of node/vertex IDs for each subgraph. For
                VoxelNeurons these are indices into `.voxels`.

    Examples
    --------
    For doctest only

    >>> import navis
    >>> n = navis.example_neurons(1, kind='skeleton')
    >>> cc = navis.graph_utils._connected_components(n)
    >>> m = navis.example_neurons(1, kind='mesh')
    >>> cc = navis.graph_utils._connected_components(m)
    >>> dp = navis.make_dotprops(n, k=5)
    >>> cc = navis.graph_utils._connected_components(dp)

    """
    assert isinstance(
        x,
        (
            core.TreeNeuron,
            core.MeshNeuron,
            core.Dotprops,
            core.VoxelNeuron,
            tm.Trimesh,
        ),
    )

    if isinstance(x, core.VoxelNeuron):
        # `sparse-cubes` labels the components straight off the sparse voxels,
        # so this never builds a graph (or the dense grid)
        ms = x.connected_components(connectivity=connectivity)
        return _group_by_label(ms)

    if isinstance(x, core.TreeNeuron) and utils.fastcore:
        # This returns for each node the ID of its root
        ms = utils.fastcore.connected_components(
            x.nodes.node_id.values, x.nodes.parent_id.values
        )
        # Translate into list of arrays of IDs
        node_ids = x.nodes.node_id.values
        cc = [node_ids[group] for group in _group_by_label(ms)]
    elif (
        isinstance(x, (core.MeshNeuron, tm.Trimesh))
        and utils.fastcore
        and hasattr(utils.fastcore, "mesh_connected_components")
    ):
        # This returns for each vertex the ID of its component
        ms = utils.fastcore.mesh_connected_components(x.faces, len(x.vertices))  # type: ignore
        # Translate into list of arrays of vertex indices
        cc = _group_by_label(ms)
    else:
        if isinstance(x, core.Dotprops):
            G: igraph.Graph = graph.neuron2igraph(x, epsilon=epsilon)
        elif isinstance(x, tm.Trimesh):
            elist, elengths = utils.mesh_unique_edges(x, return_lengths=True)
            G = igraph.Graph(elist, n=len(x.vertices), directed=False)
            G.es["weight"] = elengths
        else:
            G: igraph.Graph = x.igraph
        # Get the vertex clustering
        vc = G.components(mode="WEAK")
        # Membership maps indices to connected components
        ms = np.array(vc.membership)
        if isinstance(x, core.TreeNeuron):
            # For skeletons we need node IDs
            ids = np.array(G.vs["node_id"])
        else:
            # For MeshNeurons we can use the indices directly
            ids = np.array(G.vs.indices)

        # Extract node IDs/vertex indices for each component
        cc = [ids[ms == i] for i in np.unique(ms)]

    return cc


def _break_segments(x: "core.NeuronObject") -> list:
    """Break neuron into small segments connecting ends, branches and root.

    Parameters
    ----------
    x :         TreeNeuron | NeuronList
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
    elif isinstance(x, core.TreeNeuron):
        pass
    else:
        logger.error("Unexpected datatype: %s" % str(type(x)))
        raise ValueError

    # At this point x is TreeNeuron
    x: core.TreeNeuron

    if utils.fastcore:
        seg_list = utils.fastcore.break_segments(
            x.nodes.node_id.values, x.nodes.parent_id.values
        )
    else:
        g: igraph.Graph = x.igraph
        end = g.vs.select(_indegree=0).indices
        branch = g.vs.select(_indegree_gt=1, _outdegree=1).indices
        root = g.vs.select(_outdegree=0).indices

        # Get seeds
        seeds = branch + end
        # Remove seeds that are also roots (=disconnected single nodes)
        seeds = set(seeds) - set(root)

        # Converting to set speeds up the "parent in stops" check
        stops = set(branch + root)
        seg_list = []
        # Iterate the seeds in node table order (igraph's vertex indices are the node
        # table's row numbers). Without the `sorted` we would be walking a Python set,
        # i.e. in hash order - which is arbitrary, depends on the number of seeds, and
        # does not match the order fastcore returns. Consumers such as
        # `segment_analysis` and the NEURON interface enumerate the segments, so the
        # order ends up in their output and must not depend on the backend.
        for s in sorted(seeds):
            parent = g.successors(s)[0]
            seg = [s, parent]
            while parent not in stops:
                parent = g.successors(parent)[0]
                seg.append(parent)
            seg_list.append(seg)
        # Translate indices to node IDs
        ix_id = {
            v: n for v, n in zip(g.vs.indices, g.vs.get_attribute_values("node_id"))
        }
        seg_list = [[ix_id[n] for n in s] for s in seg_list]

    return seg_list


@utils.lock_neuron
def dist_to_root(
    x: "core.TreeNeuron", weight=None, igraph_indices: bool = False
) -> dict:
    """Calculate distance to root for each node.

    Parameters
    ----------
    x :                 TreeNeuron
    weight :            str, optional
                        Use "weight" if you want geodesic distance and `None`
                        if you want node count.
    igraph_indices :    bool
                        Whether to return igraph node indices instead of node
                        IDs. This is mainly used for internal functions.

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
    if not isinstance(x, core.TreeNeuron):
        raise TypeError(f"Expected TreeNeuron, got {type(x)}")

    G: igraph.Graph = x.igraph
    ids = np.asarray(G.vs["node_id"])

    # Note: `vs.select(node_id_in=...)` would be the idiomatic igraph here but it
    # scans every vertex in Python and costs more than the search itself.
    roots = np.where(np.isin(ids, x.root))[0]

    # Edges run child->parent; transposing gives us root->child, so a search from
    # the roots yields each node's distance to its own root.
    #
    # `min_only` is what keeps this O(N): every node reaches exactly one root, so
    # the distance to the *nearest* root is the distance to its own. Asking igraph
    # for `distances(source=roots)` instead would hand back a roots x N matrix -
    # fine for one root, hundreds of MB for a badly fragmented neuron.
    adj = _igraph_to_sparse(G, weight_attr=weight, transpose=True)

    # csgraph.dijkstra wants int32 indices/indptr
    adj.indptr = adj.indptr.astype("int32", copy=False)
    adj.indices = adj.indices.astype("int32", copy=False)

    dists = csgraph.dijkstra(adj, directed=True, indices=roots, min_only=True)

    # Unreachable nodes (i.e. those in another fragment) come back as inf and are
    # simply left out - matching the networkx behaviour this replaced.
    keys = np.arange(len(ids)) if igraph_indices else ids
    reachable = np.isfinite(dists)

    return dict(zip(keys[reachable].tolist(), dists[reachable].tolist()))


@utils.map_neuronlist(desc="Classifying", allow_parallel=True)
@utils.lock_neuron
def classify_nodes(x: "core.NeuronObject", categorical=True, inplace: bool = True):
    """Classify neuron's nodes into end nodes, branches, slabs or root.

    Adds a `'type'` column to `x.nodes` table.

    Parameters
    ----------
    x :         TreeNeuron | NeuronList
                Neuron(s) whose nodes to classify.
    categorical : bool
                If True (default), will use categorical data type which takes
                up much less memory at a small run-time overhead.
    inplace :   bool, optional
                If `False`, nodes will be classified on a copy which is then
                returned leaving the original neuron unchanged.

    Returns
    -------
    TreeNeuron/List

    Examples
    --------
    >>> import navis
    >>> nl = navis.example_neurons(2)
    >>> _ = navis.graph.classify_nodes(nl, inplace=True)

    """
    if not inplace:
        x = x.copy()

    if not isinstance(x, core.TreeNeuron):
        raise TypeError(f'Expected TreeNeuron(s), got "{type(x)}"')

    if x.nodes.empty:
        x.nodes["type"] = None
        return x

    node_ids = x.nodes.node_id.values
    parent_ids = x.nodes.parent_id.values

    # Note: we work with the integer *codes* of `NODE_TYPES` throughout and only
    # ever turn them into labels at the very end. Going via a string array (as
    # this used to) makes `pd.Categorical` factorize N strings, which costs more
    # than the classification itself.
    if utils.fastcore:
        # Fastcore uses its own order (0=root, 1=leaf, 2=branch, 3=slab)
        cl = _FASTCORE_NODE_TYPES[utils.fastcore.classify_nodes(node_ids, parent_ids)]
    else:
        # Note: I have tried to optimized the s**t out of this, i.e. every
        # single line of code here has been tested for speed. Do not
        # change anything unless you know what you're doing!

        # Turns out that numpy.isin() recently started to complain if the
        # node_ids are uint64 and the parent_ids are int64 (but strangely
        # not with 32bit integers). If that's the case we have to convert
        # the node_ids to int64.
        if node_ids.dtype == np.uint64:
            node_ids = node_ids.astype(np.int64)

        cl = np.full(len(x.nodes), NODE_TYPES.index("slab"), dtype=np.int8)
        cl[~np.isin(node_ids, parent_ids)] = NODE_TYPES.index("end")
        bp = x.nodes.parent_id.value_counts()
        bp = bp.index.values[bp.values > 1]
        cl[np.isin(node_ids, bp)] = NODE_TYPES.index("branch")
        cl[parent_ids < 0] = NODE_TYPES.index("root")

    if categorical:
        x.nodes["type"] = pd.Categorical.from_codes(
            cl, categories=NODE_TYPES, ordered=False
        )
    else:
        x.nodes["type"] = np.asarray(NODE_TYPES, dtype="<U6")[cl]

    return x


@utils.lock_neuron
def distal_to(
    x: "core.TreeNeuron",
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
    x :     TreeNeuron
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

    if not isinstance(x, core.TreeNeuron):
        raise ValueError(f"Please pass a single TreeNeuron, got {type(x)}")

    # At this point x is TreeNeuron
    x: core.TreeNeuron

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

    if utils.fastcore:
        # `targets` is what keeps this cheap: igraph will happily take a target
        # list but computes an all-sources search to honour it, and the result is
        # a len(a) x n_nodes matrix either way. Here we only ever materialise the
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
    else:
        # Grab graph once to avoid overhead from stale checks
        G: igraph.Graph = x.igraph

        # Map node ID to index
        id2ix = {n: v for v, n in zip(G.vs.indices, G.vs["node_id"])}

        # Convert node IDs to indices
        ixA = [id2ix[n] for n in tnA]  # type: ignore
        ixB = [id2ix[n] for n in tnB]  # type: ignore

        # Converting to numpy array first is ~2X as fast
        le = np.asarray(G.distances(ixA, ixB, mode="OUT"))
        reachable = le != float("inf")

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
    x :         TreeNeuron
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
    elif not isinstance(x, (core.TreeNeuron,)):
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
    limit: Union[float, int] = np.inf,
) -> pd.DataFrame:
    """Generate geodesic ("along-the-arbor") distance matrix between nodes/vertices.

    Parameters
    ----------
    x :         TreeNeuron | MeshNeuron | NeuronList
                If list, must contain a SINGLE neuron.
    from_ :     list | numpy.ndarray, optional
                Node IDs (for TreeNeurons) or vertex indices (for MeshNeurons).
                If provided, will compute distances only FROM this subset to
                all other nodes/vertices.
    to_ :       list | numpy.ndarray, optional
                Node IDs (for TreeNeurons) or vertex indices (for MeshNeurons).
                If provided, will compute distances only TO this subset. Use
                together with `from_` to get just the block you need instead of
                slicing a full matrix afterwards - that can be the difference
                between a few MB and a few GB on a large neuron.
    directed :  bool, optional
                For TreeNeurons only: if True, pairs without a child->parent
                path will be returned with `distance = "inf"`.
    weight :    'weight' | None, optional
                If "weight" distances are given as physical length.
                If `None` distance is the number of nodes.
    limit :     int | float, optional
                Use to limit distance calculations. Nodes that are not within
                `limit` will have distance `np.inf`. If neuron has its
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

    if not isinstance(x, (core.TreeNeuron, core.MeshNeuron)):
        raise ValueError(f'Unable to process data of type "{type(x)}"')

    limit = x.map_units(limit, on_error="raise")

    def _check(sel, valid):
        """Normalise a `from_`/`to_` selection and make sure it exists."""
        sel = np.unique(utils.make_iterable(sel))
        miss = sel[~np.isin(sel, valid)]
        if len(miss):
            raise ValueError(
                f"Node/vertex IDs not present: {', '.join(miss.astype(str))}"
            )
        return sel

    # Use fastcore if available
    if utils.fastcore and isinstance(x, core.TreeNeuron):
        node_ids = x.nodes.node_id.values

        # Calculate node distances
        if weight == "weight":
            weight = utils.fastcore.dag.parent_dist(
                node_ids,
                x.nodes.parent_id.values,
                x.nodes[["x", "y", "z"]].values,
                root_dist=0,
            )

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

        if limit is not None and limit is not np.inf:
            dmat[dmat > limit] = np.inf

        return pd.DataFrame(
            dmat,
            index=node_ids if from_ is None else from_,
            columns=node_ids if to_ is None else to_,
        )

    # Makes no sense to use directed for MeshNeurons
    if isinstance(x, core.MeshNeuron):
        directed = False

    if utils.fastcore and isinstance(x, core.MeshNeuron):
        vertex_ids = np.arange(len(x.vertices))

        from_ = None if from_ is None else _check(from_, vertex_ids)
        to_ = None if to_ is None else _check(to_, vertex_ids)

        dmat = utils.fastcore.geodesic_matrix_mesh(
            x.faces,
            # Without coordinates fastcore weights every edge as 1 (i.e. hop count)
            vertices=x.vertices if weight == "weight" else None,
            n_vertices=len(vertex_ids),
            sources=from_,
            targets=to_,
            # Fastcore takes `None` rather than infinity for "no limit"
            limit=None if limit is None or not np.isfinite(limit) else limit,
        )

        # Fastcore returns -1 for unreachable vertex pairs
        dmat[dmat < 0] = np.inf

        return pd.DataFrame(
            dmat,
            index=vertex_ids if from_ is None else from_,
            columns=vertex_ids if to_ is None else to_,
        )

    # Grab graph once to avoid overhead from stale checks
    G: igraph.Graph = x.igraph

    if isinstance(x, core.TreeNeuron):
        nodeList = np.array(G.vs.get_attribute_values("node_id"))
    else:
        nodeList = np.arange(len(G.vs))

    # Matrix is ordered by vertex number
    m = _igraph_to_sparse(G, weight_attr=weight)

    from_ = None if from_ is None else _check(from_, nodeList)
    to_ = None if to_ is None else _check(to_, nodeList)

    # Note: `nodeList` is in graph (i.e. node table) order, so we have to look up
    # where each requested ID sits rather than assume it is sorted. Doing this for
    # the rows as well keeps the row order identical to the fastcore path above.
    lookup = pd.Index(nodeList)
    indices = None if from_ is None else lookup.get_indexer(from_)

    # For some reason csgraph.dijkstra expects indices/indptr as int32
    m.indptr = m.indptr.astype("int32", copy=False)
    m.indices = m.indices.astype("int32", copy=False)
    dmat = csgraph.dijkstra(m, directed=directed, indices=indices, limit=limit)

    # csgraph has no notion of targets, so we have to subset after the fact. This
    # is the fallback - the fastcore path above never materialises these columns.
    if to_ is not None:
        dmat = dmat[:, lookup.get_indexer(to_)]

    return pd.DataFrame(  # type: ignore  # no stubs
        dmat,
        index=nodeList if from_ is None else from_,
        columns=nodeList if to_ is None else to_,
    )


def _geodesic_nearest(
    x: "core.TreeNeuron",
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

    Uses `navis_fastcore.geodesic_nearest` if available and falls back to a
    multi-source `scipy.sparse.csgraph.dijkstra(min_only=True)` search.

    Note
    ----
    `query` and `targets` are expected to be disjoint (the typical "assign
    unlabeled nodes to the nearest labeled node" use case). If a query node is
    itself a target the two backends may disagree on whether it matches itself
    or the nearest *other* target.

    Parameters
    ----------
    x :         TreeNeuron
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
    if not isinstance(x, core.TreeNeuron):
        raise ValueError(f'Expected TreeNeuron, got "{type(x)}"')

    node_ids = x.nodes.node_id.values
    parent_ids = x.nodes.parent_id.values
    ix = pd.Index(node_ids)

    targets = np.asarray(list(targets))
    query = node_ids if query is None else np.asarray(list(query))

    # Nothing to snap to (or nothing to snap) -> everything unreachable.
    if not len(targets) or not len(query):
        return (
            np.full(len(query), -1, dtype=node_ids.dtype),
            np.full(len(query), np.inf, dtype=float),
        )

    # Per-node distance to parent (root = 0). `None` -> unweighted (hop count).
    if weight == "weight":
        coords = x.nodes[["x", "y", "z"]].values.astype(np.float32)
        has_parent = parent_ids >= 0
        p_ix = ix.get_indexer(parent_ids)
        weights = np.zeros(len(node_ids), dtype=np.float32)
        weights[has_parent] = np.linalg.norm(
            coords[has_parent] - coords[p_ix[has_parent]], axis=1
        )
    else:
        weights = None

    # Fast path: compiled fastcore implementation (linear time, O(N) memory).
    if utils.fastcore and hasattr(utils.fastcore, "geodesic_nearest"):
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

    # Fallback: multi-source Dijkstra. `min_only=True` keeps only each node's
    # distance to the *nearest* source -> O(N) memory instead of O(N_sources*N).
    has_parent = parent_ids >= 0
    child_ix = ix.get_indexer(node_ids[has_parent])
    parent_ix = ix.get_indexer(parent_ids[has_parent])
    edge_w = (
        np.ones(child_ix.size, dtype=np.float32)
        if weights is None
        else weights[has_parent]
    )

    if directed:
        # We run the search FROM the targets (see `indices` below), so to recover
        # "distance from each query node to its nearest target travelling
        # child -> parent (towards the root)" the search must travel the *opposite*
        # way out of each target, i.e. parent -> child (towards its descendants).
        rows, cols, data = parent_ix, child_ix, edge_w
    else:
        rows = np.concatenate([child_ix, parent_ix])
        cols = np.concatenate([parent_ix, child_ix])
        data = np.concatenate([edge_w, edge_w])

    N = len(node_ids)
    adj = csr_matrix((data, (rows, cols)), shape=(N, N))
    # csgraph.dijkstra expects int32 indices/indptr
    adj.indptr = adj.indptr.astype("int32", copy=False)
    adj.indices = adj.indices.astype("int32", copy=False)

    # Sources are the targets; for each node we get its nearest target + distance.
    src = ix.get_indexer(targets)
    dist_all, _, sources = csgraph.dijkstra(
        adj,
        directed=directed,
        indices=src,
        min_only=True,
        unweighted=weights is None,
        return_predecessors=True,
    )

    q_ix = ix.get_indexer(query)
    src_node_ix = sources[q_ix]  # graph index of nearest target (< 0 if none)
    reachable = src_node_ix >= 0
    nearest = np.full(query.shape, -1, dtype=node_ids.dtype)
    nearest[reachable] = node_ids[src_node_ix[reachable]]
    distances = dist_all[q_ix].astype(float)
    distances[~reachable] = np.inf
    return nearest, distances


def geodesic_clusters(
    x: Union["core.TreeNeuron", "core.MeshNeuron"],
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

    Uses `navis_fastcore.geodesic_clusters` if available and falls back to a
    per-seed bounded `scipy.sparse.csgraph.dijkstra`. The greedy outer loop is
    inherently sequential, so the fallback is a Python loop over clusters and
    is markedly slower on neurons that produce many of them.

    Parameters
    ----------
    x :         TreeNeuron | MeshNeuron
    max_dist :  float
                Maximum distance from a cluster's seed, in the neuron's units
                (or in hops if `weight=None`). Must be finite and non-negative.
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
                `x.vertices` for a MeshNeuron) and contiguous in
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

    if _fastcore_has("geodesic_clusters"):
        labels, _ = utils.fastcore.geodesic_clusters(
            edges, n_nodes, max_dist, weights=weights, seeds=seed_ix
        )
        labels = np.asarray(labels)
    else:
        labels = _geodesic_clusters_scipy(edges, weights, n_nodes, max_dist, seed_ix)

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

    if _fastcore_has("connected_components_graph"):
        comp = utils.fastcore.connected_components_graph(intra, n_nodes)
    else:
        adj = csr_matrix(
            (
                np.ones(len(intra), dtype=np.int8),
                (intra[:, 0], intra[:, 1]),
            ),
            shape=(n_nodes, n_nodes),
        )
        comp = csgraph.connected_components(adj, directed=False)[1]

    # Relabel contiguously. N.B. this renumbers clusters - the raw growth order
    # does not survive a split anyway.
    _, out = np.unique(comp, return_inverse=True)
    return out.reshape(-1).astype(np.int32, copy=False)


def _cluster_graph(x, weight):
    """Edge list, edge weights, node count and node IDs for `geodesic_clusters`."""
    if isinstance(x, core.TreeNeuron):
        edges, ids = skeleton_edges(x)
        if weight == "weight":
            co = x.nodes[["x", "y", "z"]].values.astype(np.float64)
            w = np.linalg.norm(co[edges[:, 0]] - co[edges[:, 1]], axis=1)
        else:
            w = None
        return edges, w, len(ids), ids

    if isinstance(x, core.MeshNeuron):
        edges = np.asarray(utils.mesh_unique_edges(x.trimesh), dtype=np.int64)
        if weight == "weight":
            co = np.asarray(x.vertices, dtype=np.float64)
            w = np.linalg.norm(co[edges[:, 0]] - co[edges[:, 1]], axis=1)
        else:
            w = None
        n = len(x.vertices)
        return edges, w, n, np.arange(n)

    raise TypeError(f"Expected TreeNeuron or MeshNeuron, got {type(x)}")


def _geodesic_clusters_scipy(edges, weights, n_nodes, max_dist, seed_ix):
    """Fallback for `geodesic_clusters`: one bounded Dijkstra per cluster.

    N.B. each search runs on the *full* graph rather than on what is left
    unclaimed - that is what makes every cluster a true ball around its seed
    (see the note in `geodesic_clusters`) - and only the *assignment* skips
    nodes an earlier cluster took.
    """
    w = np.ones(len(edges), dtype=np.float64) if weights is None else np.asarray(weights)
    adj = csr_matrix(
        (
            np.concatenate([w, w]),
            (
                np.concatenate([edges[:, 0], edges[:, 1]]),
                np.concatenate([edges[:, 1], edges[:, 0]]),
            ),
        ),
        shape=(n_nodes, n_nodes),
    )
    adj.indptr = adj.indptr.astype("int32", copy=False)
    adj.indices = adj.indices.astype("int32", copy=False)

    labels = np.full(n_nodes, -1, dtype=np.int32)

    # Preferred seeds first, then everything still unassigned in index order.
    order = np.arange(n_nodes)
    if seed_ix is not None and len(seed_ix):
        order = np.concatenate([np.asarray(seed_ix), order])

    n_clusters = 0
    for seed in order:
        if labels[seed] != -1:
            continue
        dist = csgraph.dijkstra(
            adj, directed=False, indices=int(seed), limit=max_dist
        )
        labels[(dist <= max_dist) & (labels == -1)] = n_clusters
        n_clusters += 1

    return labels


@utils.lock_neuron
def segment_length(x: "core.TreeNeuron", segment: List[int]) -> float:
    """Get length of a linear segment.

    This function is superfast but has no checks - you must provide a
    valid segment.

    Parameters
    ----------
    x :         TreeNeuron
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
    if not isinstance(x, core.TreeNeuron):
        raise ValueError(f'Unable to process data of type "{type(x)}"')

    return float(segment_lengths(x, [segment])[0])


def segment_lengths(x: "core.TreeNeuron", segments: Sequence[Sequence[int]]):
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
    x :             TreeNeuron | MeshNeuron | NeuronList
                    If NeuronList must contain only a single neuron.
    a,b :           int | list of int
                    Node IDs (for TreeNeurons) or vertex indices (MeshNeurons)
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
            raise ValueError(f"Need a single TreeNeuron, got {len(x)}")

    if not isinstance(x, (core.TreeNeuron, core.MeshNeuron, igraph.Graph, nx.DiGraph)):
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

    if isinstance(x, core.TreeNeuron) and utils.fastcore:
        node_ids = x.nodes.node_id.values
        parent_ids = x.nodes.parent_id.values

        weights = utils.fastcore.dag.parent_dist(
            node_ids,
            parent_ids,
            x.nodes[["x", "y", "z"]].values,
            root_dist=0,
        )
        dist = utils.fastcore.geodesic_pairs(
            node_ids,
            parent_ids,
            pairs=np.stack((a, b), axis=1),
            weights=weights,
        ).astype(float)

        # Fastcore returns -1 for unreachable pairs
        dist[dist < 0] = np.inf
        return float(dist[0]) if scalar else dist

    G: Union[igraph.Graph, nx.DiGraph] = (
        x.igraph if isinstance(x, (core.TreeNeuron, core.MeshNeuron)) else x
    )

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

    if isinstance(x, core.TreeNeuron):
        id2ix = dict(zip(G.vs["node_id"], G.vs.indices))
        a = np.array([id2ix[i] for i in a.tolist()])
        b = np.array([id2ix[i] for i in b.tolist()])

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
    x :             TreeNeuron | NeuronList
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
    0  TreeNeuron     2572             0         170      176  475078.177926    None
    1  TreeNeuron      139             0           1        3   89983.511392  [3490]
    2  TreeNeuron     3656             0          63       66  648285.745750    None

    """
    utils.eval_param(
        method, name="method", allowed_values=("longest_neurite", "betweenness")
    )

    if not isinstance(x, core.TreeNeuron):
        raise TypeError(f'Expected TreeNeuron(s), got "{type(x)}"')

    # At this point x is TreeNeuron
    x: core.TreeNeuron

    # If no branches
    if x.nodes[x.nodes.type == "branch"].empty:
        raise ValueError("Neuron has no branch points.")

    if reroot_soma and not isinstance(x.soma, type(None)):
        x = x.reroot(x.soma, inplace=False)

    if method == "longest_neurite":
        G: igraph.Graph = x.igraph
        ids = np.asarray(G.vs["node_id"])

        # First, find longest path
        longest = _longest_weighted_path(G, weight="weight")

        # Remove it and find the second longest path through what's left
        g = G.copy()
        g.delete_vertices(longest)
        sc_longest = _longest_weighted_path(g, weight="weight")

        # Parent of the last node in sc_longest is the common branch point
        last = int(np.asarray(g.vs["node_id"])[sc_longest[-1]])
        id2ix = {nid: ix for ix, nid in enumerate(ids.tolist())}
        bp = ids[G.successors(id2ix[last])[0]]
    else:
        # Get betweenness for each node
        x = morpho.betweeness_centrality(x, directed=True, from_="branch_points")
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
def split_into_fragments(
    x: "core.NeuronObject",
    n: int = 2,
    min_size: Optional[Union[float, str]] = None,
    reroot_soma: bool = False,
) -> "core.NeuronList":
    """Split neuron into fragments.

    Cuts are based on longest neurites: the first cut is made where the second
    largest neurite merges onto the largest neurite, the second cut is made
    where the third largest neurite merges into either of the first fragments
    and so on.

    Parameters
    ----------
    x :                 TreeNeuron | MeshNeuron | NeuronList
                        Must be a single neuron.
    n :                 int, optional
                        Number of fragments to split into. Must be >1.
    min_size :          int | str, optional
                        Minimum size of fragment to be cut off. If too
                        small, will stop cutting. This takes only the longest
                        path in each fragment into account! If the neuron(s),
                        has its `.units` set, you can also pass this as a string
                        such as "10 microns".
    reroot_soma :        bool, optional
                        If True, neuron will be rerooted to soma.

    Returns
    -------
    NeuronList

    Examples
    --------
    >>> import navis
    >>> x = navis.example_neurons(1)
    >>> # Cut into two fragments
    >>> cut1 = navis.split_into_fragments(x, n=2)
    >>> # Cut into fragments of >10 um size
    >>> cut2 = navis.split_into_fragments(x, n=float('inf'), min_size=10e3)

    """
    if isinstance(x, core.NeuronList):
        if len(x) == 1:
            x = x[0]
        else:
            raise Exception(
                f"{x.shape[0]} neurons provided. Please provide only a single neuron!"
            )

    if not isinstance(x, core.TreeNeuron):
        raise TypeError(f'Expected a single TreeNeuron, got "{type(x)}"')

    if n < 2:
        raise ValueError("Number of fragments must be at least 2.")

    # At this point x is TreeNeuron
    x: core.TreeNeuron

    min_size = x.map_units(min_size, on_error="raise")

    if reroot_soma and not isinstance(x.soma, type(None)):
        x.reroot(x.soma, inplace=True)

    # Collect nodes of the n longest neurites. We work on a copy of the igraph and
    # delete each claimed neurite from it, rather than rebuilding the whole graph
    # from the node table on every iteration.
    g: igraph.Graph = x.igraph.copy()

    fragments = []
    i = 0
    while i < n and g.vcount():
        path = _longest_weighted_path(g, weight="weight")

        if not len(path):
            break

        # Check if fragment is still long enough. Note this sums the weight of
        # every edge *pointing into* the path - not just the path's own edges.
        # Preserved as-is from the networkx implementation.
        if min_size:
            edges = np.asarray(g.get_edgelist(), dtype=np.int64).reshape(-1, 2)
            weights = np.asarray(g.es["weight"])
            this_length = weights[np.isin(edges[:, 1], path)].sum()
            if this_length <= min_size:
                break

        fragments.append(np.asarray(g.vs["node_id"])[path])

        # Drop the claimed nodes so the next iteration finds the next-longest
        g.delete_vertices(path)

        i += 1

    # Next, make some virtual cuts and get the complement of nodes for each
    # fragment. The first fragment starts out as the whole neuron; every other one
    # is the sub-tree distal to its proximal-most node.
    G: igraph.Graph = x.igraph
    ids = np.asarray(G.vs["node_id"])
    id2ix = {nid: ix for ix, nid in enumerate(ids.tolist())}

    node_sets = [set(ids.tolist())]
    for fr in fragments[1:]:
        # mode="IN" walks edges backwards (child->parent reversed), i.e. collects
        # everything distal to this node - the igraph equivalent of the
        # `nx.bfs_tree(..., reverse=True)` this replaced.
        distal = G.subcomponent(id2ix[fr[-1]], mode="IN")
        node_sets.append(set(ids[distal].tolist()))

    # Remove nodes that are claimed by a subsequent (i.e. more distal) fragment
    for i, s in enumerate(node_sets):
        for s2 in node_sets[i + 1 :]:
            s -= s2

    # Now make neurons - keep node-table order for a stable result
    nl = core.NeuronList(
        [morpho.subset_neuron(x, ids[np.isin(ids, list(s))]) for s in node_sets]
    )

    return nl


def _longest_weighted_path(g: "igraph.Graph", weight="weight") -> np.ndarray:
    """Find the longest weighted path in an in-forest (edges point child->parent).

    Every maximal path in such a graph is fixed by its starting node - just follow
    the parents up to a sink - so the longest one starts at whichever node is
    furthest from its sink. That makes this a distances-to-sinks problem rather
    than the general (NP-hard) longest-path problem.
    """
    n = g.vcount()
    sinks = np.where(np.asarray(g.outdegree()) == 0)[0]
    if not len(sinks):
        return np.empty(0, dtype=int)

    # Join every sink to one virtual super-sink at zero cost and run a *single*
    # search. Each node in an in-forest reaches exactly one sink, so its distance
    # to the super-sink is the distance to its own sink. Searching from each sink
    # separately would be O(sinks x N) - and note that lopping off a neurite turns
    # every severed branch into a fresh sink, so `sinks` is not small.
    h = g.copy()
    h.add_vertices(1)
    h.add_edges([(int(s), n) for s in sinks])
    h.es[weight] = np.concatenate(
        [np.asarray(g.es[weight]), np.zeros(len(sinks))]
    ).tolist()

    dists = np.asarray(h.distances(source=[n], weights=weight, mode="IN")[0][:n])
    dists[~np.isfinite(dists)] = -1
    start = int(np.argmax(dists))

    # Path from `start` to the super-sink, minus the super-sink itself
    path = h.get_shortest_paths(start, to=n, weights=weight, mode="OUT")[0]

    return np.asarray(path[:-1])


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
    x :                 TreeNeuron | NeuronList
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
    TreeNeuron/List
                        Pruned neuron.

    See Also
    --------
    [`navis.split_into_fragments`][]
            Split neuron into fragments based on longest neurites.

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
    if not isinstance(x, core.TreeNeuron):
        raise TypeError(f'Expected TreeNeuron(s), got "{type(x)}"')

    if isinstance(n, numbers.Number) and n < 1:
        raise ValueError("Number of longest neurites to preserve must be >=1")

    # At this point x is TreeNeuron
    x: core.TreeNeuron

    if not inplace:
        x = x.copy()

    if not from_root:
        # Find the two most distal points (N.B. roots can also be "ends")
        leafs = np.unique(
            x.nodes.loc[x.nodes.type.isin(("root", "end")), "node_id"].values
        )

        if utils.fastcore:
            # We only need each leaf's distance to its farthest leaf, so there is no
            # point in materialising the full leafs x leafs matrix just to take its
            # maximum. Note fastcore also uses -1 for unreachable (i.e. fragmented).
            dists, _ = utils.fastcore.geodesic_farthest(
                x.nodes.node_id.values,
                x.nodes.parent_id.values,
                sources=leafs,
                targets=leafs,
                weights=utils.fastcore.dag.parent_dist(
                    x.nodes.node_id.values,
                    x.nodes.parent_id.values,
                    x.nodes[["x", "y", "z"]].values,
                    root_dist=0,
                ),
            )
        else:
            dmat = geodesic_matrix(x, from_=leafs, to_=leafs)
            # If the neuron is fragmented, we will have infinite distances
            dmat[dmat == np.inf] = -1
            dists = dmat.values.max(axis=1)

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
) -> "core.TreeNeuron":
    """Reroot neuron to new root.

    Parameters
    ----------
    x :        TreeNeuron | NeuronList
               List must contain only a SINGLE neuron.
    new_root : int | iterable
               Node ID(s) of node(s) to reroot to. If multiple new roots are
               provided, they will be rerooted in sequence.
    inplace :  bool, optional
               If True the input neuron will be rerooted in place. If False will
               reroot and return a copy of the original.

    Returns
    -------
    TreeNeuron
               Rerooted neuron.

    See Also
    --------
    [`navis.TreeNeuron.reroot`][]
                Quick access to reroot directly from TreeNeuron/List
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

    if not isinstance(x, core.TreeNeuron):
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

    # At this point x is TreeNeuron
    x: core.TreeNeuron
    # At this point new_roots is list of int
    new_roots: Iterable[int]

    if not inplace:
        # Make a copy
        x = x.copy()
        # Run this in a separate function so that the lock is applied to copy
        _ = reroot_skeleton(x, new_root=new_roots, inplace=True)
        return x

    # Keep track of node ID dtype
    nodeid_dtype = x.nodes.node_id.dtype

    # Go over each new root
    for new_root in new_roots:
        # Skip if new root is old root
        if any(x.root == new_root):
            continue

        # Grab graph once to avoid overhead from stale checks
        g: igraph.Graph = x.igraph

        # Map node ID -> vertex index up front. `vs.select(node_id_in=...)` and
        # `vs.find(node_id=...)` both scan every vertex in Python, and we're inside
        # a loop over the new roots.
        id2ix = {nid: ix for ix, nid in enumerate(g.vs["node_id"])}

        # Vertices of the current root(s), in the same order as x.root
        vs_roots = [id2ix[r] for r in x.root]

        # Prevent warnings in the following code - querying paths between
        # unreachable nodes will otherwise generate a runtime warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Find paths to all roots
            path = g.get_shortest_paths(id2ix[new_root], vs_roots)
            epath = g.get_shortest_paths(
                id2ix[new_root],
                vs_roots,
                output="epath",
            )

        # Extract paths that actually worked (i.e. within a continuous fragment)
        path = [p for p in path if p][0]
        epath = [p for p in epath if p][0]

        edges = [(s, t) for s, t in zip(path[:-1], path[1:])]

        weights = [g.es[e]["weight"] for e in epath]

        # Get all weights and append inversed new weights
        all_weights = g.es["weight"] + weights

        # Add inverse edges: old_root->new_root
        g.add_edges([(e[1], e[0]) for e in edges])

        # Re-set weights
        g.es["weight"] = all_weights

        # Remove new_root->old_root
        g.delete_edges(edges)

        # Get degree of old root for later categorisation
        old_root_deg = len(g.es.select(_target=path[-1]))

        # Translate path indices to node IDs
        ix2id = {
            ix: n for ix, n in zip(g.vs.indices, g.vs.get_attribute_values("node_id"))
        }
        path = [ix2id[i] for i in path]

        # Set index to node ID for later
        x.nodes.set_index("node_id", inplace=True)

        # Propagate changes in graph back to node table
        # Assign new node type to old root
        x.nodes.loc[path[1:], "parent_id"] = path[:-1]
        if old_root_deg == 1:
            x.nodes.loc[path[-1], "type"] = "slab"
        elif old_root_deg > 1:
            x.nodes.loc[path[-1], "type"] = "branch"
        else:
            x.nodes.loc[path[-1], "type"] = "end"
        # Make new root node type "root"
        x.nodes.loc[path[0], "type"] = "root"

        # Set new root's parent to None
        x.nodes.loc[new_root, "parent_id"] = -1

        # Reset index
        x.nodes.reset_index(drop=False, inplace=True)

    # Make sure node ID has the same datatype as before
    if x.nodes.node_id.dtype != nodeid_dtype:
        x.nodes["node_id"] = x.nodes.node_id.astype(nodeid_dtype)

    # Finally: only reset non-graph related attributes
    x._clear_temp_attr(exclude=["igraph", "classify_nodes"])

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
    x :        TreeNeuron | NeuronList
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
    [`navis.TreeNeuron.prune_distal_to`][]
    [`navis.TreeNeuron.prune_proximal_to`][]
            `TreeNeuron/List` shorthands to this function.
    [`navis.subset_neuron`][]
            Returns a neuron consisting of a subset of its nodes.

    """
    utils.eval_param(ret, name="ret", allowed_values=("proximal", "distal", "both"))

    if isinstance(x, core.NeuronList):
        if len(x) == 1:
            x = x[0]
        else:
            raise Exception(f"Expected a single TreeNeuron, got {len(x)}")

    if not isinstance(x, core.TreeNeuron):
        raise TypeError(f'Expected a single TreeNeuron, got "{type(x)}"')

    if x.n_trees != 1:
        raise ValueError(
            f"Unable to cut: neuron {x.id} consists of multiple "
            "disconnected trees. Use navis.heal_skeleton()"
            " to fix."
        )

    # At this point x is TreeNeuron
    x: core.TreeNeuron

    # Turn cut node into iterable
    if not utils.is_iterable(where):
        where = [where]

    # Process cut nodes (i.e. if tag)
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
        elif cn not in x.nodes.node_id.values:
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
        cut = _cut_igraph(to_cut, cn, ret)

        # If ret != 'both', we will get only a single neuron - therefore
        # make sure cut is iterable
        cut = utils.make_iterable(cut)

        # Add results back to results at same index, proximal first
        for c in cut[::-1]:
            res.insert(to_cut_ix, c)

    return core.NeuronList(res)


def _cut_igraph(
    x: "core.TreeNeuron", cut_node: int, ret: str
) -> Union["core.TreeNeuron", Tuple["core.TreeNeuron", "core.TreeNeuron"]]:
    """Use iGraph to cut a neuron."""
    # Make a copy
    g = x.igraph.copy()

    # Get vertex index
    cut_ix = g.vs.find(node_id=cut_node).index

    # Get edge to parent
    e = g.es.find(_source=cut_ix)

    # Remove edge
    g.delete_edges(e)

    # Make graph undirected -> otherwise .decompose() throws an error
    # This issue is fixed in the up-to-date branch of igraph-python
    # (which is not on PyPI O_o )
    g.to_undirected(combine_edges="first")

    # Get subgraph -> fastest way to get sets of nodes for subsetting
    a, b = g.decompose(mode="WEAK")
    # IMPORTANT: a,b are now UNDIRECTED graphs -> we must not keep using them!

    if x.root[0] in a.vs["node_id"]:
        dist_graph, prox_graph = b, a
    else:
        dist_graph, prox_graph = a, b

    if ret == "distal" or ret == "both":
        dist = morpho.subset_neuron(x, subset=dist_graph.vs["node_id"], inplace=False)

        # Change new root for dist
        dist.nodes.loc[dist.nodes.node_id == cut_node, "type"] = "root"

        # Clear other temporary attributes
        dist._clear_temp_attr(exclude=["igraph", "type", "classify_nodes"])

    if ret == "proximal" or ret == "both":
        ss: Sequence[int] = prox_graph.vs["node_id"] + [cut_node]
        prox = morpho.subset_neuron(x, subset=ss, inplace=False)

        # Change new root for dist
        prox.nodes.loc[prox.nodes.node_id == cut_node, "type"] = "end"

        # Clear other temporary attributes
        prox._clear_temp_attr(exclude=["igraph", "type", "classify_nodes"])

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
    x :     TreeNeuron | NeuronList
            If List, must contain a SINGLE neuron.

    Returns
    -------
    dict
        `{parent_id: [child_id, child_id, ...]}`

    """
    assert isinstance(x, core.TreeNeuron)

    # The node table already *is* a child->parent map, so we can invert it directly
    # instead of building a graph and asking it for `in_edges` once per node.
    nid = x.nodes.node_id.values
    pid = x.nodes.parent_id.values

    childs: Dict[int, List[int]] = {n: [] for n in nid.tolist()}
    has_parent = pid >= 0
    for c, p in zip(nid[has_parent].tolist(), pid[has_parent].tolist()):
        childs[p].append(c)

    return childs


def node_label_sorting(
    x: "core.TreeNeuron", weighted: bool = False
) -> List[Union[str, int]]:
    """Return nodes ordered by node label sorting according to Cuntz
    et al., PLoS Computational Biology (2010).

    Parameters
    ----------
    x :         TreeNeuron
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

    if not isinstance(x, core.TreeNeuron):
        raise TypeError(f'Expected a singleTreeNeuron, got "{type(x)}"')

    if len(x.root) > 1:
        raise ValueError("Unable to process multi-root neurons!")

    # Get relevant terminal nodes
    term = x.nodes[x.nodes.type == "end"].node_id.values

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
    # of the skeleton)
    G = graph.simplify_graph(x.graph)
    curr_points = sorted(
        list(G.predecessors(x.root[0])), key=sort_key(x.root[0]), reverse=True
    )

    # Walk from root towards terminals, prioritising longer branches
    nodes_walked = []
    while curr_points:
        nodes_walked.append(curr_points.pop(0))
        # If the current point is a terminal point, stop here
        if nodes_walked[-1] in term:
            pass
        else:
            new_points = sorted(
                list(G.predecessors(nodes_walked[-1])),
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


def subset_igraph(x: "core.TreeNeuron", keep) -> "igraph.Graph":
    """Induce the sub-graph of a neuron's igraph on a set of node IDs.

    The igraph equivalent of `x.graph.subgraph(keep)`, but without paying to build
    the networkx graph. `node_id` is carried over onto the new vertices.
    """
    G: igraph.Graph = x.igraph
    ids = np.asarray(G.vs["node_id"])
    keep = np.fromiter(keep, dtype=ids.dtype, count=len(keep))
    return G.subgraph(np.where(np.isin(ids, keep))[0])


def connected_components_of(x: "core.TreeNeuron", keep) -> List[Set[int]]:
    """Weakly connected components of the sub-graph induced on `keep`.

    Returns sets of node IDs, mirroring `nx.connected_components`.
    """
    if _fastcore_has("connected_components_graph"):
        # Inducing the subgraph is just dropping every edge with an endpoint
        # outside `keep` - no graph object needs building, which is what makes
        # this ~60x quicker than the igraph route below.
        edges, node_ids = skeleton_edges(x)
        # N.B. via `fromiter`, not `asarray`: `keep` is routinely a *set*, which
        # `np.asarray` turns into a 0-d object array that matches nothing.
        keep = np.fromiter(keep, dtype=node_ids.dtype, count=len(keep))
        in_keep = np.isin(node_ids, keep)

        if len(edges):
            edges = edges[in_keep[edges[:, 0]] & in_keep[edges[:, 1]]]

        labels = utils.fastcore.connected_components_graph(edges, len(node_ids))

        # Nodes outside `keep` are isolated in the induced subgraph but still
        # carry a label, so drop them before grouping.
        kept_ix = np.flatnonzero(in_keep)
        return [
            set(node_ids[kept_ix[g]].tolist())
            for g in _group_by_label(labels[kept_ix])
        ]

    sub = subset_igraph(x, keep)
    ids = np.asarray(sub.vs["node_id"])
    return [set(ids[c].tolist()) for c in sub.components(mode="WEAK")]


def _igraph_to_sparse(graph, weight_attr=None, transpose=False):
    n = graph.vcount()
    edges = np.asarray(graph.get_edgelist(), dtype=np.int64).reshape(-1, 2)

    if weight_attr is None:
        weights = np.ones(len(edges))
    else:
        weights = np.asarray(graph.es[weight_attr])

    rows, cols = edges[:, 0], edges[:, 1]
    if not graph.is_directed():
        rows, cols = np.concatenate([rows, cols]), np.concatenate([cols, rows])
        weights = np.concatenate([weights, weights])

    if transpose:
        rows, cols = cols, rows

    return csr_matrix((weights, (rows, cols)), shape=(n, n))


def connected_subgraph(
    x: Union["core.TreeNeuron", nx.DiGraph], ss: Sequence[Union[str, int]]
) -> Tuple[np.ndarray, Union[int, str]]:
    """Return set of nodes necessary to connect all nodes in subset `ss`.

    Parameters
    ----------
    x :         navis.TreeNeuron | nx.DiGraph
                Neuron (or graph thereof) to get subgraph for.
    ss :        list | array-like
                Node IDs of node to subset to.

    Returns
    -------
    np.ndarray
                Node IDs of connected subgraph.
    root ID
                ID of the node most proximal to the old root in the
                connected subgraph.

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> ends = n.nodes[n.nodes.type.isin(['end', 'root'])].node_id.values
    >>> sg, root = navis.graph.graph_utils.connected_subgraph(n, ends)
    >>> # Since we asked for a subgraph connecting all terminals + root,
    >>> # we expect to see all nodes in the subgraph
    >>> sg.shape[0] == n.nodes.shape[0]
    True

    """
    # `src` is the TreeNeuron we can pull node/parent arrays from (if any). For a
    # bare graph we fall back to reading its edges.
    src = None
    g = None
    if isinstance(x, core.NeuronList):
        if len(x) == 1:
            src = x[0]
    elif isinstance(x, core.TreeNeuron):
        src = x
    elif isinstance(x, (nx.DiGraph, igraph.Graph)):
        g = x
    else:
        raise TypeError(f'Input must be a single TreeNeuron or graph, got "{type(x)}".')

    # Build a `node -> parent` map and the set of nodes in the graph.
    # `parent.get(n)` returns None for roots (which have no parent) - this is our
    # natural walk terminator (mirrors `next(g.successors(n), None)`).
    # For a TreeNeuron we can build this straight from the node table (faster and
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


def insert_nodes(
    x: "core.TreeNeuron",
    where: List[tuple],
    coords: List[tuple] = None,
    validate: bool = True,
    inplace: bool = False,
) -> Optional["core.TreeNeuron"]:
    """Insert new nodes between existing nodes.

    Parameters
    ----------
    x :         TreeNeuron
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
    TreeNeuron

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
    utils.eval_param(x, name="x", allowed_types=(core.TreeNeuron,))

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

    return x


def remove_nodes(
    x: "core.TreeNeuron", which: List[int], inplace: bool = False
) -> Optional["core.TreeNeuron"]:
    """Drop nodes from neuron without disconnecting it.

    Dropping node 2 from 1->2->3 will lead to connectivity 1->3.

    Parameters
    ----------
    x :         TreeNeuron
                Neuron to remove nodes from.
    which :     list of node IDs
                IDs of nodes to remove.
    inplace :   bool
                If True, will rewire the neuron inplace. If False, will return
                a rewired copy of the neuron.

    Returns
    -------
    TreeNeuron

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
    utils.eval_param(x, name="x", allowed_types=(core.TreeNeuron,))

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
    x: "core.TreeNeuron",
    which: List[int],
    new_co: Iterable[Union[float, int]] = None,
    inplace: bool = False,
) -> Optional["core.TreeNeuron"]:
    """Collapse group of nodes into a single node.

    Parameters
    ----------
    x :         TreeNeuron
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
    TreeNeuron

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
    utils.eval_param(x, name="x", allowed_types=(core.TreeNeuron,))

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

    # Make igraph
    G = graph.neuron2igraph(x)

    # Mapping for old to new IDs
    mapping = np.arange(len(G.vs))
    node_ids = np.array(G.vs["node_id"])
    mapping[np.isin(node_ids, which)] = center_node

    # Contract nodes
    G.contract_vertices(mapping, combine_attrs="first")

    # Depth-first search from center node (vertex IDs/parent IDs)
    vids, pids = G.dfs(center_node, mode="all")

    # Rewire nodes
    lop = dict(zip(x.nodes.node_id.values, x.nodes.parent_id.values))
    new_node_ids = np.array(G.vs["node_id"])
    lop.update(dict(zip(new_node_ids[vids], new_node_ids[pids])))
    lop[center_node] = -1

    # Rewire neuron
    x.nodes["parent_id"] = x.nodes.node_id.map(lop)

    # Drop nodes
    keep = ~x.nodes.node_id.isin(which) | (x.nodes.node_id == center_node)
    x.nodes = x.nodes[keep].copy()

    # Check if there is a vertex map to update
    if hasattr(x, "_vertex_map"):
        x._vertex_map[np.isin(x._vertex_map, which)] = center_node

    # Clear temporary attributes
    x._clear_temp_attr()

    return x


def rewire_skeleton(
    x: "core.TreeNeuron", g: nx.Graph, root: Optional[id] = None, inplace: bool = False
) -> Optional["core.TreeNeuron"]:
    """Rewire neuron from graph.

    This function takes a graph representation of a neuron and rewires its
    node table accordingly. This is useful if we made changes to the graph
    (i.e. adding or removing edges) and want those to propagate to the node
    table.

    Parameters
    ----------
    x :         TreeNeuron
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
    TreeNeuron

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> n.n_trees
    1
    >>> # Drop one edge from graph
    >>> g = n.graph.copy()
    >>> g.remove_edge(310, 309)
    >>> # Rewire neuron
    >>> n2 = navis.rewire_skeleton(n, g, inplace=False)
    >>> n2.n_trees
    2

    """
    assert isinstance(x, core.TreeNeuron), f"Expected TreeNeuron, got {type(x)}"
    assert isinstance(g, nx.Graph), f"Expected networkx graph, got {type(g)}"

    if not inplace:
        x = x.copy()

    if g.is_directed():
        g = g.to_undirected()

    # The MST is only needed to break cycles. If the graph is already a forest
    # (which is the common case - e.g. when edges were only removed, or when
    # fragments were bridged) we can skip it: the MST of a forest is that same
    # forest, and computing it is expensive on large neurons.
    if g.number_of_edges() != (g.number_of_nodes() - nx.number_connected_components(g)):
        g = nx.minimum_spanning_tree(g, weight="weight")

    if not root:
        root = x.root[0] if x.root[0] in g.nodes else next(iter(g.nodes))

    # Generate tree for the main component
    tree = nx.dfs_tree(g, source=root)

    # Generate list of parents
    lop = {e[1]: e[0] for e in tree.edges}

    # If the graph has more than one connected component,
    # the remaining components have arbitrary roots
    if len(tree.edges) != len(g.edges):
        for cc in nx.connected_components(g):
            if root not in cc:
                tree = nx.dfs_tree(g, source=cc.pop())
                lop.update({e[1]: e[0] for e in tree.edges})

    # Update parent IDs
    x.nodes["parent_id"] = x.nodes.node_id.map(lambda x: lop.get(x, -1))

    x._clear_temp_attr()

    return x


def match_mesh_skeleton(mesh, skeleton):
    """Match vertices of MeshNeuron to nodes of TreeNeuron.

    Parameters
    ----------
    mesh :      MeshNeuron
                MeshNeuron to match.
    skeleton :  TreeNeuron
                Skeleton to match.

    Returns
    -------
    np.ndarray
                Array of skeleton node IDs for each vertex in the mesh.

    """
    if not isinstance(mesh, core.MeshNeuron):
        raise TypeError(f"Expected MeshNeuron, got {type(mesh)}")

    if not isinstance(skeleton, core.TreeNeuron):
        raise TypeError(f"Expected TreeNeuron, got {type(skeleton)}")

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
    x :         TreeNeuron | MeshNeuron
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
                applicable for TreeNeurons. If `True`, labels will only propagate
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
    if not isinstance(x, (core.TreeNeuron, core.MeshNeuron)):
        raise TypeError(f"Expected TreeNeuron or MeshNeuron, got {type(x)}")

    assert return_probs in (
        False,
        True,
        "softmax",
        "raw",
    ), f"Invalid value for return_probs: {return_probs}"
    assert max_iter > 0, "max_iter must be a positive integer"
    assert isinstance(tol, (int, float)) and tol > 0, "tol must be a positive float"

    if isinstance(labels, str):
        if isinstance(x, core.TreeNeuron):
            if labels not in x.nodes.columns:
                raise ValueError(f'No node property "{labels}" found in neuron.')
            elif getattr(x, labels).shape[0] != len(x.nodes):
                raise ValueError(
                    f'Length of node property "{labels}" does not match number of nodes ({len(x.nodes)})'
                )
            labels = dict(zip(x.nodes.node_id.values, x.nodes[labels].values))
        elif isinstance(x, core.MeshNeuron):
            if not hasattr(x, labels):
                raise ValueError(f'No vertex property "{labels}" found in neuron.')
            elif getattr(x, labels).shape[0] != len(x.vertices):
                raise ValueError(
                    f'Length of vertex property "{labels}" does not match number of vertices ({len(x.vertices)})'
                )
            labels = dict(zip(range(len(x.vertices)), getattr(x, labels)))
    elif not isinstance(labels, dict):
        if isinstance(x, core.TreeNeuron):
            if len(labels) != len(x.nodes):
                raise ValueError(
                    f"Length of labels ({len(labels)}) does not match number of nodes ({len(x.nodes)})"
                )
            labels = dict(zip(x.nodes.node_id.values, labels))
        elif isinstance(x, core.MeshNeuron):
            if len(labels) != len(x.vertices):
                raise ValueError(
                    f"Length of labels ({len(labels)}) does not match number of vertices ({len(x.vertices)})"
                )
            labels = dict(zip(range(len(x.vertices)), labels))

    # Drop missing labels from the dict
    labels = {k: v for k, v in labels.items() if not pd.isnull(v)}

    # Convert neuron to graph. Note igraph's vertex order is identical to the
    # node/vertex table, so the row order of `F` (which `return_probs` exposes)
    # is the same as it was when this used networkx.
    G: igraph.Graph = x.igraph

    nodes = G.vs["node_id"] if isinstance(x, core.TreeNeuron) else list(G.vs.indices)
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
            if isinstance(x, core.TreeNeuron):
                if weights not in x.nodes.columns:
                    raise ValueError(f'No node property "{weights}" found in neuron.')
                elif getattr(x, weights).shape[0] != len(x.nodes):
                    raise ValueError(
                        f'Length of node property "{weights}" does not match number of nodes ({len(x.nodes)})'
                    )
                per_node_weights = getattr(x, weights).values.astype(np.float32)
            elif isinstance(x, core.MeshNeuron):
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

    # Adjacency matrix (sparse, float32). `_igraph_to_sparse` already emits both
    # directions for an undirected graph (i.e. MeshNeurons); a skeleton's graph is
    # directed child->parent and needs symmetrising by hand. Skeletons have no
    # reciprocal edges, so adding the transpose can't double-count.
    A = _igraph_to_sparse(G, weight_attr="weight").astype(np.float32)
    if not directed and G.is_directed():
        A = A + A.T
    A = A.tocsr()

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
    if isinstance(x, core.TreeNeuron):
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
