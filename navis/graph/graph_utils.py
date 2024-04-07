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

import pandas as pd
import numpy as np
import networkx as nx

from typing import Union, Optional, List, Tuple, Sequence, Dict, Set, overload, Iterable
from typing_extensions import Literal

from pandas.api.types import CategoricalDtype
from scipy.sparse import csgraph, csr_matrix

from .. import graph, utils, config, core, morpho

# Set up logging
logger = config.get_logger(__name__)

__all__ = sorted(['classify_nodes', 'cut_skeleton', 'longest_neurite',
                  'split_into_fragments', 'reroot_skeleton', 'distal_to',
                  'dist_between', 'find_main_branchpoint',
                  'generate_list_of_childs', 'geodesic_matrix',
                  'node_label_sorting',
                  'segment_length', 'rewire_skeleton', 'insert_nodes',
                  'remove_nodes', 'dist_to_root'])


@utils.map_neuronlist(desc='Gen. segments', allow_parallel=True)
def _generate_segments(x: 'core.NeuronObject',
                       weight: Optional[str] = None,
                       return_lengths: bool = False) -> Union[list, Tuple[list, list]]:
    """Generate segments maximizing segment lengths.

    Parameters
    ----------
    x :         TreeNeuron | NeuronList
                May contain multiple neurons.
    weight :    'weight' | None, optional
                If ``"weight"`` use physical, geodesic length to determine
                segment length. If ``None`` use number of nodes (faster).
    return_lengths : bool
                If True, also return lengths of segments according to ``weight``.

    Returns
    -------
    segments :  list
                Segments as list of lists containing node IDs. List is
                sorted by segment lengths.
    lengths :   list
                Length for each segment according to ``weight``. Only provided
                if `return_lengths` is True.

    Examples
    --------
    This is for doctests mostly

    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> unweighted = navis.graph_utils._generate_segments(n)
    >>> weighted = navis.graph_utils._generate_segments(n, weight='weight')

    """
    if not isinstance(x, core.TreeNeuron):
        raise ValueError(f'Expected TreeNeuron, got "{type(x)}"')

    # At this point x is TreeNeuron
    x: core.TreeNeuron

    assert weight in ('weight', None), f'Unable to use weight "{weight}"'
    d = dist_to_root(x, igraph_indices=False, weight=weight)
    endNodeIDs = x.nodes[x.nodes.type == 'end'].node_id.values
    endNodeIDs = sorted(endNodeIDs, key=lambda x: d.get(x, 0), reverse=True)

    if config.use_igraph and x.igraph:
        g: igraph.Graph = x.igraph  # noqa
        # Convert endNodeIDs to indices
        id2ix = dict(zip(x.igraph.vs['node_id'], range(len(x.igraph.vs))))
        endNodeIDs = [id2ix[n] for n in endNodeIDs]
    else:
        g: nx.DiGraph = x.graph

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

    # If igraph, turn indices back to node IDs
    if config.use_igraph and x.igraph:
        ix2id = {v: k for k, v in id2ix.items()}
        sequences = [[ix2id[ix] for ix in s] for s in sequences]

    # Sort sequences by length
    lengths = [d[s[0]] - d[s[-1]] for s in sequences]
    sequences = [x for _, x in sorted(zip(lengths, sequences), reverse=True)]

    if return_lengths:
        return sequences, sorted(lengths, reverse=True)
    else:
        return sequences


def _connected_components(x: Union['core.TreeNeuron', 'core.MeshNeuron']) -> List[Set[int]]:
    """Extract the connected components within a neuron.

    Parameters
    ----------
    x :         TreeNeuron | MeshNeuron

    Returns
    -------
    list
                List containing sets of node/vertex IDs for each subgraph.

    Examples
    --------
    For doctest only

    >>> import navis
    >>> n = navis.example_neurons(1, kind='skeleton')
    >>> cc = navis.graph_utils._connected_components(n)
    >>> m = navis.example_neurons(1, kind='mesh')
    >>> cc = navis.graph_utils._connected_components(m)

    """
    assert isinstance(x, (core.TreeNeuron, core.MeshNeuron))

    if config.use_igraph and x.igraph:
        G: igraph.Graph = x.igraph  # noqa
        # Get the vertex clustering
        vc = G.components(mode='WEAK')
        # Membership maps indices to connected components
        ms = np.array(vc.membership)
        if isinstance(x, core.TreeNeuron):
            # For skeletons we need node IDs
            ids = np.array(G.vs['node_id'])
        else:
            # For MeshNeurons we can use the indices directly
            ids = np.array(G.vs.indices)

        # Extract node IDs/vertex indices for each component
        cc = [ids[ms == i] for i in np.unique(ms)]
    else:
        G: nx.DiGraph = x.graph
        cc = nx.connected_components(G.to_undirected())
        cc = list(cc)

    return cc


def _break_segments(x: 'core.NeuronObject') -> list:
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
        logger.error('Unexpected datatype: %s' % str(type(x)))
        raise ValueError

    # At this point x is TreeNeuron
    x: core.TreeNeuron

    if x.igraph and config.use_igraph:
        g: Union['igraph.Graph', 'nx.DiGraph'] = x.igraph # noqa
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
        for s in seeds:
            parent = g.successors(s)[0]
            seg = [s, parent]
            while parent not in stops:
                parent = g.successors(parent)[0]
                seg.append(parent)
            seg_list.append(seg)
        # Translate indices to node IDs
        ix_id = {v: n for v, n in zip(g.vs.indices,
                                      g.vs.get_attribute_values('node_id'))}
        seg_list = [[ix_id[n] for n in s] for s in seg_list]
    else:
        seeds = x.nodes[x.nodes.type.isin(['branch', 'end'])].node_id.values
        stops = x.nodes[x.nodes.type.isin(['branch', 'root'])].node_id.values
        # Converting to set speeds up the "parent in stops" check
        stops = set(stops)
        g = x.graph
        seg_list = []
        for s in seeds:
            parent = next(g.successors(s), None)
            seg = [s, parent]
            while parent not in stops:
                parent = next(g.successors(parent), None)
                seg.append(parent)
            seg_list.append(seg)

    return seg_list


@utils.lock_neuron
def dist_to_root(x: 'core.TreeNeuron',
                 weight=None,
                 igraph_indices: bool = False) -> dict:
    """Calculate distance to root for each node.

    Parameters
    ----------
    x :                 TreeNeuron
    weight :            str, optional
                        Use "weight" if you want geodesic distance and ``None``
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
    :func:`navis.geodesic_matrix`
                        For distances between all points.

    """
    if not isinstance(x, core.TreeNeuron):
        raise TypeError(f'Expected TreeNeuron, got {type(x)}')

    dist = {}
    for root in x.root:
        dist.update(nx.shortest_path_length(x.graph, target=root, weight=weight))

    # Map node ID to vertex index for igraph
    if igraph_indices:
        if not x.igraph:
            raise ValueError('Neuron does not have an igraph representation.')
        id2ix = dict(zip(x.igraph.vs['node_id'], range(len(x.igraph.vs))))
        dist = {id2ix[k]: v for k, v in dist.items()}

    return dist


def _edge_count_to_root_old(x: 'core.TreeNeuron') -> dict:
    """Return a map of nodeID vs number of edges.

    Starts from the first node that lacks successors (aka the root).

    """
    current_level: List[int]
    g: Union['igraph.Graph', 'nx.DiGraph']  # noqa
    if x.igraph and config.use_igraph:
        g = x.igraph
        current_level = g.vs(_outdegree=0).indices
    else:
        g = x.graph
        current_level = list(x.root)

    dist = {}
    count = 1
    next_level: List[Union[str, int]] = []
    while current_level:
        # Consume all elements in current_level
        while current_level:
            node = current_level.pop()
            dist[node] = count
            next_level.extend(g.predecessors(node))
        # Rotate lists (current_level is now empty)
        current_level, next_level = next_level, current_level  # type: ignore
        count += 1

    # Map vertex index to node ID
    if x.igraph and config.use_igraph:
        # Grab graph once to avoid overhead from stale checks
        g = x.igraph
        dist = {g.vs[k]['node_id']: v for k, v in dist.items()}

    return dist


@utils.map_neuronlist(desc='Classifying', allow_parallel=True)
@utils.lock_neuron
def _classify_nodes_old(x: 'core.NeuronObject',
                        inplace: bool = True
                        ) -> Optional['core.NeuronObject']:
    """Classify neuron's nodes into end nodes, branches, slabs or root.

    Adds ``'type'`` column to ``x.nodes``.

    Parameters
    ----------
    x :         TreeNeuron | NeuronList
                Neuron(s) whose nodes to classify nodes.
    inplace :   bool, optional
                If ``False``, nodes will be classified on a copy which is then
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

    # At this point x is TreeNeuron
    x: core.TreeNeuron

    # Make sure there are nodes to classify
    if not x.nodes.empty:
        if config.use_igraph and x.igraph:
            # Get graph representation of neuron
            vs = x.igraph.vs
            # Get branch/end nodes based on their degree of connectivity
            ends = vs.select(_indegree=0).get_attribute_values('node_id')
            branches = vs.select(_indegree_gt=1).get_attribute_values('node_id')
        else:
            # Get graph representation of neuron
            g = x.graph
            # Get branch/end nodes based on their degree of connectivity
            deg = pd.DataFrame.from_dict(dict(g.degree()), orient='index')
            # [ n for n in g.nodes if g.degree(n) == 1 ]
            ends = deg[deg.iloc[:, 0] == 1].index.values
            # [ n for n in g.nodes if g.degree(n) > 2 ]
            branches = deg[deg.iloc[:, 0] > 2].index.values

        # This also resets the column if it already exists. This is important
        # because an existing column will be categorical and if we try setting
        # types that didn't previously exist, it will throw exceptions.
        x.nodes['type'] = 'slab'

        x.nodes.loc[x.nodes.node_id.isin(ends), 'type'] = 'end'
        x.nodes.loc[x.nodes.node_id.isin(branches), 'type'] = 'branch'
        x.nodes.loc[x.nodes.parent_id < 0, 'type'] = 'root'
    else:
        x.nodes['type'] = None

    # Turn into categorical data - saves tons of memory
    # Note that we have to make sure all categories are set even if they
    # don't exist (e.g. if a neuron has no branch points)
    cat_types = CategoricalDtype(categories=["end", "branch", "root", "slab"],
                                 ordered=False)
    x.nodes['type'] = x.nodes['type'].astype(cat_types)

    return x



@utils.map_neuronlist(desc='Classifying', allow_parallel=True)
@utils.lock_neuron
def classify_nodes(x: "core.NeuronObject", categorical=True, inplace: bool = True):
    """Classify neuron's nodes into end nodes, branches, slabs or root.

    Adds a ``'type'`` column to ``x.nodes`` table.

    Parameters
    ----------
    x :         TreeNeuron | NeuronList
                Neuron(s) whose nodes to classify.
    categorical : bool
                If True (default), will use categorical data type which takes
                up much less memory at a small run-time overhead.
    inplace :   bool, optional
                If ``False``, nodes will be classified on a copy which is then
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

    # Make sure there are nodes to classify
    # Note: I have tried to optimized the s**t out of this, i.e. every
    # single line of code here has been tested for speed. Do not
    # change anything unless you know what you're doing!

    # Turns out that numpy.isin() recently started to complain if the
    # node_ids are uint64 and the parent_ids are int64 (but strangely
    # not with 32bit integers). If that's the case we have to convert
    # the node_ids to int64.
    node_ids = x.nodes.node_id.values
    parent_ids = x.nodes.parent_id.values

    if node_ids.dtype == np.uint64:
        node_ids = node_ids.astype(np.int64)

    cl = np.full(len(x.nodes), "slab", dtype="<U6")
    cl[~np.isin(node_ids, parent_ids)] = "end"
    bp = x.nodes.parent_id.value_counts()
    bp = bp.index.values[bp.values > 1]
    cl[np.isin(node_ids, bp)] = "branch"
    cl[parent_ids < 0] = "root"
    if categorical:
        cl = pd.Categorical(cl, categories=["end", "branch", "root", "slab"], ordered=False)
    x.nodes["type"] = cl

    return x


#  only this combination will return a single bool
@overload
def distal_to(x: 'core.TreeNeuron',
              a: Union[str, str],
              b: Union[str, int],
              ) -> bool:
    pass


#  if above types don't a DataFrame will be returned
@overload
def distal_to(x: 'core.TreeNeuron',
              a: Optional[List[Union[str, int]]],
              b: Optional[Union[str, int, List[Union[str, int]]]],
              ) -> pd.DataFrame:
    pass


#  if above types don't a DataFrame will be returned
@overload
def distal_to(x: 'core.TreeNeuron',
              a: Optional[Union[str, int, List[Union[str, int]]]],
              b: Optional[List[Union[str, int]]],
              ) -> pd.DataFrame:
    pass


@utils.lock_neuron
def distal_to(x: 'core.TreeNeuron',
              a: Optional[Union[str, int, List[Union[str, int]]]] = None,
              b: Optional[Union[str, int, List[Union[str, int]]]] = None,
              ) -> Union[bool, pd.DataFrame]:
    """Check if nodes A are distal to nodes B.

    Important
    ---------
    Please note that if node A is not distal to node B, this does **not**
    automatically mean it is proximal instead: if nodes are on different
    branches, they are neither distal nor proximal to one another! To test
    for this case run a->b and b->a - if both return ``False``, nodes are on
    different branches.

    Also: if a and b are the same node, this function will return ``True``!

    Parameters
    ----------
    x :     TreeNeuron
    a,b :   single node ID | list of node IDs | None, optional
            If no node IDs are provided, will consider all node. Note that for
            large sets of nodes it might be more efficient to use
            :func:`navis.geodesic_matrix` (see examples).

    Returns
    -------
    bool
            If ``a`` and ``b`` are single node IDs respectively.
    pd.DataFrame
            If ``a`` and/or ``b`` are lists of node IDs. Columns and rows
            (index) represent node IDs. Neurons ``a`` are rows, neurons
            ``b`` are columns.

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
    :func:`navis.geodesic_matrix`
            Depending on your neuron and how many nodes you're asking for,
            this function can be considerably faster! See examples.

    """
    if isinstance(x, core.NeuronList) and len(x) == 1:
        x = x[0]

    if not isinstance(x, core.TreeNeuron):
        raise ValueError(f'Please pass a single TreeNeuron, got {type(x)}')

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

    if x.igraph and config.use_igraph:
        # Map node ID to index
        id2ix = {n: v for v, n in zip(x.igraph.vs.indices,
                                      x.igraph.vs['node_id'])}

        # Convert node IDs to indices
        tnA = [id2ix[n] for n in tnA]  # type: ignore
        tnB = [id2ix[n] for n in tnB]  # type: ignore

        # Get path lengths
        le = x.igraph.shortest_paths(tnA, tnB, mode='OUT')

        # Converting to numpy array first is ~2X as fast
        le = np.asarray(le)

        # Convert to True/False
        le = le != float('inf')

        df = pd.DataFrame(le,
                          index=x.igraph.vs[tnA]['node_id'],
                          columns=x.igraph.vs[tnB]['node_id'])
    else:
        # Generate empty DataFrame
        df = pd.DataFrame(np.zeros((len(tnA), len(tnB)), dtype=bool),
                          columns=tnB, index=tnA)

        # Iterate over all targets
        # Grab graph once to avoid overhead from stale checks
        g = x.graph
        for nB in config.tqdm(tnB, desc='Querying paths',
                              disable=(len(tnB) < 1000) | config.pbar_hide,
                              leave=config.pbar_leave):
            # Get all paths TO this target. This function returns a dictionary:
            # { source1 : path_length, source2 : path_length, ... } containing
            # all nodes distal to this node.
            paths = nx.shortest_path_length(g, source=None, target=nB)
            # Check if sources are among our targets
            df[nB] = [nA in paths for nA in tnA]

    if df.shape == (1, 1):
        return df.values[0][0]
    else:
        # Return boolean
        return df


def geodesic_matrix(x: 'core.NeuronObject',
                    from_: Optional[Iterable[int]] = None,
                    directed: bool = False,
                    weight: Optional[str] = 'weight',
                    limit: Union[float, int] = np.inf
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
    directed :  bool, optional
                If True, pairs without a child->parent path will be returned
                with ``distance = "inf"``. Only relevant for ``TreeNeurons``.
    weight :    'weight' | None, optional
                If ``weight`` distances are given as physical length.
                If ``None`` distances is number of nodes.
    limit :     int | float, optional
                Use to limit distance calculations. Nodes that are not within
                ``limit`` will have distance ``np.inf``. If neuron has its
                `.units` set, you can also pass a string such as "10 microns".

    Returns
    -------
    pd.DataFrame
                Geodesic distance matrix. Distances in nanometres.

    See Also
    --------
    :func:`navis.distal_to`
        Check if a node A is distal to node B.
    :func:`navis.dist_between`
        Get point-to-point geodesic distances.
    :func:`navis.dist_to_root`
        Distances from all skeleton node to their root(s).

    Examples
    --------
    Find average geodesic distance between all leaf nodes

    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> # Generate distance matrix
    >>> m = navis.geodesic_matrix(n)
    >>> # Subset matrix to leaf nodes
    >>> leafs = n.nodes[n.nodes.type=='end'].node_id.values
    >>> l_dist = m.loc[leafs, leafs]
    >>> # Get mean
    >>> round(l_dist.mean().mean())
    12983

    """
    if isinstance(x, core.NeuronList):
        if len(x) == 1:
            x = x[0]
        else:
            raise ValueError('Cannot process more than a single neuron.')
    elif not isinstance(x, (core.TreeNeuron, core.MeshNeuron)):
        raise ValueError(f'Unable to process data of type "{type(x)}"')

    limit = x.map_units(limit, on_error='raise')

    # Makes no sense to use directed for MeshNeurons
    if isinstance(x, core.MeshNeuron):
        directed = False

    if x.igraph and config.use_igraph:
        if isinstance(x, core.TreeNeuron):
            nodeList = np.array(x.igraph.vs.get_attribute_values('node_id'))
        else:
            nodeList = np.arange(len(x.igraph.vs))

        # Matrix is ordered by vertex number
        m = _igraph_to_sparse(x.igraph, weight_attr=weight)
    else:
        nodeList = np.array(x.graph.nodes())

        if hasattr(nx, 'to_scipy_sparse_matrix'):
            m = nx.to_scipy_sparse_matrix(x.graph, nodeList,
                                          weight=weight)
        else:
            m = nx.to_scipy_sparse_array(x.graph, nodeList,
                                         weight=weight)

    if not isinstance(from_, type(None)):
        from_ = np.unique(utils.make_iterable(from_))

        miss = from_[~np.isin(from_, nodeList)].astype(str)
        if any(miss):
            raise ValueError(f'Node/vertex IDs not present: {", ".join(miss)}')

        indices = np.where(np.isin(nodeList, from_))[0]
        ix = nodeList[indices]
    else:
        indices = None
        ix = nodeList

    # For some reason csgrpah.dijkstra expects indices/indptr as int32
    # igraph seems to do that by default but networkx uses int64 for indices
    m.indptr = m.indptr.astype('int32', copy=False)
    m.indices = m.indices.astype('int32', copy=False)
    dmat = csgraph.dijkstra(m,
                            directed=directed,
                            indices=indices,
                            limit=limit)

    return pd.DataFrame(dmat, columns=nodeList, index=ix)  # type: ignore  # no stubs


@utils.lock_neuron
def segment_length(x: 'core.TreeNeuron',
                   segment: List[int]) -> float:
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
    :func:`navis.dist_between`
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

    # Get graph once to avoid overhead from validation - do NOT change this
    graph = x.graph
    dist = np.array([graph.edges[(c, p)]['weight']
                     for c, p in zip(segment[:-1], segment[1:])])
    return sum(dist)


@utils.lock_neuron
def dist_between(x: 'core.NeuronObject',
                 a: int,
                 b: int) -> float:
    """Get the geodesic distance between nodes in nanometers.

    Parameters
    ----------
    x :             TreeNeuron | MeshNeuron | NeuronList
                    If NeuronList must contain only a single neuron.
    a,b :           int
                    Node IDs (for TreeNeurons) or vertex indices (MeshNeurons)
                    to check the distance between.

    Returns
    -------
    int
                    distance in nm

    See Also
    --------
    :func:`~navis.distal_to`
        Check if a node A is distal to node B.
    :func:`~navis.geodesic_matrix`
        Get all-by-all geodesic distance matrix.
    :func:`navis.segment_length`
        Much faster if you have a linear segment and know all node IDs.

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> d = navis.dist_between(n,
    ...                        n.nodes.node_id.values[0],
    ...                        n.nodes.node_id.values[1])

    """
    if isinstance(x, core.NeuronList):
        if len(x) == 1:
            x = x[0]
        else:
            raise ValueError(f'Need a single TreeNeuron, got {len(x)}')

    if isinstance(x, (core.TreeNeuron, core.MeshNeuron)):
        G: Union['igraph.Graph',  # noqa
                 'nx.DiGraph'] = x.igraph if (x.igraph and config.use_igraph) else x.graph
    elif isinstance(x, nx.DiGraph):
        G = x
    elif 'igraph' in str(type(x.igraph)):
        # We can't use isinstance here because igraph library might not be installed
        G = x
    else:
        raise ValueError(f'Unable to process data of type {type(x)}')

    if ((utils.is_iterable(a) and len(a) > 1)  # type: ignore  # this is just a check
            or (utils.is_iterable(b) and len(b) > 1)):   # type: ignore  # this is just a check
        raise ValueError('Can only process single nodes/vertices. Use '
                         'navis.geodesic_matrix instead.')

    a = utils.make_non_iterable(a)
    b = utils.make_non_iterable(b)

    try:
        _ = int(a)
        _ = int(b)
    except BaseException:
        raise ValueError('a, b need to be node IDs or vertex indices!')

    # If we're working with network X DiGraph
    if isinstance(G, nx.DiGraph):
        return int(nx.algorithms.shortest_path_length(G.to_undirected(as_view=True),
                                                      a, b,
                                                      weight='weight'))
    else:
        if isinstance(x, core.TreeNeuron):
            a = G.vs.find(node_id=a)
            b = G.vs.find(node_id=b)

        # If not, we're assuming g is an iGraph object
        return G.shortest_paths(a, b,
                                weights='weight',
                                mode='ALL')[0][0]


@utils.map_neuronlist(desc='Searching', allow_parallel=True)
@utils.meshneuron_skeleton(method='node_to_vertex')
def find_main_branchpoint(x: 'core.NeuronObject',
                          method: Union[Literal['longest_neurite'],
                                        Literal['betweenness']] = 'betweenness',
                          threshold: float = .95,
                          reroot_soma: bool = False) -> Union[int, List[int]]:
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
    utils.eval_param(method, name='method',
                     allowed_values=('longest_neurite', 'betweenness'))

    if not isinstance(x, core.TreeNeuron):
        raise TypeError(f'Expected TreeNeuron(s), got "{type(x)}"')

    # At this point x is TreeNeuron
    x: core.TreeNeuron

    # If no branches
    if x.nodes[x.nodes.type == 'branch'].empty:
        raise ValueError('Neuron has no branch points.')

    if reroot_soma and not isinstance(x.soma, type(None)):
        x = x.reroot(x.soma, inplace=False)

    if method == 'longest_neurite':
        G = x.graph

        # First, find longest path
        longest = nx.dag_longest_path(G, weight='weight')

        # Remove longest path
        # (use subgraph to avoid editing original or copying raph)
        keep = ~np.isin(G.nodes, longest)
        G = G.subgraph(np.array(G.nodes)[keep])

        # Find second longst path
        sc_longest = nx.dag_longest_path(G, weight='weight')

        # Parent of the last node in sc_longest is the common branch point
        bp = list(x.graph.successors(sc_longest[-1]))[0]
    else:
        # Get betweenness for each node
        x = morpho.betweeness_centrality(x, directed=True, from_='branch_points')
        # Get branch points with highest centrality
        high_between = x.branch_points.betweenness >= x.branch_points.betweenness.max() * threshold
        candidates = x.branch_points[high_between]

        # If only one nodes just go with it
        if candidates.shape[0] == 1:
            bp = candidates.node_id.values[0]
        else:
            # If multiple points get the farthest one from the root
            root_dists = dist_to_root(x)
            bp = sorted(candidates.node_id.values,
                        key=lambda x: root_dists[x])[-1]


    # This makes sure we get the same data type as in the node table
    # -> Network X seems to sometimes convert integers to floats
    return x.nodes.node_id.dtype.type(bp)


@utils.meshneuron_skeleton(method='split')
def split_into_fragments(x: 'core.NeuronObject',
                         n: int = 2,
                         min_size: Optional[Union[float, str]] = None,
                         reroot_soma: bool = False) -> 'core.NeuronList':
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
            raise Exception(f'{x.shape[0]} neurons provided. Please provide '
                            'only a single neuron!')

    if not isinstance(x, core.TreeNeuron):
        raise TypeError(f'Expected a single TreeNeuron, got "{type(x)}"')

    if n < 2:
        raise ValueError('Number of fragments must be at least 2.')

    # At this point x is TreeNeuron
    x: core.TreeNeuron

    min_size = x.map_units(min_size, on_error='raise')

    if reroot_soma and not isinstance(x.soma, type(None)):
        x.reroot(x.soma, inplace=True)

    # Collect nodes of the n longest neurites
    tn_to_preserve: List[int] = []
    fragments = []
    i = 0
    while i < n:
        if tn_to_preserve:
            # Generate fresh graph
            g = graph.neuron2nx(x)

            # Remove nodes that we have already preserved
            g.remove_nodes_from(tn_to_preserve)
        else:
            g = x.graph

        # Get path
        longest_path = nx.dag_longest_path(g)

        # Check if fragment is still long enough
        if min_size:
            this_length = sum([v for k, v in nx.get_edge_attributes(
                g, 'weight').items() if k[1] in longest_path])
            if this_length <= min_size:
                break

        tn_to_preserve += longest_path
        fragments.append(longest_path)

        i += 1

    # Next, make some virtual cuts and get the complement of nodes for
    # each fragment
    graphs = [x.graph.copy()]
    # Grab graph once to avoide overhead from stale checking
    g = x.graph
    for fr in fragments[1:]:
        this_g = nx.bfs_tree(g, fr[-1], reverse=True)

        graphs.append(this_g)

    # Next, we need to remove nodes that are in subsequent graphs from
    # those graphs
    for i, g in enumerate(graphs):
        for g2 in graphs[i + 1:]:
            g.remove_nodes_from(g2.nodes)

    # Now make neurons
    nl = core.NeuronList([morpho.subset_neuron(x, g) for g in graphs])

    return nl


@utils.map_neuronlist(desc='Pruning', allow_parallel=True)
@utils.meshneuron_skeleton(method='subset')
def longest_neurite(x: 'core.NeuronObject',
                    n: int = 1,
                    reroot_soma: bool = False,
                    from_root: bool = True,
                    inverse: bool = False,
                    inplace: bool = False) -> 'core.NeuronObject':
    """Return a neuron consisting of only the longest neurite(s).

    Based on geodesic distances.

    Parameters
    ----------
    x :                 TreeNeuron | NeuronList
                        Neuron(s) to prune.
    n :                 int | slice
                        Number of longest neurites to preserve. For example:
                         - ``n=1`` keeps the longest neurites
                         - ``n=2`` keeps the two longest neurites
                         - ``n=slice(1, None)`` removes the longest neurite
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
    :func:`~navis.split_into_fragments`
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
        raise ValueError('Number of longest neurites to preserve must be >=1')

    # At this point x is TreeNeuron
    x: core.TreeNeuron

    if not inplace:
        x = x.copy()

    if not from_root:
        # Find the two most distal points
        leafs = x.leafs.node_id.values
        dists = geodesic_matrix(x, from_=leafs)[leafs]

        # This might be multiple values
        mx = np.where(dists == np.max(dists.values))
        start = dists.columns[mx[0][0]]

        # Reroot to one of the nodes that gives the longest distance
        x.reroot(start, inplace=True)
    elif reroot_soma and not isinstance(x.soma, type(None)):
        x.reroot(x.soma, inplace=True)

    segments = _generate_segments(x, weight='weight')

    if isinstance(n, (int, np.integer)):
        tn_to_preserve: List[int] = [tn for s in segments[:n] for tn in s]
    elif isinstance(n, slice):
        tn_to_preserve = [tn for s in segments[n] for tn in s]
    else:
        raise TypeError(f'Unable to use N of type "{type(n)}"')

    if not inverse:
        _ = morpho.subset_neuron(x, tn_to_preserve, inplace=True)
    else:
        _ = morpho.subset_neuron(x, ~np.isin(x.nodes.node_id.values, tn_to_preserve),
                                 inplace=True)

    return x


@utils.lock_neuron
def reroot_skeleton(x: 'core.NeuronObject',
                    new_root: Union[int, str],
                    inplace: bool = False) -> 'core.TreeNeuron':
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
    :func:`~navis.TreeNeuron.reroot`
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
            raise ValueError(f'Expected a single neuron, got {len(x)}')

    if not isinstance(x, core.TreeNeuron):
        raise ValueError(f'Unable to reroot object of type "{type(x)}"')

    # Make new root an iterable
    new_roots = utils.make_iterable(new_root)

    # Parse new roots
    for i, root in enumerate(new_roots):
        if root is None:
            raise ValueError('New root can not be <None>')

        # If new root is a tag, rather than a ID, try finding that node
        if isinstance(root, str):
            if x.tags is None:
                raise ValueError("Neuron does not have tags")

            if root not in x.tags:
                raise ValueError(f'#{x.id}: Found no nodes with tag {root}'
                                 ' - please double check!')

            elif len(x.tags[root]) > 1:
                raise ValueError(f'#{x.id}: Found multiple node with tag '
                                 f'{root} - please double check!')
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

        if x.igraph and config.use_igraph:
            # Grab graph once to avoid overhead from stale checks
            g = x.igraph

            # Prevent warnings in the following code - querying paths between
            # unreachable nodes will otherwise generate a runtime warning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Find paths to all roots
                path = g.get_shortest_paths(g.vs.find(node_id=new_root),
                                            [g.vs.find(node_id=r) for r in x.root])
                epath = g.get_shortest_paths(g.vs.find(node_id=new_root),
                                             [g.vs.find(node_id=r) for r in x.root],
                                             output='epath')

            # Extract paths that actually worked (i.e. within a continuous fragment)
            path = [p for p in path if p][0]
            epath = [p for p in epath if p][0]

            edges = [(s, t) for s, t in zip(path[:-1], path[1:])]

            weights = [g.es[e]['weight'] for e in epath]

            # Get all weights and append inversed new weights
            all_weights = g.es['weight'] + weights

            # Add inverse edges: old_root->new_root
            g.add_edges([(e[1], e[0]) for e in edges])

            # Re-set weights
            g.es['weight'] = all_weights

            # Remove new_root->old_root
            g.delete_edges(edges)

            # Get degree of old root for later categorisation
            old_root_deg = len(g.es.select(_target=path[-1]))

            # Translate path indices to node IDs
            ix2id = {ix: n for ix, n in zip(g.vs.indices,
                                            g.vs.get_attribute_values('node_id'))}
            path = [ix2id[i] for i in path]
        else:
            # Grab graph once to avoid overhead from stale checks
            g = x.graph
            # If this NetworkX graph is just an (immutable) view, turn it into a
            # full, independent graph
            nx_main_version = '.'.join(nx.__version__.split(".")[:2])
            if float(nx_main_version) < 2.2:
                if isinstance(g, nx.classes.graphviews.ReadOnlyGraph):
                    x._graph_nx = g = nx.DiGraph(g)
            elif hasattr(g, '_NODE_OK'):
                x._graph_nx = g = nx.DiGraph(g)
            elif nx.is_frozen(g):
                x._graph_nx = g = nx.DiGraph(g)

            # Walk from new root to old root and remove edges along the way
            parent = next(g.successors(new_root), None)
            if not parent:
                # new_root is already the root
                continue

            path = [new_root]
            weights = []
            while parent is not None:
                weights.append(g[path[-1]][parent]['weight'])
                g.remove_edge(path[-1], parent)
                path.append(parent)
                parent = next(g.successors(parent), None)

            # Invert path and add weights
            new_edges = [(path[i + 1], path[i],
                          {'weight': weights[i]}) for i in range(len(path) - 1)]

            # Add inverted path between old and new root
            g.add_edges_from(new_edges)

            # Get degree of old root for later categorisation
            old_root_deg = g.in_degree(path[-1])

        # Set index to node ID for later
        x.nodes.set_index('node_id', inplace=True)

        # Propagate changes in graph back to node table
        # Assign new node type to old root
        x.nodes.loc[path[1:], 'parent_id'] = path[:-1]
        if old_root_deg == 1:
            x.nodes.loc[path[-1], 'type'] = 'slab'
        elif old_root_deg > 1:
            x.nodes.loc[path[-1], 'type'] = 'branch'
        else:
            x.nodes.loc[path[-1], 'type'] = 'end'
        # Make new root node type "root"
        x.nodes.loc[path[0], 'type'] = 'root'

        # Set new root's parent to None
        x.nodes.loc[new_root, 'parent_id'] = -1

        # Reset index
        x.nodes.reset_index(drop=False, inplace=True)

    # Make sure node ID has the same datatype as before
    if x.nodes.node_id.dtype != nodeid_dtype:
        x.nodes['node_id'] = x.nodes.node_id.astype(nodeid_dtype, copy=False)

    # Finally: only reset non-graph related attributes
    if x.igraph and config.use_igraph:
        x._clear_temp_attr(exclude=['igraph', 'classify_nodes'])
    else:
        x._clear_temp_attr(exclude=['graph', 'classify_nodes'])

    return x


def cut_skeleton(x: 'core.NeuronObject',
                 where: Union[int, str, List[Union[int, str]]],
                 ret: Union[Literal['both'],
                            Literal['proximal'],
                            Literal['distal']] = 'both'
                 ) -> 'core.NeuronList':
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
               order of ``cut_node``. Fragments are ordered distal -> proximal.
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
    :func:`navis.TreeNeuron.prune_distal_to`
    :func:`navis.TreeNeuron.prune_proximal_to`
            ``TreeNeuron/List`` shorthands to this function.
    :func:`navis.subset_neuron`
            Returns a neuron consisting of a subset of its nodes.

    """
    utils.eval_param(ret, name='ret',
                     allowed_values=('proximal', 'distal', 'both'))

    if isinstance(x, core.NeuronList):
        if len(x) == 1:
            x = x[0]
        else:
            raise Exception(f'Expected a single TreeNeuron, got {len(x)}')

    if not isinstance(x, core.TreeNeuron):
        raise TypeError(f'Expected a single TreeNeuron, got "{type(x)}"')

    if x.n_trees != 1:
        raise ValueError(f'Unable to cut: neuron {x.id} consists of multiple '
                         'disconnected trees. Use navis.heal_skeleton()'
                         ' to fix.')

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
                raise ValueError(f'#{x.id}: Found no node with tag {cn}'
                                 ' - please double check!')
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
    if len(cn_ids) > 1 and ret != 'both':
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
        if x.igraph and config.use_igraph:
            cut = _cut_igraph(to_cut, cn, ret)
        else:
            cut = _cut_networkx(to_cut, cn, ret)

        # If ret != 'both', we will get only a single neuron - therefore
        # make sure cut is iterable
        cut = utils.make_iterable(cut)

        # Add results back to results at same index, proximal first
        for c in cut[::-1]:
            res.insert(to_cut_ix, c)

    return core.NeuronList(res)


def _cut_igraph(x: 'core.TreeNeuron',
                cut_node: int,
                ret: str) -> Union['core.TreeNeuron',
                                   Tuple['core.TreeNeuron',
                                         'core.TreeNeuron']]:
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
    g.to_undirected(combine_edges='first')

    # Get subgraph -> fastest way to get sets of nodes for subsetting
    a, b = g.decompose(mode='WEAK')
    # IMPORTANT: a,b are now UNDIRECTED graphs -> we must not keep using them!

    if x.root[0] in a.vs['node_id']:
        dist_graph, prox_graph = b, a
    else:
        dist_graph, prox_graph = a, b

    if ret == 'distal' or ret == 'both':
        dist = morpho.subset_neuron(x,
                                    subset=dist_graph.vs['node_id'],
                                    inplace=False)

        # Change new root for dist
        dist.nodes.loc[dist.nodes.node_id == cut_node, 'type'] = 'root'

        # Clear other temporary attributes
        dist._clear_temp_attr(exclude=['igraph', 'type', 'classify_nodes'])

    if ret == 'proximal' or ret == 'both':
        ss: Sequence[int] = prox_graph.vs['node_id'] + [cut_node]
        prox = morpho.subset_neuron(x,
                                    subset=ss,
                                    inplace=False)

        # Change new root for dist
        prox.nodes.loc[prox.nodes.node_id == cut_node, 'type'] = 'end'

        # Clear other temporary attributes
        prox._clear_temp_attr(exclude=['igraph', 'type', 'classify_nodes'])

    if ret == 'both':
        return dist, prox
    elif ret == 'distal':
        return dist
    else:  # elif ret == 'proximal':
        return prox


def _cut_networkx(x: 'core.TreeNeuron',
                  cut_node: Union[int, str],
                  ret: str) -> Union['core.TreeNeuron',
                                     Tuple['core.TreeNeuron',
                                           'core.TreeNeuron']]:
    """Use networkX graph to cut a neuron."""
    # Get subgraphs consisting of nodes distal to cut node
    dist_graph: nx.DiGraph = nx.bfs_tree(x.graph, cut_node, reverse=True)

    if ret == 'distal' or ret == 'both':
        # bfs_tree does not preserve 'weight'
        # -> need to subset original graph by those nodes
        dist_graph = x.graph.subgraph(dist_graph.nodes)

        # Generate new neurons
        # This is the actual bottleneck of the function: ~70% of time
        dist = morpho.subset_neuron(x,
                                    subset=dist_graph,
                                    inplace=False)  # type: ignore  # doesn't know nx.DiGraph

        # Change new root for dist
        dist.nodes.loc[dist.nodes.node_id == cut_node, 'parent_id'] = -1
        dist.nodes.loc[dist.nodes.node_id == cut_node, 'type'] = 'root'

        # Reassign graphs
        dist._graph_nx = dist_graph

        # Clear other temporary attributes
        dist._clear_temp_attr(exclude=['graph', 'type', 'classify_nodes'])

    if ret == 'proximal' or ret == 'both':
        # bfs_tree does not preserve 'weight'
        # need to subset original graph by those nodes
        ss_nodes = [n for n in x.graph.nodes if n not in dist_graph.nodes] + \
                   [cut_node]
        prox_graph: nx.DiGraph = x.graph.subgraph(ss_nodes)

        # Generate new neurons
        # This is the actual bottleneck of the function: ~70% of time
        prox = morpho.subset_neuron(x,
                                    subset=prox_graph,
                                    inplace=False)

        # Change cut node to end node for prox
        prox.nodes.loc[prox.nodes.node_id == cut_node, 'type'] = 'end'

        # Reassign graphs
        prox._graph_nx = prox_graph

        # Clear other temporary attributes
        prox._clear_temp_attr(exclude=['graph', 'type', 'classify_nodes'])

    # ATTENTION: prox/dist_graph contain pointers to the original graph
    # -> changes to attributes will propagate back

    if ret == 'both':
        return dist, prox
    elif ret == 'distal':
        return dist
    else:  # elif ret == 'proximal':
        return prox


def generate_list_of_childs(x: 'core.NeuronObject') -> Dict[int, List[int]]:
    """Return list of childs.

    Parameters
    ----------
    x :     TreeNeuron | NeuronList
            If List, must contain a SINGLE neuron.

    Returns
    -------
    dict
        ``{parent_id: [child_id, child_id, ...]}``

    """
    assert isinstance(x, core.TreeNeuron)
    # Grab graph once to avoid overhead from stale checks
    g = x.graph
    return {n: [e[0] for e in g.in_edges(n)] for n in g.nodes}


def node_label_sorting(x: 'core.TreeNeuron',
                       weighted: bool = False) -> List[Union[str, int]]:
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
        ``[root, node_id, node_id, ...]``

    """
    if isinstance(x, core.NeuronList) and len(x) == 1:
        x = x[0]

    if not isinstance(x, core.TreeNeuron):
        raise TypeError(f'Expected a singleTreeNeuron, got "{type(x)}"')

    if len(x.root) > 1:
        raise ValueError('Unable to process multi-root neurons!')

    # Get relevant terminal nodes
    term = x.nodes[x.nodes.type == 'end'].node_id.values

    # Get distance from terminals to all other nodes
    geo = geodesic_matrix(x, from_=term, directed=True, weight='weight' if weighted else None)
    # Set distance between unreachable points to None
    # Need to reinitialise SparseMatrix to replace float('inf') with NaN
    # dist_mat[geo == float('inf')] = None
    dist_mat = pd.DataFrame(np.where(geo == float('inf'),  # type: ignore  # no stubs for SparseDataFrame
                                     np.nan,
                                     geo),
                            columns=geo.columns,
                            index=geo.index)

    # Get starting points (i.e. branches off the root) and sort by longest
    # path to a terminal (note we're operating on the simplified version
    # of the skeleton)
    curr_points = sorted(list(x.simple.graph.predecessors(x.root[0])),
                         key=lambda n: dist_mat[n].max(),
                         reverse=True)

    # Walk from root towards terminals, prioritising longer branches
    nodes_walked = []
    while curr_points:
        nodes_walked.append(curr_points.pop(0))
        # If the current point is a terminal point, stop here
        if nodes_walked[-1] in term:
            pass
        else:
            new_points = sorted(list(x.simple.graph.predecessors(nodes_walked[-1])),
                                key=lambda n: dist_mat[n].max(),
                                reverse=True)
            curr_points = new_points + curr_points

    # Translate into segments
    node_list = [x.root[0]]
    # Note that we're inverting here so that the segments are ordered
    # proximal -> distal (i.e. root to tips)
    seg_dict = {s[0]: s[::-1] for s in _break_segments(x)}

    for n in nodes_walked:
        # Note that we're skipping the first (proximal) node to avoid double
        # counting nodes
        node_list += seg_dict[n][1:]

    return np.array(node_list)


def _igraph_to_sparse(graph, weight_attr=None):
    edges = graph.get_edgelist()
    if weight_attr is None:
        weights = [1] * len(edges)
    else:
        weights = graph.es[weight_attr]
    if not graph.is_directed():
        edges.extend([(v, u) for u, v in edges])
        weights.extend(weights)
    # Note: previously, we used a generator (weights, zip(*egdes)) as input to
    # csr_matrix but with Scipy 1.13.0 this has stopped working
    edges = np.array(edges)
    return csr_matrix((weights, (edges[:,0], edges[:,1])),
                       shape=(len(graph.vs), len(graph.vs)))


def connected_subgraph(x: Union['core.TreeNeuron', nx.DiGraph],
                       ss: Sequence[Union[str, int]]) -> Tuple[np.ndarray, Union[int, str]]:
    """Return set of nodes necessary to connect all nodes in subset ``ss``.

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
    if isinstance(x, core.NeuronList):
        if len(x) == 1:
            g = x[0].graph
    elif isinstance(x, core.TreeNeuron):
        g = x.graph
    elif isinstance(x, nx.DiGraph):
        g = x
    else:
        raise TypeError(f'Input must be a single TreeNeuron or graph, got "{type(x)}".')

    ss = set(ss)
    missing = ss - set(g.nodes)
    if np.any(missing):
        missing = np.array(list(missing)).astype(str)  # do NOT remove list() here!
        raise ValueError(f'Nodes not found: {",".join(missing)}')

    # Find nodes that are leafs WITHIN the subset
    g_ss = g.subgraph(ss)
    in_degree = dict(g_ss.in_degree)
    leafs = ss & {n for n, d in in_degree.items() if not d}

    # Run this for each connected component of the neuron
    include = set()
    new_roots = []
    for cc in nx.connected_components(g.to_undirected()):
        # Walk from each node to root and keep track of path
        paths = []
        for n in leafs & cc:
            this_path = []
            while n is not None:
                this_path.append(n)
                n = next(g.successors(n), None)
            paths.append(this_path)

        # If none of these cc in subset there won't be paths
        if not paths:
            continue

        # Find the nodes that all paths have in common
        common = set.intersection(*[set(p) for p in paths])

        # Now find the first (most distal from root) common node
        longest_path = sorted(paths, key=lambda x: len(x))[-1]
        first_common = sorted(common, key=lambda x: longest_path.index(x))[0]

        # Now go back to paths and collect all nodes until this first common node
        for p in paths:
            it = iter(p)
            n = next(it, None)
            while n is not None:
                if n in include:
                    break
                if n == first_common:
                    include.add(n)
                    break
                include.add(n)
                n = next(it, None)

        # In cases where there are even more distal common ancestors
        # (first common will typically be a branch point)
        this_ss = ss & cc
        if this_ss - include:
            # Make sure the new root is set correctly
            nr = sorted(this_ss - include,
                        key=lambda x: longest_path.index(x))[-1]
            new_roots.append(nr)
            # Add those nodes to be included
            include = set.union(include, this_ss)
        else:
            new_roots.append(first_common)

    return np.array(list(include)), new_roots


def insert_nodes(x: 'core.TreeNeuron',
                 where: List[tuple],
                 coords: List[tuple] = None,
                 validate: bool = True,
                 inplace: bool = False) -> Optional['core.TreeNeuron']:
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
                 - ``None``: new nodes will be inserted exactly between the two
                             nodes
                 - (N, 3) array of coordinates for the newly inserted nodes
                 - (N, ) array of fractional distances [0-1]: e.g. 0.25 means
                   that a new node will be inserted a quarter of the way between
                   the two nodes (from the child's perspective)
    validate :  bool
                If True, will make sure that pairs in ``where`` are always
                in (parent, child) order. If you know this to already be the
                case, set ``validate=False`` to save some time.
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
    utils.eval_param(x, name='x', allowed_types=(core.TreeNeuron, ))

    where = np.asarray(where)
    if where.ndim != 2 or where.shape[1] != 2:
        raise ValueError('Expected `where` to be a (N, 2) list of pairs. '
                         f'Got {where.shape}')

    # Validate if that's desired
    if validate:
        # Setup to get parents
        parent = x.nodes.set_index('node_id').parent_id

        # Get parents of the left and the right nodes of each pair
        parent_left = parent.loc[where[:, 0]].values
        parent_right = parent.loc[where[:, 1]].values

        # Check if the right node is parent of the left or the other way around
        correct_order = where[:, 0] == parent_right
        swapped = where[:, 1] == parent_left
        not_connected = ~(correct_order | swapped)

        if np.any(not_connected):
            raise ValueError('The following pairs are not connected: '
                             f'{where[not_connected]}')

        # Flip nodes where necessary to sure we have (parent, child) order
        if np.any(swapped):
            where[swapped, :] = where[swapped][:, [1, 0]]

    # If not provided, generate coordinates in the center between each node pair
    if isinstance(coords, type(None)):
        node_locs = x.nodes.set_index('node_id')[['x', 'y', 'z']]
        left_loc = node_locs.loc[where[:, 0]].values
        right_loc = node_locs.loc[where[:, 1]].values

        # Find center between each node
        coords = left_loc + (right_loc - left_loc) / 2

    coords = np.asarray(coords)
    # Make sure we have correct coordinates
    if coords.shape[0] != where.shape[0]:
        raise ValueError(f'Expected {where.shape[0]} coordinates or distances, '
                         f'got {coords.shape[0]}')

    # If array of fractional distances translate to coordinates
    if coords.ndim == 1:
        node_locs = x.nodes.set_index('node_id')[['x', 'y', 'z']]
        left_loc = node_locs.loc[where[:, 0]].values
        right_loc = node_locs.loc[where[:, 1]].values

        # Find center between each node
        coords = left_loc + (right_loc - left_loc) * coords.reshape(-1, 1)

    # For the moment, we will interpolate the radius
    rad = x.nodes.set_index('node_id').radius
    new_rad = (rad.loc[where[:, 0]].values + rad.loc[where[:, 1]].values) / 2

    # Generate table for new nodes
    new_nodes = pd.DataFrame()
    max_id = x.nodes.node_id.max() + 1
    new_nodes['node_id'] = np.arange(max_id, max_id + where.shape[0]).astype(int)
    new_nodes['parent_id'] = where[:, 0]
    new_nodes['x'] = coords[:, 0]
    new_nodes['y'] = coords[:, 1]
    new_nodes['z'] = coords[:, 2]
    new_nodes['radius'] = new_rad

    # Merge tables
    nodes = pd.concat([x.nodes, new_nodes],
                      join='outer', axis=0, sort=True, ignore_index=True)

    # Remap nodes
    new_parents = dict(zip(where[:, 1], new_nodes.node_id.values))
    to_rewire = nodes.node_id.isin(new_parents)
    nodes.loc[to_rewire, 'parent_id'] = nodes.loc[to_rewire, 'node_id'].map(new_parents)

    if not inplace:
        x = x.copy()

    x._nodes = nodes

    return x


def remove_nodes(x: 'core.TreeNeuron',
                 which: List[int],
                 inplace: bool = False) -> Optional['core.TreeNeuron']:
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

    """
    utils.eval_param(x, name='x', allowed_types=(core.TreeNeuron, ))

    if not utils.is_iterable(which):
        which = [which]
    which = np.asarray(which)

    miss = ~np.isin(which, x.nodes.node_id.values)
    if np.any(miss):
        raise ValueError(f'{len(miss)} node IDs not found in neuron')

    if not inplace:
        x = x.copy()

    # Generate new list of parents
    lop = dict(zip(x.nodes.node_id.values, x.nodes.parent_id.values))

    # Rewire to skip the to-be-removed nodes
    for n in which:
        lop.update({c: lop[n] for c, p in lop.items() if p == n})

    # Rewire neuron
    x.nodes['parent_id'] = x.nodes.node_id.map(lop)

    # Drop nodes
    x.nodes = x.nodes[~x.nodes.node_id.isin(which)].copy()

    # Clear temporary attributes
    x._clear_temp_attr()

    return x


def rewire_skeleton(x: 'core.TreeNeuron',
                    g: nx.Graph,
                    root: Optional[id] = None,
                    inplace: bool = False) -> Optional['core.TreeNeuron']:
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
    assert isinstance(x, core.TreeNeuron), f'Expected TreeNeuron, got {type(x)}'
    assert isinstance(g, nx.Graph), f'Expected networkx graph, got {type(g)}'

    if not inplace:
        x = x.copy()

    if g.is_directed():
        g = g.to_undirected()

    g = nx.minimum_spanning_tree(g, weight='weight')

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
    x.nodes['parent_id'] = x.nodes.node_id.map(lambda x: lop.get(x, -1))

    x._clear_temp_attr()

    return x
