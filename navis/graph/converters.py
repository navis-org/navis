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

import numpy as np
import networkx as nx
import pandas as pd
import scipy.spatial
import scipy.sparse

from typing import Union, Optional, List, Iterable

try:
    import igraph
except ImportError:
    igraph = None

from .. import config, core

# Set up logging
logger = config.get_logger(__name__)

__all__ = sorted(['network2nx', 'network2igraph', 'neuron2igraph', 'nx2neuron',
                  'neuron2nx', 'neuron2KDTree', 'neuron2tangents'])


def neuron2tangents(x: 'core.NeuronObject') -> 'core.Dotprops':
    """Turn TreeNeuron into an tangent vectors.

    This will drop zero-length vectors (i.e when node and parent occupy the
    exact same position).

    Parameters
    ----------
    x :         TreeNeuron | NeuronList

    Returns
    -------
    points :    (N, 3) array
                Midpoints for each child->parent node pair.
    vect :      (N, 3) array
                Normalized child-> parent vectors.
    length :    (N, ) array
                Distance between parent and child

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> t = navis.neuron2tangents(n)

    """
    if isinstance(x, core.NeuronList):
        return [neuron2tangents(n) for n in x]
    elif not isinstance(x, core.TreeNeuron):
        raise TypeError(f'Expected TreeNeuron/List, got "{type(x)}"')

    # Collect nodes
    nodes = x.nodes[x.nodes.parent_id >= 0]

    # Get child->parent vectors
    parent_locs = x.nodes.set_index('node_id').loc[nodes.parent_id,
                                                   ['x', 'y', 'z']].values
    child_locs = nodes[['x', 'y', 'z']].values
    vect = child_locs - parent_locs

    # Get mid point
    points = child_locs + (parent_locs - child_locs) / 2

    # Get length
    length = np.sqrt(np.sum(vect ** 2, axis=1))

    # Drop zero length points
    points = points[length != 0]
    vect = vect[length != 0]
    length = length[length != 0]

    # Normalize vector
    vect = vect / np.linalg.norm(vect, axis=1).reshape(-1, 1)

    return points, vect, length


def network2nx(x: Union[pd.DataFrame, Iterable],
               threshold: Optional[float] = None,
               group_by: Union[dict, None] = None) -> nx.DiGraph:
    """Generate NetworkX graph from edge list or adjacency.

    Parameters
    ----------
    x :                 pandas.DataFrame
                        Connectivity information:

                         1. List of edges (columns: 'source', 'target', 'weight')
                         2. Adjacency matrix (pd.DataFrame, rows=sources,
                            columns=targets)

    threshold :         float | int, optional
                        Connections weaker than this will be excluded.
    group_by :          None | dict, optional
                        Provide a dictionary ``{group_name: [skid1, skid2, ...]}``
                        to collapse sets of nodes into groups.

    Returns
    -------
    networkx.DiGraph
                        NetworkX representation of the network.

    """
    if isinstance(x, pd.DataFrame):
        present = [c in x.columns for c in ['source', 'target', 'weight']]
        if all(present):
            edges = x[['source', 'target', 'weight']].values
        else:
            # Assume it's an adjacency matrix
            ix_name = x.index.name if x.index.name else 'index'
            edges = x.reset_index(inplace=False,
                                  drop=False).melt(id_vars=ix_name).values
    elif isinstance(x, (list, np.ndarray)):
        edges = np.array(x)
    else:
        raise TypeError(f'Expected numpy array or pandas DataFrame, got "{type(x)}"')

    if edges.ndim != 2 or edges.shape[1] != 3:
        raise ValueError('Edges must be (N, 3) array containing source, '
                         'target, weight')

    if not isinstance(threshold, (type(None), bool)):
        edges = edges[edges[:, 2] >= threshold]

    # Generate graph and assign custom properties
    g = nx.DiGraph()
    g.add_weighted_edges_from(edges)

    # Group nodes
    if group_by:
        for n, skids in group_by.items():
            # First collapse all nodes into the first of each group
            for s in skids[1:]:
                g = nx.contracted_nodes(g, str(skids[0]), str(s))
            # Now relabel the first node
            g = nx.relabel_nodes(g, {str(skids[0]): str(n)})
            g.nodes[str(n)]['neuron_name'] = str(n)

    return g


def network2igraph(x: Union[pd.DataFrame, Iterable],
                   threshold: Optional[float] = None) -> 'igraph.Graph':
    """Generate iGraph graph from edge list or adjacency.

    Requires iGraph to be installed.

    Parameters
    ----------
    x :                 pandas.DataFrame | np.array
                        Connectivity information:

                         1. List of edges (columns: 'source', 'target', 'weight')
                         2. Adjacency matrix (pd.DataFrame, rows=sources,
                            columns=targets)

    threshold :         float | int, optional
                        Connections weaker than this will be excluded.

    Returns
    -------
    igraph.Graph(directed=True)
                        iGraph representation of the network.

    """
    if igraph is None:
        raise ImportError('igraph must be installed to use this function.')

    if isinstance(x, pd.DataFrame):
        present = [c in x.columns for c in ['source', 'target', 'weight']]
        if all(present):
            edges = x[['source', 'target', 'weight']].values
        else:
            edges = x.reset_index(inplace=False,
                                  drop=False).melt(id_vars='index',
                                                   inplace=False).values
    elif isinstance(x, (list, np.ndarray)):
        edges = np.array(x)
    else:
        raise TypeError(f'Expected numpy array or pandas DataFrame, got "{type(x)}"')

    if edges.ndim != 2 or edges.shape[1] != 3:
        raise ValueError('Edges must be (N, 3) array containing source, '
                         'target, weight')

    if not isinstance(threshold, (type(None), bool)):
        edges = edges[edges[:, 2] >= threshold]

    names = list(set(np.array(edges)[:, 0]) | set(np.array(edges)[:, 1]))

    edges_by_index = [[names.index(e[0]), names.index(e[1])] for e in edges]

    # Generate igraph and assign custom properties
    g = igraph.Graph(directed=True)
    g.add_vertices(len(names))
    g.add_edges(edges_by_index)

    g.vs['node_id'] = names
    # g.vs['neuron_name'] = g.vs['label'] = neuron_names
    g.es['weight'] = edges[:, 2]

    return g


def neuron2nx(x: 'core.NeuronObject') -> nx.DiGraph:
    """Turn Tree-, Mesh- or VoxelNeuron into an NetworkX graph.

    Parameters
    ----------
    x :         TreeNeuron | MeshNeuron | VoxelNeuron | NeuronList
                Uses simple 6-connectedness for voxels.

    Returns
    -------
    graph:      networkx.Graph | networkx.DiGraph
                NetworkX representation of the neuron. Returns list of graphs
                if x is multiple neurons. Graph is directed for TreeNeurons
                and undirected for Mesh- and VoxelNeurons. Graph is weighted
                for Tree- and MeshNeurons.

    """
    if isinstance(x, core.NeuronList):
        return [neuron2nx(x.loc[i]) for i in range(x.shape[0])]

    if isinstance(x, core.TreeNeuron):
        # Collect nodes
        nodes = x.nodes.set_index('node_id', inplace=False)
        # Collect edges
        edges = x.nodes[x.nodes.parent_id >= 0][['node_id', 'parent_id']].values
        # Collect weight
        weights = np.sqrt(np.sum((nodes.loc[edges[:, 0], ['x', 'y', 'z']].values.astype(float)
                                  - nodes.loc[edges[:, 1], ['x', 'y', 'z']].values.astype(float)) ** 2, axis=1))
        # Generate weight dictionary
        edge_dict = np.array([{'weight': w} for w in weights])
        # Add weights to dictionary
        edges = np.append(edges, edge_dict.reshape(len(edges), 1), axis=1)
        # Create empty directed Graph
        G = nx.DiGraph()
        # Add nodes (in case we have disconnected nodes)
        G.add_nodes_from(x.nodes.node_id.values)
        # Add edges
        G.add_edges_from(edges)
    elif isinstance(x, core.MeshNeuron):
        G = nx.Graph()
        G.add_nodes_from(np.arange(x.n_vertices))
        edges = [(e[0], e[1], l) for e, l in zip(x.trimesh.edges_unique,
                                                 x.trimesh.edges_unique_length)]
        G.add_weighted_edges_from(edges)
    elif isinstance(x, core.VoxelNeuron):
        # First we need to determine the 6-connecivity between voxels
        edges = []
        # Go over each axis
        for i in range(3):
            # Generate an offset of 1 voxel along given axis
            offset = np.zeros(3, dtype=int)
            offset[i] = 1
            # Combine real and offset voxels
            vox_off = x.voxels + offset
            # Find out which voxels overlap (i.e. count == 2 after offset)
            unique, cnt = np.unique(np.append(x.voxels, vox_off, axis=0),
                                    axis=0, return_counts=True)

            connected = unique[cnt > 1]
            for vox in connected:
                edges.append([tuple(vox), tuple(vox - offset)])
        G = nx.Graph()
        G.add_nodes_from([tuple(v) for v in x.voxels])
        G.add_edges_from(edges)
    else:
        raise ValueError(f'Unable to convert data of type "{type(x)}" to networkx graph.')

    return G


def _voxels2edges(x, connectivity=18):
    """Turn VoxelNeuron into an edges.

    This is function requires scikit-learn to be available.

    Parameters
    ----------
    x :             VoxelNeuron
    connectivity :  6 | 18 | 26
                    Connectedness:
                     - 6 = faces
                     - 18 = faces + edges
                     - 26 = faces + edges + vertices

    Returns
    -------
    edges :         (N, 2) numpy array

    """
    # The distances and metric we will use depend on the connectedness
    METRICS = {6: 'manhattan',
               18: 'euclidean',
               26: 'chebyshev'}
    DISTANCES = {6: 1,
                 18: 1.5,
                 26: 1}

    try:
        from sklearn.neighbors import KDTree
    except ImportError:
        raise ImportError('This function requires scikit-learn to be installed.')

    assert connectivity in (6, 18, 26), f'`connectivity` must be 6, 18 or 26, not "{connectivity}"'
    assert isinstance(x, core.VoxelNeuron)

    voxels = x.voxels
    # Create tree with given distance metric
    tree = KDTree(voxels, leaf_size=40, metric=METRICS[connectivity])

    # Query ball pairs
    indices = tree.query_radius(voxels, r=DISTANCES[connectivity])

    # Collected edges
    edges = []
    for i, hits in enumerate(indices):
        # Add edges
        edges += [(i, ix) for ix in hits]
    edges = np.array(edges)

    # Drop self-hits
    edges = edges[edges[:, 0] != edges[:, 1]]

    # Keep only A->B edges and drop B->A edges
    edges = np.unique(np.sort(edges, axis=1), axis=0)

    return edges


def neuron2igraph(x: 'core.NeuronObject',
                  connectivity=18,
                  raise_not_installed: bool = True) -> 'igraph.Graph':
    """Turn Tree-, Mesh- or VoxelNeuron(s) into an iGraph graph.

    Requires iGraph to be installed.

    Parameters
    ----------
    x :                     TreeNeuron | MeshNeuron | VoxelNeuron | NeuronList
                            Neuron(s) to convert.
    connectivity :          6 | 18 | 26
                            Connectedness for VoxelNeurons:
                             - 6 = faces
                             - 18 = faces + edges
                             - 26 = faces + edges + vertices
    raise_not_installed :   bool
                            If False and igraph is not installed will silently
                            return ``None``.

    Returns
    -------
    igraph.Graph
                Representation of the neuron. Returns list of graphs
                if x is multiple neurons. Directed for TreeNeurons, undirected
                for MeshNeurons.
    None
                If igraph not installed.

    """
    # If iGraph is not installed return nothing
    if igraph is None:
        if not raise_not_installed:
            return None
        else:
            raise ImportError('iGraph appears to not be installed (properly). '
                              'Make sure "import igraph" works.')

    if isinstance(x, core.NeuronList):
        return [neuron2igraph(x.loc[i],
                              connectivity=connectivity) for i in range(x.shape[0])]

    if isinstance(x, core.TreeNeuron):
        # Make sure we have correctly numbered indices
        nodes = x.nodes.reset_index(inplace=False, drop=True)

        # Generate list of vertices -> this order is retained
        vlist = nodes.node_id.values

        # Get list of edges as indices (needs to exclude root node)
        tn_index_with_parent = nodes.index.values[nodes.parent_id >= 0]
        parent_ids = nodes.parent_id.values[nodes.parent_id >= 0]
        nodes['temp_index'] = nodes.index  # add temporary index column
        try:
            parent_index = nodes.set_index('node_id', inplace=False).loc[parent_ids,
                                                                         'temp_index'].values
        except KeyError:
            miss = nodes[~nodes.parent_id.isin(nodes.node_id)].node_id.unique()
            raise KeyError(f"{len(miss)} nodes (e.g. {miss[0]}) in TreeNeuron "
                           f"{x.id} connect to non-existent parent nodes.")
        except BaseException:
            raise

        # Generate list of edges based on index of vertices
        elist = np.vstack((tn_index_with_parent, parent_index)).T

        # iGraph < 0.8.0 does not like arrays as edge list
        if getattr(igraph, '__version_info__', (0, 0, 0))[1] < 8:
            elist = elist.tolist()

        # Generate graph and assign custom properties
        G = igraph.Graph(elist, n=len(vlist), directed=True)

        G.vs['node_id'] = G.vs['name'] = nodes.node_id.values
        G.vs['parent_id'] = nodes.parent_id.values

        # Generate weights by calculating edge lengths = distance between nodes
        tn_coords = nodes[['x', 'y', 'z']].values[tn_index_with_parent, :]
        parent_coords = nodes[['x', 'y', 'z']].values[parent_index.astype(int), :]

        w = np.sqrt(np.sum((tn_coords - parent_coords) ** 2, axis=1))
        G.es['weight'] = w
    elif isinstance(x, core.MeshNeuron):
        elist = x.trimesh.edges_unique
        G = igraph.Graph(elist, n=x.n_vertices, directed=False)
        G.es['weight'] = x.trimesh.edges_unique_length
    elif isinstance(x, core.VoxelNeuron):
        edges = _voxels2edges(x, connectivity=connectivity)
        G = igraph.Graph(edges, n=len(x.voxels), directed=False)
    else:
        raise ValueError(f'Unable to convert data of type "{type(x)}" to igraph.')

    return G


def nx2neuron(g: nx.Graph,
              root: Optional[Union[int, str]] = None,
              break_cycles: bool = False,
              **kwargs
              ) -> pd.DataFrame:
    """Generate node table from NetworkX Graph.

    This function will try to generate a neuron-like tree structure from
    the Graph. Therefore the graph must not contain loops!

    Node attributes (e.g. ``x``, ``y``, ``z``, ``radius``) need
    to be properties of the graph's nodes. All node property will be added to
    the neuron's ``.nodes`` table.

    Parameters
    ----------
    g :             networkx.Graph
    root :          str | int | list, optional
                    Node in graph to use as root for neuron. If not provided,
                    will use first node in ``g.nodes``. Ignored if graph
                    consists of several disconnected components.
    break_cycles :  bool
                    The input graph must not contain cycles. We can break them
                    up at risk of disconnecting parts of the graph.
    **kwargs
                    Keyword arguments are passed to the construction of
                    :class:`~navis.TreeNeuron`.

    Returns
    -------
    TreeNeuron

    """
    # First some sanity checks
    if not isinstance(g, nx.Graph):
        raise TypeError(f'`g` must be NetworkX Graph, got "{type(g)}"')

    # We need an undirected Graph
    if isinstance(g, nx.DiGraph):
        g = g.to_undirected(as_view=True)

    if not nx.is_forest(g):
        if not break_cycles:
            raise TypeError("Graph must be tree-like. You can try setting "
                            "the `cut_cycles` parameter to True.")
        else:
            if break_cycles:
                while True:
                    try:
                        # Find cycle
                        cycle = nx.find_cycle(g)
                    except nx.exception.NetworkXNoCycle:
                        break
                    except BaseException:
                        raise

                    # Sort by degree
                    cycle = sorted(cycle, key=lambda x: g.degree[x[0]])

                    # Remove the edge with the lowest degree
                    g.remove_edge(cycle[0][0], cycle[0][1])

    # Ignore root if this is a forest
    if not nx.is_tree(g):
        root = None

    # This effectively makes sure that all edges point in the same direction
    lop = {}
    for c in nx.connected_components(g):
        sg = nx.subgraph(g, c)
        # Pick a random root if not explicitly provided
        if not root:
            r = list(sg.nodes)[0]
        elif root not in sg.nodes:
            raise ValueError(f'Node "{root}" not in graph.')
        else:
            r = root

        # Generate parent->child dictionary
        this_lop = nx.predecessor(sg, r)

        # Make sure no node has more than one parent
        if max([len(v) for v in this_lop.values()]) > 1:
            raise ValueError('Nodes with multiple parents found. Make sure graph '
                             'is tree-like.')

        # Note that we assign -1 as root's parent
        lop.update({k: v[0] if v else -1 for k, v in this_lop.items()})

    # Generate node table
    tn_table = pd.DataFrame(index=list(g.nodes))
    tn_table.index = tn_table.index.set_names('node_id', inplace=False)

    # Add parents - use -1 for root's parent
    tn_table['parent_id'] = tn_table.index.map(lop)

    try:
        tn_table.index = tn_table.index.astype(int)
        tn_table['parent_id'] = tn_table.parent_id.astype(int)
    except (ValueError, TypeError):
        raise ValueError('Node IDs must be convertible to integers.')
    except BaseException:
        raise

    # Add additional generic attribute -> will skip node_id and parent_id
    # if they exist
    all_attr = set([k for n in g.nodes for k in g.nodes[n].keys()])

    # Remove some that we don't need
    all_attr -= set(['parent_id', 'node_id'])
    # Add some that we want as columns even if they don't exist
    all_attr |= set(['x', 'y', 'z', 'radius'])

    # For some we want to have set default values
    defaults = {'x': 0, 'y': 0, 'z': 0, 'radius': -1}

    # Now map the attributes onto node table
    for at in all_attr:
        vals = nx.get_node_attributes(g, at)
        tn_table[at] = tn_table.index.map(lambda a: vals.get(a, defaults.get(at)))

    return core.TreeNeuron(tn_table.reset_index(drop=False, inplace=False),
                           **kwargs)


def _find_all_paths(g: nx.DiGraph,
                    start,
                    end,
                    mode: str = 'OUT',
                    maxlen: Optional[int] = None) -> list:
    """Find all paths between two vertices in an iGraph object.

    For some reason this function exists in R iGraph but not Python iGraph. This
    is rather slow and should not be used for large graphs.

    """
    def find_all_paths_aux(adjlist: List[set],
                           start: int,
                           end: int,
                           path: list,
                           maxlen: Optional[int] = None) -> list:
        path = path + [start]
        if start == end:
            return [path]
        paths: list = []
        if maxlen is None or len(path) <= maxlen:
            for node in adjlist[start] - set(path):
                paths.extend(find_all_paths_aux(adjlist,
                                                node,
                                                end,
                                                path,
                                                maxlen))
        return paths

    adjlist = [set(g.neighbors(node, mode=mode))
               for node in range(g.vcount())]
    all_paths: list = []
    start = start if isinstance(start, list) else [start]
    end = end if isinstance(end, list) else [end]
    for s in start:
        for e in end:
            all_paths.extend(find_all_paths_aux(adjlist, s, e, [], maxlen))
    return all_paths


def neuron2KDTree(x: 'core.NeuronObject',
                  tree_type: str = 'c',
                  data: str = 'auto',
                  **kwargs) -> Union[scipy.spatial.cKDTree,
                                     scipy.spatial.KDTree]:
    """Turn neuron into scipy KDTree.

    Parameters
    ----------
    x :         TreeNeuron | MeshNeuron | VoxelNeuron | Dotprops
                A single neuron to turn into a KDTree.
    tree_type : 'c' | 'normal'
                Type of KDTree:
                  1. ``'c'`` = ``scipy.spatial.cKDTree`` (faster)
                  2. ``'normal'`` = ``scipy.spatial.KDTree`` (more functions)
    data :      'auto' | str
                Data used to generate tree. "auto" will pick the core data
                depending on neuron type: `nodes`, `vertices`, `voxels` and
                `points` for TreeNeuron, MeshNeuron, VoxelNeuron and Dotprops,
                respectively. Other values (e.g. "connectors" or "nodes") must
                map to a neuron property that is either (N, 3) array or
                DataFrame with x/y/z columns.
    **kwargs
                Keyword arguments passed at KDTree initialization.


    Returns
    -------
    ``scipy.spatial.cKDTree`` or ``scipy.spatial.KDTree``

    """
    if tree_type not in ['c', 'normal']:
        raise ValueError('"tree_type" needs to be either "c" or "normal"')

    if isinstance(x, core.NeuronList):
        if len(x) == 1:
            x = x[0]
        else:
            raise ValueError('Need a single TreeNeuron')
    elif not isinstance(x, core.BaseNeuron):
        raise TypeError(f'Need Neuron, got "{type(x)}"')

    if data == 'auto':
        if isinstance(x, core.TreeNeuron):
            data = 'nodes'
        if isinstance(x, core.MeshNeuron):
            data = 'vertices'
        if isinstance(x, core.VoxelNeuron):
            data = 'voxels'
        if isinstance(x, core.Dotprops):
            data = 'points'

    if not hasattr(x, data):
        raise ValueError(f'Neuron does not have a {data} property')

    data = getattr(x, data)

    if isinstance(data, pd.DataFrame):
        if not all(np.isin(['x', 'y', 'z'], data.columns)):
            raise ValueError(f'"{data}" DataFrame must contain "x", "y" and '
                             '"z" columns.')
        data = data[['x', 'y', 'z']].values

    if not isinstance(data, np.ndarray) or data.ndim != 2 or data.shape[1] != 3:
        raise ValueError(f'"{data}" must be DataFrame or (N, 3) array, got {type(data)}')

    if tree_type == 'c':
        return scipy.spatial.cKDTree(data=data, **kwargs)
    else:
        return scipy.spatial.KDTree(data=data, **kwargs)
