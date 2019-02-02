#    This script is part of navis (http://www.github.com/schlegelp/navis).
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

try:
    import igraph
except ImportError:
    igraph = None

from .. import core, utils, config

# Set up logging
logger = config.logger

__all__ = sorted(['network2nx', 'network2igraph', 'neuron2igraph', 'nx2neuron',
                  'neuron2nx', 'neuron2KDTree', 'neuron2dps'])


def network2nx(x, remote_instance=None, threshold=1, group_by=None):
    """ Generates NetworkX graph for neuron connectivity.

    Parameters
    ----------
    x
                        Catmaid Neurons as:
                         1. list of skeleton IDs (int or str)
                         2. list of neuron names (str, exact match)
                         3. annotation(s): e.g. 'annotation:PN right'
                         4. NeuronList object
                         5. Adjacency matrix (pd.DataFrame, rows=sources,
                            columns=targets)
    remote_instance :   CATMAID instance, optional
                        Either pass directly to function or define globally
                        as ``remote_instance``.
    threshold :         int, optional
                        Connections weaker than this will be excluded.
    group_by :          None | dict, optional
                        Provide a dictionary ``{group_name: [skid1, skid2, ...]}``
                        to collapse sets of nodes into groups.

    Returns
    -------
    networkx.DiGraph
                        NetworkX representation of the network.

    """

    if isinstance(x, (core.NeuronList, list, np.ndarray, str)):
        remote_instance = utils._eval_remote_instance(remote_instance)
        skids = utils.eval_skids(x, remote_instance=remote_instance)

        # Fetch edges
        edges = fetch.get_edges(skids, remote_instance=remote_instance)
        # Reformat into networkx format
        edges = [[str(e.source_skid), str(e.target_skid), {'weight': e.weight}]
                 for e in edges[edges.weight >= threshold].itertuples()]
    elif isinstance(x, pd.DataFrame):
        # We have to account for the fact that some might not be skids
        skids = []
        for s in list(set(x.columns.tolist() + x.index.tolist())):
            try:
                skids.append(int(s))
            except BaseException:
                pass
        # Generate edge list
        edges = [[str(s), str(t), {'weight': x.loc[s, t]}]
                 for s in x.index.values for t in x.columns.values if x.loc[s, t] >= threshold]
    else:
        raise ValueError(
            'Unable to process data of type "{0}"'.format(type(x)))

    # Generate node dictionary
    names = fetch.get_names(skids, remote_instance=remote_instance)
    nodes = [[str(s), {'neuron_name': names.get(s, s)}] for s in skids]

    # Generate graph and assign custom properties
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)

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


def network2igraph(x, remote_instance=None, threshold=1):
    """ Generates iGraph graph for neuron connectivity.

    Requires iGraph to be installed.

    Parameters
    ----------
    x
                        Catmaid Neurons as:
                         1. list of skeleton IDs (int or str)
                         2. list of neuron names (str, exact match)
                         3. annotation(s): e.g. 'annotation:PN right'
                         4. NeuronList object
                         5. Adjacency matrix (pd.DataFrame, rows=sources,
                            columns=targets)
    remote_instance :   CATMAID instance, optional
                        Either pass directly to function or define globally
                        as 'remote_instance'.
    threshold :         int, optional
                        Connections weaker than this will be excluded .

    Returns
    -------
    igraph.Graph(directed=True)
                        NetworkX representation of the network.

    Examples
    --------
    >>> import navis
    >>> import igraph
    >>> g = navis.network2igraph('annotation:large network',
    ...                           remote_instance=rm)
    >>> # Plot graph
    >>> igraph.plot(g)
    >>> # Plot with edge width
    >>> igraph.plot(g, **{'edge_width': [ w/10 for w in g.es['weight'] ] })
    >>> # Plot with edge label
    >>> igraph.plot(g, **{'edge_label': g.es['weight'] })
    >>> # Save as graphml to import into e.g. Cytoscape
    >>> g.save('graph.graphml')

    """
    if igraph is None:
        raise ImportError('igraph must be installed to use this function.')

    if isinstance(x, (core.NeuronList, list, np.ndarray, str)):
        remote_instance = utils._eval_remote_instance(remote_instance)
        skids = utils.eval_skids(x, remote_instance=remote_instance)

        indices = {int(s): i for i, s in enumerate(skids)}

        # Fetch edges
        edges = fetch.get_edges(skids, remote_instance=remote_instance)

        # Reformat into igraph format
        edges_by_index = [[indices[e.source_skid], indices[e.target_skid]]
                          for e in edges[edges.weight >= threshold].itertuples()]
        weight = edges[edges.weight >= threshold].weight.tolist()
    elif isinstance(x, pd.DataFrame):
        skids = list(set(x.columns.tolist() + x.index.tolist()))
        # Generate edge list
        edges = [[i, j] for i in x.index.tolist()
                 for j in x.columns.tolist() if x.loc[i, j] >= threshold]
        edges_by_index = [
            [skids.index(e[0]), skids.index(e[1])] for e in edges]
        weight = [x.loc[i, j] for i in range(x.shape[0]) for j in range(
            x.shape[1]) if x.loc[i, j] >= threshold]
    else:
        raise ValueError(
            'Unable to process data of type "{0}"'.format(type(x)))

    # Generate igraph and assign custom properties
    g = igraph.Graph(directed=True)
    g.add_vertices(len(skids))
    g.add_edges(edges_by_index)

    g.vs['node_id'] = skids
    # g.vs['neuron_name'] = g.vs['label'] = neuron_names
    g.es['weight'] = weight

    return g


def neuron2nx(x):
    """ Turn TreeNeuron into an NetworkX DiGraph.

    Parameters
    ----------
    x :         TreeNeuron | NeuronList

    Returns
    -------
    networkx.DiGraph
                NetworkX representation of the neuron. Returns list of graphs
                if x is multiple neurons.

    """

    if isinstance(x, (pd.DataFrame, core.NeuronList)):
        return [neuron2nx(x.loc[i]) for i in range(x.shape[0])]
    elif isinstance(x, (pd.Series, core.TreeNeuron)):
        pass
    else:
        raise ValueError('Unable input type "{0}"'.format(type(x)))

    # Collect nodes
    nodes = x.nodes.set_index('node_id')
    # Collect edges
    edges = x.nodes[~x.nodes.parent_id.isnull()][['node_id',
                                                  'parent_id']].values
    # Collect weight
    weights = np.sqrt(np.sum((nodes.loc[edges[:, 0], ['x', 'y', 'z']].values.astype(float)
                              - nodes.loc[edges[:, 1], ['x', 'y', 'z']].values.astype(float)) ** 2, axis=1))
    # Generate weight dictionary
    edge_dict = np.array([{'weight': w} for w in weights])
    # Add weights to dictionary
    edges = np.append(edges, edge_dict.reshape(len(edges), 1), axis=1)
    # Create empty directed Graph
    g = nx.DiGraph()
    # Add nodes (in case we have disconnected nodes)
    g.add_nodes_from(x.nodes.node_id.values)
    # Add edges
    g.add_edges_from(edges)

    return g


def neuron2igraph(x):
    """ Turns TreeNeuron(s) into an iGraph graph.

    Requires iGraph to be installed.

    Parameters
    ----------
    x :         TreeNeuron | NeuronList

    Returns
    -------
    igraph.Graph(directed=True)
                Representation of the neuron. Returns list of graphs
                if x is multiple neurons.
    None
                If igraph not installed.

    """
    # If iGraph is not installed return nothing
    if igraph is None:
        return None

    if isinstance(x, (pd.DataFrame, core.NeuronList)):
        return [neuron2igraph(x.loc[i]) for i in range(x.shape[0])]
    elif isinstance(x, (pd.Series, core.TreeNeuron)):
        pass
    else:
        raise ValueError('Unable input type "{0}"'.format(type(x)))

    logger.debug('Generating graph from skeleton data...')

    # Make sure we have correctly numbered indices
    nodes = x.nodes.reset_index(drop=True)

    # Generate list of vertices -> this order is retained
    vlist = nodes.node_id.values

    # Get list of edges as indices (needs to exclude root node)
    tn_index_with_parent = nodes.loc[
        ~nodes.parent_id.isnull()].index.values
    parent_ids = nodes.loc[~nodes.parent_id.isnull()].parent_id.values
    nodes['temp_index'] = nodes.index  # add temporary index column
    parent_index = nodes.set_index('node_id').loc[parent_ids,
                                                      'temp_index'].values

    # Generate list of edges based on index of vertices
    elist = list(zip(tn_index_with_parent, parent_index.astype(int)))

    # Generate graph and assign custom properties
    g = igraph.Graph(elist, n=len(vlist), directed=True)

    g.vs['node_id'] = g.vs['name'] = nodes.node_id.values
    g.vs['parent_id'] = nodes.parent_id.values

    # Generate weights by calculating edge lengths = distance between nodes
    tn_coords = nodes.loc[[e[0] for e in elist], ['x', 'y', 'z']].values
    parent_coords = nodes.loc[[e[1] for e in elist], ['x', 'y', 'z']].values

    w = np.sqrt(np.sum((tn_coords - parent_coords) ** 2, axis=1).astype(int))
    g.es['weight'] = w

    return g


def nx2neuron(g, root=None):
    """ Generate treenode table from NetworkX Graph.

    This function will try to generate a neuron-like tree structure from
    the Graph. Therefore the graph may not contain loops!

    Treenode attributes (``x``, ``y``, ``z``, ``radius``, ``confidence``) need
    to be properties of the graph's nodes.

    Parameters
    ----------
    g :             networkx.Graph
    root :          str | int, optional
                    Node in graph to use as root for neuron. If not provided,
                    will use first node in ``g.nodes``.

    Returns
    -------
    pandas.DataFrame
    """

    # First some sanity checks
    if not isinstance(g, nx.Graph):
        raise TypeError('g must be NetworkX Graph, not "{}"'.format(type(g)))

    # We need an undirected Graph
    if isinstance(g, nx.DiGraph):
        g = g.to_undirected(as_view=True)

    if not nx.is_tree(g):
        raise TypeError("g must be tree-like. Please check for loops.")

    # Pick a randon root if not explicitly provided
    if not root:
        root = list(g.nodes)[0]
    elif root not in g.nodes:
        raise ValueError('Node "{}" not in graph.'.format(root))

    # Generate parent->child dictionary
    lop = nx.predecessor(g, root)

    # Make sure no node has more than one parent
    if max([len(v) for v in lop.values()]) > 1:
        raise ValueError('Nodes with multiple parents found. Make sure graph '
                         'is tree-like.')

    lop = {k: v[0] if v else None for k, v in lop.items()}

    # Generate treenode table
    tn_table = pd.DataFrame(index=list(g.nodes))
    tn_table.index = tn_table.index.set_names('node_id')

    # Add parents
    tn_table['parent_id'] = tn_table.index.map(lop)

    # Add coordinates
    x = nx.get_node_attributes(g, 'x')
    tn_table['x'] = tn_table.index.map(lambda a: x.get(a, 0))
    y = nx.get_node_attributes(g, 'y')
    tn_table['y'] = tn_table.index.map(lambda a: y.get(a, 0))
    z = nx.get_node_attributes(g, 'z')
    tn_table['z'] = tn_table.index.map(lambda a: z.get(a, 0))

    radii = nx.get_node_attributes(g, 'radius')
    tn_table['radius'] = tn_table.index.map(lambda x: radii.get(x, None))

    return tn_table.reset_index()


def _find_all_paths(g, start, end, mode='OUT', maxlen=None):
    """ Find all paths between two vertices in an iGraph object. For some reason
    this function exists in R iGraph but not Python iGraph. This is rather slow
    and should not be used for large graphs.
    """

    def find_all_paths_aux(adjlist, start, end, path, maxlen=None):
        path = path + [start]
        if start == end:
            return [path]
        paths = []
        if maxlen is None or len(path) <= maxlen:
            for node in adjlist[start] - set(path):
                paths.extend(find_all_paths_aux(
                    adjlist, node, end, path, maxlen))
        return paths

    adjlist = [set(g.neighbors(node, mode=mode))
               for node in range(g.vcount())]
    all_paths = []
    start = start if type(start) is list else [start]
    end = end if type(end) is list else [end]
    for s in start:
        for e in end:
            all_paths.extend(find_all_paths_aux(adjlist, s, e, [], maxlen))
    return all_paths


def neuron2KDTree(x, tree_type='c', data='treenodes', **kwargs):
    """ Turns a neuron into scipy KDTree.

    Parameters
    ----------
    x :         single TreeNeuron
    tree_type : 'c' | 'normal', optional
                Type of KDTree:
                  1. ``'c'`` = ``scipy.spatial.cKDTree`` (faster)
                  2. ``'normal'`` = ``scipy.spatial.KDTree`` (more functions)
    data :      'treenodes' | 'connectors', optional
                Data to use to generate tree.
    **kwargs
                Keyword arguments passed at KDTree initialization.


    Returns
    -------
    ``scipy.spatial.cKDTree`` or ``scipy.spatial.KDTree``

    """

    # Rarely used, so import in function
    import scipy.spatial

    if tree_type not in ['c', 'normal']:
        raise ValueError('"tree_type" needs to be either "c" or "normal"')

    if data not in ['treenodes', 'connectors']:
        raise ValueError(
            '"data" needs to be either "treenodes" or "connectors"')

    if isinstance(x, core.NeuronList):
        if len(x) == 1:
            x = x[0]
        else:
            raise ValueError('Need a single TreeNeuron')
    elif not isinstance(x, core.TreeNeuron):
        raise TypeError('Need TreeNeuron, got "{0}"'.format(type(x)))

    if data == 'treenodes':
        d = x.nodes[['x', 'y', 'z']].values
    else:
        d = x.connectors[['x', 'y', 'z']].values

    if tree_type == 'c':
        return scipy.spatial.cKDTree(data=d, **kwargs)
    else:
        return scipy.spatial.KDTree(data=d, **kwargs)


def neuron2dps(x):
    """ Converts a neuron's neurites into dotproducts.

    Dotproducts consist of a point and a vector. This works by (1) finding the
    center between child->parent treenodes and (2) getting the vector between
    them. Also returns the length of the vector.

    Parameters
    ----------
    x :         TreeNeuron
                Single neuron

    Returns
    -------
    pandas.DataFrame
            DataFrame in which each row represents a segments between two
            treenodes::

                point  vector  vec_length
             1
             2
             3

    Examples
    --------
    >>> x = navis.example_neurons()
    >>> dps = navis.to_dotproduct(x)
    >>> # Get array of all locations
    >>> locs = numpy.vstack(dps.point.values)

    See Also
    --------
    navis.TreeNeuron.dps

    """

    if isinstance(x, core.NeuronList):
        if x.shape[0] == 1:
            x = x[0]
        else:
            raise ValueError('Please pass only single TreeNeurons')

    if not isinstance(x, core.TreeNeuron):
        raise ValueError('Can only process TreeNeurons')

    # First, get a list of child -> parent locs (exclude root node!)
    tn_locs = x.nodes[~x.nodes.parent_id.isnull()][['x', 'y', 'z']].values
    pn_locs = x.nodes.set_index('node_id').loc[x.nodes[~x.nodes.parent_id.isnull(
    )].parent_id][['x', 'y', 'z']].values

    # Get centers between each pair of locs
    centers = tn_locs + (pn_locs - tn_locs) / 2

    # Get vector between points
    vec = pn_locs - tn_locs

    dps = pd.DataFrame([[c, v] for c, v in zip(centers, vec)],
                       columns=['point', 'vector'])

    # Add length of vector (for convenience)
    dps['vec_length'] = (dps.vector ** 2).apply(sum).apply(math.sqrt)

    return dps
