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

import networkx as nx
import numpy as np

from .. import core


__all__ = ['health_check', 'merge_duplicate_nodes']


def health_check(x: 'core.NeuronObject', verbose: bool = True) -> None:
    """Run a health check on Skeletons and flag potential issues.

    Parameters
    ----------
    x :         Skeleton | NeuronList
                Neuron(s) whose nodes to classify nodes.
    verbose :   bool
                If True, will print errors in addition to returning them.

    Returns
    -------
    list of issues or None

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> navis.health_check(n)
    Neuron 1734350788 seems perfectly fine.

    """
    if isinstance(x, core.NeuronList):
        for n in x:
            _ = health_check(x)
        return
    elif not isinstance(x, core.Skeleton):
        raise TypeError(f'Excepted Skeleton/List, got "{type(x)}"')

    issues = []

    # Check if neuron is not a tree
    if not x.is_tree:
        issues.append('is not a tree (navis_fastcore.has_cycles)')
    # See if there are any cycles
    if x.cycles:
        issues.append(f'has cycles (networkx.find_cycles): {str(x.cycles)}')
    # See if any node has more than one parent
    od = [n[0] for n in x.graph.out_degree if n[1] > 1]
    if od:
        issues.append(f'has nodes with multiple parents (graph.out_degree): {", ".join(od)}')

    locs, counts = np.unique(x.nodes[['x', 'y', 'z']].values,
                             axis=0,
                             return_counts=True)
    dupl = counts > 1
    if any(dupl):
        issues.append(f'has {sum(dupl)} node positions that are occupied by multiple nodes')

    if verbose:
        if issues:
            print(f'Neuron {str(x.id)} has issues:')
            for i in issues:
                print(f' - {i}')
        else:
            print(f'Neuron {str(x.id)} seems perfectly fine.')

    return issues if issues else None


def merge_duplicate_nodes(x, round=False, inplace=False):
    """Merge nodes the occupy the exact same position in space.

    Note that this might produce connections where there previously weren't
    any!

    Parameters
    ----------
    x :         Skeleton | NeuronList
                Neuron(s) to fix.
    round :     int, optional
                If provided will round node locations to given decimals. This
                can be useful if the positions are floats and not `exactly` the
                the same.
    inplace :   bool
                If True, perform operation on neuron inplace.

    Returns
    -------
    Skeleton
                Fixed neuron. Only if `inplace=False`.

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> n.nodes.loc[1, ['x', 'y' ,'z']] = n.nodes.loc[0, ['x', 'y' ,'z']]
    >>> fx = navis.graph.clinic.merge_duplicate_nodes(n)
    >>> n.n_nodes, fx.n_nodes
    (4465, 4464)

    """
    if isinstance(x, core.NeuronList):
        if not inplace:
            x = x.copy()

        for n in x:
            _ = merge_duplicate_nodes(n, round=round, inplace=True)

        if not inplace:
            return x
        return

    if not isinstance(x, core.Skeleton):
        raise TypeError(f'Expected Skeleton, got "{type(x)}"')

    if not inplace:
        x = x.copy()

    # Figure out which nodes are duplicated
    if round:
        dupl = x.nodes[['x', 'y', 'z']].round(round).duplicated(keep=False)
    else:
        dupl = x.nodes[['x', 'y', 'z']].duplicated(keep=False)

    if dupl.sum():
        # Operate on the edge list
        edges = x.nodes[['node_id', 'parent_id']].values.copy()

        # Go over each non-unique location
        ids = x.nodes.loc[dupl].groupby(['x', 'y', 'z']).node_id.apply(list)
        # Which node each collapsed one is folded into, for `relocate_refs`
        # below. Groups are one per position and so disjoint: a node is folded
        # at most once, and never into another node that is itself folded.
        folded, onto = [], []
        for i in ids:
            # Keep the first node and collapse all others into it
            edges[np.isin(edges[:, 0], i[1:]), 0] = i[0]
            edges[np.isin(edges[:, 1], i[1:]), 1] = i[0]
            folded.extend(i[1:])
            onto.extend([i[0]] * (len(i) - 1))

        # Drop self-loops
        edges = edges[edges[:, 0] != edges[:, 1]]

        # Make sure we don't have a->b and b<-a edges
        edges = np.unique(np.sort(edges, axis=1), axis=0)

        G = nx.Graph()

        # Get nodes but drop ""-1"
        nodes = edges.flatten()
        nodes = nodes[nodes >= 0]

        # Add nodes
        G.add_nodes_from(nodes)

        # Drop edges that point away from root (e.g. (1, -1))
        # Don't do this before because we would loose isolated nodes otherwise
        edges = edges[edges.min(axis=1) >= 0]

        # Add edges
        G.add_edges_from([(e[0], e[1]) for e in edges])

        # First remove cycles
        while True:
            try:
                # Find cycle
                cycle = nx.find_cycle(G)
            except nx.exception.NetworkXNoCycle:
                break
            except BaseException:
                raise

            # Sort by degree
            cycle = sorted(cycle, key=lambda x: G.degree[x[0]])

            # Remove the edge with the lowest degree
            G.remove_edge(cycle[0][0], cycle[0][1])

        # Now make sure this is a DAG, i.e. that all edges point in the same direction
        new_edges = []
        for c in nx.connected_components(G.to_undirected()):
            sg = nx.subgraph(G, c)

            # Try picking a node that was root in the original neuron
            is_present = np.isin(x.root, sg.nodes)
            if any(is_present):
                r = x.root[is_present][0]
            else:
                r = list(sg.nodes)[0]

            # Generate parent->child dictionary by graph traversal
            this_lop = nx.predecessor(sg, r)

            # Note that we assign -1 as root's parent
            new_edges += [(k, v[0]) for k, v in this_lop.items() if v]

        # We need a directed Graph for this as otherwise the child -> parent
        # order in the edges might get lost
        G2 = nx.DiGraph()
        G2.add_nodes_from(G.nodes)
        G2.add_edges_from(new_edges)

        # Generate list of parents
        new_edges = np.array(G2.edges)
        new_parents = dict(zip(new_edges[:, 0], new_edges[:, 1]))

        axis = core.schema.get_axis(x, "nodes")

        # A folded node is not gone from the neuron: it is the very same point
        # in space as the one it was folded into, so a connector, tag or soma
        # sitting on it belongs on that one. Say so first - a selection can only
        # say "gone", and would take them out with it.
        core.schema.relocate_refs(x, axis, folded, onto)

        # Now drop the nodes that aren't present anymore. No node is invented
        # here, so this is an ordinary selection and goes through the schema:
        # anything aligned to the nodes is carried, and anything still naming a
        # node that did not survive for some *other* reason is repaired.
        core.schema.apply_selection(
            x, axis, x._nodes.node_id.isin(new_edges.flatten()).values
        )

        # Rewire kept nodes. Built from the collapsed graph, so it has the answer
        # for every remaining node - including those whose parent was merged away
        # and which `apply_selection` has just turned into roots.
        x.nodes['parent_id'] = x.nodes.node_id.map(lambda i: new_parents.get(i, -1))

        # Reset temporary attributes
        x._clear_temp_attr()

    if not inplace:
        return x
