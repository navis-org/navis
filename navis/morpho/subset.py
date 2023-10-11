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

from typing import Union, Sequence

from .. import utils, config, core, graph

# Set up logging
logger = config.get_logger(__name__)

__all__ = sorted(['subset_neuron'])


@utils.lock_neuron
def subset_neuron(x: Union['core.TreeNeuron', 'core.MeshNeuron'],
                  subset: Union[Sequence[Union[int, str]],
                                nx.DiGraph,
                                pd.DataFrame],
                  inplace: bool = False,
                  keep_disc_cn: bool = False,
                  prevent_fragments: bool = False
                  ) -> 'core.NeuronObject':
    """Subset a neuron to a given set of nodes/vertices.

    Note that for ``MeshNeurons`` it is not guaranteed that all vertices in
    ``subset`` survive because we will also drop degenerate vertices that do
    not participate in any faces.

    Parameters
    ----------
    x :                   TreeNeuron | MeshNeuron | Dotprops
                          Neuron to subset.
    subset :              list-like | set | NetworkX.Graph | pandas.DataFrame
                          For TreeNeurons:
                           - node IDs to subset the neuron to
                           - a boolean mask
                           - DataFrame with ``node_id`` column
                          For MeshNeurons:
                           - vertex indices
                           - a boolean mask
                          For Dotprops:
                           - point indices
                           - a boolean mask

    keep_disc_cn :        bool, optional
                          If False, will remove disconnected connectors that
                          have "lost" their parent node/vertex.
    prevent_fragments :   bool, optional
                          If True, will add nodes/vertices to ``subset``
                          required to keep neuron from fragmenting. Ignored for
                          `Dotprops`.
    inplace :             bool, optional
                          If False, a copy of the neuron is returned.

    Returns
    -------
    TreeNeuron | MeshNeuron

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
    >>> n_short = navis.subset_neuron(n, nodes_to_keep)

    See Also
    --------
    :func:`navis.cut_skeleton`
            Cut neuron at specific points.
    :func:`navis.in_volume`
            To intersect a neuron with a volume (mesh).

    """
    if isinstance(x, core.NeuronList) and len(x) == 1:
        x = x[0]

    utils.eval_param(x, name='x',
                     allowed_types=(core.TreeNeuron, core.MeshNeuron, core.Dotprops))

    # Make a copy of the neuron
    if not inplace:
        x = x.copy()
        # We have to run this in a separate function so that the lock is applied
        # to the copy
        subset_neuron(x,
                      subset=subset,
                      inplace=True,
                      keep_disc_cn=keep_disc_cn,
                      prevent_fragments=prevent_fragments)
        return x

    if isinstance(x, core.TreeNeuron):
        x = _subset_treeneuron(x,
                               subset=subset,
                               keep_disc_cn=keep_disc_cn,
                               prevent_fragments=prevent_fragments)
    elif isinstance(x, core.MeshNeuron):
        x = _subset_meshneuron(x,
                               subset=subset,
                               keep_disc_cn=keep_disc_cn,
                               prevent_fragments=prevent_fragments)
    elif isinstance(x, core.Dotprops):
        x = _subset_dotprops(x,
                             subset=subset,
                             keep_disc_cn=keep_disc_cn)

    return x


def _subset_dotprops(x, subset, keep_disc_cn):
    """Subset Dotprops."""
    if not utils.is_iterable(subset):
        raise TypeError('Can only subset Dotprops to list, set or '
                        f'numpy.ndarray, not "{type(subset)}"')

    subset = utils.make_iterable(subset)

    # Convert indices to mask
    if subset.dtype == bool:
        if subset.shape != (x.points.shape[0], ):
            raise ValueError('Boolean mask must be of same length as points.')
        mask = subset
    else:
        mask = np.isin(np.arange(0, len(x.points)), subset)

    # Filter connectors
    if not keep_disc_cn and x.has_connectors:
        if 'point' not in x.connectors.columns:
            x.connectors['point'] = x.snap(x.connectors[['x', 'y', 'z']].values)[0]

        if subset.dtype == bool:
            subset = np.arange(0, len(x.points))[subset]

        x._connectors = x.connectors[x.connectors.point.isin(subset)].copy()
        x._connectors.reset_index(inplace=True, drop=True)

        # Make old -> new indices map
        new_ix = dict(zip(subset, np.arange(0, len(subset))))

        x.connectors['point'] = x.connectors.point.map(new_ix)

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
        raise TypeError('Can only subset MeshNeuron to list, set or '
                        f'numpy.ndarray, not "{type(subset)}"')

    subset = utils.make_iterable(subset)

    # Convert mask to indices
    if subset.dtype == bool:
        if subset.shape != (x.vertices.shape[0], ):
            raise ValueError('Boolean mask must be of same length as vertices.')
        subset = np.arange(0, len(x.vertices))[subset]

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
    if not keep_disc_cn and x.has_connectors:
        if 'vertex_id' not in x.connectors.columns:
            x.connectors['vertex'] = x.snap(x.connectors[['x', 'y', 'z']].values)[0]

        x._connectors = x.connectors[x.connectors.vertex.isin(subset)].copy()
        x._connectors.reset_index(inplace=True, drop=True)

        # Make old -> new indices map
        new_ix = dict(zip(subset, np.arange(0, len(subset))))

        x.connectors['vertex'] = x.connectors.vertex.map(new_ix)

    # Subset the mesh (by faces)
    # Build the mask bit by bit to be more efficient:
    subset_faces = np.full(len(x.faces), True)
    for i in range(3):
        subset_faces[subset_faces] = np.isin(x.faces[subset_faces, i], subset)
    subset_faces = np.where(subset_faces)[0]

    if len(subset_faces):
        submesh = x.trimesh.submesh([subset_faces], append=True)
        x.vertices, x.faces = submesh.vertices, submesh.faces
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
        raise TypeError('Can only subset to list, set, numpy.ndarray or'
                        f'networkx.Graph, not "{type(subset)}"')

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
    x._nodes = pd.concat([x.nodes.drop('parent_id', inplace=False, axis=1),  # type: ignore  # no stubs for concat
                          x.nodes[['parent_id']].where(x.nodes.parent_id.isin(x.nodes.node_id.values),
                                                       other=-1, inplace=False)],
                         axis=1)

    # Make sure any new roots or leafs are properly typed
    # We won't produce new slabs but roots and leaves might change
    x.nodes.loc[x.nodes.parent_id < 0, 'type'] = 'root'
    x.nodes.loc[(~x.nodes.node_id.isin(x.nodes.parent_id.values)
                 & (x.nodes.parent_id >= 0)), 'type'] = 'end'

    # Filter connectors
    if not keep_disc_cn and x.has_connectors:
        x._connectors = x.connectors[x.connectors.node_id.isin(x.nodes.node_id)]
        x._connectors.reset_index(inplace=True, drop=True)

    if getattr(x, 'tags', None) is not None:
        # Filter tags
        x.tags = {t: [tn for tn in x.tags[t] if tn in x.nodes.node_id.values] for t in x.tags}  # type: ignore  # TreeNeuron has no tags

        # Remove empty tags
        x.tags = {t: x.tags[t] for t in x.tags if x.tags[t]}  # type: ignore  # TreeNeuron has no tags

    # Fix graph representations
    if '_graph_nx' in x.__dict__:
        x._graph_nx = x.graph.subgraph(x.nodes.node_id.values)
    if '_igraph' in x.__dict__:
        if x.igraph and config.use_igraph:
            id2ix = {n: ix for ix, n in zip(x.igraph.vs.indices,
                                            x.igraph.vs.get_attribute_values('node_id'))}
            indices = [id2ix[n] for n in x.nodes.node_id.values]
            vs = x.igraph.vs[indices]
            x._igraph = x.igraph.subgraph(vs)

    # Reset indices of data tables
    x.nodes.reset_index(inplace=True, drop=True)

    if new_root:
        x.reroot(new_root, inplace=True)

    return x
