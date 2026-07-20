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

"""Angle-based morphometrics."""

from itertools import combinations

import numpy as np
import pandas as pd

from .. import config, graph, core, utils

logger = config.get_logger(__name__)

__all__ = sorted([
    "branch_angles",
    "path_angles",
    "root_angles",
    "soma_exit_angles",
])


def _angle_between(u, v):
    """Compute angle(s) between paired 3D vectors.

    Uses the numerically stable `arctan2(||u x v||, u . v)` formulation on the
    unit vectors. Zero-length vectors yield `NaN` rather than a spurious angle.

    Parameters
    ----------
    u,v :   (N, 3) array
            Paired vectors.

    Returns
    -------
    (N, ) array
            Angles in radians in the interval [0, pi].

    """
    u = np.asarray(u, dtype=float).reshape(-1, 3)
    v = np.asarray(v, dtype=float).reshape(-1, 3)

    nu = np.linalg.norm(u, axis=1)
    nv = np.linalg.norm(v, axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        un = u / nu[:, None]
        vn = v / nv[:, None]
        cross = np.linalg.norm(np.cross(un, vn), axis=1)
        dot = np.einsum("ij,ij->i", un, vn)
        ang = np.arctan2(cross, dot)

    # Degenerate (zero-length) vectors have no defined angle
    ang[(nu == 0) | (nv == 0)] = np.nan

    return ang


@utils.map_neuronlist_df(desc="Branch angles", allow_parallel=True, reset_index=True)
@utils.meshneuron_skeleton(method="pass_through", reroot_soma=True)
def branch_angles(x, degrees=True):
    """Compute branch (bifurcation) angles.

    At each branch point we measure the angle between the directions towards its
    child nodes. For the usual bifurcation this is a single angle between the two
    children; at higher-order branch points (3+ children) *all* pairwise angles
    are returned. The root/soma is excluded - see [`navis.soma_exit_angles`][] for
    the angles between the neurites emanating from the soma.

    Parameters
    ----------
    x :         TreeNeuron | MeshNeuron | NeuronList
                Neuron to analyze. If MeshNeuron, will generate and use a
                skeleton representation.
    degrees :   bool
                If True (default), angles are returned in degrees, otherwise in
                radians.

    Returns
    -------
    pandas.DataFrame
                One row per child pair with columns:
                  - `node_id` is the branch point the angle is measured at
                  - `branch_angle` is the angle between the two child directions

    See Also
    --------
    [`navis.path_angles`][]
                Angles measured at continuation (slab) nodes.
    [`navis.soma_exit_angles`][]
                Angles between neurites emanating from the soma.

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1, kind='skeleton')
    >>> ba = navis.branch_angles(n)
    >>> ba.columns.tolist()
    ['node_id', 'branch_angle']
    >>> bool((ba.branch_angle.dropna() <= 180).all())
    True

    """
    utils.eval_param(x, name="x", allowed_types=(core.TreeNeuron,))

    coords = x.nodes.set_index("node_id")[["x", "y", "z"]].astype(float)
    parents = x.nodes.set_index("node_id").parent_id.to_dict()
    childs = graph.generate_list_of_childs(x)

    node_ids, vec_a, vec_b = [], [], []
    for n, ch in childs.items():
        # Need a genuine branch point that is not the root/soma
        if len(ch) < 2 or parents[n] < 0:
            continue
        vecs = coords.loc[ch].values - coords.loc[n].values
        for i, j in combinations(range(len(ch)), 2):
            node_ids.append(n)
            vec_a.append(vecs[i])
            vec_b.append(vecs[j])

    if node_ids:
        ang = _angle_between(np.asarray(vec_a), np.asarray(vec_b))
    else:
        ang = np.array([], dtype=float)

    if degrees:
        ang = np.degrees(ang)

    return pd.DataFrame({"node_id": node_ids, "branch_angle": ang})


@utils.map_neuronlist_df(desc="Path angles", allow_parallel=True, reset_index=True)
@utils.meshneuron_skeleton(method="pass_through", reroot_soma=True)
def path_angles(x, degrees=True):
    """Compute path angles.

    At each continuation ("slab") node - i.e. a node with exactly one child that
    is not a root - we measure the angle between the incoming edge (from its
    parent) and the outgoing edge (to its child). A value of 0 means the path
    continues perfectly straight; larger values indicate stronger bends.

    Parameters
    ----------
    x :         TreeNeuron | MeshNeuron | NeuronList
                Neuron to analyze. If MeshNeuron, will generate and use a
                skeleton representation.
    degrees :   bool
                If True (default), angles are returned in degrees, otherwise in
                radians.

    Returns
    -------
    pandas.DataFrame
                One row per slab node with columns:
                  - `node_id` is the node the angle is measured at
                  - `path_angle` is the angle between in- and outgoing edge

    See Also
    --------
    [`navis.branch_angles`][]
                Angles between child branches at branch points.

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1, kind='skeleton')
    >>> pa = navis.path_angles(n)
    >>> pa.columns.tolist()
    ['node_id', 'path_angle']
    >>> bool((pa.path_angle.dropna() <= 180).all())
    True

    """
    utils.eval_param(x, name="x", allowed_types=(core.TreeNeuron,))

    coords = x.nodes.set_index("node_id")[["x", "y", "z"]].astype(float)
    pid = x.nodes.set_index("node_id").parent_id

    # Incoming edge vector for every node: pos(node) - pos(parent). Roots (whose
    # parent is not in the index) come out as NaN.
    incoming = pd.DataFrame(
        coords.values - coords.reindex(pid.values).values,
        index=coords.index,
        columns=["x", "y", "z"],
    )

    # A slab node has exactly one child and is not a root
    n_children = x.nodes.parent_id.value_counts().reindex(coords.index).fillna(0)
    is_slab = pd.Series(
        (n_children.values == 1) & (pid.values >= 0), index=coords.index
    )

    # For each node, check whether its parent is a slab node - if so, the parent
    # is where we measure the angle and this node is the outgoing edge.
    parent_is_slab = is_slab.reindex(pid.values).fillna(False).to_numpy(dtype=bool)
    w_ids = coords.index.values[parent_is_slab]  # distal node of the slab edge
    v_ids = pid.values[parent_is_slab]           # the slab node itself

    if len(v_ids):
        ang = _angle_between(incoming.loc[v_ids].values, incoming.loc[w_ids].values)
    else:
        ang = np.array([], dtype=float)

    if degrees:
        ang = np.degrees(ang)

    return pd.DataFrame({"node_id": v_ids, "path_angle": ang})


@utils.map_neuronlist_df(desc="Root angles", allow_parallel=True, reset_index=True)
@utils.meshneuron_skeleton(method="pass_through", reroot_soma=True)
def root_angles(x, degrees=True):
    """Compute root angles.

    For every edge `(parent -> node)` we measure the angle between the edge's
    direction and the direction from the root to the edge's proximal (parent)
    node. This captures how much each segment deviates from pointing radially
    away from the root/soma. Edges emanating directly from the root have an
    undefined reference direction and are returned as `NaN`.

    For fragmented neurons (multiple roots) each node is referenced against the
    root of its own connected component.

    Parameters
    ----------
    x :         TreeNeuron | MeshNeuron | NeuronList
                Neuron to analyze. If MeshNeuron, will generate and use a
                skeleton representation. For a meaningful result the neuron
                should be rooted at its soma (see [`navis.TreeNeuron.reroot`][]).
    degrees :   bool
                If True (default), angles are returned in degrees, otherwise in
                radians.

    Returns
    -------
    pandas.DataFrame
                One row per edge with columns:
                  - `node_id` is the distal node of the edge
                  - `root_angle` is the angle relative to the root direction

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1, kind='skeleton')
    >>> ra = navis.root_angles(n)
    >>> ra.columns.tolist()
    ['node_id', 'root_angle']
    >>> bool((ra.root_angle.dropna() <= 180).all())
    True

    """
    utils.eval_param(x, name="x", allowed_types=(core.TreeNeuron,))

    coords = x.nodes.set_index("node_id")[["x", "y", "z"]].astype(float)
    pid = x.nodes.set_index("node_id").parent_id

    # Edge direction (parent -> node) for every node
    incoming = pd.DataFrame(
        coords.values - coords.reindex(pid.values).values,
        index=coords.index,
        columns=["x", "y", "z"],
    )

    non_root = pid.values >= 0
    n2 = coords.index.values[non_root]  # distal node of the edge
    n1 = pid.values[non_root]           # proximal (parent) node of the edge

    # Map each proximal node to the root of its connected component
    root_set = set(int(r) for r in x.root)
    if len(root_set) == 1:
        root_of_n1 = np.full(len(n1), next(iter(root_set)))
    else:
        node2root = {}
        for sub in x.subtrees:
            sub = set(int(s) for s in sub)
            r = next(iter(root_set & sub))
            node2root.update({nd: r for nd in sub})
        root_of_n1 = np.array([node2root[int(nd)] for nd in n1])

    u = incoming.loc[n2].values                              # edge direction
    v = coords.loc[n1].values - coords.loc[root_of_n1].values  # root -> parent

    if len(n2):
        ang = _angle_between(u, v)
    else:
        ang = np.array([], dtype=float)

    if degrees:
        ang = np.degrees(ang)

    return pd.DataFrame({"node_id": n2, "root_angle": ang})


@utils.map_neuronlist_df(desc="Soma exit angles", allow_parallel=True, reset_index=True)
@utils.meshneuron_skeleton(method="pass_through", reroot_soma=True)
def soma_exit_angles(x, degrees=True):
    """Compute soma-exit angles.

    These are the angles between the neurites (stems) emanating from the root of
    the neuron: for each pair of stems we measure the angle between the vectors
    pointing from the root to the respective stem.

    For a meaningful result the neuron should be rooted at its soma (see
    [`navis.TreeNeuron.reroot`][]). Fragmented neurons (multiple roots) are
    handled by returning the stem angles for every root.

    Parameters
    ----------
    x :         TreeNeuron | MeshNeuron | NeuronList
                Neuron to analyze. If MeshNeuron, will generate and use a
                skeleton representation.
    degrees :   bool
                If True (default), angles are returned in degrees, otherwise in
                radians.

    Returns
    -------
    pandas.DataFrame
                One row per pair of stems with columns:
                  - `root_id` is the root the stems emanate from
                  - `soma_exit_angle` is the angle between the two stems

    See Also
    --------
    [`navis.branch_angles`][]
                Angles between child branches at (non-root) branch points.

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1, kind='skeleton')
    >>> n.reroot(n.soma, inplace=True)
    >>> sa = navis.soma_exit_angles(n)
    >>> sa.columns.tolist()
    ['root_id', 'soma_exit_angle']

    """
    utils.eval_param(x, name="x", allowed_types=(core.TreeNeuron,))

    coords = x.nodes.set_index("node_id")[["x", "y", "z"]].astype(float)
    childs = graph.generate_list_of_childs(x)

    root_ids, vec_a, vec_b = [], [], []
    for r in x.root:
        stems = childs.get(r, [])
        if len(stems) < 2:
            continue
        vecs = coords.loc[stems].values - coords.loc[r].values
        for i, j in combinations(range(len(stems)), 2):
            root_ids.append(r)
            vec_a.append(vecs[i])
            vec_b.append(vecs[j])

    if root_ids:
        ang = _angle_between(np.asarray(vec_a), np.asarray(vec_b))
    else:
        ang = np.array([], dtype=float)

    if degrees:
        ang = np.degrees(ang)

    return pd.DataFrame({"root_id": root_ids, "soma_exit_angle": ang})
