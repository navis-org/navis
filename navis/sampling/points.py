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

from .. import config, core, utils

# Set up logging
logger = config.get_logger(__name__)

__all__ = ['sample_skeleton']


# ---------------------------------------------------------------------------
# Private helpers for sample_skeleton
# ---------------------------------------------------------------------------


def _ss_build_tree(x: 'core.TreeNeuron'):
    """Return (parent_map, depth_map, root_dist, root_id, children, xyz)."""
    nodes   = x.nodes.set_index('node_id')
    root_id = x.nodes.loc[x.nodes.parent_id < 0, 'node_id'].values[0]

    # Vectorised extraction (avoids a slow per-row `.iterrows()`)
    node_ids   = nodes.index.values
    parent_ids = nodes['parent_id'].values
    coords     = nodes[['x', 'y', 'z']].values.astype(float)

    parent_map: dict = {}
    children:   dict = {nid: [] for nid in node_ids}
    xyz:        dict = {nid: coords[i] for i, nid in enumerate(node_ids)}

    for nid, pid in zip(node_ids, parent_ids):
        if pid < 0:
            parent_map[nid] = None
        else:
            parent_map[nid] = pid
            children.setdefault(pid, []).append(nid)

    depth_map  = {root_id: 0}
    root_dist  = {root_id: 0.0}
    queue = [root_id]
    while queue:
        nxt = []
        for nid in queue:
            for cid in children.get(nid, []):
                depth_map[cid] = depth_map[nid] + 1
                root_dist[cid] = root_dist[nid] + float(
                    np.linalg.norm(xyz[cid] - xyz[nid])
                )
                nxt.append(cid)
        queue = nxt

    return parent_map, depth_map, root_dist, root_id, children, xyz


def _ss_geodesic_count(root_id, children, xyz, step_size: float) -> int:
    count = 1
    stack = [(root_id, 0.0)]
    while stack:
        nid, carry = stack.pop()
        for cid in children.get(nid, []):
            L    = float(np.linalg.norm(xyz[cid] - xyz[nid]))
            dist = carry + L
            n    = int(dist / step_size)
            count += n
            stack.append((cid, dist - n * step_size))
    return count


def _ss_sample_points(x: 'core.TreeNeuron', n_points: int) -> np.ndarray:
    """Sample *n_points* at equal spacing along the arbor using DFS carry-over.

    Points on each edge are placed at ``k * step_size - carry`` from the
    parent node, where *carry* is the leftover distance from the previous
    edge in the DFS traversal.  This guarantees that consecutive samples
    are exactly ``step_size`` apart along any root-to-leaf path.
    """
    _, _, _, root_id, children, xyz = _ss_build_tree(x)

    total = sum(
        float(np.linalg.norm(xyz[cid] - xyz[nid]))
        for nid in children for cid in children[nid]
    )
    if total == 0:
        raise ValueError(f"Neuron {x.id} has zero total cable length.")

    lo = total / (n_points * 10)
    hi = total * 2
    while _ss_geodesic_count(root_id, children, xyz, lo) < n_points:
        lo /= 2.0

    for _ in range(64):
        mid = (lo + hi) / 2.0
        cnt = _ss_geodesic_count(root_id, children, xyz, mid)
        if cnt == n_points:
            break
        elif cnt > n_points:   # step too fine  → increase lo
            lo = mid
        else:                  # step too coarse → decrease hi
            hi = mid

    pts = [xyz[root_id]]
    stack = [(root_id, 0.0)]
    while stack:
        nid, carry = stack.pop()
        for cid in children.get(nid, []):
            edge_len  = float(np.linalg.norm(xyz[cid] - xyz[nid]))
            dist      = carry + edge_len
            n_new     = int(dist / mid)
            direction = (xyz[cid] - xyz[nid]) / edge_len if edge_len > 0 else np.zeros(3)
            for k in range(1, n_new + 1):
                d = min(max(k * mid - carry, 0.0), edge_len)
                pts.append(xyz[nid] + d * direction)
            stack.append((cid, dist - n_new * mid))

    pts = np.array(pts, dtype=np.float64)
    if len(pts) > n_points:
        pts = pts[:n_points]
    elif len(pts) < n_points:
        pts = np.vstack([pts, np.tile(pts[-1], (n_points - len(pts), 1))])
    return pts


# ---------------------------------------------------------------------------


@utils.map_neuronlist(desc='Sampling', allow_parallel=True)
def sample_skeleton(
    x: 'core.NeuronObject',
    n_points: int,
) -> np.ndarray:
    """Sample a fixed number of points along a skeleton.

    Points are drawn at equal spacing along the arbor using a DFS traversal
    with carry-over: the leftover distance at the end of each edge is passed
    to the next edge, ensuring consecutive samples are exactly one step apart
    along every root-to-leaf path.

    Parameters
    ----------
    x :         TreeNeuron | NeuronList
                Neuron(s) to sample.
    n_points :  int
                Number of points to draw from each neuron.

    Returns
    -------
    np.ndarray (n_points, 3)
                XYZ coordinates of sampled points. If `x` is a NeuronList,
                returns a list of such arrays - one per neuron.

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> pts = navis.sample_skeleton(n, n_points=50)
    >>> pts.shape
    (50, 3)

    >>> nl = navis.example_neurons(2)
    >>> pts = navis.sample_skeleton(nl, n_points=50)
    >>> [p.shape for p in pts]
    [(50, 3), (50, 3)]

    See Also
    --------
    [`navis.resample_skeleton`][]
                Resample a skeleton to a target node spacing (returns a neuron).
    [`navis.downsample_neuron`][]
                Reduce node count while preserving topology.

    """
    if not isinstance(x, core.TreeNeuron):
        raise TypeError(f'sample_skeleton requires a TreeNeuron, got {type(x)}.')

    return _ss_sample_points(x, n_points)
