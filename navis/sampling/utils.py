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

try:
    from pykdtree.kdtree import KDTree
except ModuleNotFoundError:
    from scipy.spatial import cKDTree as KDTree


def sample_points_uniform(points, size, output="points"):
    """Draw uniform sample from point cloud.

    This functions works by iteratively removing the point with the smallest
    distance to its nearest neighbor until the desired number of points is
    reached.

    Parameters
    ----------
    points :    (N, 3 ) array
                Point cloud to sample from.
    size :      int
                Number of samples to draw.
    output :    "points" | "indices" | "mask", optional
                If "points", returns the sampled points. If "indices", returns
                the indices of the sampled points. If "mask", returns a boolean
                mask of the sampled points.

    Returns
    -------
    See `output` parameter.

    """
    points = np.asarray(points)

    assert isinstance(points, np.ndarray) and points.ndim == 2 and points.shape[1] == 3
    assert output in ("points", "indices", "mask")
    assert (size > 0) and (size <= len(points))

    # Start with all points in the mask
    mask = np.ones(len(points), dtype=bool)

    # Generate a tree
    tree = KDTree(points)

    p_ind = np.arange(len(points))

    while mask.sum() > size:
        # Find the point with the largest distance to its nearest neighbor
        d, ind = tree.query(points[mask], k=2, mask=~mask)
        d, ind = d[:, 1], ind[:, 1]

        # Find pairs of nodes that are close to each other
        is_close = d == d.min()
        pairs = np.stack((p_ind[mask][is_close], p_ind[ind][is_close]), axis=1)

        # At this point we will have pairs show up multiple times - (a, b) and (b, a)
        pairs = np.unique(np.sort(pairs, axis=1), axis=0)

        # Imagine we have two candidate pairs for removal: (a, b) and (b, c)
        # In that case we can remove (a and c) or (b) but not (a, b) or (b, c)
        # because that might leave a hole in the point cloud
        G = nx.Graph()
        G.add_edges_from(pairs)

        to_remove = []
        for cc in nx.connected_components(G):
            # If these are two nodes, it doesn't matter which one we drop
            if len(cc) <= 2:
                to_remove.append(cc.pop())
                continue
            # If we have three or more nodes, we will simply remove the one
            # with the highest degree
            to_remove.append(sorted(cc, key=lambda x: G.degree(x))[-1])

        # Number of nodes we still need to remove
        n_remove = mask.sum() - size

        if n_remove >= len(to_remove):
            mask[to_remove] = False
        else:
            mask[to_remove[:n_remove]] = False

    if output == "mask":
        return mask
    elif output == "indices":
        return p_ind[mask]
    elif output == "points":
        return points[mask].copy()







