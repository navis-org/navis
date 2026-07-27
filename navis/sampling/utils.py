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

import heapq

import numpy as np

try:
    from pykdtree.kdtree import KDTree
    _HAS_PYKDTREE = True
except ModuleNotFoundError:
    from scipy.spatial import cKDTree as KDTree
    _HAS_PYKDTREE = False


# Farthest-point sampling is O(N * size) and produces a more uniform sample than
# decimation. We only use it while that cost stays small (roughly a few hundred
# ms) and otherwise fall back to the cheaper decimation.
_FPS_MAX_WORK = 2e7

# Number of nearest neighbours precomputed per point for decimation. Larger
# values mean fewer (expensive) tree re-queries but more up-front memory/time.
_DECIMATE_NEIGHBORS = 16


def sample_points_uniform(points, size, output="points", method="auto"):
    """Draw a uniform sample from a point cloud.

    Two strategies are available:

    - ``"fps"`` (farthest-point sampling) iteratively picks the point that is
      farthest from everything picked so far. This gives the most uniform
      coverage but costs ``O(N * size)``.
    - ``"decimate"`` iteratively removes the point with the smallest distance to
      its nearest (still-present) neighbour until ``size`` points remain. This is
      cheap (``~O(N log N)``) and scales to large point clouds.

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
    method :    "auto" | "fps" | "decimate", optional
                Sampling strategy (see above). "auto" uses farthest-point
                sampling while it is cheap and decimation otherwise.

    Returns
    -------
    See `output` parameter. Indices/points are returned in ascending index
    order regardless of the sampling method.

    See Also
    --------
    [`navis.ml.chunk_neuron`][]
                Break a neuron into fixed-size, spatially-coherent fragments
                (tiling/oversampling) rather than drawing a single uniform
                subsample. The complementary primitive for preparing ML inputs.
    [`navis.ml.normalize_neuron`][]
                Canonicalize a neuron's pose (center/orient/scale) - a common
                preprocessing step before sampling points for a model.

    """
    points = np.asarray(points)

    assert isinstance(points, np.ndarray) and points.ndim == 2 and points.shape[1] == 3
    assert output in ("points", "indices", "mask")
    assert method in ("auto", "fps", "decimate")
    assert (size > 0) and (size <= len(points))

    N = len(points)

    if size == N:
        mask = np.ones(N, dtype=bool)
    else:
        if method == "auto":
            # Use farthest-point sampling while its O(N * size) cost stays small,
            # otherwise fall back to the much cheaper decimation.
            method = "fps" if (N * size) <= _FPS_MAX_WORK else "decimate"

        if method == "fps":
            mask = _sample_fps(points, size)
        else:
            mask = _sample_decimate(points, size)

    if output == "mask":
        return mask
    elif output == "indices":
        return np.arange(N)[mask]
    return points[mask].copy()


def estimate_spacing(points, aggregate="median"):
    """Estimate the characteristic point spacing of a cloud.

    Measures, for every point, the distance to its nearest neighbour and returns a
    single representative value. This is the empirical inverse of the `spacing` knob
    in [`navis.ml.sample_surface`][]: for a roughly even *surface* sample the areal
    density relates to the spacing by ``density ~ 1 / spacing**2``. Use it to audit
    an existing cloud or to pick a `spacing`/`density` for a whole dataset.

    Parameters
    ----------
    points :        (N, 3) array
                    Point cloud to measure.
    aggregate :     "median" | "mean" | "min", optional
                    How to reduce the per-point nearest-neighbour distances.
                    "median" (default) is robust to a few very close pairs.

    Returns
    -------
    float
                    The aggregated nearest-neighbour distance (the characteristic
                    inter-point spacing).

    See Also
    --------
    [`navis.ml.sample_surface`][]
                    Sample a mesh at a target `spacing`/`density`; this is the
                    inverse measurement.
    [`navis.ml.sample_points_uniform`][]
                    Subsample a cloud to a target size.

    Examples
    --------
    >>> import navis
    >>> import numpy as np
    >>> pts = np.stack(np.meshgrid(np.arange(10), np.arange(10), 0), -1).reshape(-1, 3)
    >>> float(navis.ml.estimate_spacing(pts.astype(float)))   # unit grid -> spacing 1
    1.0

    """
    points = np.asarray(points, dtype=np.float64)
    assert points.ndim == 2 and points.shape[1] == 3, "`points` must be an (N, 3) array."
    assert aggregate in ("median", "mean", "min"), "invalid `aggregate`"
    if len(points) < 2:
        raise ValueError("Need at least 2 points to estimate spacing.")

    tree = KDTree(points)
    # k=2: column 0 is the point itself (distance 0), column 1 its nearest neighbour.
    dd, _ = tree.query(points, k=2)
    nn = np.asarray(dd)[:, 1]

    # `aggregate` is asserted above to be one of median/mean/min - all np functions.
    return float(getattr(np, aggregate)(nn))


def _sample_fps(points, size):
    """Farthest-point sampling.

    Returns a boolean mask of the ``size`` selected points. Starts from the first
    point and repeatedly adds the point with the largest distance to the set of
    already-selected points.
    """
    points = np.asarray(points, dtype=np.float64)
    N = len(points)

    sel = np.empty(size, dtype=np.int64)
    sel[0] = 0
    # Squared distance of every point to the selected set (sqrt is monotonic, so
    # we can compare squared distances and skip it in the hot loop).
    dist = np.sum((points - points[0]) ** 2, axis=1)
    dist[0] = -np.inf  # never re-select an already-selected point
    for k in range(1, size):
        j = int(np.argmax(dist))
        sel[k] = j
        # A point's distance to the selected set can only shrink as we add points.
        np.minimum(dist, np.sum((points - points[j]) ** 2, axis=1), out=dist)
        # Coincident points sit at distance 0 from a selected point; mark this one
        # so argmax can't hand it back on the next iteration.
        dist[j] = -np.inf

    mask = np.zeros(N, dtype=bool)
    mask[sel] = True
    return mask


def _sample_decimate(points, size):
    """Decimate a point cloud down to ``size`` points.

    Returns a boolean mask of the surviving points. Repeatedly removes the point
    with the smallest distance to its nearest surviving neighbour.

    Rather than recomputing all nearest-neighbour distances every time a point is
    removed (which is ``O(N)`` per removal), we keep them in a min-heap and only
    recompute the distance for a point once its recorded nearest neighbour has
    itself been removed. The distance to the nearest surviving neighbour can only
    grow as points are removed, so this lazy update always yields the correct
    global minimum.
    """
    N = len(points)
    alive = np.ones(N, dtype=bool)

    tree = KDTree(points)

    # Precompute the K nearest neighbours of every point once. As points are
    # removed we walk down this list to find the nearest surviving neighbour and
    # only fall back to a fresh tree query once the list is exhausted.
    K = int(min(_DECIMATE_NEIGHBORS, N - 1))
    dd, ii = tree.query(points, k=K + 1)

    # Drop the self-match from every row. It is usually in column 0 but can be
    # elsewhere when points are coincident (distance 0), so locate it explicitly.
    rows = np.arange(N)
    is_self = ii == rows[:, None]
    keep = ~is_self
    keep[~is_self.any(axis=1), -1] = False  # no self found -> just drop last col
    neigh_idx = ii[keep].reshape(N, K)
    neigh_d = dd[keep].reshape(N, K)

    ptr = np.zeros(N, dtype=np.int64)   # position of each point's nearest survivor
    nn = neigh_idx[:, 0].copy()         # current nearest surviving neighbour

    def nearest_alive(i):
        """Nearest surviving neighbour of ``i`` as (distance, index)."""
        p = ptr[i]
        row = neigh_idx[i]
        while p < K and not alive[row[p]]:
            p += 1
        ptr[i] = p
        if p < K:
            return float(neigh_d[i, p]), int(row[p])
        # Precomputed neighbours exhausted -> query the tree for the nearest
        # point that is neither ``i`` nor already removed.
        if _HAS_PYKDTREE:
            masked = ~alive
            masked[i] = True
            d1, j1 = tree.query(points[i:i + 1], k=1, mask=masked)
            return float(d1[0]), int(j1[0])
        kf = min(K * 2, N)
        while True:
            d1, j1 = tree.query(points[i:i + 1], k=kf)
            for dv, jv in zip(np.atleast_1d(d1[0]), np.atleast_1d(j1[0])):
                if jv != i and alive[jv]:
                    return float(dv), int(jv)
            if kf >= N:
                return np.inf, i
            kf = min(kf * 2, N)

    heap = [(float(neigh_d[i, 0]), int(i)) for i in range(N)]
    heapq.heapify(heap)

    n_alive = N
    while n_alive > size:
        d, i = heapq.heappop(heap)
        if not alive[i]:
            continue
        if not alive[nn[i]]:
            # Recorded nearest neighbour is gone -> refresh and re-insert. The new
            # distance is >= the old one, so it is safe to defer i's removal.
            dnew, jnew = nearest_alive(i)
            nn[i] = jnew
            heapq.heappush(heap, (dnew, i))
            continue
        # i has the globally smallest nearest-neighbour distance -> remove it.
        alive[i] = False
        n_alive -= 1

    return alive
