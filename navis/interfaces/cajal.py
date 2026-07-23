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

"""Interface with CAJAL (https://github.com/CamaraLab/CAJAL) for computing
Gromov-Wasserstein distances between neurons.

CAJAL works by:
  1. Sampling N uniformly-spaced points along a neuron's arbor.
  2. Computing an NxN intracellular distance matrix (ICDM) between those
     points (Euclidean or geodesic).
  3. Computing pairwise Gromov-Wasserstein (GW) distances between neurons
     from their ICDMs.

This module mirrors CAJAL's sampling approach (binary-search for step_size +
edge interpolation) and delegates GW computation to ``cajal.run_gw``.
"""

from __future__ import annotations

import os
import multiprocessing as mp
import numpy as np
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed
from textwrap import dedent
from typing import Union, Optional
from typing_extensions import Literal

from scipy.spatial.distance import cdist, pdist, squareform

from .. import config, utils
from ..core import TreeNeuron, NeuronList

logger = config.get_logger(__name__)

__all__ = ["cajal_gw"]


# ---------------------------------------------------------------------------
# Module-level worker for parallel ICDM computation
# (must be at module level to be picklable by ProcessPoolExecutor)
# ---------------------------------------------------------------------------


def _icdm_worker(args: tuple) -> tuple:
    """Compute the ICDM (and optional features) for a single neuron.

    Parameters
    ----------
    args : (TreeNeuron, int, str, list or None)
        Neuron, n_points, metric ('euclidean' or 'geodesic'), feature_cols.

    Returns
    -------
    (neuron_id, np.ndarray, np.ndarray or None)
    """
    neuron, n_points, metric, feature_cols = args
    icdm_fn = _icdm_geodesic if metric == "geodesic" else _icdm_euclidean
    D, feat = icdm_fn(neuron, n_points, feature_cols=feature_cols)
    return (neuron.id, D, feat)


# ---------------------------------------------------------------------------
# Internal helpers: edge extraction
# ---------------------------------------------------------------------------


def _get_skeleton_edges(x: TreeNeuron):
    """Return a list of edges as (parent_id, parent_xyz, child_id, child_xyz, length) tuples.

    Returns
    -------
    edges : list of (int, np.ndarray (3,), int, np.ndarray (3,), float)
        Each tuple is (parent_id, parent_xyz, child_id, child_xyz, edge_length).
    root_id : int
        Node ID of the root.
    root_xyz : np.ndarray shape (3,)
        XYZ of the root node.
    """
    nodes = x.nodes.set_index("node_id")
    root_id = x.nodes.loc[x.nodes.parent_id < 0, "node_id"].values[0]
    root_xyz = nodes.loc[root_id, ["x", "y", "z"]].values.astype(float)

    edges = []
    non_root = x.nodes[x.nodes.parent_id >= 0]
    for _, row in non_root.iterrows():
        child_id = row.node_id
        parent_id = row.parent_id
        child_xyz = nodes.loc[child_id, ["x", "y", "z"]].values.astype(float)
        parent_xyz = nodes.loc[parent_id, ["x", "y", "z"]].values.astype(float)
        L = float(np.linalg.norm(child_xyz - parent_xyz))
        if L > 0:
            edges.append((parent_id, parent_xyz, child_id, child_xyz, L))

    return edges, root_id, root_xyz


# ---------------------------------------------------------------------------
# Internal helpers: Euclidean sampling (mirrors CAJAL's sample_swc approach)
# ---------------------------------------------------------------------------


def _count_euclidean_samples(edges, step_size: float) -> int:
    """Count how many points would be sampled with the given step_size."""
    count = 1  # always include root
    for _pid, _parent, _cid, _child, L in edges:
        count += int(L / step_size)
    return count


def _collect_euclidean_points(edges, root_id, root_xyz, step_size: float):
    """Sample points along skeleton edges at equal Euclidean spacing.

    Includes the root node.  For each edge, places samples at offsets
    ``step_size, 2*step_size, ...`` from the parent toward the child so
    branch points are not double-counted.

    Parameters
    ----------
    edges :     list of (parent_id, parent_xyz, child_id, child_xyz, length)
    root_id :   int
    root_xyz :  np.ndarray (3,)
    step_size : float

    Returns
    -------
    pts :        np.ndarray (N, 3)
    parent_ids : list of int  — skeleton parent node for each sampled point
    child_ids :  list of int  — skeleton child node for each sampled point
    fractions :  list of float — interpolation fraction along the edge (0=parent)
    """
    pts = [root_xyz]
    parent_ids = [root_id]
    child_ids = [root_id]
    fractions = [0.0]
    for pid, parent_xyz, cid, child_xyz, L in edges:
        direction = (child_xyz - parent_xyz) / L
        n_samples = int(L / step_size)
        for k in range(1, n_samples + 1):
            t = k * step_size / L
            pts.append(parent_xyz + k * step_size * direction)
            parent_ids.append(pid)
            child_ids.append(cid)
            fractions.append(min(t, 1.0))
    return np.array(pts, dtype=np.float64), parent_ids, child_ids, fractions


def _icdm_euclidean(x: TreeNeuron, n_points: int, feature_cols=None):
    """Compute an NxN Euclidean intracellular distance matrix.

    Samples ``n_points`` evenly-spaced points along the skeleton's edges
    using a binary search for the appropriate step size (mirroring CAJAL's
    ``icdm_euclidean``).

    Parameters
    ----------
    x :            TreeNeuron
    n_points :     int
    feature_cols : list of str or None

    Returns
    -------
    (np.ndarray (n_points, n_points), np.ndarray (n_points, n_feats) or None)
    """
    edges, root_id, root_xyz = _get_skeleton_edges(x)

    if not edges:
        raise ValueError(f"Neuron {x.id} has no edges (single node).")

    total_length = sum(L for _, _, _, _, L in edges)
    min_edge = min(L for _, _, _, _, L in edges)

    # Binary search for step_size that yields exactly n_points samples.
    lo = min_edge / (n_points + 1)
    hi = total_length  # very large step → 1 sample (root)

    # Ensure lo actually gives enough points
    if _count_euclidean_samples(edges, lo) < n_points:
        raise ValueError(
            f"Neuron {x.id}: cannot sample {n_points} points "
            f"(max ≈ {_count_euclidean_samples(edges, lo)})."
        )

    for _ in range(64):
        mid = (lo + hi) / 2.0
        cnt = _count_euclidean_samples(edges, mid)
        if cnt == n_points:
            break
        elif cnt > n_points:
            lo = mid  # step too fine → more samples → too many
        else:
            hi = mid  # step too coarse → fewer samples → too few

    pts, parent_ids, child_ids, fractions = _collect_euclidean_points(
        edges, root_id, root_xyz, mid
    )

    # Truncate or pad to exactly n_points (binary search may be ±1 off)
    if len(pts) > n_points:
        pts = pts[:n_points]
        parent_ids = parent_ids[:n_points]
        child_ids = child_ids[:n_points]
        fractions = fractions[:n_points]
    elif len(pts) < n_points:
        pad = n_points - len(pts)
        pts = np.vstack([pts, np.tile(pts[-1], (pad, 1))])
        parent_ids = parent_ids + [parent_ids[-1]] * pad
        child_ids = child_ids + [child_ids[-1]] * pad
        fractions = fractions + [fractions[-1]] * pad

    feat = (
        _extract_features(x, parent_ids, child_ids, fractions, feature_cols)
        if feature_cols is not None
        else None
    )
    return squareform(pdist(pts)).astype(np.float64), feat


# ---------------------------------------------------------------------------
# Internal helpers: Geodesic sampling (mirrors CAJAL's geodesic approach)
# ---------------------------------------------------------------------------


def _build_tree_structure(x: TreeNeuron):
    """Build auxiliary tree data structures for geodesic computations.

    Returns
    -------
    parent_map : dict  {node_id -> parent_id}  (root -> None)
    depth_map  : dict  {node_id -> int hop count from root}
    root_dist  : dict  {node_id -> float cumulative Euclidean dist from root}
    root_id    : int/str
    children   : dict  {node_id -> [child_node_id, ...]}
    xyz        : dict  {node_id -> np.ndarray (3,)}
    """
    nodes = x.nodes.set_index("node_id")
    root_id = x.nodes.loc[x.nodes.parent_id < 0, "node_id"].values[0]

    parent_map = {}
    children = {}
    xyz = {}

    for nid, row in nodes.iterrows():
        xyz[nid] = row[["x", "y", "z"]].values.astype(float)
        children.setdefault(nid, [])
        if row.parent_id < 0:
            parent_map[nid] = None
        else:
            parent_map[nid] = row.parent_id
            children.setdefault(row.parent_id, []).append(nid)

    # BFS from root to compute depth and cumulative root distances
    depth_map = {root_id: 0}
    root_dist = {root_id: 0.0}
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


def _lca_root_dist(a, b, parent_map, depth_map, root_dist):
    """Return root_dist of the lowest common ancestor of nodes *a* and *b*."""
    da, db = depth_map[a], depth_map[b]
    while da > db:
        a = parent_map[a]
        da -= 1
    while db > da:
        b = parent_map[b]
        db -= 1
    while a != b:
        a = parent_map[a]
        b = parent_map[b]
    return root_dist[a]


def _count_geodesic_samples(root_id, children, xyz, step_size: float) -> int:
    """Count samples produced by a DFS traversal with *step_size*.

    CAJAL resets the carry-over distance at each branch point so that each
    sub-branch is sampled independently.  The root counts as one sample.
    """
    count = 1  # root
    # stack: (node_id, carry_distance_since_last_sample_on_this_branch)
    stack = [(root_id, 0.0)]
    while stack:
        nid, carry = stack.pop()
        for cid in children.get(nid, []):
            edge_len = float(np.linalg.norm(xyz[cid] - xyz[nid]))
            dist = carry + edge_len
            n_new = int(dist / step_size)
            count += n_new
            stack.append((cid, dist - n_new * step_size))
    return count


def _collect_geodesic_samples(root_id, children, root_dist, xyz, step_size: float):
    """DFS traversal collecting sampled points.

    Each sample is a tuple
    ``(parent_node_id, child_node_id, fraction, dist_from_root, xyz)``.
    The root is included as the first sample (fraction=0, parent==child==root).
    """
    samples = [(root_id, root_id, 0.0, root_dist[root_id], xyz[root_id])]

    # stack: (node_id, carry, last_sample_node_id)
    # carry = accumulated distance along the current branch since the last sample
    stack = [(root_id, 0.0, root_id)]
    while stack:
        nid, carry, last_sample_node = stack.pop()
        for cid in children.get(nid, []):
            edge_len = float(np.linalg.norm(xyz[cid] - xyz[nid]))
            dist = carry + edge_len
            n_new = int(dist / step_size)
            direction = (
                (xyz[cid] - xyz[nid]) / edge_len if edge_len > 0 else np.zeros(3)
            )

            for k in range(1, n_new + 1):
                # Distance from nid of the k-th new sample on this edge
                dist_from_nid = k * step_size - carry
                dist_from_nid = min(max(dist_from_nid, 0.0), edge_len)
                t = dist_from_nid / edge_len if edge_len > 0 else 0.0
                pt = xyz[nid] + dist_from_nid * direction
                rd = root_dist[nid] + dist_from_nid
                samples.append((nid, cid, t, rd, pt))
                last_sample_node = nid

            stack.append((cid, dist - n_new * step_size, last_sample_node))

    return samples


def _icdm_geodesic(x: TreeNeuron, n_points: int, feature_cols=None):
    """Compute an NxN geodesic intracellular distance matrix.

    Samples ``n_points`` evenly-spaced points along the skeleton's arbor
    using a DFS with binary-searched step_size (mirroring CAJAL's geodesic
    sampling).  The geodesic distance between two sampled points p_i and p_j
    that lie on edges (u→u') and (v→v') respectively is:

        dist(i, j) = rd_i + rd_j - 2 · rd(LCA(u, v))

    where rd is cumulative root distance and LCA is the lowest common ancestor
    of the nearest skeleton nodes.

    Parameters
    ----------
    x :            TreeNeuron
    n_points :     int
    feature_cols : list of str or None

    Returns
    -------
    (np.ndarray (n_points, n_points), np.ndarray (n_points, n_feats) or None)
    """
    if not x.is_tree:
        raise ValueError(
            f"Neuron {x.id} is not a single connected tree. "
            "Use metric='euclidean' or repair/reroot the neuron first."
        )

    parent_map, depth_map, root_dist, root_id, children, xyz = _build_tree_structure(x)

    total_length = sum(
        float(np.linalg.norm(xyz[cid] - xyz[nid]))
        for nid in children
        for cid in children[nid]
    )
    if total_length == 0:
        raise ValueError(f"Neuron {x.id} has zero total cable length.")

    # Binary search for step_size
    lo = total_length / (n_points * 10)
    hi = total_length * 2

    # Make sure lo actually gives enough points
    while _count_geodesic_samples(root_id, children, xyz, lo) < n_points:
        lo /= 2.0

    for _ in range(64):
        mid = (lo + hi) / 2.0
        cnt = _count_geodesic_samples(root_id, children, xyz, mid)
        if cnt == n_points:
            break
        elif cnt > n_points:
            lo = mid
        else:
            hi = mid

    samples = _collect_geodesic_samples(root_id, children, root_dist, xyz, mid)

    # Truncate / pad to exactly n_points
    if len(samples) > n_points:
        samples = samples[:n_points]
    elif len(samples) < n_points:
        samples = samples + [samples[-1]] * (n_points - len(samples))

    # Build N×N geodesic distance matrix
    # dist(i,j) = rd_i + rd_j - 2 * root_dist(LCA(nearest_node_i, nearest_node_j))
    # s = (parent_nid, child_nid, fraction, rd, xyz_pt)
    nearest_nodes = [s[0] for s in samples]
    rd_vals = np.array([s[3] for s in samples])

    N = n_points
    D = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i + 1, N):
            lca_rd = _lca_root_dist(
                nearest_nodes[i], nearest_nodes[j], parent_map, depth_map, root_dist
            )
            d = rd_vals[i] + rd_vals[j] - 2.0 * lca_rd
            D[i, j] = d
            D[j, i] = d

    feat = None
    if feature_cols is not None:
        parent_ids = [s[0] for s in samples]
        child_ids = [s[1] for s in samples]
        fractions = [s[2] for s in samples]
        feat = _extract_features(x, parent_ids, child_ids, fractions, feature_cols)

    return D, feat


# ---------------------------------------------------------------------------
# Internal helpers: feature extraction for FGW
# ---------------------------------------------------------------------------


def _extract_features(x: TreeNeuron, parent_ids, child_ids, fractions, feature_cols):
    """Interpolate node-table feature columns onto sampled points.

    For float columns the value is linearly interpolated between the parent
    and child node of the edge on which each sample lies.  For all other
    column types the parent node's value is used (parent-snap).

    Parameters
    ----------
    x :            TreeNeuron
    parent_ids :   list of node IDs (length N)
    child_ids :    list of node IDs (length N)
    fractions :    list of float    (length N)  — 0=parent end, 1=child end
    feature_cols : list of str

    Returns
    -------
    np.ndarray (N, len(feature_cols)), dtype float64
    """
    nodes = x.nodes.set_index("node_id")
    N = len(parent_ids)
    feat = np.empty((N, len(feature_cols)), dtype=np.float64)

    for k, col in enumerate(feature_cols):
        col_data = nodes[col]
        is_float = col_data.dtype.kind == "f"
        if is_float:
            p_vals = col_data.loc[parent_ids].values.astype(np.float64)
            c_vals = col_data.loc[child_ids].values.astype(np.float64)
            t = np.array(fractions, dtype=np.float64)
            feat[:, k] = (1.0 - t) * p_vals + t * c_vals
        else:
            # Label-encode: convert to pandas Categorical then use integer codes.
            # This handles string, object, and existing Categorical dtypes uniformly.
            cat = col_data.astype("category")
            p_vals = cat.loc[parent_ids].cat.codes.values.astype(np.float64)
            feat[:, k] = p_vals

    return feat


# ---------------------------------------------------------------------------
# Module-level worker for parallel FGW computation
# (must be at module level to be picklable by ProcessPoolExecutor)
# ---------------------------------------------------------------------------


def _fgw_worker(args: tuple) -> float:
    """Compute FGW distance between one pair of (ICDM, features).

    Parameters
    ----------
    args : (D_A, f_A, D_B, f_B, alpha, feature_metric)
        D_A, D_B : globally-normalised ICDMs (np.ndarray, square)
        f_A, f_B : globally-normalised feature matrices (np.ndarray (N, K))
        alpha    : float — GW structural weight (1 = pure GW, 0 = pure features)
        feature_metric : str — scipy cdist metric

    Returns
    -------
    float — FGW distance
    """
    try:
        import ot
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "python-ot is not installed. Install it with:\n\n"
            "    pip install pot\n\n"
            "See https://pythonot.github.io for details."
        )

    D_A, f_A, D_B, f_B, alpha, feature_metric, epsilon = args

    n, m = D_A.shape[0], D_B.shape[0]
    p = np.ones(n, dtype=np.float64) / n
    q = np.ones(m, dtype=np.float64) / m

    M = cdist(f_A, f_B, metric=feature_metric)
    M_max = M.max()
    if M_max > 0:
        M /= M_max

    # Limit BLAS threads per worker to prevent oversubscription when running
    # inside a ProcessPoolExecutor.  Falls back gracefully if threadpoolctl is
    # not installed.
    try:
        from threadpoolctl import threadpool_limits as _tpl
        _thread_ctx = _tpl(limits=1, user_api="blas")
    except ImportError:
        from contextlib import nullcontext
        _thread_ctx = nullcontext()

    with _thread_ctx:
        if epsilon is not None:
            # Entropic (Sinkhorn-regularised) FGW — O(n²) per iteration,
            # faster than exact EMD-based FGW but returns a regularised
            # approximation.  No warm start needed.
            fgw_dist = ot.gromov.entropic_fused_gromov_wasserstein2(
                M, D_A, D_B, p, q, loss_fun="square_loss",
                epsilon=epsilon, alpha=alpha,
            )
        else:
            # Exact FGW warm-started from the GW transport plan (mirrors
            # CAJAL's approach).  The GW plan is a far better initial guess
            # than the uniform outer product, typically halving the number of
            # Frank-Wolfe iterations needed.
            T_gw, _ = ot.gromov.gromov_wasserstein(
                D_A, D_B, p, q, loss_fun="square_loss", log=True
            )
            fgw_dist = ot.gromov.fused_gromov_wasserstein2(
                M, D_A, D_B, p, q, G0=T_gw, alpha=alpha,
            )
    return float(fgw_dist)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cajal_gw(
    query: Union[TreeNeuron, NeuronList],
    target: Optional[Union[TreeNeuron, NeuronList]] = None,
    n_points: int = 50,
    metric: Literal["euclidean", "geodesic"] = "geodesic",
    n_cores: int = max(1, os.cpu_count() // 2),
    progress: bool = True,
    slb: bool = False,
    features: Optional[Union[str, list]] = None,
    alpha: float = 0.5,
    feature_metric: str = "sqeuclidean",
    epsilon: Optional[float] = None,
) -> pd.DataFrame:
    """Compute pairwise Gromov-Wasserstein (GW) distances between neurons.

    Uses CAJAL's GW algorithm (Peyre et al. ICML 2016) to compare neuron
    morphologies in a metric-space-preserving way. Each neuron is first
    reduced to an NxN intracellular distance matrix (ICDM) by sampling
    ``n_points`` evenly-spaced points along its arbor. Pairwise GW distances
    are then computed between all pairs of ICDMs.

    When ``features`` is provided, uses Fused Gromov-Wasserstein (FGW) which
    augments the structural GW cost with a per-point feature cost via POT's
    ``ot.gromov.fused_gromov_wasserstein``.

    Parameters
    ----------
    query :          TreeNeuron | NeuronList
                     Query neuron(s).
    target :         TreeNeuron | NeuronList, optional
                     Target neuron(s).  If not provided, runs an all-by-all
                     comparison of ``query`` against itself.
    n_points :       int, optional
                     Number of sample points per neuron when building the ICDM.
                     More points → higher fidelity but slower GW computation
                     (scales as O(n_points²·N²)).  Default 50 matches CAJAL's
                     recommended range of 50-100.
    metric :         'euclidean' | 'geodesic'
                     Distance metric for the ICDM.
                     ``'euclidean'`` uses straight-line distances between sampled
                     points (fast, ignores topology).
                     ``'geodesic'`` uses along-the-arbor distances (slower but
                     topology-aware; requires a connected tree).
    n_cores :        int, optional
                     Number of parallel processes for GW computation.  Defaults
                     to ``os.cpu_count() // 2``.
    progress :       bool, optional
                     Whether to show a progress bar while computing ICDMs.
                     Note that when `slb=True`, no progress bar is shown.
    slb :            bool, optional
                     If ``True``, compute the Second Lower Bound (SLB) instead of
                     the full Gromov-Wasserstein distance.  The SLB is a fast
                     lower bound to the GW distance (typically computed in seconds
                     rather than minutes) and is useful as a proxy for GW in
                     clustering and dimensionality-reduction workflows.  Only
                     supported for all-by-all comparisons (``target`` must be
                     ``None``).  Incompatible with ``features``.  Defaults to
                     ``False``.
    features :       str | list of str, optional
                     Column name(s) from the skeleton node table to use as
                     per-point features for Fused GW (FGW).  Float columns are
                     linearly interpolated between the two endpoint nodes of the
                     edge on which each sampled point lies; all other column
                     types snap to the parent node's value.  Features and ICDMs
                     are globally normalised before FGW so that ``alpha`` has a
                     consistent scale-invariant meaning.  Requires ``python-ot``
                     (``pip install pot``).
    alpha :          float, optional
                     Structural weight for FGW (only used when ``features`` is
                     set).  ``alpha=1`` is equivalent to pure GW (no feature
                     cost); ``alpha=0`` is pure Wasserstein on the features.
                     Default 0.5.
    feature_metric : str, optional
                     scipy ``cdist`` metric used to build the per-pair feature
                     cost matrix M.  Default ``'sqeuclidean'``.
    epsilon :        float, optional
                     If provided, switches the FGW solver from the exact
                     Frank-Wolfe / EMD solver to POT's entropic
                     (Sinkhorn-regularised) ``entropic_fused_gromov_wasserstein2``
                     with this regularisation strength.  Entropic FGW is
                     typically 3-10x faster (O(n²) vs O(n³) per inner
                     iteration) but returns a regularised approximation rather
                     than the exact FGW distance.  Smaller values (e.g.
                     ``1e-3``) yield closer approximations at the cost of more
                     Sinkhorn iterations; larger values (e.g. ``1e-1``) converge
                     faster but introduce more bias.  Only used when ``features``
                     is set.  Default ``None`` (exact solver).

    Returns
    -------
    pd.DataFrame
                     Matrix of pairwise GW (or FGW / SLB) distances.  Rows are
                     query neurons, columns are target neurons (or both when no
                     target is given).  Labels are neuron ``.id`` values.  For
                     an all-by-all run the matrix is symmetric with zeros on the
                     diagonal.

    References
    ----------
    Peyre G, Cuturi M, Solomon J. Gromov-Wasserstein averaging of kernel and
    distance matrices. ICML 2016.

    Nickel CL et al. CAJAL enables analysis and integration of
    single-cell morphological data using metric geometry. Nat Commun 2023.

    Vayer T et al. Fused Gromov-Wasserstein distance for structured objects.
    Algorithms 2020.

    Examples
    --------
    >>> import navis
    >>> from navis.interfaces.cajal import cajal_gw
    >>> nl = navis.example_neurons(n=5, kind='skeleton')
    >>> # All-by-all GW distances
    >>> scores = cajal_gw(nl[:3], n_points=30)
    >>> scores.shape
    (3, 3)
    >>> # Query-vs-target
    >>> scores = cajal_gw(nl[:2], nl[3:], n_points=30)
    >>> scores.shape
    (2, 2)
    >>> # Fast SLB lower bound (all-by-all only)
    >>> scores = cajal_gw(nl[:3], n_points=30, slb=True)
    >>> scores.shape
    (3, 3)
    >>> # Fused GW with node radius as a feature
    >>> scores = cajal_gw(nl[:3], n_points=30, features='radius')
    >>> scores.shape
    (3, 3)

    See Also
    --------
    [`navis.nblast`][]
            NBLAST: a fast point-cloud-based neuron similarity measure.

    """
    utils.eval_param(metric, name="metric", allowed_values=("euclidean", "geodesic"))

    # Normalise features parameter
    feature_cols: Optional[list] = None
    if features is not None:
        feature_cols = [features] if isinstance(features, str) else list(features)

    # Normalise input
    query_nl = NeuronList(query)
    allbyall = target is None
    target_nl = query_nl if allbyall else NeuronList(target)

    # Validate neuron types
    for nl, label in [(query_nl, "query"), (target_nl, "target")]:
        for n in nl:
            if not isinstance(n, TreeNeuron):
                raise TypeError(
                    f"{label} neuron {n.id!r} is not a TreeNeuron. "
                    "cajal_gw currently supports TreeNeurons only."
                )

    # Require unique IDs
    combined = list(query_nl) + ([] if allbyall else list(target_nl))
    ids = [n.id for n in combined]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate neuron IDs found. All neuron IDs must be unique.")

    # Feature-specific validation (before any heavy imports)
    if feature_cols is not None:
        if slb:
            raise ValueError("slb=True is incompatible with features.")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be between 0 and 1, got {alpha!r}.")
        if epsilon is not None and epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon!r}.")
        for n in combined:
            missing = [c for c in feature_cols if c not in n.nodes.columns]
            if missing:
                raise ValueError(
                    f"Neuron {n.id!r}: column(s) {missing} not found in node table. "
                    f"Available columns: {list(n.nodes.columns)}"
                )

    try:
        from cajal.run_gw import gw_pairwise_parallel, gw_query_target_parallel
        from cajal.utilities import uniform
    except ModuleNotFoundError:
        raise ModuleNotFoundError(dedent("""
            CAJAL is not installed. Install it with:

                pip install cajal

            See https://github.com/CamaraLab/CAJAL for details.
            """).strip())

    # Collect unique neurons to process (each computed once)
    unique_neurons = list(query_nl) if allbyall else list(query_nl) + list(target_nl)
    seen: set = set()
    to_compute = []
    for n in unique_neurons:
        if n.id not in seen:
            seen.add(n.id)
            to_compute.append(n)

    icdm_cache: dict = {}
    feat_cache: dict = {}

    if n_cores > 1 and len(to_compute) > 10:
        # Parallel ICDM computation — uses spawn to keep memory footprint low
        worker_args = [(n, n_points, metric, feature_cols) for n in to_compute]
        with ProcessPoolExecutor(
            max_workers=n_cores,
            mp_context=mp.get_context("spawn"),
        ) as pool:
            futures = {pool.submit(_icdm_worker, a): a[0].id for a in worker_args}
            for f in config.tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Computing ICDMs ({metric})",
                disable=not progress,
                leave=False,
            ):
                nid, D, feat = f.result()
                icdm_cache[nid] = D
                if feat is not None:
                    feat_cache[nid] = feat
    else:
        icdm_fn = _icdm_geodesic if metric == "geodesic" else _icdm_euclidean
        for n in config.tqdm(
            to_compute,
            desc=f"Computing ICDMs ({metric})",
            disable=not progress,
            leave=False,
        ):
            D, feat = icdm_fn(n, n_points, feature_cols=feature_cols)
            icdm_cache[n.id] = D
            if feat is not None:
                feat_cache[n.id] = feat

    def _cell(nid):
        D = icdm_cache[nid]
        return (D, uniform(D.shape[0]))

    query_ids = [n.id for n in query_nl]
    target_ids = [n.id for n in target_nl]

    # FGW dispatch — uses globally-normalised ICDMs and features
    if feature_cols is not None:
        # Global ICDM normalisation
        global_icdm_max = max(D.max() for D in icdm_cache.values())
        if global_icdm_max == 0:
            global_icdm_max = 1.0
        icdm_norm = {nid: D / global_icdm_max for nid, D in icdm_cache.items()}

        # Global per-column feature min-max normalisation
        all_feats = np.vstack(list(feat_cache.values()))
        feat_min = all_feats.min(axis=0)
        feat_range = all_feats.max(axis=0) - feat_min
        feat_range[feat_range == 0] = 1.0
        feat_norm = {
            nid: (f - feat_min) / feat_range for nid, f in feat_cache.items()
        }

        # Build list of (i_id, j_id) pairs to compute
        if allbyall:
            pairs = [
                (query_ids[i], query_ids[j])
                for i in range(len(query_ids))
                for j in range(i + 1, len(query_ids))
            ]
        else:
            pairs = [(qi, ti) for qi in query_ids for ti in target_ids]

        def _make_fgw_arg(qi, ti):
            return (
                icdm_norm[qi], feat_norm[qi],
                icdm_norm[ti], feat_norm[ti],
                alpha, feature_metric, epsilon,
            )

        n_pairs = len(pairs)
        mat = np.zeros((len(query_ids), len(target_ids)), dtype=np.float64)

        if n_cores > 1 and n_pairs > 1:
            fgw_args = [_make_fgw_arg(qi, ti) for qi, ti in pairs]
            with ProcessPoolExecutor(
                max_workers=n_cores,
                mp_context=mp.get_context("spawn"),
            ) as pool:
                fgw_futures = {
                    pool.submit(_fgw_worker, a): idx
                    for idx, a in enumerate(fgw_args)
                }
                for f in config.tqdm(
                    as_completed(fgw_futures),
                    total=n_pairs,
                    desc="Computing FGW distances",
                    disable=not progress,
                    leave=False,
                ):
                    idx = fgw_futures[f]
                    qi, ti = pairs[idx]
                    val = f.result()
                    row = query_ids.index(qi)
                    col = target_ids.index(ti)
                    mat[row, col] = val
                    if allbyall:
                        mat[col, row] = val
        else:
            for qi, ti in config.tqdm(
                pairs,
                desc="Computing FGW distances",
                disable=not progress,
                leave=False,
            ):
                val = _fgw_worker(_make_fgw_arg(qi, ti))
                row = query_ids.index(qi)
                col = target_ids.index(ti)
                mat[row, col] = val
                if allbyall:
                    mat[col, row] = val

    elif slb:
        if not allbyall:
            raise ValueError(
                "slb=True only supports all-by-all comparisons; "
                "target must be None when slb=True."
            )
        from cajal.qgw import slb_parallel_memory
        cell_dms = [icdm_cache[nid] for nid in query_ids]
        cell_dists = [uniform(D.shape[0]) for D in cell_dms]
        mat = slb_parallel_memory(cell_dms, cell_dists, num_processes=n_cores)
    elif allbyall:
        cells = [_cell(nid) for nid in query_ids]
        mat, _ = gw_pairwise_parallel(cells, num_processes=n_cores)
    else:
        queries = [_cell(nid) for nid in query_ids]
        targets = [_cell(nid) for nid in target_ids]
        mat, _ = gw_query_target_parallel(queries, targets, num_processes=n_cores)

    df = pd.DataFrame(mat, index=query_ids, columns=target_ids)
    df.index.name = "query"
    df.columns.name = "target"
    return df

