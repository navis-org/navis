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
import pandas as pd

from .. import config, core, utils

# Set up logging
logger = config.get_logger(__name__)

# Only `sample_skeleton` is a top-level `navis.*` function. `sample_cable` and
# `sample_surface` are defined here but exposed solely via `navis.ml` (imported
# explicitly there), so they are deliberately kept out of `__all__`.
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
    [`navis.ml.sample_cable`][]
                Arclength-uniform sampling that also interpolates radius/other
                node attributes and randomizes (stratified) - the richer sibling
                for preparing model inputs.
    [`navis.resample_skeleton`][]
                Resample a skeleton to a target node spacing (returns a neuron).
    [`navis.downsample_neuron`][]
                Reduce node count while preserving topology.

    """
    if not isinstance(x, core.TreeNeuron):
        raise TypeError(f'sample_skeleton requires a TreeNeuron, got {type(x)}.')

    return _ss_sample_points(x, n_points)


# ---------------------------------------------------------------------------
# Arclength-uniform cable sampling with attribute transfer
# ---------------------------------------------------------------------------


# Node columns that describe structure/geometry rather than a per-node feature -
# excluded when `interpolate=True` grabs "everything".
_STRUCTURAL_COLS = {"node_id", "parent_id", "x", "y", "z", "type"}


@utils.map_neuronlist(desc="Sampling", allow_parallel=True)
def sample_cable(
    x: "core.NeuronObject",
    n_points: int,
    interpolate=None,
    weights=None,
    random_state=None,
) -> "pd.DataFrame":
    """Sample points at uniform arclength along a skeleton, carrying attributes.

    Skeleton nodes are rarely spaced evenly along the cable, so drawing nodes at
    random over-weights densely-noded regions. This instead samples points at
    uniform arclength along the *edges* - the skeleton analogue of area-weighted
    surface sampling (see [`navis.ml.sample_surface`][]) - and interpolates node
    attributes onto each sample. Draws are **stratified** (one jittered sample per
    equal-measure bin), giving exactly even coverage rather than the clumping of
    i.i.d. draws.

    This is the "resample cable -> point cloud + features" primitive for feeding
    skeletons to a model: unlike [`navis.sample_skeleton`][] it interpolates
    `radius`/other columns and records provenance, and unlike
    [`navis.ml.sample_points_uniform`][] it samples *new* points along the cable
    rather than subsampling existing nodes.

    Parameters
    ----------
    x :             TreeNeuron | NeuronList
    n_points :      int
                    Number of points to draw.
    interpolate :   None | True | str | list of str
                    Node columns to attach to each sample:

                    - **float** columns (e.g. `radius`) are linearly interpolated
                      between the edge's two endpoints;
                    - **integer/categorical** columns (e.g. a label) are carried
                      from the segment's source node (the distal/child end, which
                      owns the segment).

                    `True` attaches every non-structural column. `None` (default)
                    attaches none - you still get `source_id` for provenance.
    weights :       None | str | (n_nodes,) array-like
                    Per-node weighting of the sampling density. Each edge's sampling
                    measure is ``length * mean(weight)`` over its endpoints, so:

                    - None (default): measure = length -> uniform along the cable;
                    - ``"radius"``: measure ~ length x radius -> denser on thick
                      cable (a lateral-surface-area-like weighting);
                    - any column/array: your own measure.

                    Must be non-negative.
    random_state :  int | np.random.Generator, optional
                    Seed/generator for the stratified jitter. Omit for a fresh
                    random tiling each call; pass a seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
                    One row per sample with columns ``x``, ``y``, ``z``, then any
                    requested `interpolate` columns, then ``source_id`` (the
                    ``node_id`` of the segment's source/child node - use it to join
                    any other per-node attribute). A `NeuronList` returns a list of
                    such DataFrames.

    See Also
    --------
    [`navis.ml.sample_surface`][]
                    The mesh analogue: area-weighted surface sampling with
                    per-vertex provenance.
    [`navis.sample_skeleton`][]
                    Deterministic equal-spacing sampling (points only, no
                    attributes).
    [`navis.ml.sample_points_uniform`][]
                    Uniformly subsample an existing point cloud.

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1, kind="skeleton")
    >>> pts = navis.ml.sample_cable(n, 1000, interpolate="radius", random_state=0)
    >>> pts.shape
    (1000, 5)
    >>> list(pts.columns)
    ['x', 'y', 'z', 'radius', 'source_id']

    """
    if not isinstance(x, core.TreeNeuron):
        raise TypeError(f"sample_cable requires a TreeNeuron, got {type(x)}.")
    if int(n_points) != n_points or n_points <= 0:
        raise ValueError(f"`n_points` must be a positive integer, got {n_points!r}")
    n_points = int(n_points)

    nodes = x.nodes
    cols = _resolve_attr_cols(nodes, interpolate)
    rng = np.random.default_rng(random_state)

    pos = nodes[["x", "y", "z"]].to_numpy(np.float64)
    node_id = nodes["node_id"].to_numpy()

    # Edges are child -> parent. Map parent ids to row indices.
    row_of_id = pd.Series(np.arange(len(nodes)), index=node_id)
    parent_row = nodes["parent_id"].map(row_of_id)
    has_parent = parent_row.notna().to_numpy()
    child = np.nonzero(has_parent)[0]
    parent = parent_row.to_numpy()[has_parent].astype(np.int64)

    seg = pos[child] - pos[parent]
    length = np.linalg.norm(seg, axis=1)

    w = _resolve_weights(nodes, weights)
    if w is None:
        measure = length.copy()
    else:
        measure = length * 0.5 * (w[child] + w[parent])

    keep = measure > 0
    child, parent, seg, measure = child[keep], parent[keep], seg[keep], measure[keep]

    if len(measure) == 0 or measure.sum() == 0:
        # Degenerate skeleton (no positive-measure cable): fall back to the nodes.
        logger.warning(
            f"Neuron {x.id} has no positive-length/weight cable; sampling nodes at random."
        )
        sel = rng.integers(0, len(nodes), n_points)
        return _cable_frame(pos[sel], nodes, cols, sel, sel, np.zeros(n_points), node_id)

    cum = np.cumsum(measure)
    total = cum[-1]

    # Stratified (jittered-grid) draw over the cumulative measure: exactly one
    # sample per equal-measure bin -> even coverage, unlike i.i.d. draws.
    s = (np.arange(n_points) + rng.random(n_points)) / n_points * total
    e = np.clip(np.searchsorted(cum, s, side="right"), 0, len(measure) - 1)
    start = cum[e] - measure[e]
    t = np.where(measure[e] > 0, (s - start) / measure[e], 0.0)
    t = np.clip(t, 0.0, 1.0)

    points = pos[parent[e]] + t[:, None] * seg[e]
    return _cable_frame(points, nodes, cols, child[e], parent[e], t, node_id)


def _resolve_attr_cols(nodes, interpolate):
    """Resolve the `interpolate` argument to a list of existing node columns."""
    if interpolate is None:
        return []
    if interpolate is True:
        return [c for c in nodes.columns if c not in _STRUCTURAL_COLS]
    cols = [interpolate] if isinstance(interpolate, str) else list(interpolate)
    missing = [c for c in cols if c not in nodes.columns]
    if missing:
        raise ValueError(f"`interpolate` columns not in nodes table: {missing}")
    return cols


def _resolve_weights(nodes, weights):
    """Resolve `weights` to a per-node non-negative float array (or None)."""
    if weights is None:
        return None
    if isinstance(weights, str):
        if weights not in nodes.columns:
            raise ValueError(f"`weights` column {weights!r} not in nodes table.")
        w = nodes[weights].to_numpy(np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (len(nodes),):
            raise ValueError(
                f"`weights` array must have one value per node ({len(nodes)}), "
                f"got shape {w.shape}."
            )
    if np.any(w < 0) or np.any(~np.isfinite(w)):
        raise ValueError("`weights` must be finite and non-negative.")
    return w


def _cable_frame(points, nodes, cols, child_rows, parent_rows, t, node_id):
    """Assemble the output DataFrame: x/y/z, interpolated/carried cols, source_id."""
    out = {
        "x": points[:, 0],
        "y": points[:, 1],
        "z": points[:, 2],
    }
    for c in cols:
        vals = nodes[c].to_numpy()
        if pd.api.types.is_float_dtype(nodes[c]):
            # Linear interpolation between the edge's endpoints.
            out[c] = vals[parent_rows] * (1 - t) + vals[child_rows] * t
        else:
            # Carry from the source (child) node - the distal end owns the segment.
            out[c] = vals[child_rows]
    out["source_id"] = node_id[child_rows]
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Area-weighted surface sampling with per-vertex provenance
# ---------------------------------------------------------------------------


@utils.map_neuronlist(desc="Sampling", allow_parallel=True)
def sample_surface(
    x: "core.NeuronObject",
    n_points: int,
    mode: str = "even",
    attributes=None,
    random_state=None,
) -> "pd.DataFrame":
    """Sample points on a mesh surface, keeping each point's source vertex.

    Mesh vertex density tracks surface detail rather than the arbor, so meshes are
    best sampled by **surface area**. Each sample records the nearest source vertex,
    so per-vertex labels/attributes transfer to the sampled cloud - the mesh
    analogue of [`navis.ml.sample_cable`][].

    Parameters
    ----------
    x :             MeshNeuron | NeuronList
    n_points :      int
                    Number of points to draw.
    mode :          "even" | "surface" | "vertex"
                    - "even" (default): area-weighted *even* (blue-noise) sampling
                      via `trimesh`, topped up with plain area-weighted samples if
                      it under-delivers (it does, by design).
                    - "surface": plain area-weighted random sampling.
                    - "vertex": draw existing vertices (without replacement while
                      enough remain).
    attributes :    dict | pandas.DataFrame, optional
                    Per-vertex values to transfer onto the samples (length =
                    number of vertices). Each becomes a column, taken from each
                    sample's source vertex. Intended for single meshes.
    random_state :  int | np.random.Generator, optional

    Returns
    -------
    pandas.DataFrame
                    One row per sample with columns ``x``, ``y``, ``z``, any
                    transferred `attributes`, ``source_id`` (source vertex index -
                    join per-vertex data with it) and ``face`` (source face index,
                    ``-1`` in "vertex" mode). A `NeuronList` returns a list of such
                    DataFrames.

    See Also
    --------
    [`navis.ml.sample_cable`][]
                    The skeleton analogue: arclength-uniform sampling with
                    attribute interpolation.

    Examples
    --------
    >>> import navis
    >>> import numpy as np
    >>> m = navis.example_neurons(1, kind="mesh")
    >>> pts = navis.ml.sample_surface(m, 1000, random_state=0)
    >>> pts.shape
    (1000, 5)
    >>> list(pts.columns)
    ['x', 'y', 'z', 'source_id', 'face']
    >>> # transfer a per-vertex label:
    >>> lab = np.zeros(len(m.vertices), dtype=int)
    >>> pts = navis.ml.sample_surface(m, 1000, attributes={"label": lab}, random_state=0)
    >>> "label" in pts.columns
    True

    """
    import trimesh

    if not isinstance(x, core.MeshNeuron):
        raise TypeError(f"sample_surface requires a MeshNeuron, got {type(x)}.")
    if int(n_points) != n_points or n_points <= 0:
        raise ValueError(f"`n_points` must be a positive integer, got {n_points!r}")
    n_points = int(n_points)
    if mode not in ("even", "surface", "vertex"):
        raise ValueError(f'`mode` must be "even", "surface" or "vertex", got {mode!r}')

    verts = np.asarray(x.vertices, dtype=np.float64)
    faces = np.asarray(x.faces)
    rng = np.random.default_rng(random_state)

    if mode == "vertex":
        V = len(verts)
        if V >= n_points:
            idx = rng.choice(V, n_points, replace=False)
        else:
            idx = rng.integers(0, V, n_points)
        points = verts[idx]
        source_id = idx.astype(np.int64)
        face = np.full(n_points, -1, dtype=np.int64)
    else:
        if len(faces) == 0:
            raise ValueError(
                f"MeshNeuron {x.id} has no faces; use mode='vertex' to sample vertices."
            )
        # process=False keeps vertex indices aligned with `x.vertices` so the
        # returned `source_id` indexes straight into per-vertex attributes.
        tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        points, face = _sample_surface_pts(tm, n_points, mode, rng)
        # Provenance: nearest of the source face's three corners.
        tri = faces[face]
        d = ((verts[tri] - points[:, None, :]) ** 2).sum(-1)
        source_id = tri[np.arange(len(tri)), d.argmin(1)].astype(np.int64)

    out = {"x": points[:, 0], "y": points[:, 1], "z": points[:, 2]}
    for name, vals in _resolve_vertex_attrs(attributes, len(verts)).items():
        out[name] = vals[source_id]
    out["source_id"] = source_id
    out["face"] = face
    return pd.DataFrame(out)


def _sample_surface_pts(tm, n_points, mode, rng):
    """Return (points, face_idx) from a trimesh via the requested `mode`."""
    import logging

    import trimesh

    seed = int(rng.integers(0, 2**31 - 1))
    if mode == "even":
        # `sample_surface_even` under-delivers by design and logs a warning about
        # it; we top up the shortfall ourselves, so silence that expected noise.
        tlog = logging.getLogger("trimesh")
        prev = tlog.level
        tlog.setLevel(logging.ERROR)
        try:
            pts, face_idx = trimesh.sample.sample_surface_even(tm, n_points, seed=seed)
        finally:
            tlog.setLevel(prev)
        if len(pts) < n_points:
            extra_pts, extra_faces = trimesh.sample.sample_surface(
                tm, n_points - len(pts), seed=seed
            )
            pts = np.vstack([pts, extra_pts])
            face_idx = np.concatenate([face_idx, extra_faces])
    else:  # "surface"
        pts, face_idx = trimesh.sample.sample_surface(tm, n_points, seed=seed)

    return np.asarray(pts, dtype=np.float64)[:n_points], np.asarray(face_idx)[:n_points]


def _resolve_vertex_attrs(attributes, n_verts):
    """Coerce `attributes` to an ordered {name: (n_verts,) array} mapping."""
    if attributes is None:
        return {}
    if isinstance(attributes, pd.DataFrame):
        items = [(c, attributes[c].to_numpy()) for c in attributes.columns]
    elif isinstance(attributes, dict):
        items = [(k, np.asarray(v)) for k, v in attributes.items()]
    else:
        raise TypeError("`attributes` must be a dict or DataFrame of per-vertex values.")
    out = {}
    for name, vals in items:
        if vals.shape[0] != n_verts:
            raise ValueError(
                f"`attributes[{name!r}]` has {vals.shape[0]} values but the mesh has "
                f"{n_verts} vertices."
            )
        out[name] = vals
    return out
