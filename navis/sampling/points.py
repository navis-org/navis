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


def _ss_build_tree(x: 'core.Skeleton'):
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


def _ss_total_length(children, xyz) -> float:
    """Total cable length (sum of edge lengths) of the arbor."""
    return sum(
        float(np.linalg.norm(xyz[cid] - xyz[nid]))
        for nid in children for cid in children[nid]
    )


def _ss_place_points(root_id, children, xyz, step_size: float) -> np.ndarray:
    """Place points at `step_size` arclength intervals via DFS carry-over.

    Points on each edge are placed at ``k * step_size - carry`` from the parent
    node, where *carry* is the leftover distance from the previous edge in the DFS
    traversal. This guarantees that consecutive samples are exactly ``step_size``
    apart along any root-to-leaf path. The number of points returned depends on the
    arbor and the step size.
    """
    pts = [xyz[root_id]]
    stack = [(root_id, 0.0)]
    while stack:
        nid, carry = stack.pop()
        for cid in children.get(nid, []):
            edge_len  = float(np.linalg.norm(xyz[cid] - xyz[nid]))
            dist      = carry + edge_len
            n_new     = int(dist / step_size)
            direction = (xyz[cid] - xyz[nid]) / edge_len if edge_len > 0 else np.zeros(3)
            for k in range(1, n_new + 1):
                d = min(max(k * step_size - carry, 0.0), edge_len)
                pts.append(xyz[nid] + d * direction)
            stack.append((cid, dist - n_new * step_size))
    return np.array(pts, dtype=np.float64)


def _ss_sample_n(root_id, children, xyz, total: float, n_points: int) -> np.ndarray:
    """Sample exactly `n_points` at equal spacing along the arbor.

    Binary-searches the step size whose equal-spacing placement yields `n_points`,
    then pads/truncates any off-by-a-few to land on exactly `n_points`.
    """
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

    pts = _ss_place_points(root_id, children, xyz, mid)
    if len(pts) > n_points:
        pts = pts[:n_points]
    elif len(pts) < n_points:
        pts = np.vstack([pts, np.tile(pts[-1], (n_points - len(pts), 1))])
    return pts


# ---------------------------------------------------------------------------


@utils.map_neuronlist(desc='Sampling', allow_parallel=True)
def sample_skeleton(
    x: 'core.NeuronObject',
    *,
    n_points: int = None,
    density: float = None,
    spacing: float = None,
) -> np.ndarray:
    """Sample points at equal spacing along a skeleton.

    Points are drawn at equal spacing along the arbor using a DFS traversal
    with carry-over: the leftover distance at the end of each edge is passed
    to the next edge, ensuring consecutive samples are exactly one step apart
    along every root-to-leaf path.

    How many points to draw is set by **exactly one** of three mutually-exclusive
    knobs - a skeleton's cable length ties count and spacing together, so you can
    only hold one constant:

    - **`n_points`** fixes the *count*: every neuron yields the same number of
      points, so the step size varies with total cable length. Use when a model
      needs fixed-size inputs.
    - **`density`** fixes points *per unit cable length* (count = ``round(density *
      length)``, so it varies with the neuron). Use when the sampling resolution
      must mean the same physical thing across neurons.
    - **`spacing`** fixes the *step size* directly (the exact arclength between
      consecutive samples). The count then falls out of the arbor and varies. For a
      1-D cable spacing and density are reciprocals (``density = 1 / spacing``).

    Parameters
    ----------
    x :         Skeleton | NeuronList
                Neuron(s) to sample.
    n_points :  int, optional
                Fixed number of points to draw. Mutually exclusive with
                `density`/`spacing` - give exactly one of the three.
    density :   float, optional
                Target points per unit cable length; count = ``round(density *
                length)``.
    spacing :   float, optional
                Target arclength between consecutive samples (the step size).

    Returns
    -------
    np.ndarray (n, 3)
                XYZ coordinates of sampled points. `n` is `n_points` for the
                `n_points`/`density` knobs and arbor-dependent (variable) for
                `spacing`. If `x` is a NeuronList, returns a list of such arrays -
                one per neuron.

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

    >>> # fixed step size instead of fixed count (count varies with cable length):
    >>> pts = navis.sample_skeleton(n, spacing=1000)
    >>> pts.shape[1]
    3

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
    if not isinstance(x, core.Skeleton):
        raise TypeError(f'sample_skeleton requires a Skeleton, got {type(x)}.')
    which = _one_of(n_points, density, spacing)

    _, _, _, root_id, children, xyz = _ss_build_tree(x)
    total = _ss_total_length(children, xyz)
    if total == 0:
        raise ValueError(f"Neuron {x.id} has zero total cable length.")

    if which == "spacing":
        # Exact step size -> honour it directly; the count falls out of the arbor.
        return _ss_place_points(root_id, children, xyz, _check_positive_number(spacing, "spacing"))

    n_points = _count_from_measure(n_points, density, total)
    return _ss_sample_n(root_id, children, xyz, total, n_points)


# ---------------------------------------------------------------------------
# Arclength-uniform cable sampling with attribute transfer
# ---------------------------------------------------------------------------


# Node columns that describe structure/geometry rather than a per-node feature -
# excluded when `interpolate=True` grabs "everything".
_STRUCTURAL_COLS = {"node_id", "parent_id", "x", "y", "z", "type"}


@utils.map_neuronlist(desc="Sampling", allow_parallel=True)
def sample_cable(
    x: "core.NeuronObject",
    *,
    n_points: int = None,
    density: float = None,
    spacing: float = None,
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

    How many points to draw is set by **exactly one** of three mutually-exclusive
    knobs (the cable length ties count and spacing together):

    - **`n_points`** fixes the *count* - same number of points for every neuron
      (spacing then varies with cable length). Use for fixed-size model inputs.
    - **`density`** fixes points *per unit cable length* (count = ``round(density *
      length)``). Based on raw arclength; `weights` only redistribute the points.
    - **`spacing`** fixes the *mean* arclength between samples (count = ``round(
      length / spacing)``; ``density = 1 / spacing``). Because draws are stratified
      (not rejection-based) this is an average, not a hard minimum - for a strict
      equal-spacing guarantee use [`navis.sample_skeleton`][] with `spacing`.

    Unlike [`navis.ml.sample_surface`][], all three knobs here yield a deterministic
    count (the stratified sampler always fills exactly its bins).

    Parameters
    ----------
    x :             Skeleton | NeuronList
    n_points :      int, optional
                    Fixed number of points to draw. Mutually exclusive with
                    `density`/`spacing` - give exactly one of the three.
    density :       float, optional
                    Target points per unit cable length; count = ``round(density *
                    length)``.
    spacing :       float, optional
                    Target mean arclength between samples; count = ``round(length /
                    spacing)``.
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
    >>> pts = navis.ml.sample_cable(n, n_points=1000, interpolate="radius", random_state=0)
    >>> pts.shape
    (1000, 5)
    >>> list(pts.columns)
    ['x', 'y', 'z', 'radius', 'source_id']

    """
    if not isinstance(x, core.Skeleton):
        raise TypeError(f"sample_cable requires a Skeleton, got {type(x)}.")
    if _one_of(n_points, density, spacing) == "spacing":
        # spacing and density are reciprocal along a 1-D cable, so convert now and
        # let the shared `_count_from_measure` (below) resolve the count either way.
        density = 1.0 / _check_positive_number(spacing, "spacing")
    # The count is resolved below, once total cable length is known.

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

    # `density` sets the count from total cable length; `weights` then only
    # redistribute those points (they don't change how many there are).
    n_points = _count_from_measure(n_points, density, float(length.sum()))

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
    *,
    n_points: int = None,
    density: float = None,
    spacing: float = None,
    mode: str = "even",
    attributes=None,
    random_state=None,
) -> "pd.DataFrame":
    """Sample points on a mesh surface, keeping each point's source vertex.

    Mesh vertex density tracks surface detail rather than the arbor, so meshes are
    best sampled by **surface area**. Each sample records the nearest source vertex,
    so per-vertex labels/attributes transfer to the sampled cloud - the mesh
    analogue of [`navis.ml.sample_cable`][].

    How many points to draw is set by **exactly one** of three mutually-exclusive
    knobs - a mesh's area ties count and density together, so you can only hold one
    constant:

    - **`n_points`** fixes the *count*: every mesh yields the same number of points,
      so the achieved density varies with mesh size (a big neuron is sampled
      sparsely, a small one densely). Use when a model needs fixed-size inputs.
    - **`density`** fixes points *per unit surface area* (count = ``round(density *
      area)``, so the count varies with mesh size). Use when local neighbourhoods
      (k-NN / ball queries) must span the same *physical* scale across neurons.
    - **`spacing`** fixes the minimum *inter-point distance* (mode="even" only).
      Poisson-disk rejection thins the sample to honour it, so the count varies and
      may fall a little below the internal target rather than diluting the spacing
      guarantee by topping up. Related to density by ``density ~ 1 / spacing**2``.

    The achieved ``area``/``density``/``spacing`` are recorded on the returned
    frame's ``.attrs`` (measure any cloud with [`navis.ml.estimate_spacing`][]).

    Parameters
    ----------
    x :             Mesh | NeuronList
    n_points :      int, optional
                    Fixed number of points to draw. Mutually exclusive with
                    `density`/`spacing` - give exactly one of the three.
    density :       float, optional
                    Target points per unit surface area; count = ``round(density *
                    area)``. Requires a mesh with faces.
    spacing :       float, optional
                    Target minimum inter-point distance (mode="even" only), enforced
                    by Poisson-disk rejection. Requires a mesh with faces.
    mode :          "even" | "surface" | "vertex"
                    - "even" (default): area-weighted *even* (blue-noise) sampling
                      via `trimesh`. With `n_points`/`density` it is topped up with
                      plain area-weighted samples if it under-delivers (it does, by
                      design); with `spacing` it is **not** topped up.
                    - "surface": plain area-weighted random sampling.
                    - "vertex": draw existing vertices (without replacement while
                      enough remain). Not compatible with `spacing`.
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
                    ``-1`` in "vertex" mode). ``df.attrs`` carries the achieved
                    ``area``/``density``/``spacing``. A `NeuronList` returns a list
                    of such DataFrames.

    See Also
    --------
    [`navis.ml.sample_cable`][]
                    The skeleton analogue: arclength-uniform sampling with
                    attribute interpolation.
    [`navis.ml.estimate_spacing`][]
                    Measure the nearest-neighbour spacing (hence density) of an
                    existing cloud - the inverse of the `spacing`/`density` knobs.

    Examples
    --------
    >>> import navis
    >>> import numpy as np
    >>> m = navis.example_neurons(1, kind="mesh")
    >>> pts = navis.ml.sample_surface(m, n_points=1000, random_state=0)
    >>> pts.shape
    (1000, 5)
    >>> list(pts.columns)
    ['x', 'y', 'z', 'source_id', 'face']
    >>> # constant density instead of constant count (count now varies with area):
    >>> pts = navis.ml.sample_surface(m, density=1e-5, random_state=0)
    >>> len(pts)
    644
    >>> bool(pts.attrs["density"] > 0)
    True
    >>> # transfer a per-vertex label:
    >>> lab = np.zeros(len(m.vertices), dtype=int)
    >>> pts = navis.ml.sample_surface(m, n_points=1000, attributes={"label": lab}, random_state=0)
    >>> "label" in pts.columns
    True

    """
    import trimesh

    if not isinstance(x, core.Mesh):
        raise TypeError(f"sample_surface requires a Mesh, got {type(x)}.")
    if mode not in ("even", "surface", "vertex"):
        raise ValueError(f'`mode` must be "even", "surface" or "vertex", got {mode!r}')

    verts = np.asarray(x.vertices, dtype=np.float64)
    faces = np.asarray(x.faces)
    rng = np.random.default_rng(random_state)

    # Build the trimesh once (reused for the surface area and for area-based
    # sampling). process=False keeps vertex indices aligned with `x.vertices` so
    # the returned `source_id` indexes straight into per-vertex attributes.
    tm = None
    area = None
    if len(faces):
        tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        area = float(tm.area)

    # Resolve the mutually-exclusive count/density/spacing knobs into a concrete
    # request: how many points, an optional Poisson-disk radius, and whether to top
    # up to hit the count exactly.
    n_points, radius, top_up = _resolve_surface_target(
        area, n_points, density, spacing, mode, x.id
    )

    if mode == "vertex":
        V = len(verts)
        if V >= n_points:
            idx = rng.choice(V, n_points, replace=False)
        else:
            idx = rng.integers(0, V, n_points)
        points = verts[idx]
        source_id = idx.astype(np.int64)
        face = np.full(len(points), -1, dtype=np.int64)
    else:
        if tm is None:
            raise ValueError(
                f"Mesh {x.id} has no faces; use mode='vertex' to sample vertices."
            )
        points, face = _sample_surface_pts(tm, n_points, mode, rng, radius, top_up)
        # Provenance: nearest of the source face's three corners.
        tri = faces[face]
        d = ((verts[tri] - points[:, None, :]) ** 2).sum(-1)
        source_id = tri[np.arange(len(tri)), d.argmin(1)].astype(np.int64)

    out = {"x": points[:, 0], "y": points[:, 1], "z": points[:, 2]}
    for name, vals in _resolve_vertex_attrs(attributes, len(verts)).items():
        out[name] = vals[source_id]
    out["source_id"] = source_id
    out["face"] = face
    df = pd.DataFrame(out)
    _attach_density(df, area)
    return df


def _resolve_surface_target(area, n_points, density, spacing, mode, neuron_id):
    """Resolve the mutually-exclusive count/density/spacing knobs.

    Returns ``(n_points, radius, top_up)``:

    - ``n_points`` - number of samples to request from the sampler,
    - ``radius``   - Poisson-disk minimum spacing to enforce ("even" mode only), or
      None,
    - ``top_up``   - whether to backfill any shortfall to hit ``n_points`` exactly.
    """
    which = _one_of(n_points, density, spacing)
    if which in ("density", "spacing") and area is None:
        raise ValueError(
            f"Mesh {neuron_id} has no faces, so its surface area is undefined; "
            f"`{which}` needs it. Pass `n_points` instead."
        )

    if which != "spacing":
        return _count_from_measure(n_points, density, area), None, True

    if mode != "even":
        raise ValueError(
            '`spacing` enforces a minimum inter-point distance and is only available '
            f'for mode="even" (got mode={mode!r}).'
        )
    spacing = _check_positive_number(spacing, "spacing")
    # trimesh relates area, count and rejection radius by ``radius = sqrt(area/(3n))``.
    # Invert it to budget the count that fills the surface at this spacing; Poisson-
    # disk rejection then thins it to the requested minimum distance. No top-up, so
    # the spacing guarantee holds and the returned count varies with the mesh.
    n = max(1, int(np.ceil(area / (3.0 * spacing ** 2))))
    return n, spacing, False


def _one_of(n_points, density, spacing):
    """Validate that exactly one of the three knobs is set; return its name.

    Shared by the mesh (`sample_surface`) and skeleton (`sample_skeleton`,
    `sample_cable`) samplers - the count, density and spacing are tied together by
    the neuron's size, so exactly one may be held constant.
    """
    given = [
        name
        for name, val in (("n_points", n_points), ("density", density), ("spacing", spacing))
        if val is not None
    ]
    if len(given) != 1:
        raise ValueError(
            "Provide exactly one of `n_points`, `density` or `spacing` (got "
            f"{given or 'none'}). They are mutually exclusive: `n_points` fixes the "
            "count, `density` fixes points per unit size, `spacing` fixes the "
            "inter-point distance."
        )
    return given[0]


def _check_positive_int(value, name):
    """Validate that `value` is a positive integer, returning it as an int."""
    if int(value) != value or value <= 0:
        raise ValueError(f"`{name}` must be a positive integer, got {value!r}")
    return int(value)


def _check_positive_number(value, name):
    """Validate that `value` is a positive, finite number, returning it as a float."""
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"`{name}` must be positive and finite, got {value!r}")
    return float(value)


def _count_from_measure(n_points, density, measure):
    """Resolve the `n_points` (fixed count) or `density` (points per unit `measure`)
    knob to a sample count. `measure` is the neuron's size in the relevant dimension
    (cable length or surface area). Shared by all three samplers; `spacing` is left
    per-sampler because its geometry differs (exact step / 1-D reciprocal / 2-D
    Poisson radius).
    """
    if n_points is not None:
        return _check_positive_int(n_points, "n_points")
    return max(1, int(round(_check_positive_number(density, "density") * measure)))


def _attach_density(df, area):
    """Record the achieved sampling density/spacing on `df.attrs` (best-effort).

    Stored as metadata (may be dropped by later pandas ops). Reporting what was
    actually produced matters now that `density`/`spacing` let the returned count
    vary with mesh size. Skipped when the surface area is unknown (no faces).
    """
    if area is None or area <= 0 or len(df) == 0:
        return
    n = len(df)
    df.attrs["area"] = float(area)
    df.attrs["density"] = n / float(area)            # points per unit area
    df.attrs["spacing"] = float(np.sqrt(area / n))   # ~ inter-point distance


def _sample_surface_pts(tm, n_points, mode, rng, radius=None, top_up=True):
    """Return (points, face_idx) from a trimesh via the requested `mode`.

    `radius` ("even" mode only) enforces a Poisson-disk minimum spacing. When
    `top_up` is True any shortfall is backfilled with plain area-weighted samples so
    exactly `n_points` come back; when False (spacing mode) the shortfall is kept so
    the spacing guarantee is not diluted and fewer than `n_points` may be returned.
    """
    import logging

    import trimesh

    seed = int(rng.integers(0, 2**31 - 1))
    if mode == "even":
        # `sample_surface_even` under-delivers by design and logs a warning about
        # it; we handle the shortfall ourselves, so silence that expected noise.
        tlog = logging.getLogger("trimesh")
        prev = tlog.level
        tlog.setLevel(logging.ERROR)
        try:
            pts, face_idx = trimesh.sample.sample_surface_even(
                tm, n_points, radius=radius, seed=seed
            )
        finally:
            tlog.setLevel(prev)
        if top_up and len(pts) < n_points:
            extra_pts, extra_faces = trimesh.sample.sample_surface(
                tm, n_points - len(pts), seed=seed
            )
            pts = np.vstack([pts, extra_pts])
            face_idx = np.concatenate([face_idx, extra_faces])
    else:  # "surface"
        pts, face_idx = trimesh.sample.sample_surface(tm, n_points, seed=seed)

    pts = np.asarray(pts, dtype=np.float64)
    face_idx = np.asarray(face_idx)
    if top_up:
        # Cap to the exact count (even mode can slightly overshoot after top-up).
        pts, face_idx = pts[:n_points], face_idx[:n_points]
    return pts, face_idx


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
