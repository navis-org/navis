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
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import csgraph
from scipy.spatial import cKDTree

from .. import core, config
from ..graph.graph_utils import _cluster_graph
from ..sampling.points import (
    sample_surface,
    sample_cable,
    _check_positive_int,
    _check_positive_number,
)

logger = config.get_logger(__name__)

__all__ = ["chunk_neuron", "sample_patches"]


def chunk_neuron(
    x,
    size,
    mode="partition",
    connected=True,
    k=None,
    sampling_factor=1,
    weight="weight",
    undersized="pad",
    pad_value=-1,
    random_state=None,
):
    """Break a neuron into evenly sized fragments.

    This is useful for batching neurons into fixed-size inputs for a neural network.

    Parameters
    ----------
    x :         TreeNeuron | MeshNeuron
    size :      int
                Number of nodes (skeleton) / vertices (mesh) per fragment.
    mode :      "partition" | "cover" | "random" | "spaced"
                - "partition": non-overlapping; the remainder that can't fill a
                  whole fragment is handled per `undersized`.
                - "cover": overlapping; cover every node at least once.
                - "random": fragments from random seeds (may overlap).
                - "spaced": like "random" but seeds are evenly spread by
                  farthest-point sampling instead of drawn at random.
    connected : bool
                If True (default) grow along the arbor by geodesic distance -
                every fragment is connected. If False grow by Euclidean distance
                (KD-tree over coordinates) - fragments are the `size` nearest
                points in space and need not be connected. `weight` is ignored
                when False.
    k :         int, optional
                For ``mode="random"``/``"spaced"``: exact number of fragments.
                Overrides `sampling_factor`. Ignored by "partition"/"cover".
    sampling_factor : float
                For ``mode="random"``/``"spaced"`` when `k` is not given: how
                many times over to sample the neuron. The fragment count is
                ``ceil(sampling_factor * n_nodes / size)``, so 1 (default) is
                roughly single coverage and e.g. 3 oversamples ~3x. Ignored by
                "partition"/"cover".
    weight :    "weight" | None
                Only for ``connected=True``: grow by physical geodesic distance
                ("weight") or by hops (None). Ignored when ``connected=False``.
                For ``random``/``spaced`` the count is pinned by `k`/
                `sampling_factor`, so `weight` only changes *which* `size` nodes
                land in a fragment. For ``partition``/``cover`` the count is
                emergent (it depends on how the geodesic balls carve up the
                arbor), so switching metric can change the number of fragments
                too.
    undersized : "pad" | "keep" | "discard"
                What to do with a fragment that can't reach `size` nodes (the
                remainder in "partition"; a too-small connected component in
                "cover"/"random"). Full-size fragments are never affected.
                - "pad" (default): fill up to `size` with `pad_value`, so every
                  returned fragment has length `size` (stackable).
                - "keep": return it at its natural length - the output list is
                  then ragged and `np.stack` won't work.
                - "discard": drop it. In "cover"/"random" this means those nodes
                  go uncovered/unsampled.
    pad_value : int
                Fill value used when ``undersized="pad"`` (default -1, a natural
                mask token). Keep it negative (or otherwise outside ``[0,
                n_nodes)``) so the ``chunk >= 0`` mask below can tell pad slots
                from real indices; a non-negative in-range `pad_value` is itself
                a valid index and would slip through that mask (use
                ``chunk != pad_value`` instead).
    random_state : int | np.random.Generator, optional
                Seed/generator for ``mode="random"``. For ``mode="spaced"`` it
                only randomises the first seed (varies the even tiling); omit
                for a fully deterministic tiling.

    Returns
    -------
    list of np.ndarray
                One array of **positional indices** per fragment, each of
                length `size` when ``undersized="pad"``. Indices address
                `x.nodes` rows (skeleton) or `x.vertices` (mesh).

                With ``undersized="pad"`` a fragment may contain `pad_value`
                (default -1) in its unfilled slots. Do **not** index coordinates
                with a padded chunk directly: a negative `pad_value` is a valid
                (wrong) index and silently returns real rows. Mask first::

                    co = n.nodes[["x", "y", "z"]].values      # skeleton
                    co = m.vertices                            # mesh
                    chunk = chunks[0]
                    real = chunk[chunk >= 0]                   # drop pad slots
                    patch = co[real]                           # (<= size, 3)

                Map back to node IDs (skeletons) with
                ``n.nodes.node_id.values[real]``.

    See Also
    --------
    [`navis.ml.sample_points_uniform`][]
                Draw a uniform sample of points from a cloud (whole-neuron
                subsampling) rather than tiling it into fixed-size fragments.
                The complementary primitive for preparing ML inputs.
    [`navis.ml.normalize_neuron`][]
                Canonicalize a neuron's pose (center/orient/scale) - usually run
                before chunking so fragments live in a normalized frame.

    Examples
    --------
    >>> import navis
    >>> from navis.ml import chunk_neuron
    >>> import numpy as np
    >>> n = navis.example_neurons(1, kind="skeleton")
    >>> parts = chunk_neuron(n, size=50, mode="partition")
    >>> np.stack(parts).shape[1]
    50
    >>> # Euclidean packing wastes at most `size - 1` nodes:
    >>> tight = chunk_neuron(n, size=50, mode="partition", connected=False,
    ...                      undersized="discard")
    >>> n.n_nodes - sum(len(c) for c in tight) < 50
    True
    >>> # Evenly-spaced oversampling: ~2 passes over the neuron
    >>> over = chunk_neuron(n, size=50, mode="spaced", sampling_factor=2)
    >>> len(over) == int(np.ceil(2 * n.n_nodes / 50))
    True

    """
    size = _validate_tiling_args(mode, size, undersized, sampling_factor, k)
    if weight not in ("weight", None):
        raise ValueError(f'`weight` must be "weight" or None, got {weight!r}')

    n_nodes = _n_nodes(x)
    if n_nodes == 0:
        return []

    backend = _geodesic_backend(x, weight) if connected else _euclidean_backend(x)

    chunks = _chunk_indices(
        backend, n_nodes, size, mode, k, sampling_factor,
        undersized, pad_value, random_state,
    )
    # `_chunk_indices` carries a per-point distance-to-seed alongside each fragment
    # (see `_finalize`); only `sample_patches` surfaces it, so drop it here.
    return [idx for idx, _, _ in chunks]


def _validate_tiling_args(mode, size, undersized, sampling_factor, k, size_name="size"):
    """Validate the tiling contract shared by `chunk_neuron` and `sample_patches`.

    Returns `size` coerced to int. Each public caller validates only its own extra
    arguments on top (`weight` for `chunk_neuron`; `density`/`spacing` for
    `sample_patches`).
    """
    if mode not in ("partition", "cover", "random", "spaced"):
        raise ValueError(
            f'`mode` must be "partition", "cover", "random" or "spaced", got {mode!r}'
        )
    if int(size) != size or size <= 0:
        raise ValueError(f"`{size_name}` must be a positive integer, got {size!r}")
    if undersized not in ("pad", "keep", "discard"):
        raise ValueError(
            f'`undersized` must be "pad", "keep" or "discard", got {undersized!r}'
        )
    if sampling_factor <= 0:
        raise ValueError(f"`sampling_factor` must be > 0, got {sampling_factor!r}")
    if k is not None and int(k) <= 0:
        raise ValueError(f"`k` must be a positive integer, got {k!r}")
    return int(size)


def _check_foveate_mode(mode):
    """`partition`/`cover` are incompatible with foveation - fail loudly.

    Both drivers use what `grow` returns as their bookkeeping: `_partition` marks it
    `assigned`, `_cover` marks it `covered`. A foveated patch reports only its thinned
    selection, so the gaps between its peripheral points would stay unassigned
    (`partition` re-grows the same territory forever) or count as covered on the
    strength of one distant sample (`cover` silently under-covers).

    For `partition` that is fundamental: a disjoint tiling whose patches deliberately
    skip most of their own territory is not a partition. For `cover` it is only an
    implementation limit - `_Foveated.grow` still holds the full pre-thinning pool and
    could report it as "claimed" separately from what the patch contains, which would
    make cover-at-multiple-scales work. That needs a wider `grow` contract across every
    backend, so for now both fail loudly.
    """
    if mode not in ("random", "spaced"):
        raise ValueError(
            f'`foveate` requires `mode="spaced"` or `mode="random"`, got {mode!r}. '
            "Foveated patches deliberately overlap and do not tile, so the "
            '"partition" (disjoint) and "cover" (every point included) guarantees '
            "cannot hold."
        )


def _resolve_foveate(foveate):
    """Normalise `foveate` to a `falloff` for `_radial_thin`: None = scale-free."""
    if foveate is True or foveate == "scale-free":
        return None
    return _check_positive_number(foveate, "foveate")



def _chunk_indices(backend, n_nodes, size, mode, k, sampling_factor,
                   undersized, pad_value, random_state):
    """Dispatch to the mode driver, returning one `(indices, distances)` pair per
    fragment - positional indices plus each point's distance to the fragment's seed
    (the metric is the backend's own: geodesic or Euclidean).

    Shared by `chunk_neuron` (neuron-backed: geodesic or Euclidean backend) and
    `sample_patches` (cloud-backed: an `_Euclidean`/`_ConnectedCloud` over a resampled
    point cloud, optionally wrapped in `_Foveated`). Assumes its arguments are already
    validated by the caller.
    """
    if mode == "partition":
        return _partition(backend, n_nodes, size, undersized, pad_value)
    if mode == "cover":
        return _cover(backend, n_nodes, size, undersized, pad_value)
    count = _num_fragments(k, sampling_factor, n_nodes, size)
    driver = _random if mode == "random" else _spaced
    return driver(backend, n_nodes, size, count, undersized, pad_value, random_state)


def sample_patches(
    x,
    *,
    n_points,
    density=None,
    spacing=None,
    mode="spaced",
    connected=True,
    foveate=None,
    reach=32,
    fovea=0,
    k=None,
    sampling_factor=1,
    undersized="keep",
    pad_value=-1,
    interpolate=None,
    weights=None,
    surface_mode="even",
    attributes=None,
    random_state=None,
):
    """Resample a neuron at a target density, then tile it into fixed-size patches.

    This is the one primitive that fixes **both** a per-patch point count *and* a
    physical resolution - the combination [`navis.ml.chunk_neuron`][] cannot give
    you. `chunk_neuron` indexes the neuron's *existing* nodes/vertices, so a patch's
    density is whatever the reconstruction happened to have; and count-and-density
    are over-constrained on a fixed point set anyway. `sample_patches` instead
    **resamples** the neuron to a uniform cloud at the requested `density`/`spacing`
    (via [`navis.ml.sample_surface`][] for meshes, [`navis.ml.sample_cable`][] for
    skeletons) and then groups that cloud into spatial patches of `n_points` each.

    Because the cloud is uniform, `n_points` + `density` pin each patch's physical
    extent, and - the property that actually matters for a ball/k-NN point model -
    that extent is the same across patches and across neurons. As a *flat-surface*
    approximation the patch radius is ``~ sqrt(n_points / (pi * density))``; this is
    accurate where the surface is locally flat (e.g. a soma or a large mesh) but
    **under-estimates the Euclidean extent on thin cable/tubes**, where a patch is a
    long axial sleeve rather than a flat disk and can run several times longer than
    the formula suggests (the thinner the neurite, the larger the factor). Treat it
    as a scaling/dimensional guide, not the achieved radius; if you need the real
    extent, measure it straight off each patch's coordinates
    (e.g. ``g[["x", "y", "z"]]`` per ``chunk_id`` group).

    Like [`navis.ml.chunk_neuron`][], patches grow **along the neuron** by default
    (`connected=True`): each sample is tied to a structural element via its
    `source_id` (skeleton node / mesh vertex) and growth is geodesic on that native
    graph, so a patch follows one branch/surface region instead of a Euclidean ball
    that can straddle two branches passing close in space. Set `connected=False` for
    plain Euclidean (spatial-ball) patches.

    Parameters
    ----------
    x :                 TreeNeuron | MeshNeuron
    n_points :          int
                        Points per patch (fixed count). Keyword-only.
    density :           float, optional
                        Resample resolution: points per unit area (mesh) / length
                        (skeleton). Exactly one of `density`/`spacing` is required.
    spacing :           float, optional
                        Resample resolution as an inter-point distance instead.
    mode :              "spaced" | "partition" | "cover" | "random"
                        How patches tile the cloud (as in [`navis.ml.chunk_neuron`][]):
                        "spaced" (default) and "random" scatter evenly-spread /
                        random, possibly-overlapping patches - good for oversampling,
                        but they do **not** guarantee full coverage (raise
                        `sampling_factor`, or use "cover" for that). "cover" overlaps
                        to cover every point at least once. "partition" tiles
                        disjointly; with `connected=True`, carving connected patches
                        strands leftover points into small fragments (drop them with
                        `undersized="discard"` for clean full-size patches).
    connected :         bool
                        If True (default) grow patches geodesically along the neuron
                        (see above) so each is a single connected piece; if False
                        grow Euclidean balls (the `n_points` nearest points in space),
                        which pack tighter but can span nearby branches.
    foveate :           None | True | "scale-free" | float
                        Spend the point budget unevenly: dense at the patch centre,
                        thinning outwards, so one patch carries fine local detail
                        *and* long-range context. `None`/`False` (default) is the
                        uniform behaviour. ``True``/``"scale-free"`` places points
                        geometrically in radial rank, which is a ``1 / r**D`` falloff
                        for a locally `D`-dimensional cloud - equal points per octave
                        of radius - self-calibrating per patch. A float instead
                        applies a literal ``1 / r**foveate`` weighting (see
                        `_radial_thin`); prefer the default unless you specifically
                        need that physical density, because the measured `D` runs
                        from ~1 on a thin neurite to ~2 on a soma *within one mesh*,
                        so a fixed exponent splits the budget differently patch to
                        patch. Requires ``mode="spaced"``/``"random"``: foveated
                        patches deliberately overlap and cannot tile.
    reach :             int | None
                        Only with `foveate`: grow ``reach * n_points`` candidates per
                        patch before thinning, so this sets how far the periphery
                        extends. `None` reaches over the whole connected component
                        (expensive). Cost scales with `reach`, and it is the dominant
                        cost of foveation - the thinning itself is free.
    fovea :             int
                        Only with `foveate`: keep the innermost `fovea` candidates at
                        full cloud density before the falloff starts, for a genuinely
                        full-resolution core. 0 (default) still yields a small dense
                        centre, because the falloff cannot ask for gaps narrower than
                        the cloud (see `_spread`).
    k :                 int, optional
                        For "random"/"spaced": exact number of patches (overrides
                        `sampling_factor`).
    sampling_factor :   float
                        For "random"/"spaced": how many times over to sample the
                        cloud (patch count ``~ ceil(sampling_factor * N / n_points)``).
    undersized :        "keep" | "pad" | "discard"
                        What to do with a patch that can't reach `n_points` (only the
                        cloud being smaller than `n_points`, or the "partition"
                        remainder). "keep" (default) leaves it short - natural in this
                        long/tidy output; "pad" appends filler rows (coords/attrs
                        ``NaN``, ``source_id`` = `pad_value`) so every patch has
                        `n_points` rows; "discard" drops it.
    pad_value :         int
                        `source_id` marker for padded rows when ``undersized="pad"``.
    interpolate :       None | True | str | list of str
                        Skeletons only: node columns to interpolate/carry onto each
                        sample (see [`navis.ml.sample_cable`][]).
    weights :           None | str | array-like
                        Skeletons only: per-node sampling weights (see
                        [`navis.ml.sample_cable`][]). Redistributes points; the count
                        still comes from `density`/`spacing`.
    surface_mode :      "even" | "surface"
                        Meshes only: the surface sampling mode (see
                        [`navis.ml.sample_surface`][]). `spacing` needs "even".
    attributes :        dict | pandas.DataFrame, optional
                        Meshes only: per-vertex values to transfer onto samples.
    random_state :      int | np.random.Generator, optional
                        Seeds both the resampling jitter and the patch seeding
                        (split internally so they stay independent). Omit for a fresh
                        tiling each call.

    Returns
    -------
    pandas.DataFrame
                        Long/tidy form: the resampled cloud's columns (``x``, ``y``,
                        ``z``, any transferred/interpolated columns, ``source_id``)
                        plus a ``chunk_id`` column giving each row's patch. Overlapping
                        patches (cover/random/spaced) **duplicate** a point's row once
                        per patch it belongs to, so ``groupby("chunk_id")`` yields the
                        patches. Points in no patch are simply absent. Rows within a
                        patch are ordered seed-first, by increasing distance.

                        With `foveate` there are two extra columns:

                        - ``chunk_dist`` - the row's distance to its patch's seed
                          (geodesic when ``connected=True``, else Euclidean). Foveated
                          patches span orders of magnitude in radius, so feed this to
                          the model as the radial coordinate rather than inferring
                          scale from the coordinates.
                        - ``chunk_focus`` - how "in focus" the row is: the fraction of
                          the cloud kept around it, ``1.0`` where the patch is at full
                          `density` (the fovea) and falling towards 0 out in the
                          sparse periphery. Use it to weight points by how much
                          resolution actually backs them, or threshold it
                          (``chunk_focus == 1``) to recover just the crisp core.

    See Also
    --------
    [`navis.ml.chunk_neuron`][]
                        Tile a neuron's *existing* nodes/vertices into fixed-count
                        fragments (no resampling, no density control).
    [`navis.ml.sample_surface`][] / [`navis.ml.sample_cable`][]
                        The whole-neuron resamplers this builds on.

    Examples
    --------
    >>> import navis
    >>> m = navis.example_neurons(1, kind="mesh")
    >>> patches = navis.ml.sample_patches(m, n_points=64, density=1e-5, random_state=0)
    >>> "chunk_id" in patches.columns
    True
    >>> patches.groupby("chunk_id").size().unique().tolist()   # every patch full-size
    [64]
    >>> # stack a fixed-size batch tensor from the groups:
    >>> import numpy as np
    >>> batch = np.stack([g[["x", "y", "z"]].values
    ...                   for _, g in patches.groupby("chunk_id")])
    >>> batch.shape[1:]
    (64, 3)
    >>> # foveated: same 64 points, but reaching much further out
    >>> fov = navis.ml.sample_patches(m, n_points=64, density=1e-5, mode="spaced",
    ...                               foveate=True, reach=8, random_state=0)
    >>> fov.groupby("chunk_id").size().unique().tolist()   # still exactly 64
    [64]
    >>> reach = lambda d: np.mean([g.chunk_dist.max()
    ...                            for _, g in d.groupby("chunk_id")])
    >>> extent = lambda d: np.mean([np.linalg.norm(
    ...     g[["x", "y", "z"]].values - g[["x", "y", "z"]].values[0], axis=1).max()
    ...     for _, g in d.groupby("chunk_id")])
    >>> bool(reach(fov) > 2 * extent(patches))             # far longer reach
    True

    """
    n_points = _validate_tiling_args(mode, n_points, undersized, sampling_factor, k,
                                     size_name="n_points")
    if (density is None) == (spacing is None):
        raise ValueError(
            "Provide exactly one of `density` or `spacing` to set the resample "
            "resolution (they are mutually exclusive)."
        )

    # Resolve the foveation arguments *before* resampling: they are pure argument
    # checks, and the resample below is the expensive part of this call.
    fov = foveate is not None and foveate is not False
    falloff = None
    if fov:
        _check_foveate_mode(mode)
        falloff = _resolve_foveate(foveate)
        reach = None if reach is None else _check_positive_int(reach, "reach")
        if int(fovea) != fovea or fovea < 0:
            raise ValueError(f"`fovea` must be a non-negative integer, got {fovea!r}")
        fovea = int(fovea)

    # Split the RNG so the resampling jitter, the patch seeding and the foveal
    # thinning are independent yet jointly reproducible. `None` stays `None` so each
    # step keeps its own "fresh each call" / deterministic-tiling default (see
    # `sample_*`/`_spaced`). `spawn` is sequential, so the first two children are
    # unchanged by the third being drawn.
    if random_state is None:
        rs_sample = rs_chunk = rs_fov = None
    else:
        rs_sample, rs_chunk, rs_fov = np.random.default_rng(random_state).spawn(3)

    if isinstance(x, core.MeshNeuron):
        cloud = sample_surface(
            x, density=density, spacing=spacing, mode=surface_mode,
            attributes=attributes, random_state=rs_sample,
        )
    elif isinstance(x, core.TreeNeuron):
        cloud = sample_cable(
            x, density=density, spacing=spacing, interpolate=interpolate,
            weights=weights, random_state=rs_sample,
        )
    else:
        raise TypeError(
            f"sample_patches requires a TreeNeuron or MeshNeuron, got {type(x)}."
        )

    coords = cloud[["x", "y", "z"]].to_numpy(dtype=float)
    if len(coords) == 0:
        cloud["chunk_id"] = np.zeros(0, dtype=np.int64)
        return cloud

    backend = _connected_cloud_backend(x, cloud) if connected else _Euclidean(coords)
    if fov:
        # `reach=None` means "the whole reachable component": a pool of every point
        # in the cloud, which `grow` fills only as far as the region actually goes.
        pool = len(coords) if reach is None else n_points * reach
        backend = _Foveated(backend, pool, fovea, falloff, rs_fov)
    chunks = _chunk_indices(
        backend, len(coords), n_points, mode, k, sampling_factor,
        undersized, pad_value, rs_chunk,
    )
    return _patch_frame(cloud, chunks, pad_value, with_dist=fov)


def _patch_frame(cloud, chunks, pad_value, with_dist=False):
    """Assemble the long-form patch DataFrame: the cloud's columns + `chunk_id`.

    One row per (point, patch) membership - overlapping patches duplicate a point's
    row under each `chunk_id`. Padded slots (present only when ``undersized="pad"``,
    encoded as negative indices) become filler rows: coords/attrs ``NaN``,
    `source_id` = `pad_value`.

    `with_dist` additionally emits the two per-point patch columns, `chunk_dist` and
    `chunk_focus`. Only foveated patches ask for them: they span orders of magnitude
    in radius and in local resolution, so a model needs both as explicit features. They
    stay off otherwise, where they would be a constant 0 and 1 - and so the
    uniform-patch output keeps its established column set.
    """
    frames, ids, dists, focuses = [], [], [], []
    for cid, (idx, dist, focus) in enumerate(chunks):
        idx = np.asarray(idx)
        sub = cloud.iloc[idx[idx >= 0]]
        n_pad = int((idx < 0).sum())
        if n_pad:
            sub = pd.concat([sub, _pad_rows(cloud, n_pad, pad_value)], ignore_index=True)
        frames.append(sub)
        ids.append(np.full(len(sub), cid, dtype=np.int64))
        if with_dist:
            dists.append(np.asarray(dist, dtype=float))
            focuses.append(np.asarray(focus, dtype=float))

    if not frames:
        out = cloud.iloc[:0].copy()
        out["chunk_id"] = np.zeros(0, dtype=np.int64)
        if with_dist:
            out["chunk_dist"] = np.zeros(0, dtype=float)
            out["chunk_focus"] = np.zeros(0, dtype=float)
        return out
    # Assign `chunk_id` on the concatenated frame (always a fresh object) rather
    # than on each `cloud.iloc[...]` slice: the latter raises SettingWithCopyWarning
    # on pandas builds without copy-on-write. Also avoids a per-chunk copy.
    out = pd.concat(frames, ignore_index=True)
    out["chunk_id"] = np.concatenate(ids)
    if with_dist:
        out["chunk_dist"] = np.concatenate(dists)
        out["chunk_focus"] = np.concatenate(focuses)
    return out


def _pad_rows(cloud, n, pad_value):
    """`n` filler rows matching `cloud`'s columns: NaN everywhere except
    `source_id`, set to `pad_value` so padded rows are identifiable."""
    out = {}
    for c in cloud.columns:
        if c == "source_id":
            out[c] = np.full(n, pad_value, dtype=np.int64)
        else:
            out[c] = np.full(n, np.nan)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Mode drivers - metric-agnostic, they only call backend.grow / backend.seed
# --------------------------------------------------------------------------- #
def _partition(backend, n_nodes, size, undersized, pad_value):
    """Non-overlapping fragments.

    `connected=True`: carving compact *connected* balls out of a branchy arbor
    strands sub-`size` pockets, so 30-40% of nodes typically can't fill a whole
    fragment (inherent to an exact-size connected partition, not a bug - no
    seeding order fixes it). Note this is a fraction of *nodes*; because the
    stranded nodes scatter into many tiny fragments, the fraction of padded
    *slots* under ``undersized="pad"`` is much higher and grows with `size`
    (e.g. >90% at `size=500` on the example neurons) - measure at your real
    `size` before relying on connected partition for fixed-size tiles.
    `connected=False`: each round simply takes the `size` nearest unassigned
    points, so only the final remainder is ever short (tight packing, but
    fragments bleed across spatially-adjacent branches). Short fragments are
    handled per `undersized`.

    Seeds are just the first unassigned node (`_first_unset`), not the
    farthest-point `backend.seed`: partition covers every node regardless of
    order, so the spread buys nothing here and would cost a full graph traversal
    per fragment.
    """
    assigned = np.zeros(n_nodes, dtype=bool)
    chunks = []
    while not assigned.all():
        seed = _first_unset(assigned)
        region, dist, focus = backend.grow(seed, size, forbidden=assigned)
        # Mark everything the growth touched as done either way, so a pocket
        # smaller than `size` is not retried forever.
        assigned[region] = True
        out = _finalize(region, dist, focus, size, undersized, pad_value)
        if out is not None:
            chunks.append(out)
    return chunks


def _cover(backend, n_nodes, size, undersized, pad_value):
    """Overlapping fragments covering every node at least once.

    With ``connected=True`` a fragment can't span components, so cover is forced
    to emit at least one fragment per connected component - on a mesh with many
    tiny specks that is one padded fragment each. ``undersized="discard"`` drops
    those (and any other sub-`size` fragment), which trades the full-coverage
    guarantee for full-size, unpadded output - usually the right call when the
    specks are fluff. Seeding tiles the largest component first (see
    `_Geodesic.seed`), so the discarded fragments are exactly the small ones.
    """
    covered = np.zeros(n_nodes, dtype=bool)
    chunks = []
    while not covered.all():
        seed = backend.seed(covered)
        # No `forbidden`: a fragment may reuse already-covered nodes to fill up
        # to `size` - that reuse is exactly the overlap that lets us keep every
        # fragment full-sized while still discarding nothing.
        region, dist, focus = backend.grow(seed, size)
        covered[region] = True
        out = _finalize(region, dist, focus, size, undersized, pad_value)
        if out is not None:
            chunks.append(out)
    return chunks


def _random(backend, n_nodes, size, count, undersized, pad_value, random_state):
    """`count` fragments from random seeds (independent, may overlap)."""
    rng = np.random.default_rng(random_state)
    seeds = rng.integers(0, n_nodes, size=count)
    return _grow_all(backend, seeds, size, undersized, pad_value)


def _spaced(backend, n_nodes, size, count, undersized, pad_value, random_state):
    """`count` evenly-spaced fragments - the deterministic sibling of `_random`.

    Seeds are chosen by farthest-point sampling: the first seed, then repeatedly
    the point farthest (geodesic or Euclidean, per the backend) from every seed
    picked so far. That spreads the fragments evenly instead of clumping them at
    random. The first seed comes from the backend's largest component (so a
    fluff-ridden mesh doesn't start on a disconnected speck); `random_state`, if
    given, only randomises that first seed to vary the tiling between epochs.
    """
    chosen = np.zeros(n_nodes, dtype=bool)
    if random_state is None:
        first = backend.seed(chosen)  # largest component (nothing chosen yet)
    else:
        first = backend.random_seed(np.random.default_rng(random_state))
    seeds = [first]
    chosen[first] = True
    while len(seeds) < count and not chosen.all():
        s = backend.seed(chosen)  # farthest point from all seeds so far
        seeds.append(s)
        chosen[s] = True
    if len(seeds) < count:
        logger.warning(
            f"`spaced` can place at most one seed per node: asked for {count} "
            f"fragments but only {len(seeds)} unique seeds exist ({n_nodes} nodes)."
        )
    return _grow_all(backend, seeds, size, undersized, pad_value)


def _grow_all(backend, seeds, size, undersized, pad_value):
    """Grow one fragment per seed and apply the `undersized` policy to each.

    Shared tail of `_random`/`_spaced` (the seed-driven modes): unlike
    `partition`/`cover` they keep no cross-fragment bookkeeping, so growing is just
    a map over the seeds. Dropped fragments (``undersized="discard"``) vanish here,
    which is why distances travel *with* each fragment rather than in a parallel
    list - there is no stable index to re-align them by.
    """
    out = []
    for s in seeds:
        region, dist, focus = backend.grow(s, size)
        chunk = _finalize(region, dist, focus, size, undersized, pad_value)
        if chunk is not None:
            out.append(chunk)
    return out


def _num_fragments(k, sampling_factor, n_nodes, size):
    """Fragment count for `random`/`spaced`: explicit `k` wins, else
    `sampling_factor` passes over the neuron (~`n_nodes / size` fragments each).
    """
    if k is not None:
        return int(k)
    return max(1, int(np.ceil(sampling_factor * n_nodes / size)))


# --------------------------------------------------------------------------- #
# Backends: each exposes grow(seed, size, forbidden) and seed(done)
# --------------------------------------------------------------------------- #
class _Geodesic:
    """Grow connected geodesic balls along the arbor (the future fastcore fn)."""

    def __init__(self, indptr, indices, data, csr):
        self.indptr, self.indices, self.data, self.csr = indptr, indices, data, csr
        # Connected components let seeding stay on the reachable arbor and prefer
        # the largest unfinished component - essential on meshes, which are
        # riddled with tiny disconnected specks (see `seed`).
        self.n_comp, self.labels = csgraph.connected_components(csr, directed=False)
        # Incremental farthest-point-sampling state (see `seed`): running geodesic
        # distance from every node to its nearest source, and the sources already
        # folded in. `None` until the first `seed` call.
        self._fps_min = None
        self._fps_seen = None

    def grow(self, seed, size, forbidden=None):
        """Flood-fill a connected region of up to `size` nodes from `seed`.

        Dijkstra-style: settle nodes in order of increasing geodesic distance,
        stopping at `size`. Every settled node (bar the seed) is reached through
        an already-settled neighbour, so the region is connected. `forbidden`
        keeps partition fragments disjoint (growth stays in the unassigned
        subgraph). Returns ``(indices, distances)`` seed-first, in increasing-
        distance order; `distances` is the geodesic distance to the seed, which
        Dijkstra has already computed (it is the heap key each node settles at).
        """
        indptr, indices, data = self.indptr, self.indices, self.data
        region, dists, settled = [], [], set()
        heap = [(0.0, int(seed))]
        while heap and len(region) < size:
            d, u = heapq.heappop(heap)
            if u in settled:
                continue
            settled.add(u)
            region.append(u)
            dists.append(d)
            for j in range(indptr[u], indptr[u + 1]):
                v = int(indices[j])
                if v in settled:
                    continue
                if forbidden is not None and forbidden[v]:
                    continue
                heapq.heappush(heap, (d + float(data[j]), v))
        return np.array(region, dtype=np.int64), np.array(dists, dtype=float), None

    def seed(self, done):
        """Next farthest-point seed, kept on the reachable arbor.

        Among nodes not in `done`, return the one geodesically farthest from
        everything in `done` **that is actually reachable from it** (finite
        distance). Unreachable nodes have ``inf`` distance; picking those would
        seed every disconnected speck before the main arbor (a mesh can have
        hundreds of 4-vertex components), so they are excluded. Only once the
        reachable frontier is exhausted - or nothing is seeded yet - do we jump
        to a fresh component, largest first.

        The distance field is maintained incrementally: `done` only ever grows
        (both callers set bits, never clear them), so each call folds in a
        Dijkstra from just the *newly* added sources rather than re-running a
        multi-source Dijkstra over the whole `done` set every time.
        """
        if done.any():
            self._fps_fold(done)
            reachable = np.isfinite(self._fps_min) & ~done
            if reachable.any():
                return int(np.argmax(np.where(reachable, self._fps_min, -np.inf)))
        return self._largest_unset(done)

    def _fps_fold(self, done):
        """Fold sources newly present in `done` into the running distance field.

        `self._fps_min[i]` ends up equal to the geodesic distance from `i` to its
        nearest node in `done` - identical to a multi-source Dijkstra over all of
        `done`, because ``min`` over sources is associative and `done` only grows.
        """
        if self._fps_min is None:
            self._fps_min = np.full(done.shape[0], np.inf)
            self._fps_seen = np.zeros(done.shape[0], dtype=bool)
        new = done & ~self._fps_seen
        if new.any():
            d = csgraph.dijkstra(
                self.csr, directed=False, indices=np.where(new)[0], min_only=True
            )
            np.minimum(self._fps_min, d, out=self._fps_min)
            self._fps_seen |= done

    def _largest_unset(self, done):
        """First node of the largest component that still has an unset node."""
        unset = ~done
        counts = np.bincount(self.labels[unset], minlength=self.n_comp)
        best = int(np.argmax(counts))
        return int(np.flatnonzero(unset & (self.labels == best))[0])

    def random_seed(self, rng):
        """A random seed from the largest component - varies `spaced`'s tiling
        (per `random_state`) without dropping the first fragment on a speck.
        """
        largest = int(np.argmax(np.bincount(self.labels, minlength=self.n_comp)))
        pool = np.flatnonzero(self.labels == largest)
        return int(pool[rng.integers(0, len(pool))])


class _Euclidean:
    """Grow Euclidean balls in space, ignoring the arbor (KD-tree)."""

    def __init__(self, coords):
        self.coords = coords
        self.tree = cKDTree(coords)
        # Incremental farthest-point-sampling state (see `seed`).
        self._fps_min = None
        self._fps_seen = None

    def grow(self, seed, size, forbidden=None):
        """The `size` nearest points to `seed` in Euclidean space.

        With `forbidden` (partition) restrict to allowed points; else a straight
        KD-tree k-NN query. Returns ``(indices, distances)`` seed-first, in
        increasing-distance order; both queries produce the distances anyway.
        Fragments need not be graph-connected.
        """
        if forbidden is None:
            d, idx = self.tree.query(self.coords[seed], k=min(size, len(self.coords)))
            return (np.atleast_1d(idx).astype(np.int64),
                    np.atleast_1d(d).astype(float), None)

        allowed = np.where(~forbidden)[0]
        d = np.linalg.norm(self.coords[allowed] - self.coords[seed], axis=1)
        if len(allowed) > size:
            keep = np.argpartition(d, size - 1)[:size]
            allowed, d = allowed[keep], d[keep]
        order = np.argsort(d)
        return allowed[order].astype(np.int64), d[order], None

    def seed(self, done):
        """Next seed: the unset point Euclidean-farthest from everything done.

        Like `_Geodesic.seed`, the distance-to-nearest-source field is kept
        incrementally: `done` only grows, so each call queries only the *newly*
        added sources against every point and mins them into the running field -
        far cheaper than rebuilding a KD-tree over the whole (growing) `done` set
        each time.
        """
        if not done.any():
            return 0
        self._fps_fold(done)
        return int(np.argmax(np.where(~done, self._fps_min, -np.inf)))

    def _fps_fold(self, done):
        """Fold sources newly present in `done` into the running distance field.

        `self._fps_min[i]` ends up equal to the distance from `i` to its nearest
        node in `done` - min over sources is associative, so folding the new
        sources in matches a fresh nearest-neighbour query over all of `done`.
        """
        if self._fps_min is None:
            self._fps_min = np.full(len(self.coords), np.inf)
            self._fps_seen = np.zeros(len(self.coords), dtype=bool)
        new = np.where(done & ~self._fps_seen)[0]
        if not len(new):
            return
        self._fps_seen |= done
        if len(new) == 1:
            # Single new source (the `spaced` pattern, one seed per call): a plain
            # full-array update is cheapest - no fancy-index gather/scatter.
            d = np.linalg.norm(self.coords - self.coords[new[0]], axis=1)
            np.minimum(self._fps_min, d, out=self._fps_min)
        else:
            # A batch of new sources (the `cover` pattern, a whole region per
            # call): only un-done points can still be selected, and that set
            # shrinks as coverage grows, so query just those against the new
            # sources. A point keeps getting folded for as long as it is
            # selectable, so its distance is exact when it is finally picked.
            un = np.where(~done)[0]
            if len(un):
                d, _ = cKDTree(self.coords[new]).query(self.coords[un])
                self._fps_min[un] = np.minimum(self._fps_min[un], d)

    def random_seed(self, rng):
        """A uniformly random seed. Euclidean growth ignores connectivity, so
        there is no disconnected-fluff trap to avoid here (unlike `_Geodesic`).
        """
        return int(rng.integers(0, len(self.coords)))


class _ConnectedCloud(_Euclidean):
    """Grow patches over a resampled cloud along the neuron's own topology - the
    cloud analogue of ``chunk_neuron(connected=True)``.

    Each sample is attached to a structural vertex (skeleton node / mesh vertex) via
    its ``source_id``; a patch is grown by geodesic flood-fill on that native graph,
    collecting the samples on visited vertices until ``size`` are gathered. Because
    growth follows the arbor/surface, a patch never jumps between branches that only
    pass close in space, and it stays connected even when the cloud is far sparser
    than the mesh/skeleton (empty vertices simply conduct without contributing).

    Only geodesic *growth* differs from `_Euclidean`; FPS / random *seeding* is
    purely spatial, so it is inherited unchanged. Build via `_connected_cloud_backend`,
    which resolves the graph and the per-sample vertex array (keeping neuron-type
    knowledge out of this numeric peer of `_Geodesic` / `_Euclidean`).
    """

    def __init__(self, indptr, indices, data, svtx, coords):
        super().__init__(coords)                     # KD-tree + spatial FPS seeding
        self.indptr, self.indices, self.data = indptr, indices, data
        self.svtx = svtx
        # vertex -> sample indices (sparse: only vertices that carry a sample)
        by_vtx: dict = {}
        for i, v in enumerate(svtx):
            by_vtx.setdefault(int(v), []).append(i)
        self.by_vtx = by_vtx

    def grow(self, seed, size, forbidden=None):
        """`size` samples in a connected structural region around sample `seed`.

        Dijkstra out from the seed's vertex, gathering the samples on each settled
        vertex (skipping `forbidden` ones) in order of increasing geodesic distance.
        Returns fewer only when the reachable region runs out of eligible samples.
        Mirrors `_Geodesic.grow`, but counts *samples* attached to each vertex.

        For a disjoint partition (`forbidden` given), a vertex whose samples are
        *all* forbidden acts as a wall - growth does not cross it - so each fragment
        stays a connected region of still-unassigned samples. Empty vertices (which
        carry no samples) always conduct, bridging the gaps in a sparse cloud.

        The returned distances are the *vertex's* geodesic distance to the seed's
        vertex, so every sample riding on one vertex shares a distance (ties are
        common in a cloud denser than the mesh). That quantisation is at the scale
        of one edge and is what `_Foveated` ranks by.
        """
        indptr, indices, data, by_vtx = self.indptr, self.indices, self.data, self.by_vtx
        settled = set()
        got: list = []
        gdist: list = []
        heap = [(0.0, int(self.svtx[seed]))]
        while heap and len(got) < size:
            d, u = heapq.heappop(heap)
            if u in settled:
                continue
            settled.add(u)
            for s in by_vtx.get(u, ()):
                if forbidden is None or not forbidden[s]:
                    got.append(s)
                    gdist.append(d)
            for j in range(indptr[u], indptr[u + 1]):
                v = int(indices[j])
                if v in settled:
                    continue
                if forbidden is not None:
                    sv = by_vtx.get(v)
                    if sv is not None and all(forbidden[s] for s in sv):
                        continue                    # fully-assigned vertex: a wall
                heapq.heappush(heap, (d + float(data[j]), v))
        return (np.array(got[:size], dtype=np.int64),
                np.array(gdist[:size], dtype=float), None)


class _Foveated:
    """Wrap any backend so patches get a dense core and a sparse, far-reaching halo.

    A uniform patch spends its whole budget at one resolution, so `n_points` fixes
    both the detail *and* the extent. This grows a `reach`-times oversized candidate
    pool and thins it back to `n_points` with a density that falls off with distance
    - fine local detail plus long-range context for the same point count.

    The thinning is **rank-based** (see `_radial_thin`), which is why this is a
    generic wrapper: it needs only the seed-first, increasing-distance ordering that
    every backend's `grow` already guarantees, and works the same on a Euclidean
    ball or a geodesic region. Seeding is untouched - it delegates straight through,
    so the patch *centres* stay evenly spread exactly as before.

    Only `grow` differs, so the mode drivers, `_finalize` and `_patch_frame` are
    unchanged. Note this composes only with the seed-driven modes: see
    `_check_foveate_mode` for why `partition`/`cover` cannot use it.
    """

    def __init__(self, inner, pool, fovea, falloff, random_state):
        self.inner = inner
        self.pool = pool
        self.fovea = fovea
        self.falloff = falloff
        self.rng = np.random.default_rng(random_state)

    def grow(self, seed, size, forbidden=None):
        idx, dist, _ = self.inner.grow(seed, self.pool, forbidden)
        sel, focus = _radial_thin(len(idx), size, self.fovea, self.falloff, dist,
                                  self.rng)
        return idx[sel], dist[sel], focus

    def seed(self, done):
        return self.inner.seed(done)

    def random_seed(self, rng):
        return self.inner.random_seed(rng)


def _connected_cloud_backend(x, cloud):
    """Backend that grows patches along `x`'s native topology over the resampled
    `cloud`, locating each sample by its `source_id`. The neuron-type dispatch lives
    here (mirroring `_geodesic_backend`/`_euclidean_backend`) so `_ConnectedCloud`
    stays a purely numeric peer.
    """
    edges, weights, n, ids = _cluster_graph(x, "weight")
    indptr, indices, data = _build_csr(edges, weights, n)
    src = cloud["source_id"].to_numpy()
    if isinstance(x, core.TreeNeuron):
        # `source_id` is a node id; `ids` maps graph vertex -> node id.
        svtx = pd.Index(ids).get_indexer(src).astype(np.int64)
    else:
        svtx = src.astype(np.int64)                  # mesh: `source_id` is the vertex
    coords = cloud[["x", "y", "z"]].to_numpy(np.float64)
    return _ConnectedCloud(indptr, indices, data, svtx, coords)


def _geodesic_backend(x, weight):
    edges, weights, n_nodes, _ids = _cluster_graph(x, weight)
    indptr, indices, data = _build_csr(edges, weights, n_nodes)
    csr = csr_matrix((data, indices, indptr), shape=(n_nodes, n_nodes))
    return _Geodesic(indptr, indices, data, csr)


def _euclidean_backend(x):
    return _Euclidean(_node_coords(x))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _radial_thin(m, size, fovea, falloff, dist, rng):
    """Pick `size` positions out of `m` distance-sorted candidates, thinned radially.

    Returns ``(positions, focus)``. `positions` index the candidate list, strictly
    increasing, so the patch keeps its seed-first / increasing-distance order.
    `focus` is each kept point's **local keep fraction** - see `_focus`.

    Default (``falloff=None``) is **scale-free**: positions are spaced geometrically
    in rank. Because a candidate's rank is the number of cloud points within its
    distance - i.e. the cumulative measure, ``~ r**D`` for a locally `D`-dimensional
    cloud - geometric-in-rank *is* a ``1 / r**D`` density falloff, equal points per
    octave of radius, without ever estimating `D` (which is what makes it stable
    across wildly different local geometry; see `sample_patches`' `foveate` docs).

    A float `falloff` instead applies a literal ``1 / r**falloff`` weighting, keyed on
    true distance rather than rank and softened at the fovea edge so the centre stays
    finite.

    `fovea` takes the innermost candidates verbatim, at full cloud density, before
    the falloff starts.
    """
    if m <= size:
        sel = np.arange(m)                       # pool ran dry: nothing to thin
        return sel, _focus(sel)

    k = min(int(fovea), size)
    n = size - k
    if n == 0:
        sel = np.arange(k)
        return sel, _focus(sel)

    # Stratified (jittered-grid) draw, mirroring `sample_cable`'s sampling of the
    # cumulative measure: one position per equal-share bin rather than n evenly
    # spaced ones. Gives even coverage of the halo and decorrelates the peripheral
    # picks between epochs (a free augmentation) instead of always hitting the
    # same ranks.
    u = (np.arange(n) + rng.random(n)) / n

    if falloff is None:
        # Geometric in rank over the candidates outside the fovea: u=0 -> the first,
        # u=1 -> the last.
        sel = k + np.round((m - k) ** u).astype(np.int64) - 1
    else:
        r = dist[k:]
        # Soften at the fovea edge so `r -> 0` cannot blow up the weight. `dist` has
        # genuine zeros (the seed itself; and under `connected=True` every sample on
        # the seed's vertex), so fall back to the smallest positive distance - the
        # cloud's local spacing - when the edge itself sits at zero.
        pos = r[r > 0]
        r0 = float(r[0]) if r[0] > 0 else (float(pos[0]) if len(pos) else 1.0)
        w = (r0**2 + r**2) ** (-falloff / 2)
        cum = np.cumsum(w)
        sel = k + np.searchsorted(cum, u * cum[-1], side="right")

    sel = np.concatenate([np.arange(k), _spread(np.clip(sel, k, m - 1), k, m)])
    return sel, _focus(sel)


def _focus(sel):
    """How "in focus" each kept point is: the local fraction of candidates kept.

    `sel` is strictly increasing, so the gap to its neighbours says how heavily the
    cloud was thinned around it: a gap of 1 means every candidate at that radius
    survived - full `density` resolution - while a gap of 20 means one in twenty did.
    Focus is the reciprocal of that gap, hence **1 across the full-density core and
    falling towards 0 out in the periphery**.

    `np.gradient` gives the gap as a central difference (one-sided at the ends), which
    both centres the estimate on the point and smooths the transition out of the core.
    Since `sel` increases by at least 1 each step, the gap is always >= 1 and focus
    always lands in ``(0, 1]``.

    The value is the *realised* local density, not the requested one, so it inherits
    the stratified jitter's noise in the sparse periphery - that is honest: it is what
    the patch actually sampled there.
    """
    if len(sel) < 2:
        return np.ones(len(sel), dtype=float)
    return 1.0 / np.gradient(sel.astype(float))


def _spread(sel, lo, hi):
    """Force `sel` strictly increasing inside ``[lo, hi)``, preserving its shape.

    Both rules above can repeat a position wherever the target spacing drops below
    one candidate - near the centre, where a geometric ladder takes many steps to
    advance a whole rank. Nudging those apart is what turns the singularity into a
    *fully sampled core*: the falloff only starts biting once it asks for gaps wider
    than the cloud itself. Without this the repeats would collapse into duplicate
    points and the patch would come up short.

    Caller guarantees ``sel`` is sorted inside ``[lo, hi)`` and that
    ``len(sel) <= hi - lo``, so spreading always fits: the running max keeps every
    position ``>= sel[0] >= lo``, and the ceiling never drops below ``lo + i``.
    """
    i = np.arange(len(sel))
    out = np.maximum.accumulate(sel - i) + i     # strictly increasing, >= sel
    return np.minimum(out, hi - len(sel) + i)    # ...and still inside the pool



def _first_unset(done):
    """Index of the first ``False`` in `done` - the cheap seed for `partition`.

    Caller guarantees at least one unset node (the loop runs while
    ``not done.all()``), so the argmax always lands on a genuine ``False``.
    """
    return int(np.argmax(~done))


def _finalize(region, dist, focus, size, undersized, pad_value):
    """Apply the `undersized` policy to one fragment.

    Returns an ``(indices, distances, focus)`` triple kept in lockstep. `distances`
    holds each point's distance to the fragment's seed in the backend's own metric;
    `focus` its local keep fraction, or None from the backends that never thin. Both
    pad with ``NaN`` wherever `indices` pads with `pad_value`. Full-size fragments
    pass through unchanged; undersized ones are padded to `size`, kept ragged, or
    dropped (returns None -> caller skips the fragment entirely).
    """
    if len(region) >= size or undersized == "keep":
        return region, dist, focus
    if undersized == "pad":
        n_pad = size - len(region)
        pad = np.full(n_pad, np.nan)
        return (np.concatenate([region, np.full(n_pad, pad_value, dtype=np.int64)]),
                np.concatenate([dist, pad]),
                None if focus is None else np.concatenate([focus, pad]))
    return None  # "discard"


def _node_coords(x):
    """(N, 3) float coordinates aligned with positional indices."""
    if isinstance(x, core.TreeNeuron):
        return x.nodes[["x", "y", "z"]].values.astype(float)
    if isinstance(x, core.MeshNeuron):
        return np.asarray(x.vertices, dtype=float)
    raise TypeError(f"Expected TreeNeuron or MeshNeuron, got {type(x)}")


def _n_nodes(x):
    if isinstance(x, core.TreeNeuron):
        return len(x.nodes)
    if isinstance(x, core.MeshNeuron):
        return len(x.vertices)
    raise TypeError(f"Expected TreeNeuron or MeshNeuron, got {type(x)}")


def _build_csr(edges, weights, n_nodes):
    """Symmetric CSR (indptr, indices, data) for the undirected graph."""
    if len(edges):
        w = np.ones(len(edges)) if weights is None else np.asarray(weights, float)
        rows = np.concatenate([edges[:, 0], edges[:, 1]])
        cols = np.concatenate([edges[:, 1], edges[:, 0]])
        data = np.concatenate([w, w])
    else:
        rows = cols = np.zeros(0, dtype=np.int64)
        data = np.zeros(0, dtype=float)
    m = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    m.sum_duplicates()
    return m.indptr, m.indices, m.data
