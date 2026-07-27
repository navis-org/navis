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
from ..sampling.points import sample_surface, sample_cable

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

    return _chunk_indices(
        backend, n_nodes, size, mode, k, sampling_factor,
        undersized, pad_value, random_state,
    )


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


def _chunk_indices(backend, n_nodes, size, mode, k, sampling_factor,
                   undersized, pad_value, random_state):
    """Dispatch to the mode driver, returning a list of positional-index arrays.

    Shared by `chunk_neuron` (neuron-backed: geodesic or Euclidean backend) and
    `sample_patches` (cloud-backed: an `_Euclidean` over a resampled point cloud).
    Assumes its arguments are already validated by the caller.
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
    extent (radius ``~ sqrt(n_points / (pi * density))`` on a surface), and that
    extent is the same across patches and across neurons - the scale-consistency a
    ball/k-NN point model wants. Patches are grown in **Euclidean** space (spatial
    balls); there is no geodesic option, since the resampled cloud carries no graph.

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
                        How patches tile the cloud (see [`navis.ml.chunk_neuron`][]).
                        "spaced" (default) places evenly-spread, possibly-overlapping
                        patches; "partition" is non-overlapping; "cover" overlaps to
                        cover every point; "random" seeds at random.
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
                        patches. Points in no patch are simply absent.

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

    """
    n_points = _validate_tiling_args(mode, n_points, undersized, sampling_factor, k,
                                     size_name="n_points")
    if (density is None) == (spacing is None):
        raise ValueError(
            "Provide exactly one of `density` or `spacing` to set the resample "
            "resolution (they are mutually exclusive)."
        )

    # Split the RNG so the resampling jitter and the patch seeding are independent
    # yet jointly reproducible. `None` stays `None` so each step keeps its own
    # "fresh each call" / deterministic-tiling default (see `sample_*`/`_spaced`).
    if random_state is None:
        rs_sample = rs_chunk = None
    else:
        rs_sample, rs_chunk = np.random.default_rng(random_state).spawn(2)

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

    backend = _Euclidean(coords)
    chunks = _chunk_indices(
        backend, len(coords), n_points, mode, k, sampling_factor,
        undersized, pad_value, rs_chunk,
    )
    return _patch_frame(cloud, chunks, pad_value)


def _patch_frame(cloud, chunks, pad_value):
    """Assemble the long-form patch DataFrame: the cloud's columns + `chunk_id`.

    One row per (point, patch) membership - overlapping patches duplicate a point's
    row under each `chunk_id`. Padded slots (present only when ``undersized="pad"``,
    encoded as negative indices) become filler rows: coords/attrs ``NaN``,
    `source_id` = `pad_value`.
    """
    frames = []
    for cid, idx in enumerate(chunks):
        idx = np.asarray(idx)
        # `cloud.iloc[array]` already returns a fresh frame, so the column assign
        # below is safe (copy-on-write) without an extra `.copy()`.
        sub = cloud.iloc[idx[idx >= 0]]
        n_pad = int((idx < 0).sum())
        if n_pad:
            sub = pd.concat([sub, _pad_rows(cloud, n_pad, pad_value)], ignore_index=True)
        sub["chunk_id"] = cid
        frames.append(sub)

    if not frames:
        out = cloud.iloc[:0].copy()
        out["chunk_id"] = np.zeros(0, dtype=np.int64)
        return out
    return pd.concat(frames, ignore_index=True)


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
        region = backend.grow(seed, size, forbidden=assigned)
        # Mark everything the growth touched as done either way, so a pocket
        # smaller than `size` is not retried forever.
        assigned[region] = True
        out = _finalize(region, size, undersized, pad_value)
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
        region = backend.grow(seed, size)
        covered[region] = True
        out = _finalize(region, size, undersized, pad_value)
        if out is not None:
            chunks.append(out)
    return chunks


def _random(backend, n_nodes, size, count, undersized, pad_value, random_state):
    """`count` fragments from random seeds (independent, may overlap)."""
    rng = np.random.default_rng(random_state)
    seeds = rng.integers(0, n_nodes, size=count)
    chunks = [
        _finalize(backend.grow(s, size), size, undersized, pad_value) for s in seeds
    ]
    return [c for c in chunks if c is not None]


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
    chunks = [
        _finalize(backend.grow(s, size), size, undersized, pad_value) for s in seeds
    ]
    return [c for c in chunks if c is not None]


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
        subgraph). Returns indices seed-first, in increasing-distance order.
        """
        indptr, indices, data = self.indptr, self.indices, self.data
        region, settled = [], set()
        heap = [(0.0, int(seed))]
        while heap and len(region) < size:
            d, u = heapq.heappop(heap)
            if u in settled:
                continue
            settled.add(u)
            region.append(u)
            for j in range(indptr[u], indptr[u + 1]):
                v = int(indices[j])
                if v in settled:
                    continue
                if forbidden is not None and forbidden[v]:
                    continue
                heapq.heappush(heap, (d + float(data[j]), v))
        return np.array(region, dtype=np.int64)

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
        KD-tree k-NN query. Returns indices seed-first, in increasing-distance
        order. Fragments need not be graph-connected.
        """
        if forbidden is None:
            _, idx = self.tree.query(self.coords[seed], k=min(size, len(self.coords)))
            return np.atleast_1d(idx).astype(np.int64)

        allowed = np.where(~forbidden)[0]
        d = np.linalg.norm(self.coords[allowed] - self.coords[seed], axis=1)
        if len(allowed) > size:
            allowed = allowed[np.argpartition(d, size - 1)[:size]]
            d = np.linalg.norm(self.coords[allowed] - self.coords[seed], axis=1)
        return allowed[np.argsort(d)].astype(np.int64)

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
def _first_unset(done):
    """Index of the first ``False`` in `done` - the cheap seed for `partition`.

    Caller guarantees at least one unset node (the loop runs while
    ``not done.all()``), so the argmax always lands on a genuine ``False``.
    """
    return int(np.argmax(~done))


def _finalize(region, size, undersized, pad_value):
    """Apply the `undersized` policy to one fragment.

    Full-size fragments pass through unchanged. Undersized ones are padded to
    `size`, kept ragged, or dropped (returns None -> caller skips it).
    """
    if len(region) >= size:
        return region
    if undersized == "pad":
        fill = np.full(size - len(region), pad_value, dtype=np.int64)
        return np.concatenate([region, fill])
    if undersized == "keep":
        return region
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
