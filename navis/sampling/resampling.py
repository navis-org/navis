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

import warnings

import trimesh as tm
import pandas as pd
import numpy as np
import scipy.spatial
import scipy.interpolate

from typing import Union, Optional
from typing_extensions import Literal

from .. import config, core, utils, graph
from ..core import schema

# `method` is handed straight to `scipy.interpolate.interp1d(kind=...)`. Check
# it here: an unknown value otherwise surfaces from deep inside scipy as
# "NotImplementedError: <method> is unsupported: Use fitpack routines for other
# types", which names neither navis nor the parameter, and points at a library
# the caller never asked for.
INTERP_METHODS = (
    "linear",
    "nearest",
    "nearest-up",
    "zero",
    "slinear",
    "quadratic",
    "cubic",
    "previous",
    "next",
)

# Set up logging
logger = config.get_logger(__name__)

__all__ = ['resample_skeleton', 'resample_along_axis']


@utils.map_neuronlist(desc='Resampling', allow_parallel=True)
@utils.rebuilds('nodes')
def resample_skeleton(x: 'core.NeuronObject',
                      resample_to: Union[int, str],
                      inplace: bool = False,
                      method: str = 'linear',
                      map_columns: Optional[list] = None,
                      skip_errors: bool = True,
                      preserve_volume: bool = False
                      ) -> Optional['core.NeuronObject']:
    """Resample skeleton(s) to given resolution.

    Preserves root, leafs and branchpoints. Soma, connectors and node tags
    (if present) are mapped onto the closest node in the resampled neuron.

    Important
    ---------
    A few things to keep in mind:
      - This generates an entirely new set of node IDs! They will be unique
        within a neuron, but you may encounter duplicates across neurons.
      - Any non-standard node table columns (e.g. "labels") will be lost.
      - Soma(s) will be pinned to the closest node in the resampled neuron.
      - We may end up upcasting the data type for node and parent IDs to
        accommodate the new node IDs.

    Also: be aware that high-resolution neurons will use A LOT of memory.

    Parameters
    ----------
    x :                 Skeleton | NeuronList
                        Neuron(s) to resample.
    resample_to :       int | float | str
                        Target sampling resolution, i.e. one node every
                        N units of cable. Note that hitting the exact
                        sampling resolution might not be possible e.g. if
                        a branch is shorter than the target resolution. If
                        neuron(s) have their `.units` parameter, you can also
                        pass a string such as "1 micron".
    method :            str, optional
                        See `scipy.interpolate.interp1d` for possible
                        options. By default, we're using linear interpolation.
    map_columns :       list of str, optional
                        Names of additional columns to carry over to the resampled
                        neuron. Numerical columns will be interpolated according to
                        `method`. Non-numerical columns will be interpolated
                        using nearest neighbour interpolation.
    inplace :           bool, optional
                        If True, will modify original neuron. If False, a
                        resampled copy is returned.
    skip_errors :       bool, optional
                        If True, will skip errors during interpolation and
                        only print summary.
    preserve_volume :   bool, optional
                        By default, radii are interpolated point-wise like any
                        other numeric column, which does *not* conserve a
                        segment's radius-based volume (`.volume`): resampling
                        skips over kinks in the radius profile and cuts corners
                        of the path. If True, the radii of the newly inserted
                        (interior) nodes are rescaled per small segment so that
                        each segment matches its original frustum volume. The
                        segment endpoints (roots, leafs, branch points) keep
                        their interpolated radii - they are shared between
                        segments, so scaling them would make a branch point's
                        radius ambiguous. Segments with no interior nodes (e.g.
                        collapsed short segments) cannot be corrected and are
                        left as-is.

    Returns
    -------
    Skeleton/List
                        Downsampled neuron(s).

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> # Check sampling resolution (nodes/cable)
    >>> round(n.sampling_resolution)
    60
    >>> # Resample to 1 micron (example neurons are in 8x8x8nm)
    >>> n_rs = navis.resample_skeleton(n,
    ...                                resample_to=1000 / 8,
    ...                                inplace=False)
    >>> round(n_rs.sampling_resolution)
    112

    See Also
    --------
    [`navis.downsample_neuron`][]
                        This function reduces the number of nodes instead of
                        resample to certain resolution. Useful if you are
                        just after some simplification - e.g. for speeding up
                        your calculations or you want to preserve node IDs.
    [`navis.resample_along_axis`][]
                        Resample neuron along a single axis such that nodes
                        align with given 1-dimensional grid.

    """
    if not isinstance(x, core.Skeleton):
        raise TypeError(f'Unable to resample data of type "{type(x)}"')

    if method not in INTERP_METHODS and not isinstance(method, (int, np.integer)):
        raise ValueError(
            f'Unknown method "{method}". Allowed values: '
            f"{', '.join(repr(m) for m in INTERP_METHODS)}, or an int giving "
            "the spline order."
        )

    # Map units (non-str are just passed through)
    resample_to = x.map_units(resample_to, on_error="raise")

    if not inplace:
        x = x.copy()

    num_cols = ["x", "y", "z", "radius"]
    non_num_cols = []

    if map_columns:
        if isinstance(map_columns, str):
            map_columns = [map_columns]

        for col in map_columns:
            if col in num_cols or col in non_num_cols:
                continue
            if col not in x.nodes.columns:
                raise ValueError(f'Column "{col}" not found in node table')
            if pd.api.types.is_numeric_dtype(x.nodes[col].dtype):
                num_cols.append(col)
            else:
                non_num_cols.append(col)

    # Capture per-segment volumes now, while `x` still holds the original nodes
    # and topology - we restore them onto the resampled neuron further down.
    orig_vols = _segment_volumes(x) if preserve_volume else None

    new_nodes = _resample_segments(
        x, resample_to, method, num_cols, non_num_cols, skip_errors
    )

    # At this point, new node and parent IDs will be 64 bit integers and x/y/z columns will
    # be float 64. We will convert them back to the original dtypes but we have to
    # be careful with node & parent IDs to avoid overflows if the original datatype
    # can't accommodate the new IDs.

    # Gather the original dtypes
    dtypes = {
        k: x.nodes[k].dtype for k in ["node_id", "parent_id"] + num_cols + non_num_cols
    }

    # Check for overflow
    for col in ("node_id", "parent_id"):
        # No need for checks if we're not changing the dtype
        if new_nodes[col].dtype == dtypes[col]:
            continue

        # If there is an overflow downcast to smallest possible dtype
        # N.B. we could also check for underflow but that's less likely
        if new_nodes[col].max() >= np.iinfo(np.int32).max:
            new_nodes[col] = pd.to_numeric(new_nodes[col], downcast="integer")
            dtypes[col] = new_nodes[col].dtype  # Update dtype

    # Now cast the rest
    new_nodes = new_nodes.astype(dtypes, errors="ignore")

    # Remove duplicate nodes (branch points)
    new_nodes = new_nodes[~new_nodes.node_id.duplicated()]

    # `.soma` may be a detection function, and the radii it goes by change under
    # resampling - so resolve it now and pin the result. This has to happen
    # *before* the snap map is built below: a callable is a rule for finding the
    # soma rather than a reference to a node, so `referenced_values` rightly does
    # not report one - and a soma that is not in the map has nowhere to move to.
    soma = x.soma
    if np.any(soma):
        # `_soma` to skip the checks - the new nodes are not assigned yet
        x._soma = list(soma) if utils.is_iterable(soma) else soma
    else:
        # If `._soma` was (read: is) a function but it didn't detect anything in
        # the original neuron, this makes sure the resampled one has no soma
        # either
        x.soma = None

    # Anything naming a node - the soma, connectors, tags, a link somebody
    # attached - is pinned to whichever node in the resampled neuron ended up
    # closest to its original position. Resampling does not remove any part of
    # the neuron, so that node is the same place on the same branch. Handing the
    # map back rather than applying it here is what lets all of them follow it,
    # each by the policy it was declared with (see `schema.Rebuild`).
    # Only the nodes something actually names: the query is priced per point,
    # and a skeleton with no soma, connectors or tags has nothing to ask about.
    named = schema.referenced_values(x, schema.get_axis(x, "nodes"))
    if len(named):
        tree = scipy.spatial.cKDTree(new_nodes[["x", "y", "z"]].values)
        old_pos = x.nodes.set_index("node_id", inplace=False)
        _, ix = tree.query(old_pos.loc[named, ["x", "y", "z"]].values)
        snap = (named, new_nodes.node_id.values[ix])
    else:
        snap = None

    # The soma comes out of the way for the swap. `_clear_temp_attr` drops one
    # whose node is not in the table, and that is exactly the state a rebuild
    # leaves it in between the new nodes going on and `apply_rebuild` snapping
    # the references - so leaving it there would delete the very soma we just
    # pinned. Connectors and tags are not repaired from here, so they can stay.
    soma_ref, x._soma = x._soma, None

    # Set nodes (avoid setting on copy warning)
    x.nodes = new_nodes.copy()

    # Clear and regenerate temporary attributes
    x._clear_temp_attr()

    # Restore each segment's original volume by rescaling interior radii. This
    # must run after `_clear_temp_attr` so that `.small_segments` reflects the
    # resampled topology.
    if preserve_volume:
        _conserve_segment_volumes(x, orig_vols)

    # Put it back for `apply_rebuild` to move onto the node it now belongs on -
    # last, so that nothing in between can take it for a dangling reference.
    x._soma = soma_ref

    # No `kept`: resampling re-samples, so a node that comes out carrying an old
    # ID is not thereby the old node, and claiming otherwise would carry data
    # onto points it does not describe. Anything aligned to the nodes is dropped
    # (loudly) rather than guessed at.
    return x, schema.Rebuild(snap=snap)


def _flatten_segments(x):
    """Flatten `x`'s small segments into aligned arrays for vectorised per-segment math.

    Returns `(seg_n_nodes, offsets, flat, radius, step)`, following the same
    layout as `_resample_segments`: `flat` is the concatenation of all small
    segments (branch points appear once per segment they belong to), `radius` the
    matching float64 radii, and `step[i]` the length of the edge from `flat[i-1]`
    to `flat[i]` with segment-boundary steps zeroed - so edges never straddle two
    segments.
    """
    segs = x.small_segments
    seg_n_nodes = np.array([len(s) for s in segs], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(seg_n_nodes)))
    if len(segs):
        flat = np.concatenate([np.asarray(s) for s in segs])
    else:
        flat = np.empty(0, dtype=np.int64)

    row_ix = pd.Index(x.nodes.node_id.values).get_indexer(flat)
    radius = x.nodes.radius.values.astype(np.float64)[row_ix]
    xyz = x.nodes[["x", "y", "z"]].values.astype(np.float64)[row_ix]

    step = np.zeros(len(flat))
    if len(flat):
        step[1:] = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        step[offsets[:-1]] = 0.0

    return seg_n_nodes, offsets, flat, radius, step


def _reduce_segment_volumes(radius, step, offsets):
    """Frustum volume of each segment, summed from its per-edge contributions.

    Truncated-cone model: edge `i` (from `flat[i-1]` to `flat[i]`) contributes
    `(π step[i] / 3)(r_{i-1}² + r_{i-1} r_i + r_i²)`. Segment-boundary rows carry
    `step == 0`, so cross-segment radii never enter the sum. NaN radii are treated
    as zero-volume, matching `np.nansum`.
    """
    r_prev = np.empty_like(radius)
    r_prev[0] = 0.0
    r_prev[1:] = radius[:-1]
    edge = (np.pi * step / 3.0) * (r_prev ** 2 + r_prev * radius + radius ** 2)
    edge = np.nan_to_num(edge, nan=0.0)
    edge[offsets[:-1]] = 0.0
    return np.add.reduceat(edge, offsets[:-1])


def _segment_volumes(x):
    """Radius-based volume of each of `x`'s small segments.

    Uses the same truncated-cone (frustum) model as `Skeleton.volume`: an edge
    of length L between nodes of radius r1 and r2 has volume
    `(π L / 3)(r1² + r1 r2 + r2²)`. This is the exact `∫ π r(s)² ds` for a radius
    that varies linearly along the edge - i.e. the quantity `preserve_volume`
    conserves. Segments are keyed by their `(distal, proximal)` endpoint node IDs
    (ID-sorted so the key is independent of traversal direction); those endpoints
    are preserved by resampling, which is what lets us match new segments back
    onto their originals.
    """
    seg_n_nodes, offsets, flat, radius, step = _flatten_segments(x)
    if not len(flat):
        return {}

    vols = _reduce_segment_volumes(radius, step, offsets)
    first = flat[offsets[:-1]]
    last = flat[offsets[1:] - 1]
    return {tuple(sorted((fst, lst))): v for fst, lst, v in zip(first, last, vols)}


def _conserve_segment_volumes(x, orig_vols):
    """Rescale interior node radii so each small segment matches its original volume.

    The endpoints of every small segment (roots, leafs, branch points) are
    preserved by resampling and keep their interpolated radii - they are shared
    between segments, so scaling them would make a branch point's radius
    ambiguous. Only the newly inserted *interior* nodes are scaled, by a single
    per-segment factor `f`. Because every radius in the frustum sum is then either
    fixed (endpoint) or `f·ρ` (interior), the segment volume is a quadratic
    `V(f) = A f² + B f + C`; we recover the coefficients by evaluating `V` at
    `f = 0, 1, 2` and solve `V(f) = V_orig` for the positive root nearest 1. All
    of this is done across every segment at once.
    """
    seg_n_nodes, offsets, flat, radius, step = _flatten_segments(x)
    if not len(flat):
        return

    # Interior = every node that is neither the first nor last of its segment
    interior = np.ones(len(flat), dtype=bool)
    interior[offsets[:-1]] = False
    interior[offsets[1:] - 1] = False

    # V(f) for a scalar f, evaluated for all segments at once
    def seg_vols(f):
        return _reduce_segment_volumes(np.where(interior, radius * f, radius),
                                       step, offsets)

    c = seg_vols(0.0)
    v1 = seg_vols(1.0)
    v2 = seg_vols(2.0)
    a = (v2 - 2.0 * v1 + c) / 2.0
    b = v1 - c - a

    # Match each segment onto its original volume via the preserved endpoints
    first = flat[offsets[:-1]]
    last = flat[offsets[1:] - 1]
    target = np.array([orig_vols.get(tuple(sorted((fst, lst))), np.nan)
                       for fst, lst in zip(first, last)])

    f = _solve_scale(a, b, c - target)

    # A segment is correctable only if it has interior nodes, matched an original
    # and yielded a positive scale factor. Segments with no interior nodes (e.g.
    # collapsed short segments) are expected and left as-is without warning.
    want = (seg_n_nodes > 2) & ~np.isnan(target)
    applied = want & ~np.isnan(f)
    skipped = int(np.sum(want & np.isnan(f)))

    seg_scale = np.ones(len(seg_n_nodes))
    seg_scale[applied] = f[applied]
    upd = interior & np.repeat(applied, seg_n_nodes)
    if upd.any():
        new_r = radius * np.repeat(seg_scale, seg_n_nodes)
        dtype = x.nodes["radius"].dtype
        mapped = x.nodes["node_id"].map(pd.Series(new_r[upd], index=flat[upd]))
        x.nodes["radius"] = mapped.fillna(x.nodes["radius"]).astype(dtype)

    if skipped:
        logger.warning(
            f"{skipped} segment(s) could not be volume-corrected (no positive "
            "scale factor found); their volume may have changed."
        )


def _solve_scale(a, b, c, lo=0.1, hi=10.0):
    """Positive root of `a f² + b f + c = 0` nearest 1, clamped to [lo, hi].

    Vectorised over segments: `a`, `b`, `c` are arrays and the return value is an
    array with NaN wherever there is no positive (real) root - the caller then
    leaves that segment's radii untouched.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        # Two quadratic roots (NaN where the discriminant is negative) plus the
        # single linear root for the degenerate a≈0 case.
        sq = np.sqrt(np.where((disc := b * b - 4.0 * a * c) >= 0, disc, np.nan))
        cand = np.stack([(-b + sq) / (2.0 * a),
                         (-b - sq) / (2.0 * a),
                         np.where(np.abs(b) > 1e-12, -c / b, np.nan)], axis=-1)
        # Keep only the meaningful candidate(s): quadratic roots unless a≈0, in
        # which case only the linear root applies.
        lin = np.abs(a) < 1e-12
        cand[lin, :2] = np.nan
        cand[~lin, 2] = np.nan

    cand = np.where(cand > 0, cand, np.nan)     # positive roots only
    dist = np.where(np.isnan(cand), np.inf, np.abs(cand - 1.0))
    chosen = np.take_along_axis(cand, np.argmin(dist, axis=-1)[..., None], axis=-1)[..., 0]
    return np.clip(chosen, lo, hi)              # NaN (no positive root) stays NaN


def _resample_segments(x, resample_to, method, num_cols, non_num_cols, skip_errors):
    """Build the resampled node table for `x`.

    Each of the neuron's small segments is resampled independently at
    `resample_to` intervals. The first and last node of a segment are always
    kept, which is what preserves roots, leafs and branch points.

    Returns a DataFrame with columns `["node_id", "parent_id"] + num_cols +
    non_num_cols`. Node/parent IDs are int64 and all other columns float64 -
    the caller casts them back to the neuron's original dtypes.
    """
    nodes = x.nodes
    cols = num_cols + non_num_cols
    n_num = len(num_cols)
    segs = x.small_segments

    # Pack every column into a single float matrix so that we can interpolate
    # them all with the same machinery. Non-numerical columns are label-encoded
    # and decoded again at the very end.
    # N.B. `use_na_sentinel=False` makes NaN a category in its own right - without
    # it NaN would encode to -1 and decode back to whatever the *last* category is.
    num2cat = {}
    encoded = []
    for col in cols:
        if col in non_num_cols:
            codes, num2cat[col] = pd.factorize(nodes[col].values, use_na_sentinel=False)
            encoded.append(codes.astype(np.float64))
        else:
            encoded.append(nodes[col].values.astype(np.float64))
    vals = np.stack(encoded, axis=1)

    if not len(segs):
        new_nodes = _empty_rows(cols)
    else:
        # Flatten the segments into one long array so that we only have to
        # translate node IDs into row indices once.
        seg_n_nodes = np.array([len(s) for s in segs])
        offsets = np.concatenate(([0], np.cumsum(seg_n_nodes)))
        flat = np.concatenate([np.asarray(s) for s in segs])
        vals_flat = vals[pd.Index(nodes.node_id.values).get_indexer(flat)]

        # Cumulative distance along each segment, computed for all segments at
        # once: take the norm between consecutive rows of the flattened array,
        # zero out the steps that straddle a segment boundary, then subtract
        # each segment's starting offset from the global cumulative sum.
        step = np.zeros(len(flat))
        step[1:] = np.linalg.norm(np.diff(vals_flat[:, :3], axis=0), axis=1)
        step[offsets[:-1]] = 0
        cum = np.cumsum(step)
        dist = cum - np.repeat(cum[offsets[:-1]], seg_n_nodes)
        seg_cable = dist[offsets[1:] - 1]

        # Segments that are shorter than the target resolution are collapsed to
        # their first and last node. Same for segments with too few nodes to fit
        # a cubic spline.
        keep = seg_cable < resample_to
        if method == "cubic":
            keep |= seg_n_nodes <= 3

        resampler = _resample_linear if method == "linear" else _resample_nonlinear
        new_nodes = resampler(
            flat, vals_flat, dist, seg_cable, seg_n_nodes, offsets, keep,
            resample_to, nodes, cols, n_num, method, skip_errors
        )

    # Add the root node(s): they only ever appear as a segment's *last* node
    # and are therefore not among the rows generated above.
    root_ix = pd.Index(nodes.node_id.values).get_indexer(utils.make_iterable(x.root))
    new_nodes = {
        "node_id": np.concatenate([new_nodes["node_id"], nodes.node_id.values[root_ix]]),
        "parent_id": np.concatenate([new_nodes["parent_id"], nodes.parent_id.values[root_ix]]),
        "values": np.vstack([new_nodes["values"], vals[root_ix]]),
    }

    data = {"node_id": new_nodes["node_id"], "parent_id": new_nodes["parent_id"]}
    for i, col in enumerate(cols):
        if col in non_num_cols:
            data[col] = num2cat[col][new_nodes["values"][:, i].astype(int)]
        else:
            data[col] = new_nodes["values"][:, i]

    return pd.DataFrame(data)


def _empty_rows(cols):
    """Empty row collection in the format used by the `_resample_*` helpers."""
    return {
        "node_id": np.empty(0, dtype=np.int64),
        "parent_id": np.empty(0, dtype=np.int64),
        "values": np.empty((0, len(cols)), dtype=np.float64),
    }


def _sample_counts(seg_cable, resample_to):
    """Number of sample points for each segment to be resampled.

    A segment of length L sampled at N points spans N - 1 intervals, so hitting a
    spacing of `resample_to` takes `round(L / resample_to) + 1` points. Note the
    caller only ever resamples segments with `L >= resample_to`, so this is always
    at least 2 (i.e. the segment's start and end node).
    """
    return np.round(seg_cable / resample_to).astype(np.int64) + 1


def _new_ids(first_id, last_id, n_pts, max_id):
    """Generate node and parent IDs for the resampled segments.

    The first and last node ID of each segment are preserved; the nodes in
    between get fresh IDs starting at `max_id`. Each segment emits `n_pts - 1`
    rows: node IDs `[first, new_0, ..., new_n-3]` with parents shifted by one,
    `[new_0, ..., new_n-3, last]`.
    """
    # N.B. we consume only `n_pts - 2` IDs per segment but advance the counter by
    # `n_pts` - this leaves gaps but keeps the IDs unique, which is all we need
    id_base = max_id + np.concatenate(([0], np.cumsum(n_pts)[:-1]))

    n_new = n_pts - 2
    new_off = np.concatenate(([0], np.cumsum(n_new)))
    interm = np.repeat(id_base, n_new) + (
        np.arange(new_off[-1]) - np.repeat(new_off[:-1], n_new)
    )

    n_rows = n_pts - 1
    row_off = np.concatenate(([0], np.cumsum(n_rows)))
    is_first = np.zeros(row_off[-1], dtype=bool)
    is_first[row_off[:-1]] = True
    is_last = np.zeros(row_off[-1], dtype=bool)
    is_last[row_off[1:] - 1] = True

    node_id = np.empty(row_off[-1], dtype=np.int64)
    node_id[is_first] = first_id
    node_id[~is_first] = interm

    parent_id = np.empty(row_off[-1], dtype=np.int64)
    parent_id[~is_last] = interm
    parent_id[is_last] = last_id

    return node_id, parent_id


def _resample_linear(flat, vals_flat, dist, seg_cable, seg_n_nodes, offsets, keep,
                     resample_to, nodes, cols, n_num, method, skip_errors):
    """Resample all segments at once using linear interpolation.

    Rather than fitting an interpolator per segment, we map the segments onto a
    single, strictly monotonic axis - segment `i` occupies `[i, i + 0.5]`, which
    leaves a gap before segment `i + 1` - and interpolate each column with a
    single `np.interp` call across the whole neuron.

    Note this only works because linear interpolation is *local*: only the two
    knots bracketing a sample contribute to it, so no segment can bleed into its
    neighbours across the gaps. See `_resample_nonlinear` for the methods where
    that does not hold.
    """
    resample = ~keep
    n_pts = _sample_counts(seg_cable[resample], resample_to)
    n_rows = n_pts - 1

    # Rows are laid out in segment order (i.e. short and resampled segments
    # interleaved) so that the row order matches the order of `.small_segments`
    rows_per_seg = np.ones(len(seg_n_nodes), dtype=np.int64)
    rows_per_seg[resample] = n_rows
    seg_row_off = np.concatenate(([0], np.cumsum(rows_per_seg)))

    node_id = np.empty(seg_row_off[-1], dtype=np.int64)
    parent_id = np.empty(seg_row_off[-1], dtype=np.int64)
    values = np.empty((seg_row_off[-1], len(cols)), dtype=np.float64)

    first_id = flat[offsets[:-1]]
    last_id = flat[offsets[1:] - 1]

    # Segments that are too short collapse into a single row
    short = seg_row_off[:-1][keep]
    node_id[short] = first_id[keep]
    parent_id[short] = last_id[keep]
    values[short] = vals_flat[offsets[:-1][keep]]

    if resample.any():
        # Normalised sampling axis. N.B. we normalise by segment length rather
        # than offsetting by the cumulative cable length: the latter grows large
        # enough to eat into float64's mantissa, which would shift the
        # tie-breaking of the nearest-neighbour interpolation below.
        base = np.arange(len(seg_n_nodes), dtype=np.float64)
        safe_cable = np.where(seg_cable > 0, seg_cable, 1)
        xp = np.repeat(base, seg_n_nodes) + 0.5 * dist / np.repeat(safe_cable, seg_n_nodes)

        # Sample points, i.e. `base + 0.5 * linspace(0, 1, n_pts)` per segment
        pt_off = np.concatenate(([0], np.cumsum(n_pts)))
        within = np.arange(pt_off[-1]) - np.repeat(pt_off[:-1], n_pts)
        x_new = np.repeat(base[resample], n_pts) + 0.5 * within / np.repeat(n_pts - 1, n_pts)

        samples = np.empty((pt_off[-1], len(cols)), dtype=np.float64)
        for i in range(n_num):
            samples[:, i] = np.interp(x_new, xp, vals_flat[:, i])

        # Non-numerical columns are mapped by nearest neighbour, on the raw (i.e.
        # un-normalised) distances and one segment at a time. That's a hair slower
        # but reproduces scipy's tie-breaking at coincident nodes exactly, which
        # the normalised axis would not.
        if n_num < len(cols):
            starts, ends = offsets[:-1][resample], offsets[1:][resample]
            for i in range(len(n_pts)):
                s, e = starts[i], ends[i]
                samples[pt_off[i]:pt_off[i + 1], n_num:] = _nearest(
                    dist[s:e],
                    vals_flat[s:e, n_num:],
                    np.linspace(0, seg_cable[resample][i], n_pts[i]),
                )

        # Each segment's last sample is only ever used as a parent, never as a
        # node in its own right - so drop its values
        not_last = np.ones(pt_off[-1], dtype=bool)
        not_last[pt_off[1:] - 1] = False

        rows = _rows_for(seg_row_off[:-1][resample], n_rows)
        node_id[rows], parent_id[rows] = _new_ids(
            first_id[resample], last_id[resample], n_pts, nodes.node_id.max() + 1
        )
        values[rows] = samples[not_last]

    return {"node_id": node_id, "parent_id": parent_id, "values": values}


def _resample_nonlinear(flat, vals_flat, dist, seg_cable, seg_n_nodes, offsets, keep,
                        resample_to, nodes, cols, n_num, method, skip_errors):
    """Resample all segments using a non-linear interpolation method.

    Unlike linear (and nearest) interpolation, methods such as "cubic" are
    *global*: the fitted curve is continuous across all knots. We must therefore
    fit them one segment at a time - but we can at least fit all numerical
    columns in one go by interpolating along `axis=0`.
    """
    resample = ~keep
    n_pts = _sample_counts(seg_cable[resample], resample_to)

    first_id = flat[offsets[:-1]]
    last_id = flat[offsets[1:] - 1]

    # Pre-generate the IDs for every segment we intend to resample, then hand
    # each segment its slice below
    node_id, parent_id = _new_ids(
        first_id[resample], last_id[resample], n_pts, nodes.node_id.max() + 1
    )
    row_off = np.concatenate(([0], np.cumsum(n_pts - 1)))
    # Maps a segment index onto its position among the resampled segments
    pos = np.zeros(len(seg_n_nodes), dtype=np.int64)
    pos[resample] = np.arange(len(n_pts))

    errors = 0
    chunks = []
    for i in range(len(seg_n_nodes)):
        s, e = offsets[i], offsets[i + 1]

        # Segments that are too short collapse into a single row
        if keep[i]:
            chunks.append((first_id[i:i + 1], last_id[i:i + 1], vals_flat[s:s + 1]))
            continue

        j = pos[i]
        lo, hi = row_off[j], row_off[j + 1]
        # N.B. we drop the last sample point: it is only ever used as a parent ID,
        # never as a node in its own right
        new_dist = np.linspace(0, seg_cable[i], n_pts[j])[:-1]

        values = np.empty((hi - lo, len(cols)), dtype=np.float64)
        try:
            values[:, :n_num] = scipy.interpolate.interp1d(
                dist[s:e], vals_flat[s:e, :n_num], kind=method, axis=0
            )(new_dist)
        except ValueError:
            if not skip_errors:
                raise
            # Fall back to keeping the segment's original nodes
            errors += 1
            chunks.append((flat[s:e - 1], flat[s + 1:e], vals_flat[s:e - 1]))
            continue

        if n_num < len(cols):
            values[:, n_num:] = _nearest(
                dist[s:e], vals_flat[s:e, n_num:], new_dist
            )

        chunks.append((node_id[lo:hi], parent_id[lo:hi], values))

    if errors:
        logger.warning(
            f"{errors} ({errors / len(seg_n_nodes):.0%}) segments skipped due to errors"
        )

    return {
        "node_id": np.concatenate([c[0] for c in chunks]).astype(np.int64),
        "parent_id": np.concatenate([c[1] for c in chunks]).astype(np.int64),
        "values": np.vstack([c[2] for c in chunks]),
    }


def _rows_for(row_starts, n_rows):
    """Row indices occupied by each segment, flattened."""
    off = np.concatenate(([0], np.cumsum(n_rows)))
    return np.repeat(row_starts, n_rows) + (
        np.arange(off[-1]) - np.repeat(off[:-1], n_rows)
    )


def _nearest(dist, values, new_dist):
    """Nearest-neighbour interpolation.

    Equivalent to `scipy.interpolate.interp1d(dist, values, kind="nearest")` -
    including the halving-before-adding, which matters for reproducing scipy's
    tie-breaking at coincident nodes.
    """
    half = dist / 2.0
    bounds = half[1:] + half[:-1]
    ix = np.searchsorted(bounds, new_dist, side="left").clip(0, len(dist) - 1)
    return values[ix]


@utils.map_neuronlist(desc='Binning', allow_parallel=True)
def resample_along_axis(x: 'core.Skeleton',
                        interval: Union[int, float, str],
                        axis: int = 2,
                        old_nodes: Union[Literal['remove'],
                                         Literal['keep'],
                                         Literal['snap']] = 'remove',
                        inplace: bool = False
                        ) -> Optional['core.Skeleton']:
    """Resample neuron such that nodes lie exactly on given 1d grid.

    This function does not simply snap nodes to the closest grid line but
    instead adds new nodes where edges between existing nodes intersect
    with the planes defined by the grid.

    Parameters
    ----------
    x :             Skeleton | NeuronList
                    Neuron(s) to resample.
    interval :      float | int | str
                    Intervals defining a 1-dimensional grid along given axes
                    (see examples). If neuron(s) have `.units` set, you can also
                    pass a string such as "50 nm".
    axis :           0 | 1 | 2
                    Along which axes (x/y/z) to resample.
    old_nodes :     "remove" | "keep" | "snap"
                    Existing nodes are unlikely to intersect with the planes as
                    defined by the grid interval. There are three possible ways
                    to deal with them:
                     - "remove" (default) will simply drop old nodes: this
                       guarantees all remaining nodes will lie on a plane
                     - "keep" will keep old nodes without changing them
                     - "snap" will snap those nodes to the closest coordinate
                       on the grid without interpolation

    inplace :       bool
                    If False, will resample and return a copy of the original. If
                    True, will resample input neuron in place.

    Returns
    -------
    Skeleton/List
                    The resampled neuron(s).

    See Also
    --------
    [`navis.resample_skeleton`][]
                        Resample neuron such that edges between nodes have a
                        given length.
    [`navis.downsample_neuron`][]
                        This function reduces the number of nodes instead of
                        resample to certain resolution. Useful if you are
                        just after some simplification e.g. for speeding up
                        your calculations or you want to preserve node IDs.

    Examples
    --------
    Resample neuron such that we have one node in every 40nm slice along z axis

    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> n.n_nodes
    4465
    >>> res = navis.resample_along_axis(n, interval='40 nm',
    ...                                 axis=2, old_nodes='remove')
    >>> res.n_nodes < n.n_nodes
    True

    """
    utils.eval_param(axis, name='axis', allowed_values=(0, 1, 2))
    utils.eval_param(old_nodes, name='old_nodes',
                     allowed_values=("remove", "keep", "snap"))
    utils.eval_param(x, name='x', allowed_types=(core.Skeleton, ))

    interval = x.map_units(interval, on_error='raise')

    if not inplace:
        x = x.copy()

    # Collect coordinates of nodes and their parents
    nodes = x.nodes
    not_root = nodes.loc[nodes.parent_id >= 0]
    node_locs = not_root[['x', 'y', 'z']].values
    parent_locs = nodes.set_index('node_id').loc[not_root.parent_id.values,
                                                 ['x', 'y', 'z']].values

    # Get all vectors
    vecs = parent_locs - node_locs

    # Get coordinates along this axis
    loc1 = node_locs[:, axis]
    loc2 = parent_locs[:, axis]

    # This prevents runtime warnings e.g. from division by zero
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Find out which grid interval these are on
        int1 = (loc1 / interval).astype(int)
        int2 = (loc2 / interval).astype(int)

        # Difference in bin between both locs
        diff = int2 - int1
        sign = diff / np.abs(diff)

        # Figure out by how far we are from the gridline
        dist = np.zeros(diff.shape[0])
        dist[diff < 0] = loc1[diff < 0] % interval
        dist[diff > 0] = -loc1[diff > 0] % interval

        # Now we need to calculate the new position
        # Get other axes
        other_axes = list({0, 1, 2} - {axis})
        # Normalize other vectors by this vector
        other_vecs_norm = vecs[:, other_axes] / vecs[:, [axis]]

        # Get offset for other axis
        other_offset = other_vecs_norm * dist.reshape(dist.shape[0], 1)

        # Offset for this axis
        this_offset = dist * sign

    # Apply offsets
    new_coords = node_locs.copy()
    new_coords[:, other_axes] += other_offset * sign.reshape(sign.shape[0], 1)
    new_coords[:, [axis]] += this_offset.reshape(this_offset.shape[0], 1)

    # Now extract nodes that need to be inserted
    insert_between = not_root.loc[diff != 0, ['node_id', 'parent_id']].values
    new_coords = new_coords[diff != 0]

    # Insert nodes
    graph.insert_nodes(x, where=insert_between, coords=new_coords, inplace=True)

    # Figure out what to do with nodes that are not on the grid
    if old_nodes == 'remove':
        mod = x.nodes[['x', 'y', 'z'][axis]].values % interval
        not_lined_up = mod != 0
        to_remove = x.nodes.loc[not_lined_up, 'node_id'].values
    elif old_nodes == 'keep':
        to_remove = insert_between[:, 0]
    elif old_nodes == 'snap':
        not_lined_up = x.nodes[['x', 'y', 'z']].values[:, axis] % interval != 0
        to_snap = x.nodes.loc[not_lined_up, ['x', 'y', 'z'][axis]].values
        snapped = (to_snap / interval).round() * interval
        x.nodes.loc[not_lined_up, ['x', 'y', 'z'][axis]] = snapped
        to_remove = []

    if np.any(to_remove):
        graph.remove_nodes(x, which=to_remove, inplace=True)

    return x


def _make_grid(interval, axis, neuron):
    """Generate Volume visualizing 1d grid."""
    assert axis in (0, 1, 2)
    bounds = neuron.bbox

    # Generate a box for each plane - just a face won't render properly
    b = tm.primitives.Box()
    box_verts = np.array(b.vertices)
    box_faces = np.array(b.faces)
    for i in range(3):
        is_low = box_verts[:, i] < 0
        box_verts[is_low, i] = bounds[i][0]
        box_verts[~is_low, i] = bounds[i][1]

    is_low = b.vertices[:, axis] < 0

    start = (bounds[axis][0] / interval).astype(int) * interval
    end = ((bounds[axis][1] / interval).astype(int) + 1) * interval
    depth = np.arange(start, end + interval, interval)

    faces = []
    vertices = []
    for i, d in enumerate(depth):
        this_verts = box_verts.copy()
        this_faces = box_faces.copy()

        this_verts[is_low, axis] = d - 0.01 * interval
        this_verts[~is_low, axis] = d + 0.01 * interval

        this_faces += this_verts.shape[0] * i

        vertices.append(this_verts)
        faces.append(this_faces)
    faces = np.vstack(faces)
    vertices = np.vstack(vertices)

    return core.Volume(vertices=vertices, faces=faces, color=(1, 1, 1, .1))
