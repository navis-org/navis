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

"""Declarative description of a neuron's element axes.

Every neuron type stores its data along one or more *axes*: skeletons along
`nodes`, meshes along `vertices`, dotprops along `points`. Lots of other data is
either aligned to one of those axes (a dotprops' tangent vectors are one per
point) or *references* it (a connector table's `node_id` column, a face's three
vertex indices, the soma).

Selecting a subset of a neuron therefore always means the same three things:

1. subset every attribute aligned to the axis,
2. drop or repair everything that references elements which did not survive,
3. remember where the survivors came from, so edits can be mapped back.

Only step 3 is new; steps 1 and 2 used to be hand-written once per neuron type,
which is how `subset_neuron` came to filter a skeleton's `tags` but not a
dotprops' `soma`. Here they are written once and driven by the `AXES`
declaration each neuron class carries, next to `TEMP_ATTR` and `CORE_DATA`.

Notes
-----
Axes come in two flavours, and the distinction drives everything below:

- **id-bearing** (`Axis.ids` is set): elements carry a stable identifier, e.g. a
  skeleton node's `node_id`. References store those IDs, so a subset only has to
  *filter* them - surviving IDs do not change.
- **positional** (`Axis.ids` is `None`): elements are identified by their index,
  e.g. a mesh vertex. References store indices, so a subset has to filter *and*
  remap them.

"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = []

#: Sentinel for "this element did not survive".
DROPPED = -1


@dataclass(frozen=True)
class Ref:
    """Something whose *values* point at elements of an axis.

    Parameters
    ----------
    attr :          str
                    Name of the attribute holding the reference. Always the
                    private name (e.g. `_connectors`) so that repairs bypass
                    the property setters' validation, which we don't need for
                    data we know to be well-formed.
    kind :          "column" | "index_array" | "id_lists" | "scalar"
                    How to read the reference:
                      - `column`: a column of a DataFrame
                      - `index_array`: an (N, M) integer array, each row of
                        which references M elements (faces, extra edges)
                      - `id_lists`: a `{key: [element, ...]}` dict (tags)
                      - `scalar`: a single element, an array of elements,
                        `None`, or a callable (soma)
    column :        str, optional
                    Column name, for `kind="column"`.
    null :          Any, optional
                    What to do with references to elements that did not
                    survive: drop the row/entry (the default), or - if `null` is
                    given - keep the row and blank the reference out with this
                    value. Only honoured by `kind="column"`.
    write_attr :    str, optional
                    Attribute to write the repaired value back to. Defaults to
                    `attr`; set it to the public name where the property setter
                    does bookkeeping we want (`extra_edges` validates and
                    maintains `SUMMARY_PROPS`).

    """

    attr: str
    kind: str
    column: Optional[str] = None
    null: Any = None
    write_attr: Optional[str] = None

    @property
    def target(self) -> str:
        """Attribute to write the repaired reference back to."""
        return self.write_attr or self.attr

    def __post_init__(self):
        if self.kind not in _REF_HANDLERS:
            raise ValueError(f'Unknown ref kind "{self.kind}"')
        if self.kind == "column" and not self.column:
            raise ValueError('`column` is required for kind="column"')


@dataclass(frozen=True)
class Axis:
    """One axis along which a neuron's elements are stored.

    Parameters
    ----------
    name :          str
                    Name of the axis, e.g. "nodes".
    data :          tuple of str
                    Attributes aligned 1:1 to this axis, in the sense that their
                    first dimension *is* the axis. The first entry is the primary
                    table and is the one `ids` refers to.
    ids :           str, optional
                    Column of `data[0]` holding stable element IDs. `None` makes
                    this a positional axis (see module docstring).
    refs :          tuple of Ref
                    Everything referencing elements of this axis - including,
                    for meshes, `_faces`, whose three columns are vertex indices.
    invalidates :   tuple of str
                    Caches derived from this axis' data, set to `None` when it
                    changes. These are attributes the corresponding *public*
                    setter would have reset for us (a dotprops' `_tree`), which
                    subsetting bypasses by writing the private name.

    """

    name: str
    data: Tuple[str, ...] = ()
    ids: Optional[str] = None
    refs: Tuple[Ref, ...] = ()
    invalidates: Tuple[str, ...] = ()

    @property
    def positional(self) -> bool:
        """Whether elements are identified by index rather than by ID."""
        return self.ids is None


def axes(*items: Axis) -> Dict[str, Axis]:
    """Build a neuron class' `AXES` mapping, keyed by axis name."""
    return {axis.name: axis for axis in items}


@dataclass
class Survivors:
    """Which elements of an axis survived a selection, and where they went.

    Built by whoever performs the actual subsetting, because only they know the
    truth: a mesh's `submesh` drops degenerate vertices that the requested mask
    wanted to keep, so a `Survivors` derived from that mask would be a lie.
    """

    #: Surviving IDs, in their new order. Only for id-bearing axes.
    ids: Optional[np.ndarray] = None
    #: Old index -> new index, `DROPPED` where the element did not survive.
    #: Only for positional axes.
    old2new: Optional[np.ndarray] = None
    #: New index -> old index. Only for positional axes. This is the
    #: provenance: element `i` of the result came from element `kept[i]`.
    kept: Optional[np.ndarray] = None

    @classmethod
    def from_ids(cls, ids: Sequence) -> "Survivors":
        """For an id-bearing axis: the IDs that survived, in their new order."""
        return cls(ids=np.asarray(ids))

    @classmethod
    def from_kept(cls, n_old: int, kept: Sequence[int]) -> "Survivors":
        """For a positional axis: old indices that survived, in their new order.

        This is exactly what `submesh(..., return_map=True)` hands back.
        """
        kept = np.asarray(kept, dtype=np.int64)
        old2new = np.full(n_old, DROPPED, dtype=np.int64)
        old2new[kept] = np.arange(len(kept))
        return cls(old2new=old2new, kept=kept)

    @classmethod
    def from_mask(cls, keep: np.ndarray) -> "Survivors":
        """For a positional axis: a boolean mask over the old elements."""
        return cls.from_kept(len(keep), np.where(keep)[0])


# ---------------------------------------------------------------------------
# Reading the schema
# ---------------------------------------------------------------------------


def get_axis(neuron, name: str) -> Axis:
    """Return a single axis declaration, raising if the neuron has no such axis."""
    declared = getattr(neuron, "AXES", {})
    if name not in declared:
        raise KeyError(
            f'{type(neuron).__name__} has no "{name}" axis '
            f"(has: {sorted(declared) or 'none'})"
        )
    return declared[name]


def axis_ids(neuron, axis: Axis) -> np.ndarray:
    """Return the current IDs along an id-bearing axis."""
    table = getattr(neuron, axis.data[0], None)
    if table is None:
        return np.array([])
    return table[axis.ids].values


def axis_length(neuron, axis: Axis) -> int:
    """Return the current number of elements along an axis."""
    primary = getattr(neuron, axis.data[0], None)
    return 0 if primary is None else len(primary)


# ---------------------------------------------------------------------------
# Selecting
# ---------------------------------------------------------------------------


def resolve_selection(neuron, axis: Axis, selection) -> np.ndarray:
    """Normalise a user-supplied selection into a boolean mask over an axis.

    Accepts, for any axis: a boolean mask of the right length, or a list of
    element indices. For id-bearing axes also: a list of element IDs, or a
    DataFrame carrying the ID column. Callers that accept richer inputs (a
    networkx graph, a callable) resolve those first.

    Returns
    -------
    np.ndarray
                Boolean mask over the axis, `True` where the element is kept.

    """
    n = axis_length(neuron, axis)

    if isinstance(selection, pd.DataFrame):
        if axis.positional or axis.ids not in selection.columns:
            raise ValueError(
                f'DataFrame selection for axis "{axis.name}" must have a '
                f'"{axis.ids}" column'
            )
        selection = selection[axis.ids].values
    elif isinstance(selection, pd.Series):
        selection = selection.values
    elif isinstance(selection, (set, frozenset)):
        selection = list(selection)

    selection = np.asarray(selection)

    # Empty selections are ambiguous (no dtype to go by) - treat as "keep none"
    if selection.size == 0:
        return np.zeros(n, dtype=bool)

    if selection.dtype == bool:
        if selection.shape != (n,):
            raise ValueError(
                f'Boolean mask for axis "{axis.name}" has length '
                f"{selection.shape[0]}, expected {n}"
            )
        return selection

    if axis.positional:
        # Values are indices into the axis. Scatter rather than
        # `np.isin(np.arange(n), selection)`, which would materialise an int64
        # array the length of the axis just to throw it away.
        keep = np.zeros(n, dtype=bool)
        in_range = selection[(selection >= 0) & (selection < n)]
        keep[in_range.astype(np.intp, copy=False)] = True
        return keep

    # Values are element IDs
    return np.isin(axis_ids(neuron, axis), selection)


def apply_selection(
    neuron,
    axis: Axis,
    keep: Optional[np.ndarray] = None,
    *,
    survivors: Optional[Survivors] = None,
    skip: Sequence[str] = (),
) -> Survivors:
    """Subset an axis and repair everything referencing it, in place.

    This is the whole of a selection for an axis, and everything that selects
    should come through here - the repair and the provenance carry are easy to
    forget and silent when forgotten.

    Parameters
    ----------
    keep :      np.ndarray, optional
                Boolean mask over the axis. Subsets the axis' data as well.
    survivors : Survivors, optional
                Pass instead of `keep` when the caller has already subset the
                data and knows better than we do what came through - a mesh's
                `submesh` resolves faces and vertices together and drops
                degenerate vertices this layer cannot predict.

    Returns
    -------
    Survivors
                Which elements made it through - the input to
                `record_provenance`.

    """
    if survivors is None:
        subset_axis_data(neuron, axis, keep)
        # For an id-bearing axis the survivors are read back off the subsetted
        # data rather than derived from `keep`, so they cannot disagree with it.
        survivors = (
            Survivors.from_mask(keep)
            if axis.positional
            else Survivors.from_ids(axis_ids(neuron, axis))
        )

    repair_refs(neuron, axis, survivors, skip=skip)
    carry_provenance(neuron, axis, survivors)
    return survivors


def carry_provenance(neuron, axis: Axis, survivors: Survivors) -> None:
    """Keep an existing provenance aligned through a further selection.

    Only positional axes need this, and the asymmetry is the point. There,
    provenance *is* the elements' identity, so it has to be subset alongside
    them or a later merge silently maps them back to the wrong place. On an
    id-bearing axis the origin is the *set* of parent elements the selection
    covered: shrinking it would claim the elements the child dropped were never
    selected, and their deletion would then not propagate on merge.
    """
    if survivors.kept is None:
        # Not a record of which elements survived, so there is nothing to
        # follow. Indexing by `None` would quietly add an axis instead.
        return
    _remap_origin(neuron, axis, lambda origin: origin[survivors.kept])


def subset_axis_data(neuron, axis: Axis, keep: np.ndarray) -> None:
    """Subset every attribute aligned to an axis, in place.

    Attributes that are absent or `None` (a dotprops' `_alpha`, say) are
    skipped rather than being brought into existence.
    """
    for attr in axis.data:
        value = getattr(neuron, attr, None)
        if value is None:
            continue
        if isinstance(value, pd.DataFrame):
            # `reset_index` both hands back a standalone frame - ref repair
            # writes columns in place and pandas would otherwise (rightly) warn
            # about writing to a slice - and drops the gappy index, which breaks
            # igraph conversion downstream.
            setattr(neuron, attr, value.loc[keep].reset_index(drop=True))
        else:
            setattr(neuron, attr, np.asarray(value)[keep])

    for attr in axis.invalidates:
        setattr(neuron, attr, None)


def repair_refs(
    neuron, axis: Axis, survivors: Survivors, skip: Sequence[str] = ()
) -> None:
    """Filter and remap everything referencing an axis, in place.

    Parameters
    ----------
    survivors : Survivors
                Which elements survived. Must have been built from what
                *actually* happened, not from the requested selection.
    skip :      sequence of str
                `Ref.attr` names to leave alone - either because the caller
                already handled them (a mesh's `_faces`, remapped by `submesh`)
                or because it was asked to (`keep_disc_cn`).

    """
    translate = _translator(axis, survivors)
    skip = set(skip)
    for ref in axis.refs:
        if ref.attr not in skip:
            _REF_HANDLERS[ref.kind](neuron, ref, translate)


# ---------------------------------------------------------------------------
# Ref handlers
#
# A handler reads its reference, hands the values to `translate` and writes the
# result back. `translate(values) -> (new_values, ok)`, where `ok` is False for
# references that no longer point at anything. Building the translator once per
# axis keeps the id-vs-positional question out of the per-reference path.
# ---------------------------------------------------------------------------


def _translator(axis: Axis, survivors: Survivors):
    """Map reference values onto the post-selection axis."""
    if not axis.positional:
        # IDs are stable, so surviving references keep their value
        return lambda values: (values, np.isin(values, survivors.ids))

    old2new = survivors.old2new

    def translate(values):
        values = np.asarray(values)
        # Guard against out-of-range references rather than letting them index
        # `old2new` and silently pick up an unrelated element.
        in_range = (values >= 0) & (values < len(old2new))
        new = np.full(values.shape, DROPPED, dtype=np.int64)
        new[in_range] = old2new[values[in_range]]
        return new, new >= 0

    return translate


def _repair_column(neuron, ref, translate):
    df = getattr(neuron, ref.attr, None)
    if not isinstance(df, pd.DataFrame) or ref.column not in df.columns:
        return

    new, ok = translate(df[ref.column].values)

    if ref.null is not None:
        # Keep every row, blank out the dangling references. Note we assign the
        # column back in place, so column order is preserved.
        df[ref.column] = np.where(ok, new, ref.null)
        setattr(neuron, ref.target, df)
        return

    df = df.loc[ok].copy()
    df[ref.column] = new[ok]
    df.reset_index(drop=True, inplace=True)
    setattr(neuron, ref.target, df)


def _repair_index_array(neuron, ref, translate):
    arr = getattr(neuron, ref.attr, None)
    if arr is None or not len(arr):
        return

    new, ok = translate(np.asarray(arr))
    # A row survives only if *every* element it references survived: a face
    # missing one of its corners is not a face.
    row_ok = ok.all(axis=1) if new.ndim > 1 else ok
    setattr(neuron, ref.target, new[row_ok])


def _repair_id_lists(neuron, ref, translate):
    mapping = getattr(neuron, ref.attr, None)
    if not mapping:
        return

    # Translate every list in one go and split the result back up. Doing it per
    # key instead would rescan the (large) survivor array once per key, which
    # for a skeleton with a few dozen tags dominates the whole subset.
    keys = [k for k, values in mapping.items() if len(values)]
    if not keys:
        setattr(neuron, ref.target, {})
        return

    lists = [np.asarray(mapping[k]) for k in keys]
    bounds = np.cumsum([len(values) for values in lists])
    new, ok = translate(np.concatenate(lists))

    repaired = {}
    for key, chunk_new, chunk_ok in zip(
        keys, np.split(new, bounds[:-1]), np.split(ok, bounds[:-1])
    ):
        kept = chunk_new[chunk_ok]
        if len(kept):
            repaired[key] = kept.tolist()
    setattr(neuron, ref.target, repaired)


def _repair_scalar(neuron, ref, translate):
    value = getattr(neuron, ref.attr, None)

    # A callable is a *rule* for finding the element (e.g. the default
    # `find_soma`), not a reference to one - it stays valid across a subset.
    if value is None or callable(value):
        return

    scalar = not isinstance(value, (list, tuple, np.ndarray, pd.Series))
    new, ok = translate(np.atleast_1d(np.asarray(value)))
    kept = new[ok]

    if scalar:
        setattr(neuron, ref.target, kept[0] if len(kept) else None)
    else:
        setattr(neuron, ref.target, kept if len(kept) else None)


_REF_HANDLERS = {
    "column": _repair_column,
    "index_array": _repair_index_array,
    "id_lists": _repair_id_lists,
    "scalar": _repair_scalar,
}


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


@dataclass
class Provenance:
    """Where a selected neuron's elements came from.

    Recorded by `record_provenance`, and what lets edits made to a selection be
    mapped back onto the neuron it came from. Deliberately records the *parent's*
    id and a hash of its state rather than a reference to the parent object, so
    it survives pickling and does not keep a large neuron alive.
    """

    #: `.id` of the neuron this was selected from.
    parent_id: Any
    #: The parent's `core_md5` at the time of selection. A merge checks this and
    #: refuses rather than guessing if the parent has been restructured since -
    #: a hash cannot miss a mutation the way a counter someone forgot to bump
    #: can, and merging against the wrong parent is the failure mode this whole
    #: design exists to make impossible.
    parent_epoch: Optional[str] = None
    #: Per axis: for element `i` of this neuron, the parent element it came
    #: from - an ID for id-bearing axes, an index for positional ones. Kept
    #: aligned as the selection is edited further, so it always describes the
    #: elements that are here *now*.
    origin: Dict[str, np.ndarray] = field(default_factory=dict)
    #: Per axis: the parent elements this selection took responsibility for,
    #: fixed at selection time. Distinct from `origin` and it has to be: once
    #: the selection deletes some of its elements, `origin` no longer mentions
    #: them, and a merge that went by `origin` alone would quietly restore them
    #: from the parent instead of honouring the deletion.
    covered: Dict[str, np.ndarray] = field(default_factory=dict)

    def __copy__(self) -> "Provenance":
        """Copy the mappings, share the arrays in them.

        `Neuron.copy()` copies each attribute with `copy.copy`, which for a
        dataclass would hand the copy the same two dicts - and then subsetting
        one neuron would rewrite the other's origins. The arrays inside are
        always replaced wholesale rather than written into, so they can safely
        stay shared.
        """
        return Provenance(
            parent_id=self.parent_id,
            parent_epoch=self.parent_epoch,
            origin=dict(self.origin),
            covered=dict(self.covered),
        )


class MergeError(Exception):
    """Raised when a selection cannot be safely folded back into its parent."""


def record_provenance(
    child, parent_id, parent_epoch, axis: Axis, survivors: Survivors
) -> None:
    """Attach (or extend) the provenance of `child` relative to its parent.

    Extending only makes sense for a *second axis of the same selection* - a
    neuron with more than one axis records one footprint per axis against one
    parent. Anything else is a new selection and needs a new `Provenance`.

    Note the epoch is part of what identifies the parent, not just its `.id`.
    Selecting from a selection is the case that matters: masking works in place,
    so the child carries the same `.id` as the parent, and going by `.id` alone
    would keep the *previous* parent's epoch while overwriting the footprints
    with ones relative to the new parent. A later merge would then check itself
    against the grandparent, pass, and map the elements back onto the wrong
    place.
    """
    prov = getattr(child, "_prov", None)
    if (
        prov is None
        or prov.parent_id != parent_id
        or prov.parent_epoch != parent_epoch
    ):
        prov = Provenance(parent_id=parent_id, parent_epoch=parent_epoch)
        child._prov = prov

    footprint = np.asarray(survivors.kept if axis.positional else survivors.ids)
    prov.origin[axis.name] = footprint
    prov.covered[axis.name] = footprint


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------


def check_provenance(parent, child) -> Provenance:
    """Verify that `child` is a selection of `parent` that can still be merged.

    Raises rather than returning a verdict: every caller wants to stop here.
    """
    prov = getattr(child, "_prov", None)
    if prov is None:
        raise MergeError(
            "Neuron carries no provenance - it was not created by a tracked "
            "selection (see `subset_neuron(..., track=True)`)."
        )
    if prov.parent_id != parent.id:
        raise MergeError(
            f"Neuron was selected from {prov.parent_id}, not from {parent.id}."
        )
    if prov.parent_epoch != parent.core_md5:
        raise MergeError(
            f"{type(parent).__name__} {parent.id} has been modified since the "
            "selection was made, so there is no longer a reliable mapping back "
            "onto it. Re-select and redo the edits, or apply them to a copy."
        )
    return prov


def merge_selection(
    parent, child, axis: Axis, origin: np.ndarray, covered: np.ndarray
) -> None:
    """Fold a selection's elements back into `parent`, in place.

    The rule is the same for both kinds of axis: the parent keeps every element
    that was *not* in the selection, the child supplies every element that was,
    and then everything referencing the axis is repaired against the result. An
    element the child dropped simply doesn't come back, which is what makes
    "prune the axon, then unmask" work.

    References are taken from the parent and repaired, not merged from both
    sides: connectors, tags and the soma follow their elements. References the
    child added or removed *without* changing elements are not carried over.

    """
    if axis.positional:
        survivors = _merge_positional(parent, child, axis, origin, covered)
    else:
        survivors = _merge_by_id(parent, child, axis, covered)

    repair_refs(parent, axis, survivors)


def _merge_by_id(parent, child, axis: Axis, covered: np.ndarray) -> Survivors:
    """Merge an id-bearing axis. Identity is in the data, so order is all we set."""
    parent_ids = axis_ids(parent, axis)
    selected = np.isin(parent_ids, covered)
    retained = parent_ids[~selected]

    # An element the child kept or invented must not clash with one we are
    # holding on to - that would make two different elements share an ID.
    clash = np.intersect1d(axis_ids(child, axis), retained)
    if len(clash):
        raise MergeError(
            f"{len(clash)} element(s) of the selection (e.g. {clash[0]}) collide "
            f"with '{axis.name}' that were not selected. Merging would leave the "
            "neuron with duplicate IDs."
        )

    # Remember the parent's own topology before we overwrite it
    links = {
        ref.column: pd.Series(
            getattr(parent, ref.attr)[ref.column].values, index=parent_ids
        )
        for ref in _self_refs(axis)
    }

    _concat_axis_data(parent, child, axis, ~selected)
    merged_ids = axis_ids(parent, axis)

    for ref in _self_refs(axis):
        _reattach(parent, ref, links[ref.column], merged_ids, covered)

    return Survivors.from_ids(merged_ids)


def _self_refs(axis: Axis):
    """Refs that live *in* the axis' own data and describe its topology.

    A skeleton's `parent_id` is the case that matters: it rides along with the
    node table rather than being a separate structure, and it is the one thing a
    selection unavoidably damages - the element at the edge of the selection
    loses its parent and becomes a root.
    """
    return [r for r in axis.refs if r.attr in axis.data and r.null is not None]


def _reattach(
    neuron, ref: Ref, links: pd.Series, merged_ids: np.ndarray, covered: np.ndarray
) -> None:
    """Restore links that *the selection* severed, and only those.

    Selecting part of a neuron cuts the elements at its border loose. Merging
    puts their old neighbours back within reach, so those cuts should heal.
    Nothing else should: a break the selection did not cause is an edit, and
    healing it would silently undo the very thing the caller went into the mask
    to do.

    The two are told apart by where the old neighbour was. Outside the
    selection's footprint means only the selection can have severed the link -
    the element could not reach its neighbour from in there. Inside it means the
    child had the neighbour all along and is telling us the link is gone, either
    because it deleted the neighbour (pruning a twig) or because it re-pointed
    the element itself (rerooting). Both must stand.
    """
    table = getattr(neuron, ref.attr)
    values = table[ref.column].values
    severed = values == ref.null
    if not severed.any():
        return

    was = links.reindex(merged_ids[severed])
    known = was.notna().values
    restore = np.zeros(len(was), dtype=bool)
    restore[known] = np.isin(was.values[known], merged_ids) & ~np.isin(
        was.values[known], covered
    )
    if not restore.any():
        return

    values = values.copy()
    values[np.where(severed)[0][restore]] = was.values[restore].astype(values.dtype)
    table[ref.column] = values


def _merge_positional(
    parent, child, axis: Axis, origin: np.ndarray, covered: np.ndarray
) -> Survivors:
    """Merge a positional axis.

    Elements have no identity of their own, so `origin` *is* their identity and
    has to still line up with the child - anything that restructured the child
    without maintaining it makes the merge unsafe, and we say so.
    """
    n_parent = axis_length(parent, axis)
    n_child = axis_length(child, axis)

    if len(origin) != n_child:
        raise MergeError(
            f"Provenance for '{axis.name}' describes {len(origin)} element(s) "
            f"but the selection now has {n_child}. It was restructured by "
            "something that does not maintain provenance, so its elements can "
            "no longer be traced back."
        )

    # The parent gives up everything the selection *covered*, not merely what
    # survived in it: elements the selection deleted must not come back.
    selected = np.zeros(n_parent, dtype=bool)
    selected[covered] = True
    in_parent = origin >= 0

    _concat_axis_data(parent, child, axis, ~selected)

    # Where each *original* parent element ended up: retained ones keep their
    # relative order at the front, selected ones move to wherever the child put
    # them. Elements the child dropped stay `DROPPED`, so references to them
    # (including faces straddling the selection boundary) are pruned.
    n_retained = int((~selected).sum())
    old2new = np.full(n_parent, DROPPED, dtype=np.int64)
    old2new[~selected] = np.arange(n_retained)
    old2new[origin[in_parent]] = n_retained + np.where(in_parent)[0]

    carry_provenance_through_merge(parent, axis, ~selected, origin)

    return Survivors(old2new=old2new)


def carry_provenance_through_merge(
    parent, axis: Axis, retained: np.ndarray, origin: np.ndarray
) -> None:
    """Keep the parent's *own* provenance aligned through a merge.

    The mirror of `carry_provenance`, and needed for the same reason: on a
    positional axis, provenance is the elements' identity, so any operation that
    reorders or drops them has to bring it along. A parent that is itself a
    selection - the middle neuron of a nested mask - has one, and merging its
    child rebuilds its elements underneath it.

    The composition is one hop: a retained element still came from wherever it
    did, and an element the child supplied came from whatever *the parent's*
    element it replaced came from. Elements the child invented have no origin
    beyond the parent, so they are `DROPPED` here too.
    """

    def compose(was):
        if len(was) != len(retained):
            # The parent's own provenance is already out of step with its
            # elements; composing on top of it would only launder the problem.
            # Leave it be - merging it in turn will notice and refuse.
            return None
        from_child = np.where(origin >= 0, was[np.clip(origin, 0, None)], DROPPED)
        return np.concatenate([was[retained], from_child])

    _remap_origin(parent, axis, compose)


def _remap_origin(neuron, axis: Axis, remap) -> None:
    """Rewrite a neuron's recorded origins for one axis, if it has any.

    The single place that knows origins are a positional-axis concern: on an
    id-bearing axis the origin is a *set* of parent elements rather than a
    per-element mapping, so nothing that reorders or drops elements has to touch
    it. `remap` may return `None` to decline.
    """
    prov = getattr(neuron, "_prov", None)
    if prov is None or not axis.positional or axis.name not in prov.origin:
        return

    new = remap(np.asarray(prov.origin[axis.name]))
    if new is not None:
        prov.origin[axis.name] = new


def clear_provenance(neuron) -> None:
    """Forget where a neuron's elements came from."""
    neuron.__dict__.pop("_prov", None)


def _concat_axis_data(parent, child, axis: Axis, keep: np.ndarray) -> None:
    """Replace the parent's axis data with `parent[keep]` followed by the child's."""
    for attr in axis.data:
        p_value = getattr(parent, attr, None)
        c_value = getattr(child, attr, None)

        if p_value is None:
            continue
        if c_value is None:
            # Child dropped this entirely (e.g. tangent vectors); so must we,
            # or it would be left misaligned with the merged elements
            setattr(parent, attr, None)
            continue

        if isinstance(p_value, pd.DataFrame):
            merged = pd.concat([p_value.loc[keep], c_value], ignore_index=True)
        else:
            merged = np.concatenate([np.asarray(p_value)[keep], np.asarray(c_value)])
        setattr(parent, attr, merged)

    for attr in axis.invalidates:
        setattr(parent, attr, None)
