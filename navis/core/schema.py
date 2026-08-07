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

A *link* is the same declaration pointed at a second axis. Where a `Ref` says
"these values name elements of my own axis", a `Link` says "this array is
aligned to my axis and its values name elements of *another* one" - possibly on
another object. A mesh's vertex-to-node map is exactly that, and saying so is
what lets one selection carry a mesh, its skeleton and anything hung off them
through in one step. See the "Links" section below.

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

import hashlib

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .. import config

try:
    import xxhash
except ModuleNotFoundError:
    xxhash = None

logger = config.get_logger(__name__)

__all__ = []

#: Sentinel for "this element did not survive".
DROPPED = -1

#: Stands in for a link's `LinkState` between it being carried through a
#: selection and its endpoints settling - see `refresh_links`. Deliberately a
#: string so it pickles and can never be mistaken for a real state.
PENDING = "<pending>"

#: Distinguishes "not passed" from "passed as None" for `apply_selection`'s
#: `links`, where the two mean different things.
_REQUIRED = object()

#: What a *rebuild* of an axis does to data aligned to it. A rebuild is not a
#: selection: some of the elements are new, so there is no value to carry onto
#: them and no way to invent one that is not a guess.
REBUILDS = (
    #: Let it go, with a warning saying so. The safe answer, and the default,
    #: because a wrong value that is the right length is worse than no value.
    "drop",
    #: Keep the values of the elements the rebuild says it kept. Only possible
    #: when it says so - `Rebuild.kept` - and only when every new element has a
    #: counterpart; otherwise this falls back to dropping.
    "carry",
)

#: What a rebuild does to something whose *values* name elements of the axis -
#: a `Ref` or a `Link`. The mirror of `REBUILDS`, and a separate question:
#: a reference does not need a value invented for it, only somewhere to point.
REBUILD_REFS = (
    #: Treat the old element as gone, i.e. exactly as a selection would.
    "drop",
    #: Follow it to wherever the rebuild says a reference should now point.
    #: A connector's node was thinned away, but the nearest survivor is the same
    #: stretch of the same branch, so the connector still means something.
    "snap",
    #: The rebuild replaced this along with the elements, so leave it exactly as
    #: it was handed over. A skeleton's `parent_id` and a mesh's faces are the
    #: axis' own topology - a face *is* three vertex indices - so whatever
    #: re-made the elements necessarily re-made them, and repairing them here
    #: would overwrite the rebuild's answer with ours.
    "rebuilt",
)


def _check_rebuild_policy(policy: str) -> None:
    """One policy, one validator - `Ref` and `Link` offer the same choice."""
    if policy not in REBUILD_REFS:
        raise ValueError(
            f'Unknown rebuild policy "{policy}", expected one of {REBUILD_REFS}'
        )


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
    on_rebuild :    "drop" | "snap"
                    What a *rebuild* of the axis does to this reference; see
                    `REBUILD_REFS`. Distinct from `null`, which is about an
                    element that is gone: under a rebuild it may merely have
                    moved, and `"snap"` follows it there.

    """

    attr: str
    kind: str
    column: Optional[str] = None
    null: Any = None
    write_attr: Optional[str] = None
    on_rebuild: str = "drop"

    @property
    def target(self) -> str:
        """Attribute to write the repaired reference back to."""
        return self.write_attr or self.attr

    def __post_init__(self):
        if self.kind not in _REF_HANDLERS:
            raise ValueError(f'Unknown ref kind "{self.kind}"')
        if self.kind == "column" and not self.column:
            raise ValueError('`column` is required for kind="column"')
        _check_rebuild_policy(self.on_rebuild)


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
    carried :       frozenset of str
                    Entries of `data` that a *rebuild* should carry onto the
                    elements it kept rather than drop - i.e. those attached with
                    `on_rebuild="carry"`. Everything else in `data` is dropped by
                    a rebuild; see `REBUILDS`.

    """

    name: str
    data: Tuple[str, ...] = ()
    ids: Optional[str] = None
    refs: Tuple[Ref, ...] = ()
    invalidates: Tuple[str, ...] = ()
    carried: FrozenSet[str] = frozenset()

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


def declared_axes(neuron) -> Dict[str, Axis]:
    """Every axis this neuron has - from its class, plus any attached to it.

    The instance half is what `BaseNeuron.attach` writes, both for axes a user
    brought along whole and for extra data aligned to an axis the class already
    declared.
    """
    extra = neuron.__dict__.get("_axes")
    return {**neuron.AXES, **extra} if extra else neuron.AXES


def get_axis(neuron, name: str) -> Axis:
    """Return a single axis declaration, raising if the neuron has no such axis."""
    declared = declared_axes(neuron)
    if name not in declared:
        raise KeyError(
            f'{type(neuron).__name__} has no "{name}" axis '
            f"(has: {sorted(declared) or 'none'})"
        )
    return declared[name]


def declare_axis(neuron, axis: Axis) -> None:
    """Add or replace an axis on this neuron alone."""
    neuron.__dict__.setdefault("_axes", {})[axis.name] = axis


def declare_aligned(neuron, axis_name: str, attr: str, on_rebuild: str = "drop") -> None:
    """Record that `attr` holds one value per element of an axis."""
    if on_rebuild not in REBUILDS:
        raise ValueError(
            f'Unknown rebuild policy "{on_rebuild}", expected one of {REBUILDS}'
        )
    axis = get_axis(neuron, axis_name)
    carried = axis.carried | {attr} if on_rebuild == "carry" else axis.carried - {attr}
    data = axis.data if attr in axis.data else axis.data + (attr,)
    # The no-op guard matters: it is what stops `n.connectors = ...` copying the
    # class' axis onto the instance on every assignment.
    if (data, carried) != (axis.data, axis.carried):
        declare_axis(neuron, replace(axis, data=data, carried=carried))


def undeclare_aligned(neuron, axis_name: str, attr: str) -> None:
    """Forget that `attr` was aligned to an axis."""
    axis = get_axis(neuron, axis_name)
    if attr in axis.data:
        declare_axis(
            neuron,
            replace(
                axis,
                data=tuple(a for a in axis.data if a != attr),
                carried=axis.carried - {attr},
            ),
        )


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


def axis_names(neuron, axis: Axis) -> np.ndarray:
    """What an axis' elements are called: IDs, or positions where there are none.

    The one place the id-vs-positional distinction is turned into concrete
    values, so that everything asking "which elements are these" - a link's
    mapping, a rebuild's provenance, `get_mapping`'s identity case - agrees.
    """
    if axis.positional:
        return np.arange(axis_length(neuron, axis))
    return np.asarray(axis_ids(neuron, axis))


def referenced_values(neuron, axis: Axis) -> np.ndarray:
    """Every element of an axis that something outside it names.

    A connector's node, a tag's nodes, the soma, a link somebody attached. What
    a caller wants this for is to answer "where did these go" for the elements
    that actually matter, rather than for the whole axis: on a 200k-node
    skeleton with fifty connectors, the difference is the whole cost.
    """
    values = []
    for ref in axis.refs:
        if ref.on_rebuild == "rebuilt":
            # The axis' own topology: whatever replaces the elements replaces
            # this too, so its values are the new ones already
            continue
        found = _REF_READERS[ref.kind](neuron, ref)
        if found is not None and len(found):
            values.append(np.asarray(found).ravel())

    for link in declared_links(neuron):
        if not _roles(link, axis)[1]:
            continue
        mapping = link_mapping(neuron, link)
        if mapping is not None and len(mapping):
            values.append(np.asarray(mapping).ravel())

    if not values:
        return np.array([], dtype=np.int64)
    return np.unique(np.concatenate(values))


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


#: Neurons whose orphan handling is currently stood down, by `id()`. Keyed that
#: way rather than held on the neuron because `copy()` takes the whole `__dict__`
#: - a neuron copied inside a `replacing` block would carry the suppression for
#: the rest of its life, and every later assignment to that axis would silently
#: skip the check. The neuron is alive for the whole block, so its `id` is stable.
_REPLACING: Dict[int, FrozenSet[str]] = {}


def is_replacing(neuron, axis_name: str) -> bool:
    """Whether somebody has said they are driving this axis' replacement."""
    return axis_name in _REPLACING.get(id(neuron), ())


@contextmanager
def replacing(neuron, *axis_names: str):
    """Stand the setters' orphan handling down while a selection does the work.

    `_subset_meshneuron` subsets the vertices itself - `submesh` resolves
    vertices and faces together and drops degenerate vertices only it knows
    about - and hands the result to `apply_selection`, which carries everything
    aligned to them. The public setter it assigns through cannot tell that from
    a caller replacing the elements outright, and guessing wrong either way is
    silent, so the caller says which it is.
    """
    key = id(neuron)
    was = _REPLACING.get(key, frozenset())
    _REPLACING[key] = was | frozenset(axis_names)
    try:
        yield
    finally:
        if was:
            _REPLACING[key] = was
        else:
            _REPLACING.pop(key, None)


def apply_selection(
    neuron,
    axis: Axis,
    keep: Optional[np.ndarray] = None,
    *,
    survivors: Optional[Survivors] = None,
    skip: Sequence[str] = (),
    links: Optional[Sequence["LiveLink"]] = _REQUIRED,
) -> Survivors:
    """Subset an axis and repair everything referencing it, in place.

    This is the whole of a selection for an axis, and everything that selects
    should come through here - the repair, the provenance carry and the link
    cascade are easy to forget and silent when forgotten.

    Parameters
    ----------
    keep :      np.ndarray, optional
                Boolean mask over the axis. Subsets the axis' data as well.
    survivors : Survivors, optional
                Pass instead of `keep` when the caller has already subset the
                data and knows better than we do what came through - a mesh's
                `submesh` resolves faces and vertices together and drops
                degenerate vertices this layer cannot predict.
    links :     sequence of LiveLink or None
                Links to carry through this selection, from `snapshot_links`.
                Whether a link can still be followed is judged against the state
                the neuron is in *before* anything moves, so a caller that
                subsets the data itself - i.e. one that passes `survivors` - has
                to take the snapshot before it does and hand it in here, and is
                made to say so: pass `None` to mean "I checked, there were
                none". Callers that pass `keep` need not bother, since nothing
                has changed by the time we are called.

    Returns
    -------
    Survivors
                Which elements made it through - the input to
                `record_provenance`.

    """
    if links is _REQUIRED:
        if survivors is not None:
            # We cannot take the snapshot ourselves - the data has already
            # moved, so every link would read as dead whether it is or not, and
            # silently dropping them all is indistinguishable from there being
            # none to drop. Only the caller knows, so the caller must say.
            raise TypeError(
                "`apply_selection(survivors=...)` must also be given `links`: "
                "take `snapshot_links(neuron, axis)` before subsetting the "
                "data, or pass `links=None` if there are none to carry."
            )
        links = snapshot_links(neuron, axis)

    # Applied here rather than at the snapshot, so that one parameter has one
    # owner: a caller that snapshots for itself (`_subset_meshneuron`) would
    # otherwise have to remember to pass `skip` twice, and forgetting the
    # second is silent - which is how `keep_disc_cn` came to be ignored for
    # meshes.
    links = [e for e in (links or ()) if e.link.attr not in skip]

    if survivors is None:
        subset_axis_data(neuron, axis, keep, skip=skip)
        # For an id-bearing axis the survivors are read back off the subsetted
        # data rather than derived from `keep`, so they cannot disagree with it.
        survivors = (
            Survivors.from_mask(keep)
            if axis.positional
            else Survivors.from_ids(axis_ids(neuron, axis))
        )
    else:
        # Passing `survivors` *means* the caller replaced the axis' primary
        # itself - it could not have built them otherwise - so that one needs no
        # naming. Anything else aligned to the axis is still ours to carry.
        subset_axis_data(
            neuron, axis, survivors=survivors, skip=(*skip, axis.data[0])
        )

    repair_refs(neuron, axis, survivors, skip=skip)
    carry_provenance(neuron, axis, survivors)
    follow_links(neuron, axis, links, keep, survivors)
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


def subset_axis_data(
    neuron,
    axis: Axis,
    keep: Optional[np.ndarray] = None,
    *,
    survivors: Optional[Survivors] = None,
    skip: Sequence[str] = (),
) -> None:
    """Subset every attribute aligned to an axis, in place.

    Attributes that are absent or `None` (a dotprops' `_alpha`, say) are
    skipped rather than being brought into existence, and so are those the
    caller says it has dealt with itself - a mesh's `submesh` resolves vertices
    and faces together, but a user's per-vertex labels are still ours to carry.
    """
    for attr in axis.data:
        value = getattr(neuron, attr, None)
        if value is None or attr in skip:
            continue
        setattr(neuron, attr, _select_aligned(value, keep, survivors))

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


def _read_column(neuron, ref):
    df = getattr(neuron, ref.attr, None)
    if not isinstance(df, pd.DataFrame) or ref.column not in df.columns:
        return None
    return df[ref.column].values


def _repair_column(neuron, ref, translate):
    values = _read_column(neuron, ref)
    if values is None:
        return
    df = getattr(neuron, ref.attr)

    new, ok = translate(values)

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


def _read_index_array(neuron, ref):
    arr = getattr(neuron, ref.attr, None)
    return None if arr is None or not len(arr) else np.asarray(arr)


def _repair_index_array(neuron, ref, translate):
    arr = _read_index_array(neuron, ref)
    if arr is None:
        return

    new, ok = translate(arr)
    # A row survives only if *every* element it references survived: a face
    # missing one of its corners is not a face.
    row_ok = ok.all(axis=1) if new.ndim > 1 else ok
    setattr(neuron, ref.target, new[row_ok])


def _id_list_keys(neuron, ref):
    """The non-empty keys of an `id_lists` ref, and the arrays behind them."""
    mapping = getattr(neuron, ref.attr, None)
    if not mapping:
        return None, None
    keys = [k for k, values in mapping.items() if len(values)]
    return keys, [np.asarray(mapping[k]) for k in keys]


def _read_id_lists(neuron, ref):
    _, lists = _id_list_keys(neuron, ref)
    return np.concatenate(lists) if lists else None


def _repair_id_lists(neuron, ref, translate):
    keys, lists = _id_list_keys(neuron, ref)
    if keys is None:
        return
    if not keys:
        setattr(neuron, ref.target, {})
        return

    # Translate every list in one go and split the result back up. Doing it per
    # key instead would rescan the (large) survivor array once per key, which
    # for a skeleton with a few dozen tags dominates the whole subset.
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


def _read_scalar(neuron, ref):
    value = getattr(neuron, ref.attr, None)
    # A callable is a *rule* for finding the element (e.g. the default
    # `find_soma`), not a reference to one - it stays valid across a subset.
    if value is None or callable(value):
        return None
    return np.atleast_1d(np.asarray(value))


def _repair_scalar(neuron, ref, translate):
    values = _read_scalar(neuron, ref)
    if values is None:
        return

    value = getattr(neuron, ref.attr)
    scalar = not isinstance(value, (list, tuple, np.ndarray, pd.Series))
    new, ok = translate(values)
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

#: The reading half of each handler on its own, for callers that want to know
#: *which* elements are referenced without repairing anything.
_REF_READERS = {
    "column": _read_column,
    "index_array": _read_index_array,
    "id_lists": _read_id_lists,
    "scalar": _read_scalar,
}


# ---------------------------------------------------------------------------
# Rebuilding
#
# A selection takes elements away; a rebuild replaces them. `resample_skeleton`
# is the case that makes the difference concrete: it does not remove any part of
# the neuron, it re-samples it, so a connector still sits on the arbour even
# though the node it named is gone.
#
# That asks two questions, and everything here turns on keeping them apart:
#
#   - where should a *reference* to an old element now point?  (`Rebuild.snap`)
#   - which of the new elements *is* an old one?                (`Rebuild.kept`)
#
# The first is about position and the second about identity, and the second may
# never be inferred from the first. A rebuild is free to mint element IDs however
# it likes, so an ID that happens to be reused is not thereby the same point -
# only the function that did the rebuilding knows, and it has to say.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rebuild:
    """How a rebuilt axis' elements relate to the ones they replaced.

    Built by whoever did the rebuilding, because only they can know. Handed to
    `apply_rebuild` - in practice by returning it alongside the neuron from a
    function wrapped in `navis.utils.rebuilds`.

    Parameters
    ----------
    snap :  (old, new) arrays, optional
            For an old element, the element a *reference* to it should now name.
            IDs for an id-bearing axis, indices for a positional one. May be
            partial, and what that means depends on the axis: on an id-bearing
            one an unlisted element keeps its value and then stands or falls on
            whether it is still there, while on a positional one it is simply
            gone - an index is not a name, so there is nothing to ask. `None`
            means nothing moved anywhere recoverable, and every reference into
            the axis is then repaired as if the elements had been removed.
    kept :  array, optional
            For each *new* element, in order, the name of the old element it
            **is** - or `DROPPED` where it is genuinely new. This is a claim
            about identity and is the only thing that lets aligned data be
            carried; `None` means no such claim, and everything aligned to the
            axis is dropped. Do not derive it from `snap`.

    """

    snap: Optional[Tuple[Sequence, Sequence]] = None
    kept: Optional[Sequence] = None


@dataclass
class RebuildState:
    """An axis as it was, captured before a rebuild replaced it.

    A rebuild assigns the new elements through the public setter, which - not
    knowing any better - drops the data attached to the old ones. So it has to
    be taken out of the way first, and put back by `apply_rebuild`.
    """

    axis: Axis
    #: Names of the old elements: IDs, or positions for a positional axis.
    names: np.ndarray
    #: Attached attribute -> its value, before the rebuild took it away.
    aligned: Dict[str, Any] = field(default_factory=dict)


def capture_rebuild(neuron, axis_name: str) -> Optional[RebuildState]:
    """Take an axis' identity and attached data out of a rebuild's way.

    Returns `None` for a neuron with no such axis, which is what makes the
    decorator safe to put on functions that also take types with no schema.
    """
    axis = declared_axes(neuron).get(axis_name)
    if axis is None:
        return None

    names = axis_names(neuron, axis)
    aligned = {
        attr: getattr(neuron, attr)
        for attr in neuron._attached_aligned(axis)
    }
    return RebuildState(axis=axis, names=names, aligned=aligned)


def _rebuild_translator(neuron, axis: Axis, snap):
    """Map reference values onto the axis as it is after a rebuild.

    Same contract as `_translator`: values in, `(new_values, ok)` out. A value
    the rebuild moved is followed; what becomes of one it did not mention is
    where the two kinds of axis part company, and the asymmetry is the point.

    An ID names the same element wherever the rebuild put it, so an unmentioned
    one can simply be asked whether it is still there. An *index* names no
    element at all - position 3 of the rebuilt axis is whatever now sits at 3 -
    so an unmentioned one is gone, in exactly the sense that `Rebuild.kept=None`
    means everything aligned to the axis is gone. Asking whether it is "still
    there" would only ever be a bounds check, and would leave it pointing at an
    unrelated element: the plausible-but-wrong answer this module exists to
    refuse.
    """
    live = None if axis.positional else axis_names(neuron, axis)
    n_live = axis_length(neuron, axis)
    old, new = (np.asarray(snap[0]), np.asarray(snap[1])) if snap else (None, None)
    lookup = None if old is None or not len(old) else pd.Index(old)

    def translate(values):
        values = np.asarray(values)
        moved = np.zeros(values.shape, dtype=bool)
        if lookup is not None:
            # `get_indexer` takes one dimension, and a single reference can be a
            # row of several - a face, an extra edge - so ask flat and put the
            # answer back into shape.
            where = lookup.get_indexer(values.ravel()).reshape(values.shape)
            moved = where >= 0
            if moved.any():
                # Write into a copy rather than `np.where`ing over a reindex: the
                # latter goes through float and quietly widens integer IDs.
                values = values.copy()
                values[moved] = new[where[moved]]

        if axis.positional:
            # Only what the rebuild explicitly moved still means anything - see
            # above. The bounds check is belt and braces: a rebuild that snapped
            # to an element it does not have is a bug, not a reference to repair.
            return values, moved & (values >= 0) & (values < n_live)
        return values, np.isin(values, live)

    return translate


def apply_rebuild(neuron, state: RebuildState, rebuild: Rebuild) -> None:
    """Put an axis' references and attached data back after a rebuild.

    The mirror of `apply_selection`, and deliberately a separate entry point:
    a selection knows which elements survived and can carry everything by
    construction, while a rebuild knows only what it chose to record.
    """
    axis = get_axis(neuron, state.axis.name)

    # One translator per policy: `"snap"` follows the rebuild, `"drop"` treats
    # the old element as gone, exactly as a selection would.
    translators = {
        "snap": _rebuild_translator(neuron, axis, rebuild.snap),
        "drop": _rebuild_translator(neuron, axis, None),
    }

    for ref in axis.refs:
        if ref.on_rebuild != "rebuilt":
            _REF_HANDLERS[ref.kind](neuron, ref, translators[ref.on_rebuild])

    # Same rule as a selection about what becomes of a source element left
    # naming nothing - a connector with nowhere to sit still goes - so this is
    # the selection's own code with a different idea of where things went.
    live = []
    for link in declared_links(neuron):
        mapping = link_mapping(neuron, link) if _roles(link, axis)[1] else None
        if mapping is not None:
            live.append(LiveLink(link, np.asarray(mapping), vouched=False))
    repair_links(neuron, axis, live, None, translate=translators)

    _restore_aligned(neuron, axis, state, rebuild)
    stamp_links(neuron, axis.name)


def relocate_refs(neuron, axis: Axis, old, new) -> None:
    """Point everything naming `old` at `new` instead, in place.

    A selection can only say that an element is *gone*. Sometimes it went
    somewhere: `merge_duplicate_nodes` folds a node into the one it sits on top
    of, which is the very same point in space, so a connector on it belongs on
    the survivor rather than in the bin.

    This is the `snap` half of `apply_rebuild` on its own, and deliberately only
    that half. Nothing here is a rebuild: the elements are still about to be
    taken away by an ordinary selection, so everything *aligned* to them still
    carries by construction and none of it should be dropped or second-guessed.
    Only what *references* the elements needs telling where they went, and only
    those declarations that asked to be snapped are moved - the rest are left
    exactly as they were, for the selection to repair as it always would.

    Parameters
    ----------
    old, new :  array-like
                Element names, pairwise: whatever named `old[i]` comes to name
                `new[i]`. Anything not listed is left alone.

    """
    old, new = np.asarray(old), np.asarray(new)
    if not len(old):
        return

    # Built against the axis as it stands, i.e. with every element still on it,
    # so nothing is dropped here on the way past - what survives is the
    # selection's business and it has not run yet.
    translate = _rebuild_translator(neuron, axis, (old, new))

    for ref in axis.refs:
        if ref.on_rebuild == "snap":
            _REF_HANDLERS[ref.kind](neuron, ref, translate)

    live = []
    for link in declared_links(neuron):
        if link.on_rebuild != "snap" or not _roles(link, axis)[1]:
            continue
        mapping = link_mapping(neuron, link)
        if mapping is not None:
            live.append(LiveLink(link, np.asarray(mapping), vouched=False))
    repair_links(neuron, axis, live, None, translate={"snap": translate})


def _carry_positions(neuron, axis: Axis, state: "RebuildState", rebuild: Rebuild):
    """For each new element, the old one whose value it inherits - or why not.

    Carrying needs the rebuild to have said which new elements are old ones
    *and* for every new element to be one: there is no value to give an element
    that was not there before, and inventing one is the kind of
    plausible-but-wrong answer this module exists to refuse.
    """
    if rebuild.kept is None:
        return None, "it did not record which of the new elements are old ones"

    kept, n = np.asarray(rebuild.kept), axis_length(neuron, axis)
    if len(kept) != n:
        return None, f"it recorded {len(kept)} element(s) of provenance for {n} element(s)"

    where = pd.Index(state.names).get_indexer(kept)
    if (where < 0).any():
        return None, f"{int((where < 0).sum())} of the new elements are new"
    return where, None


def _restore_aligned(neuron, axis: Axis, state: "RebuildState", rebuild: Rebuild) -> None:
    """Put back what was aligned to the old elements, where that is honest."""
    if not state.aligned:
        return

    carried = state.axis.carried & set(state.aligned)
    positions, why = (
        _carry_positions(neuron, axis, state, rebuild) if carried else (None, None)
    )

    for attr, value in state.aligned.items():
        if attr in carried and positions is not None:
            # Through `attach`, not `setattr`: with `inplace=False` the result is
            # a copy taken after the data was moved aside, so the declaration
            # that says what this describes has to go back on too.
            neuron.attach(
                attr,
                _select_aligned(value, positions, None),
                axis.name,
                on_rebuild="carry",
            )
        else:
            neuron.detach(attr)

    if why is not None:
        logger.warning(
            f"{type(neuron).__name__} {neuron.id}: dropped "
            f"{', '.join(sorted(carried))}, which asked to be carried through a "
            f"rebuild of '{axis.name}' - but {why}."
        )
    lost = sorted(set(state.aligned) - carried)
    if lost:
        logger.warning(
            f"{type(neuron).__name__} {neuron.id}: rebuilding '{axis.name}' "
            f"dropped {', '.join(lost)}, which described the old elements. "
            'Attach with `on_rebuild="carry"` to keep such data where the '
            "rebuild can say which elements it kept."
        )


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


class MappingError(Exception):
    """Raised when a link cannot be followed because it is out of date."""


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
    # Before the merge rebuilds the axis underneath them
    live = snapshot_links(parent, axis)

    if axis.positional:
        survivors = _merge_positional(parent, child, axis, origin, covered)
    else:
        survivors = _merge_by_id(parent, child, axis, covered)

    repair_refs(parent, axis, survivors)
    repair_links(parent, axis, live, survivors)


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


# ---------------------------------------------------------------------------
# Links
#
# A link (see the module docstring) is two declarations at once, and both are
# already implemented above:
#
#   - aligned to the source axis  -> subset it like `Axis.data`
#   - names target elements       -> filter and remap it like `Axis.refs`
#
# so following one through a selection is `subset_axis_data` in one direction
# and `_translator` in the other, with a cascade in between deciding what the
# far end keeps. Nothing here is new machinery; it is the existing machinery
# aimed at a second axis.
#
# Endpoints are named in the *neuron's own* namespace: its axis names, plus one
# name per link for the far end. A chain (connectors -> vertices -> nodes) is
# therefore declared on the neuron that owns it, and composing across it never
# has to guess whose axis a name belongs to.
# ---------------------------------------------------------------------------


#: Cascade policies - what a selection on the source does to the far end.
CASCADES = (
    #: Subset the target to the elements the surviving source elements still
    #: map to. A mesh's skeleton follows its vertices this way.
    "propagate",
    #: Leave the target alone and only maintain the mapping. Right when the
    #: source is the smaller, dependent thing: dropping connectors should not
    #: drop the vertices they sat on.
    "keep",
)

#: Dangling policies - what becomes of a source element whose target is gone.
#: The mirror of `Ref.null`, and the same two choices it offers.
DANGLING = (
    #: The source element goes too. A connector whose node was pruned is not a
    #: connector of anything.
    "drop",
    #: Keep it, and blank the mapping to `DROPPED`. Right when the element
    #: stands on its own and the correspondence was only ever an annotation.
    "blank",
)


@dataclass(frozen=True)
class Link:
    """A correspondence between the elements of two axes.

    One array wearing two hats: aligned to the source axis, and naming elements
    of the target one. See the module docstring.

    Parameters
    ----------
    name :          str
                    Name of the far endpoint, e.g. `"skeleton"`. Together with
                    `source` this identifies the link, and it is what
                    `get_mapping` and friends are addressed by.
    source :        str
                    Endpoint the mapping is aligned to - an axis of this neuron,
                    or the `name` of another link on it.
    mapping :       str
                    Dotted path to the mapping, e.g. `"_skeleton._vertex_map"`,
                    or - with `column` - to the DataFrame holding it. One value
                    per source element, naming a target element: an ID for an
                    id-bearing target axis, an index for a positional one,
                    `DROPPED` for source elements that map to nothing. Always
                    private names, for the same reason `Ref` uses them: repairs
                    write data we already know to be well-formed and do not need
                    the setters' checks.
    target_axis :   str
                    Axis on the target object.
    target :        str, optional
                    Dotted path to the object holding `target_axis`. Empty means
                    this neuron - a link between two of its own axes.
    column :        str, optional
                    Column of the DataFrame at `mapping` that holds the values -
                    a connector table's `node_id`. Without it `mapping` is taken
                    to be the array itself.
    cascade :       str
                    One of `CASCADES`; see there.
    dangling :      str
                    One of `DANGLING`; see there.
    on_rebuild :    str
                    One of `REBUILD_REFS`; see there. Distinct from `dangling`,
                    which is about a target that is *gone*: under a rebuild it
                    may merely have moved, and `"snap"` follows it.

    """

    name: str
    source: str
    mapping: str
    target_axis: str
    target: str = ""
    column: Optional[str] = None
    cascade: str = "propagate"
    dangling: str = "drop"
    on_rebuild: str = "drop"

    def __post_init__(self):
        if self.cascade not in CASCADES:
            raise ValueError(
                f'Unknown cascade "{self.cascade}", expected one of {CASCADES}'
            )
        if self.dangling not in DANGLING:
            raise ValueError(
                f'Unknown dangling policy "{self.dangling}", expected one of '
                f"{DANGLING}"
            )
        _check_rebuild_policy(self.on_rebuild)

    @property
    def key(self) -> str:
        """Stable identity for this link, used to key its bookkeeping."""
        return f"{self.source}->{self.name}"

    @property
    def attr(self) -> str:
        """The attribute the mapping hangs off, whatever the path to it."""
        return self.mapping.split(".")[0]

    @property
    def where(self) -> str:
        """Where the values live, spelled out for a message."""
        if self.column is None:
            return self.mapping
        return f'{self.mapping}["{self.column}"]'


#: Every neuron type stores connectors the same way, so they get one
#: declaration rather than one per class. Positional on purpose: nothing
#: references a connector, so there is no identity to keep - and minting one
#: would mean writing a column into the table the user handed us, which
#: `combine_neurons` would then happily duplicate.
CONNECTOR_AXIS = Axis(name="connectors", data=("_connectors",))


def connector_link(target_axis: str, column: str) -> Link:
    """The link saying which element of `target_axis` each connector sits on.

    A connector does not carry its node away with it (`cascade="keep"`), but a
    connector whose node was pruned is not a connector of anything, so it goes
    (`dangling="drop"`).

    A *rebuild* is the other story. Thinning or resampling a skeleton does not
    remove any part of it, so the connector is still on the neuron - it is the
    node under it that moved. Hence `on_rebuild="snap"`, which is what
    `downsample_neuron` and `resample_skeleton` have always done by hand.
    """
    return Link(
        name=target_axis,
        source="connectors",
        mapping="_connectors",
        column=column,
        target_axis=target_axis,
        cascade="keep",
        dangling="drop",
        on_rebuild="snap",
    )


def links(*items: Link) -> Tuple[Link, ...]:
    """Build a neuron class' `LINKS` declaration."""
    seen = set()
    for link in items:
        if link.key in seen:
            raise ValueError(f'Duplicate link "{link.key}"')
        seen.add(link.key)
    return tuple(items)


@dataclass(frozen=True)
class LinkState:
    """What both ends of a link looked like when it was last known good.

    Frozen, and always replaced rather than edited, so a neuron and its copies
    can share the objects without one rewriting the other's bookkeeping.
    """

    #: `axis_epoch` of the source axis.
    source: Optional[str]
    #: `axis_epoch` of the target axis.
    target: Optional[str]


@dataclass(frozen=True)
class LiveLink:
    """A link a selection touches, with its mapping as it was beforehand."""

    link: Link
    mapping: np.ndarray
    #: Whether the mapping was still vouched for when the snapshot was taken.
    #: Only the *aligned* role needs this - see `snapshot_links`.
    vouched: bool


def _roles(link: Link, axis: Axis) -> Tuple[bool, bool]:
    """The two hats a link can wear in a selection of `axis`.

    It may be the axis the mapping is *aligned* to (so it goes wherever the
    elements went), the one its values *name* (so they have to be filtered and
    remapped), or - for a link between two axes of one neuron - both at once.
    Only a link between two axes of *this* neuron can have its values repaired
    from here; anything else we cannot reach.
    """
    return link.source == axis.name, not link.target and link.target_axis == axis.name


# ---------------------------------------------------------------------------
# Reading the declaration
# ---------------------------------------------------------------------------


def declared_links(neuron) -> Tuple[Link, ...]:
    """Every link this neuron has - from its class, plus any attached to it.

    The instance half is what `BaseNeuron.attach_link` writes, and it *shadows*
    the class half by key, exactly as `declared_axes` does - two links with one
    key would share one slot of `_link_state` and repair the same values twice
    under conflicting policies.
    """
    attached = neuron.__dict__.get("_links")
    if not attached:
        return tuple(neuron.LINKS)
    shadowed = {link.key for link in attached}
    return tuple(lk for lk in neuron.LINKS if lk.key not in shadowed) + tuple(attached)


def declare_link(neuron, link: Link) -> None:
    """Add or replace a link on this neuron alone."""
    kept = tuple(lk for lk in neuron.__dict__.get("_links", ()) if lk.key != link.key)
    neuron.__dict__["_links"] = kept + (link,)


def undeclare_link(neuron, attr: str) -> None:
    """Forget every link whose mapping hangs off `attr`."""
    attached = neuron.__dict__.get("_links")
    if attached:
        neuron.__dict__["_links"] = tuple(lk for lk in attached if lk.attr != attr)


def get_link(neuron, name: str, source: Optional[str] = None) -> Link:
    """Return a single link by the name of its far endpoint.

    Pass `source` where a link has to be identified beyond doubt: a user is free
    to attach another link of the same name from somewhere else, and code that
    means *its own* link (`Mesh.skeleton`) must not be at the mercy of that.
    """
    found = [
        link
        for link in declared_links(neuron)
        if link.name == name and source in (None, link.source)
    ]
    if len(found) == 1:
        return found[0]
    if found:
        raise KeyError(
            f'{type(neuron).__name__} has several links named "{name}"; say '
            f"which by passing `source` (one of "
            f"{sorted(link.source for link in found)})."
        )
    known = sorted(link.key for link in declared_links(neuron))
    raise KeyError(
        f'{type(neuron).__name__} has no link to "{name}" (has: {known or "none"})'
    )


def _read_path(obj, path: str):
    """Read a dotted attribute path, `None` if any step is missing."""
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _write_path(obj, path: str, value) -> None:
    """Write a dotted attribute path, silently if the holder isn't there."""
    holder, _, attr = path.rpartition(".")
    obj = _read_path(obj, holder) if holder else obj
    if obj is not None:
        setattr(obj, attr, value)


def link_target(neuron, link: Link):
    """The object holding the far end of `link`, `None` if not attached."""
    return neuron if not link.target else _read_path(neuron, link.target)


def link_mapping(neuron, link: Link) -> Optional[np.ndarray]:
    """The mapping values of `link`, `None` if not there."""
    holder = _read_path(neuron, link.mapping)
    if holder is None:
        return None
    if link.column is None:
        return np.asarray(holder)
    if link.column not in holder.columns:
        # A mesh's connectors only learn which vertex they sit on when
        # something asks; until then there is no mapping to speak of.
        return None
    return holder[link.column].values


def set_link_mapping(neuron, link: Link, values) -> None:
    """Write the mapping values of `link` back."""
    if link.column is None:
        _write_path(neuron, link.mapping, values)
        return
    holder = _read_path(neuron, link.mapping)
    if holder is not None:
        # Assign into the frame rather than `setattr`, which pandas would take
        # for a new attribute rather than a column
        holder[link.column] = values


def resolve_endpoint(neuron, name: str) -> Tuple[Any, Axis]:
    """Resolve an endpoint name to the object and axis it stands for."""
    declared = declared_axes(neuron)
    if name in declared:
        return neuron, declared[name]

    link = get_link(neuron, name)
    target = link_target(neuron, link)
    if target is None:
        raise KeyError(
            f'{type(neuron).__name__} declares a link to "{name}" but nothing '
            "is attached there."
        )
    return target, get_axis(target, link.target_axis)


# ---------------------------------------------------------------------------
# Epochs
#
# A link is only as good as the two axes it spans, so every check here is
# ultimately "are they still the axes this was built against".
# ---------------------------------------------------------------------------


def hash_array(data) -> str:
    """Content hash of an array, the fast way if `xxhash` is installed."""
    data = np.ascontiguousarray(data)
    if xxhash:
        return xxhash.xxh128(data).hexdigest()
    return hashlib.md5(data).hexdigest()


def axis_epoch(neuron, axis: Axis) -> Optional[str]:
    """Hash of everything about an axis that the values of a link depend on.

    For an id-bearing axis that is the IDs and nothing else. A link into a
    skeleton stores node IDs, so rerooting it, moving it or relabelling its
    nodes all leave every link into it perfectly valid - only adding or removing
    nodes can break one, and only that shows up here.

    A positional axis has no identity to hash: element 3 is whatever now sits at
    index 3, so there is nothing to go on but the axis' own data. Note that is
    the axis' data and not the neuron's `CORE_DATA` - a mesh's faces say nothing
    about which vertex is where, so capping holes must not invalidate a skeleton
    it cannot have affected (and hashing them would cost twice as much again).

    Conservative in one specific way: a transform invalidates links out of a
    mesh even though it only moved the vertices and left the correspondence
    intact. The fix for that is to teach transforms to carry links, not to
    loosen the check.
    """
    if not axis.positional:
        return hash_array(axis_ids(neuron, axis))

    # Only the primary: everything else aligned to the axis is length-tied to
    # it and says nothing about *which* elements these are. Hashing it too
    # would also mean that attaching data changed the epoch, and so that
    # `attach` had to re-stamp - which would quietly bless a link that had gone
    # stale for an entirely unrelated reason.
    primary = getattr(neuron, axis.data[0], None)
    return None if primary is None else _hash_part(primary)


def _hash_part(value) -> str:
    """Describe one attribute of an axis well enough to tell it from another.

    An array is hashed outright - a mesh's vertices *are* its elements, and
    nothing else says which ones they are.

    A table is described by its shape and column names instead, which is O(1)
    rather than O(rows). That is a deliberately weaker claim, and it is the
    right one here: a table-backed axis is positional, so the only thing that
    can be aligned to it from outside is data `_orphan_aligned` already guards
    by count, while a mapping *out* of it lives in a column of the table itself
    and so cannot come adrift from the rows it describes. Hashing 100k
    connectors on every `.connectors =` - which the plotting loop does per
    neuron - would buy nothing for it.
    """
    if isinstance(value, pd.DataFrame):
        return hash_array(np.asarray(value.shape)) + hash_array(
            np.asarray([str(c) for c in value.columns], dtype="S")
        )
    return hash_array(value)


def _endpoints(neuron, link: Link):
    """Both ends of a link as `(object, axis)` pairs, `None` if not attached.

    The resolution `stamp_link` and `target_is_current` share; they take the
    epochs themselves so that the cheap end can be compared before the dear one.
    """
    target = link_target(neuron, link)
    if target is None:
        return None
    return (
        (neuron, get_axis(neuron, link.source)),
        (target, get_axis(target, link.target_axis)),
    )


def stamp_link(neuron, link: Link, reuse_target: bool = False) -> None:
    """Record the state both ends are in now, making the link current.

    `reuse_target` keeps the stored target epoch instead of recomputing it, for
    callers that know only the *source* moved. Hashing the far end is priced by
    its size, not by what changed: assigning a neuron's connectors would
    otherwise re-hash every node ID, which on a million-node skeleton costs half
    a millisecond per assignment to answer a question nobody asked.
    """
    state = neuron.__dict__.setdefault("_link_state", {})
    ends = _endpoints(neuron, link)
    if ends is None:
        # Nothing attached, so there is no state to describe
        state.pop(link.key, None)
        return
    (source, source_axis), (target, target_axis) = ends
    known = state.get(link.key) if reuse_target else None
    state[link.key] = LinkState(
        source=axis_epoch(source, source_axis),
        target=known.target if isinstance(known, LinkState) else axis_epoch(target, target_axis),
    )


def stamp_links(neuron, axis_name: str) -> None:
    """Stamp every link that touches an axis whose data has just been set.

    Both directions: a link out of the axis has new elements to describe, and
    one into it has just had its values replaced along with the table they live
    in. Links whose mapping is not there yet are dropped from the bookkeeping
    rather than stamped, so they read as absent instead of as current.
    """
    axis = get_axis(neuron, axis_name)
    for link in declared_links(neuron):
        aligned, names = _roles(link, axis)
        if not (aligned or names):
            continue
        if fitted_mapping(neuron, link) is None:
            neuron.__dict__.get("_link_state", {}).pop(link.key, None)
        else:
            # A link that merely *starts* here cannot have had its far end
            # changed by what happened to this axis
            stamp_link(neuron, link, reuse_target=aligned and not names)


def target_is_current(neuron, link: Link) -> bool:
    """Whether the far end still describes this neuron.

    This is the question `Mesh.skeleton` asks before handing its cached skeleton
    back, and it deliberately does not require a mapping: a skeleton attached by
    hand has no vertex map and is still a perfectly good skeleton of *this* mesh
    - right up until the mesh changes underneath it.
    """
    current = (neuron.__dict__.get("_link_state") or {}).get(link.key)
    if current is None or current is PENDING:
        return False

    ends = _endpoints(neuron, link)
    if ends is None:
        return False
    (source, source_axis), (target, target_axis) = ends

    # Target first: it is the id-bearing end and so the cheap hash, and getting
    # a `False` from it saves hashing the source's coordinates at all.
    if current.target != axis_epoch(target, target_axis):
        return False
    return current.source == axis_epoch(source, source_axis)


def fitted_mapping(neuron, link: Link) -> Optional[np.ndarray]:
    """The mapping of `link` if it is there and has one value per element.

    `None` otherwise, which is what "there is no mapping to speak of" means
    everywhere: a mesh's connectors only learn which vertex they sit on when
    something asks, and a mapping that no longer fits its axis describes
    elements that are gone.
    """
    mapping = link_mapping(neuron, link)
    if mapping is None or len(mapping) != axis_length(
        neuron, get_axis(neuron, link.source)
    ):
        return None
    return mapping


def mapping_is_current(neuron, link: Link) -> bool:
    """Whether the mapping can still be trusted to say what maps to what."""
    return fitted_mapping(neuron, link) is not None and target_is_current(neuron, link)


# ---------------------------------------------------------------------------
# Carrying links through a selection
# ---------------------------------------------------------------------------


def snapshot_links(neuron, axis: Axis) -> Tuple[LiveLink, ...]:
    """Links this selection touches, with their mappings as they are now.

    Must be taken *before* the selection touches anything: what a mapping says
    is only true of the elements it was built against, and selecting is
    precisely what ends that.

    The two roles are held to different standards, and deliberately so. The
    *aligned* role drives a cascade - it decides which of the far end's elements
    to keep - so it may only act on a mapping we can still vouch for. The
    *names* role only repairs values that point into the axis being selected,
    which is what `repair_refs` does for every `Ref` unconditionally; refusing to
    do it for a stale mapping would leave connectors hanging off nodes that no
    longer exist.

    Note what is *not* here either way: a link declared on some other object that
    names our elements - a mesh's link into the skeleton, when it is the skeleton
    being selected. We have no way to reach the mesh from here, so that link goes
    out of date instead, which `target_is_current` will report and the mesh will
    answer by regenerating.
    """
    live = []
    for link in declared_links(neuron):
        aligned, names = _roles(link, axis)
        if not (aligned or names):
            continue

        mapping = fitted_mapping(neuron, link)
        if mapping is None:
            continue

        # Only the aligned role has to be vouched for, and only it pays for the
        # answer: a names-only link is repaired either way, and comes out of
        # that repair current by construction.
        vouched = aligned and target_is_current(neuron, link)
        if not (vouched or names):
            continue
        live.append(LiveLink(link, mapping, vouched))
    return tuple(live)


def follow_links(
    neuron,
    axis: Axis,
    live: Sequence[LiveLink],
    keep: Optional[np.ndarray],
    survivors: Survivors,
) -> None:
    """Carry links through a selection, in place.

    Leaves the epochs unstamped - see `refresh_links` for why that is a separate
    step.
    """
    live = list(live)
    for i, entry in enumerate(live):
        link = entry.link
        aligned, names = _roles(link, axis)
        mapping = entry.mapping

        if aligned:
            mapping = _carry_aligned(neuron, entry, keep, survivors, mapping)
        if names:
            mapping, dropped = _repair_naming(neuron, link, axis, survivors, mapping)
            _drop_from_siblings(live, i, link, dropped)

        set_link_mapping(neuron, link, mapping)
        if entry.vouched or names:
            # Carried, so it can be vouched for once the caller has finished
            # moving things around - `refresh_links` does that. A repaired
            # names-only link qualifies without having been vouched for on the
            # way in: repairing it is what makes it true.
            neuron.__dict__.setdefault("_link_state", {})[link.key] = PENDING


def repair_links(
    neuron,
    axis: Axis,
    live: Sequence[LiveLink],
    survivors: Optional[Survivors],
    translate=None,
) -> None:
    """Repair links whose values name elements of an axis that has changed.

    The half of `follow_links` that is about the axis' elements rather than
    about a *selection* of them, and so the half a merge needs too: elements
    coming back rebuilds the axis just as thoroughly as elements going away, and
    leaves a connector pointing at whatever now sits where its node used to.

    The other half has no meaning in a merge - nothing was selected, so there is
    nothing to carry a mapping through and nothing to cascade.

    `translate` overrides how values are moved onto the axis as it is now: a
    dict of translator per `on_rebuild` policy, which is how a *rebuild* reuses
    this - same rule about what becomes of a source element left naming nothing,
    different idea of where things went.
    """
    live = list(live)
    for i, entry in enumerate(live):
        if _roles(entry.link, axis)[1]:
            mapping, dropped = _repair_naming(
                neuron,
                entry.link,
                axis,
                survivors,
                entry.mapping,
                translate=translate[entry.link.on_rebuild] if translate else None,
            )
            _drop_from_siblings(live, i, entry.link, dropped)
            set_link_mapping(neuron, entry.link, mapping)


def _drop_from_siblings(live: List[LiveLink], i: int, link: Link, dropped) -> None:
    """Take the links still to come through a drop one of them just made.

    `dangling="drop"` re-selects the source axis, which updates every other
    mapping out of it on the neuron - but not the snapshots taken before any of
    this started, and those are what the rest of the loop writes back.
    """
    if dropped is None:
        return
    for j in range(i + 1, len(live)):
        if live[j].link.source == link.source:
            live[j] = replace(
                live[j], mapping=_select_aligned(live[j].mapping, dropped, None)
            )


def _carry_aligned(neuron, entry: LiveLink, keep, survivors, mapping) -> np.ndarray:
    """Take a mapping through a selection of the axis it is aligned to."""
    link = entry.link
    # Aligned to the axis, so it goes wherever the elements went
    mapping = _select_aligned(mapping, keep, survivors)

    target = link_target(neuron, link)
    if entry.vouched and target is not None and link.cascade == "propagate":
        target_axis = get_axis(target, link.target_axis)
        # Selecting the far end goes through `apply_selection` like any other
        # selection, which is what repairs a skeleton's parent IDs, connectors,
        # tags and soma - and follows any links *it* has.
        target_survivors = apply_selection(
            target, target_axis, _target_mask(target, target_axis, mapping)
        )
        mapping = _retarget(target_axis, target_survivors, mapping)
        # A neuron subsetted directly has a caller to tidy up after it; one
        # reached through a link has none, so give it the same chance to fix up
        # derived state (a skeleton reclassifies its nodes, drops a vanished
        # soma and its stale graph views).
        target._clear_temp_attr()
    return mapping


def _repair_naming(
    neuron, link: Link, axis: Axis, survivors, mapping, translate=None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Point a mapping's values at the axis they name as it is now.

    Exactly what `repair_refs` does for a `Ref`, down to the two things that can
    become of a value with nothing left to name. Returns the repaired mapping
    and, where source elements were dropped, the mask saying which survived -
    see `_drop_from_siblings` for who needs it.
    """
    new, ok = (translate or _translator(axis, survivors))(mapping)
    if link.dangling == "blank" or ok.all():
        return np.where(ok, new, DROPPED), None

    # The element loses its reason to exist, so it goes - and it goes through
    # `apply_selection`, so that whatever else is aligned to *its* axis (a
    # user's per-connector labels) follows it out.
    apply_selection(neuron, get_axis(neuron, link.source), ok)
    return new[ok], ok


def _select_aligned(values, keep, survivors: Optional[Survivors]):
    """Subset anything aligned to an axis, the same way the axis was subset.

    `survivors` wins where it has an answer: it is what *actually* came through,
    in the order it came through in, which for a mesh is not the same as the
    mask that was asked for.
    """
    index = None if survivors is None else survivors.kept
    if index is None:
        index = keep
    if index is None:
        raise ValueError(
            "Cannot carry data through a selection that says neither which "
            "elements it kept nor which it wanted - pass `keep`, or build "
            "`Survivors` with `from_kept`/`from_mask`."
        )

    if isinstance(values, pd.DataFrame):
        # `iloc` because the index may be positions rather than labels.
        # `reset_index` both hands back a standalone frame - ref repair writes
        # columns in place and pandas would otherwise (rightly) warn about
        # writing to a slice - and drops the gappy index, which breaks igraph
        # conversion downstream.
        return values.iloc[index].reset_index(drop=True)
    return np.asarray(values)[index]


def _target_mask(target, axis: Axis, mapping: np.ndarray) -> np.ndarray:
    """Which target elements the surviving source elements still map to.

    The mapping names elements exactly the way a caller-supplied selection does,
    so this is `resolve_selection` and nothing else. Note it is handed the
    values with their duplicates intact: both of its branches cope, and
    de-duplicating first costs an order of magnitude more than it saves.
    """
    return resolve_selection(target, axis, mapping[mapping != DROPPED])


def _retarget(axis: Axis, survivors: Survivors, mapping: np.ndarray) -> np.ndarray:
    """Point a mapping at the target axis as it is *after* it was selected."""
    new, ok = _translator(axis, survivors)(np.asarray(mapping))
    return np.where(ok, new, DROPPED)


def refresh_links(neuron, _seen: Optional[set] = None) -> None:
    """Stamp links that were just carried with the state they now describe.

    Split from `follow_links` because the selection finishing is not the caller
    finishing: `subset_neuron` reroots a skeleton *after* subsetting it, which
    moves the very data an epoch is taken from. So the cascade only marks links
    as carried and this puts the epochs on once everything has settled.

    A link left marked - because nobody called this - reads as out of date,
    which costs a needless regeneration and never a wrong answer. That is the
    right way round for something easy to forget.
    """
    if _seen is None:
        _seen = set()
    if id(neuron) in _seen:
        return
    _seen.add(id(neuron))

    state = neuron.__dict__.get("_link_state") or {}
    for link in declared_links(neuron):
        target = link_target(neuron, link)
        if target is not None and target is not neuron:
            refresh_links(target, _seen)

        if state.get(link.key) is PENDING:
            stamp_link(neuron, link)


# ---------------------------------------------------------------------------
# The link graph
# ---------------------------------------------------------------------------


def link_graph(neuron) -> Dict[str, List[Link]]:
    """Links by the endpoint they start from."""
    graph: Dict[str, List[Link]] = {}
    for link in declared_links(neuron):
        graph.setdefault(link.source, []).append(link)
    return graph


def link_path(neuron, source: str, target: str) -> List[Link]:
    """Shortest chain of links leading from one endpoint to another.

    Links are directed, and deliberately only followed forwards. Backwards is
    not the same question: a mesh vertex has one skeleton node, but a node has
    many vertices, so a "mapping" the other way would have to either invent an
    answer or quietly hand back more rows than it was asked about. Ask
    `select_across` for the backwards direction instead - a *selection* is
    well defined in both.
    """
    if source == target:
        return []

    graph = link_graph(neuron)
    queue = deque([(source, [])])
    seen = {source}
    while queue:
        node, path = queue.popleft()
        for link in graph.get(node, ()):
            if link.name in seen:
                continue
            step = path + [link]
            if link.name == target:
                return step
            seen.add(link.name)
            queue.append((link.name, step))

    msg = f'No links lead from "{source}" to "{target}"'
    if any(link.name == source for link in declared_links(neuron)):
        # Note the argument order: `select_across` still reads along the link,
        # it is what it *returns* that goes the other way.
        msg += (
            f' - but "{target}" links to "{source}". Links only compose '
            f'forwards; ask which "{target}" elements sit on a selection of '
            f'"{source}" with `neuron.select_across("{target}", "{source}", '
            "selection)`"
        )
    raise KeyError(f"{msg}.")


def _positions(obj, axis: Axis, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Where the named elements sit along an axis, and which were found.

    Returns positions clamped into range wherever the element was *not* found,
    so the caller can index with them unconditionally and mask afterwards.
    """
    values = np.asarray(values)
    n = axis_length(obj, axis)
    if not n:
        return np.zeros(len(values), dtype=np.intp), np.zeros(len(values), dtype=bool)

    if axis.positional:
        ok = (values >= 0) & (values < n)
        return np.where(ok, values, 0).astype(np.intp, copy=False), ok

    # `get_indexer` is a hash join and already answers `-1` for values that are
    # not there - an order of magnitude cheaper than sorting the IDs to
    # `searchsorted` into them.
    where = pd.Index(axis_ids(obj, axis)).get_indexer(values)
    ok = where >= 0
    return np.where(ok, where, 0).astype(np.intp, copy=False), ok


def get_mapping(neuron, source: str, target: str) -> np.ndarray:
    """Map every element of one endpoint onto an element of another.

    Composes links along `link_path`, so a mapping that was never declared
    directly - connectors to skeleton nodes, via the vertices they sit on - is
    still available.

    Returns
    -------
    np.ndarray
                One entry per element of `source`, in its order, naming a
                `target` element (an ID for an id-bearing target axis, an index
                for a positional one). `DROPPED` where an element maps to
                nothing.

    """
    path = link_path(neuron, source, target)
    if not path:
        obj, axis = resolve_endpoint(neuron, source)
        return axis_names(obj, axis)

    values = _checked_mapping(neuron, path[0])
    for link in path[1:]:
        # Values name elements of this link's source endpoint; the mapping is
        # aligned to it, so they have to be turned back into positions first.
        obj, axis = resolve_endpoint(neuron, link.source)
        where, ok = _positions(obj, axis, values)
        mapping = _checked_mapping(neuron, link)
        values = np.where(ok & (values != DROPPED), mapping[where], DROPPED)

    return values


def _checked_mapping(neuron, link: Link) -> np.ndarray:
    """A link's mapping, or a `MappingError` saying which way it is unusable.

    Two different problems, and telling them apart is the difference between
    "build it" and "rebuild it".
    """
    mapping = fitted_mapping(neuron, link)
    if mapping is None:
        raise MappingError(
            f'There is no "{link.key}" mapping: {link.where} is not set, or '
            f'does not have one value per "{link.source}" element. Links '
            "describe what is there - they do not generate it."
        )
    if not target_is_current(neuron, link):
        raise MappingError(
            f'The "{link.key}" mapping is out of date - the elements it was '
            "built against have changed since. Regenerate it before mapping "
            "across it."
        )
    return mapping


def select_across(neuron, source: str, target: str, selection) -> np.ndarray:
    """Which `source` elements map into a selection of `target` elements.

    The backwards direction, as a *selection* rather than a mapping - see
    `link_path` for why the two are not the same question.

    Returns
    -------
    np.ndarray
                Boolean mask over `source`, ready to hand to `subset_neuron` or
                `mask`.

    """
    obj, axis = resolve_endpoint(neuron, target)
    wanted = resolve_selection(obj, axis, selection)
    names = axis_names(obj, axis)[wanted]
    return np.isin(get_mapping(neuron, source, target), names)


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
