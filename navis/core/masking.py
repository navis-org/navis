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

"""Temporarily restricting a neuron to part of itself.

A mask is not a new kind of operation - it is a selection you can undo. So this
is a thin layer over the two primitives that already exist: `subset_neuron`
takes the selection and records where its elements came from, and
`merge_subset` folds whatever you did to them back in. Everything here is the
bookkeeping that lets that happen in place, so a neuron you already hold a
reference to becomes the selection for a while and then becomes whole again.

Masks nest. Each `mask()` pushes the pre-mask state onto a stack and each
`unmask()` pops one, so an inner mask does not have to know an outer one
exists.

"""

from contextlib import contextmanager

import numpy as np

from .. import config

__all__ = ["masked"]

logger = config.get_logger(__name__)


class MaskingError(Exception):
    """Raised when a neuron cannot be masked or unmasked as asked."""


def _count_cut_branches(whole, masked) -> int:
    """How many nodes the mask turned into terminals that are not terminals.

    A mask that follows a compartment keeps whole subtrees, so its leaves are the
    neuron's own. A mask that cuts across branches - a coordinate threshold, a
    geodesic radius - leaves nodes whose children fell outside it, and those look
    exactly like the ends of the arbour to anything that reads the node table.

    Nothing downstream can tell the two apart: since the fastcore migration
    `prune_twigs`, `strahler_index` and friends derive terminality straight from
    the `parent_id` array rather than from the `type` column, so there is no
    label we could set here that they would honour. Hence a warning rather than a
    fix.

    Cheap by construction: the masked neuron has just been classified, so its
    leaves are already known and only they have to be looked up.
    """
    # Skeletons only - meshes and dotprops have no notion of a terminal
    nodes = getattr(masked, "_nodes", None)
    if nodes is None or nodes.empty or "type" not in nodes.columns:
        return 0

    leafs = nodes.node_id.values[(nodes["type"] == "end").values]
    if not len(leafs):
        return 0
    # A leaf that was somebody's parent before the mask had children, and lost
    # every one of them
    return int(np.isin(leafs, whole.nodes.parent_id.values).sum())


def _warn_cut_branches(counts) -> None:
    """Warn once about masks that cut across branches, if any did.

    Takes counts for every neuron masked in one go so a `NeuronList` gets one
    warning rather than one per neuron.
    """
    affected = [(neuron, n) for neuron, n in counts if n]
    if not affected:
        return

    total = sum(n for _, n in affected)
    if len(counts) == 1:
        where = f"Mask on {type(affected[0][0]).__name__} {affected[0][0].id}"
    else:
        where = f"Masks on {len(affected)} of {len(counts)} neurons"

    logger.warning(
        f"{where} cut across {total} branch(es): those nodes lost their children "
        "and now look like the ends of the arbour. Leaf-sensitive functions "
        "(`prune_twigs`, `strahler_index`, ...) will treat them as real endings "
        "and can erode the mask boundary. Many such functions take a `mask` "
        "argument that does the same job on the intact neuron - see the masking "
        "tutorial. Pass `warn_cut=False` to silence this."
    )


def _n_roots(x) -> int:
    """Number of roots, i.e. of connected pieces. 0 for types that have none."""
    nodes = getattr(x, "_nodes", None)
    if nodes is None or nodes.empty:
        return 0
    return int((nodes.parent_id.values < 0).sum())


def _warn_severed(neuron, gained: int) -> None:
    """Warn when folding a mask back left the neuron in more pieces than before.

    The sharp end of `_count_cut_branches`: that one warns that a mask *might*
    mislead, this one fires only once something actually came apart, and is
    exact - a skeleton's roots are its connected components, so a rise in their
    number is a rise in the number of pieces.

    Deleting elements inside a mask on purpose (cutting, say) trips this too, and
    that is why it is a warning rather than an error - but the usual cause is a
    leaf-sensitive edit eating through the mask boundary.
    """
    if gained <= 0:
        return

    logger.warning(
        f"Merging the mask back into {type(neuron).__name__} {neuron.id} left it "
        f"in {gained} more piece(s) than it was: nodes that held it together were "
        "deleted while it was masked. If that was not deliberate, it is likely a "
        "leaf-sensitive edit (`prune_twigs` and friends) eroding the mask "
        "boundary - see the masking tutorial. Pass `warn_cut=False` to silence "
        "this."
    )


def _pop_mask(x):
    """Take the innermost mask off the stack, or say why we cannot.

    Returns the snapshot and the stack it came off, without touching the
    neuron: whoever called us may still fail, and a half-popped stack is worse
    than none of the work done.
    """
    stack = list(getattr(x, "_mask_stack", ()))
    if not stack:
        raise MaskingError(f"{type(x).__name__} {x.id} is not masked.")
    return stack.pop(), stack


def mask_neuron(x, mask, inplace: bool = False, warn_cut: bool = True):
    """Restrict a neuron to part of itself. See `BaseNeuron.mask`."""
    from ..morpho import subset_neuron

    if not getattr(x, "AXES", None):
        raise MaskingError(
            f"{type(x).__name__} cannot be masked: masking works off a "
            "declaration of how a neuron's elements are laid out, and this "
            "neuron type does not have one."
        )

    n = x if inplace else x.copy()

    # The state to come back to. Taken before the selection because the
    # selection is what destroys it.
    snapshot = n.copy()

    try:
        subset_neuron(n, mask, inplace=True, track=True)
    except BaseException:
        # Do not leave a half-masked neuron behind
        n._adopt(snapshot)
        raise

    # A fresh list rather than an append: a neuron and its copies otherwise
    # share one stack, and unmasking either would corrupt the other
    n._mask_stack = [*getattr(n, "_mask_stack", ()), snapshot]
    # The neuron shrank *and* took a whole copy of itself along, and neither
    # showed: `memory_usage` caches, and the selection ran with the neuron
    # locked, which is precisely when `_clear_temp_attr` declines to do anything
    n.__dict__.pop("_memory_usage", None)

    if warn_cut:
        _warn_cut_branches([(n, _count_cut_branches(snapshot, n))])

    return n


def unmask_neuron(x, reset: bool = True, warn_cut: bool = True):
    """Undo the innermost mask. See `BaseNeuron.unmask`."""
    from ..morpho import merge_subset

    snapshot, stack = _pop_mask(x)

    # Copy on the way out rather than on the way in. `Neuron.copy()` hands the
    # copy the same snapshot objects, and consuming one destroys it - `_adopt`
    # gives its tables away and `merge_subset` writes into it - so the copy has
    # to happen somewhere. Here it costs one per unmask; in `copy()` it would
    # cost one per copy per nesting level, and copying is what every non-inplace
    # navis call does inside a mask.
    snapshot = snapshot.copy()
    if reset:
        restored = snapshot
    else:
        # Counted before the merge: it writes into the snapshot, so afterwards
        # there is nothing left to compare against
        was_in_pieces = _n_roots(snapshot)
        restored = merge_subset(snapshot, x, inplace=True)
        if warn_cut:
            _warn_severed(restored, _n_roots(restored) - was_in_pieces)

    # No need to clear `_prov` when the stack empties: the snapshot carries
    # whatever provenance the neuron had before it was masked, which may well be
    # a tracked subset of something else, and `_adopt` has just restored it.
    x._adopt(restored)
    x._mask_stack = stack
    # `_adopt` took the snapshot's cached size with the rest of its state, which
    # describes the neuron as it was before the mask - true again for
    # `reset=True`, a lie for a merge
    x.__dict__.pop("_memory_usage", None)

    return x


def apply_mask_neuron(x, inplace: bool = False):
    """Make the innermost mask permanent. See `BaseNeuron.apply_mask`."""
    from .schema import clear_provenance

    n = x if inplace else x.copy()

    _, stack = _pop_mask(n)
    n._mask_stack = stack
    if not stack:
        # Nothing left to merge back into, so the provenance is dead weight
        clear_provenance(n)

    return n


def _resolve(mask, neuron):
    """Pick this neuron's mask out of whatever the caller supplied."""
    if isinstance(mask, dict):
        if neuron.id not in mask:
            raise MaskingError(f"No mask given for neuron {neuron.id}.")
        return mask[neuron.id]
    return mask


@contextmanager
def masked(x, mask, reset: bool = True, warn_cut: bool = True):
    """Work on part of a neuron, then put it back.

    Inside the block the neuron(s) *are* the masked region: every navis function
    and every property sees only that part, including through references you
    already hold. On the way out they become whole again.

    Parameters
    ----------
    x :         Neuron | NeuronList
                Neuron(s) to mask. Masked in place, so the objects you passed in
                are the ones that change.
    mask :      see [`navis.subset_neuron`][]
                Anything `subset_neuron` accepts - node IDs, a boolean mask,
                vertex indices, or a callable taking a neuron and returning one
                of those. May also be a dict keyed by neuron ID.
    reset :     bool
                If True, discard anything done inside the block and restore the
                neurons exactly as they were. If False, edits made to the masked
                region are folded back into the whole neuron - but only if the
                block runs to completion. An exception always resets, because
                half an edit is worse than none of it.
    warn_cut :  bool
                Warn if the mask cuts across branches, leaving nodes that look
                like the ends of the arbour but are not - and again on the way
                out if that actually cost the neuron its connectivity. Silent for
                masks that keep whole subtrees, e.g. a compartment.

    Examples
    --------
    Measure part of a neuron without changing it

    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> _ = navis.split_axon_dendrite(n, label_only=True)
    >>> with navis.masked(n, lambda x: x.nodes.compartment == 'axon'):
    ...     axon_cable = n.cable_length
    >>> bool(axon_cable < n.cable_length)
    True

    Edit part of a neuron and keep the edit

    >>> before = n.n_nodes
    >>> with navis.masked(n, lambda x: x.nodes.compartment == 'axon', reset=False):
    ...     _ = navis.prune_twigs(n, 5000, inplace=True)
    >>> bool(n.n_nodes < before)
    True

    See Also
    --------
    [`navis.BaseNeuron.mask`][]
                The underlying method, if you want to hold a mask open across
                function boundaries rather than in a block.

    """
    from .neuronlist import NeuronList

    neurons = x.neurons if isinstance(x, NeuronList) else [x]

    masked_so_far = []
    try:
        for n in neurons:
            # Warn once for the whole list below rather than once per neuron
            mask_neuron(n, _resolve(mask, n), inplace=True, warn_cut=False)
            masked_so_far.append(n)
        if warn_cut:
            _warn_cut_branches(
                [(n, _count_cut_branches(n._mask_stack[-1], n)) for n in masked_so_far]
            )
        yield x
    except BaseException:
        # A block that did not finish did not finish its edit either, and
        # nothing here can tell how far it got. Keeping half of an edit is worse
        # than keeping none of it, so this comes off as though `reset=True` -
        # and quietly, because the exception on its way out is the useful one.
        _unwind(masked_so_far, reset=True, quiet=True, warn_cut=warn_cut)
        raise
    else:
        _unwind(masked_so_far, reset=reset, quiet=False, warn_cut=warn_cut)


def _unwind(neurons, reset: bool, quiet: bool, warn_cut: bool = True) -> None:
    """Unmask every neuron, whatever happens to any one of them.

    Two things have to hold when this returns: no neuron is left masked, and a
    failure on one did not stop another from being put back. Restoring needs no
    provenance and so cannot fail, which is what makes the second guarantee
    affordable - a merge we are unable to do falls back to it.

    Parameters
    ----------
    quiet :     bool
                Report failures through the log rather than by raising. Set when
                an exception is already on its way out: whatever went wrong in
                the caller's block is the more useful of the two, and raising
                from a `finally` would bury it.

    """
    failed = []
    for n in reversed(neurons):
        try:
            unmask_neuron(n, reset=reset, warn_cut=warn_cut)
        except BaseException as exc:
            failed.append((n, exc))
            if not reset:
                try:
                    unmask_neuron(n, reset=True)
                except BaseException:
                    pass  # out of options for this one; reported below

    if not failed:
        return

    ids = ", ".join(str(n.id) for n, _ in failed)
    msg = (
        f"{len(failed)} of {len(neurons)} neuron(s) could not be unmasked as "
        f"asked ({ids}) and were restored to their pre-mask state instead. "
        "Check `.is_masked` for any that could not even be restored."
    )
    if quiet:
        logger.warning(f"{msg} Cause: {failed[0][1]!r}")
    else:
        raise MaskingError(msg) from failed[0][1]
