"""Tests for the fastcore-backed graph primitives.

`connected_components_of` and `_component_labels` run on
`navis_fastcore.connected_components_graph` when available and fall back to
igraph / scipy otherwise. Both paths must produce the same *partition* - the
label values themselves are arbitrary and differ between backends (fastcore
labels a component by its smallest member index, scipy numbers them in
discovery order), so nothing may depend on them.
"""

import numpy as np
import pytest

import navis
from navis import utils
from navis.graph.graph_utils import connected_components_of, skeleton_edges
from navis.morpho.manipulation import _component_labels

HAS_FASTCORE_GRAPH = utils.fastcore is not None and hasattr(
    utils.fastcore, "connected_components_graph"
)

needs_fastcore = pytest.mark.skipif(
    not HAS_FASTCORE_GRAPH,
    reason="navis-fastcore has no `connected_components_graph`",
)


@pytest.fixture
def no_fastcore(monkeypatch):
    monkeypatch.setattr(utils, "fastcore", None)


@pytest.fixture(scope="module")
def neuron():
    return navis.example_neurons(1, kind="skeleton")


@pytest.fixture(scope="module")
def fragmented(neuron):
    """A neuron broken into three fragments, by orphaning two nodes."""
    n = neuron.copy()
    ids = n.nodes.node_id.values
    n.nodes.loc[n.nodes.node_id.isin([ids[500], ids[1500]]), "parent_id"] = -1
    n._clear_temp_attr()
    return n


def _partition(components):
    """Normalise a list of node-ID sets for comparison."""
    return sorted(sorted(int(i) for i in c) for c in components)


def _partition_from_labels(labels, n):
    return sorted(sorted(np.flatnonzero(labels == i).tolist()) for i in range(n))


# ------------------------------------------------------- connected_components_of


@needs_fastcore
@pytest.mark.parametrize("step", [1, 2, 3])
def test_connected_components_of_parity(neuron, step, monkeypatch):
    keep = neuron.nodes.node_id.values[::step]

    fast = connected_components_of(neuron, keep)
    monkeypatch.setattr(utils, "fastcore", None)
    slow = connected_components_of(neuron, keep)

    assert _partition(fast) == _partition(slow)


@needs_fastcore
def test_connected_components_of_parity_when_fragmented(fragmented, monkeypatch):
    keep = fragmented.nodes.node_id.values[::3]

    fast = connected_components_of(fragmented, keep)
    monkeypatch.setattr(utils, "fastcore", None)
    slow = connected_components_of(fragmented, keep)

    assert _partition(fast) == _partition(slow)


@needs_fastcore
def test_connected_components_of_accepts_a_set(neuron, monkeypatch):
    """Regression: `keep` is routinely a set.

    `np.asarray(a_set)` produces a 0-d *object* array which `np.isin` matches
    against nothing, so the fastcore path silently returned no components at
    all - which surfaced as `max() arg is an empty sequence` in
    `find_soma_label`, not as anything pointing here.
    """
    keep = set(neuron.nodes.node_id.values[::2].tolist())

    fast = connected_components_of(neuron, keep)
    assert len(fast) > 0

    monkeypatch.setattr(utils, "fastcore", None)
    assert _partition(fast) == _partition(connected_components_of(neuron, keep))


def test_connected_components_of_covers_exactly_keep(neuron):
    """Every kept node lands in exactly one component; nothing else does."""
    keep = set(neuron.nodes.node_id.values[::2].tolist())
    comps = connected_components_of(neuron, keep)

    flat = [n for c in comps for n in c]
    assert len(flat) == len(set(flat)), "a node appeared in two components"
    assert set(flat) == keep


def test_connected_components_of_empty_keep(neuron):
    assert connected_components_of(neuron, []) == []


# ----------------------------------------------------------- _component_labels


@needs_fastcore
def test_component_labels_parity(fragmented, monkeypatch):
    labels_f, n_f = _component_labels(fragmented)
    monkeypatch.setattr(utils, "fastcore", None)
    labels_s, n_s = _component_labels(fragmented)

    assert n_f == n_s
    assert _partition_from_labels(labels_f, n_f) == _partition_from_labels(labels_s, n_s)


@needs_fastcore
def test_component_labels_are_contiguous(fragmented):
    """Callers index a `bincount` with these, so they must be 0..n-1.

    fastcore labels each component by its smallest member index, which is *not*
    contiguous - the wiring has to relabel.
    """
    labels, n = _component_labels(fragmented)

    assert set(np.unique(labels).tolist()) == set(range(n))
    assert len(labels) == fragmented.n_nodes
    # This is what `heal_skeleton` actually does with them
    assert np.bincount(labels, minlength=n).sum() == fragmented.n_nodes


def test_component_labels_single_component(neuron):
    labels, n = _component_labels(neuron)
    assert n == 1
    assert (labels == 0).all()


# ---------------------------------------------------------------- edge helper


def test_skeleton_edges_shape_and_mapping(neuron):
    edges, node_ids = skeleton_edges(neuron)

    assert len(node_ids) == neuron.n_nodes
    # One edge per non-root node
    assert len(edges) == neuron.n_nodes - len(neuron.root)
    assert edges.min() >= 0 and edges.max() < neuron.n_nodes

    # Spot-check: the edge for a known node points at its parent
    nid = neuron.nodes.node_id.values[100]
    pid = neuron.nodes.parent_id.values[100]
    ix = np.flatnonzero(node_ids == nid)[0]
    row = edges[edges[:, 0] == ix][0]
    assert node_ids[row[1]] == pid


# -------------------------------------------------------------- end-to-end


@needs_fastcore
def test_heal_skeleton_parity(fragmented, monkeypatch):
    """`_component_labels` feeds healing - the result must not change."""
    assert fragmented.n_trees > 1

    fast = navis.heal_skeleton(fragmented, inplace=False)
    monkeypatch.setattr(utils, "fastcore", None)
    slow = navis.heal_skeleton(fragmented, inplace=False)

    assert fast.n_trees == slow.n_trees == 1
    assert np.isclose(fast.cable_length, slow.cable_length)


# N.B. the label-only soma path is the other consumer of
# `connected_components_of` and is covered end-to-end by `tests/test_soma.py` -
# which is what caught the set-input bug above.
