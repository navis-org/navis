"""Tests for `navis.graph.graph_utils.propagate_labels`.

These pin down the *return type* of `propagate_labels`, which used to be inferred by
pandas/numpy and therefore varied with the installed pandas version: `Series.map()` on
string labels yields an `ArrowStringArray` under pandas >= 3 (where `future.infer_string`
defaults to True) but a plain object array under pandas < 3. The MeshNeuron branch had a
related problem: `np.array` of an all-labeled vertex list gives a fixed-width `<U*` array
that cannot hold NaN.

The contract is now: an object-dtype `np.ndarray`, with `np.nan` for nodes/vertices that
were never reached.
"""

import navis
import networkx as nx
import numpy as np
import pandas as pd

import pytest

from navis.graph import graph_utils as gu


@pytest.fixture
def skeleton():
    return navis.example_neurons(1, kind="skeleton")


@pytest.fixture
def mesh():
    return navis.example_neurons(1, kind="mesh")


def _synapse_labels(n):
    """Label nodes by whether they are pre- or postsynaptic sites."""
    pre_nodes = n.snap(n.presynapses[["x", "y", "z"]].values)[0]
    post_nodes = n.snap(n.postsynapses[["x", "y", "z"]].values)[0]

    labels = np.full(n.n_nodes, np.nan, dtype=object)
    labels[post_nodes] = "post"
    labels[pre_nodes] = "pre"
    return labels


def test_propagate_labels_skeleton_dtype(skeleton):
    """TreeNeuron: must return an object-dtype ndarray, not an ArrowStringArray."""
    n = skeleton
    prop = gu.propagate_labels(n, _synapse_labels(n), clamping=False)

    assert isinstance(prop, np.ndarray)
    assert prop.dtype == object
    assert len(prop) == n.n_nodes
    assert set(pd.unique(prop)) <= {"pre", "post"}


def test_propagate_labels_mesh_dtype(mesh):
    """MeshNeuron: must be object-dtype even when *every* vertex ends up labeled.

    Regression guard for the fixed-width `<U4` array that `np.array([...])` produced
    when no vertex was left unlabeled. The example mesh has ~70 connected components,
    so we seed a label in each of them - and clamp, so the seeds can't decay - to reach
    the all-labeled case.
    """
    m = mesh

    labels = np.full(len(m.vertices), np.nan, dtype=object)
    for i, cc in enumerate(nx.connected_components(m.graph)):
        labels[next(iter(cc))] = "pre" if i % 2 else "post"

    prop = gu.propagate_labels(m, labels, clamping=True, tol=1e-6, max_iter=10000)

    assert isinstance(prop, np.ndarray)
    assert prop.dtype == object  # NOT '<U4'
    assert len(prop) == len(m.vertices)
    assert not pd.isnull(prop).any()  # sanity: this is the all-labeled case
    assert set(pd.unique(prop)) <= {"pre", "post"}


def test_propagate_labels_unvisited_is_nan():
    """Nodes disconnected from any label must come back as NaN, as documented."""
    # Two disconnected 3-node trees.
    nodes = pd.DataFrame(
        {
            "node_id": [1, 2, 3, 10, 11, 12],
            "parent_id": [-1, 1, 2, -1, 10, 11],
            "x": [0.0, 1.0, 2.0, 100.0, 101.0, 102.0],
            "y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    n = navis.TreeNeuron(nodes)
    assert n.n_trees == 2  # sanity

    # Only label the first component.
    labels = {1: "pre", 2: "pre"}

    prop = gu.propagate_labels(n, labels)

    assert isinstance(prop, np.ndarray)
    assert prop.dtype == object

    is_first = n.nodes.node_id.isin([1, 2, 3]).values
    # The labeled component propagates its label to all of its nodes...
    assert (prop[is_first] == "pre").all()
    # ...while the component that never saw a label comes back as NaN.
    assert pd.isnull(prop[~is_first]).all()
