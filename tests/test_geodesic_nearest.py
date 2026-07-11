"""Tests for the memory-efficient `graph._geodesic_nearest` helper and its use in
`split_axon_dendrite` for assigning orphan nodes to the nearest labeled node.
"""

import navis
import numpy as np

import pytest

from navis import graph
from navis.graph import graph_utils as gu


@pytest.fixture
def neuron():
    return navis.example_neurons(1, kind="skeleton")


def _old_nearest(n, miss, labeled):
    """Reference: nearest-labeled distances via the full geodesic matrix."""
    M = gu.geodesic_matrix(n, directed=False, weight=None, from_=miss)
    dist = M.loc[:, labeled].values.min(axis=1)
    lookup = dict(zip(M.index.values, dist))
    return np.array([lookup[m] for m in miss])


@pytest.mark.parametrize("seed", range(5))
def test_geodesic_nearest_matches_geodesic_matrix(neuron, seed):
    """Nearest-target distances must match the old full-matrix argmin approach."""
    n = neuron
    ids = n.nodes.node_id.values
    rng = np.random.default_rng(seed)
    miss = rng.choice(ids, size=600, replace=False)
    labeled = ids[~np.isin(ids, miss)]

    nearest, dist = graph._geodesic_nearest(
        n, targets=labeled, query=miss, weight=None, directed=False
    )

    assert np.all(nearest >= 0)  # everything reachable in a connected neuron
    assert np.all(np.isin(nearest, labeled))  # snaps only to labeled nodes
    assert np.allclose(dist, _old_nearest(n, miss, labeled))


def test_geodesic_nearest_weighted(neuron):
    """Physical-length weighting must match `geodesic_matrix(weight='weight')`."""
    n = neuron
    ids = n.nodes.node_id.values
    rng = np.random.default_rng(0)
    miss = rng.choice(ids, size=300, replace=False)
    labeled = ids[~np.isin(ids, miss)]

    nearest, dist = graph._geodesic_nearest(
        n, targets=labeled, query=miss, weight="weight", directed=False
    )
    M = gu.geodesic_matrix(n, directed=False, weight="weight", from_=miss)
    old_dist = M.loc[:, labeled].values.min(axis=1)
    lookup = dict(zip(M.index.values, old_dist))
    old_dist = np.array([lookup[m] for m in miss])

    assert np.allclose(dist, old_dist, rtol=1e-4)


def test_geodesic_nearest_empty_targets(neuron):
    """No targets at all -> everything unreachable (-1 / inf), no crash."""
    n = neuron
    nearest, dist = graph._geodesic_nearest(
        n, targets=[], query=n.nodes.node_id.values[:50]
    )
    assert np.all(nearest == -1)
    assert np.all(np.isinf(dist))


def test_geodesic_nearest_disconnected():
    """Query nodes in a component with no reachable target -> -1 / inf."""
    import pandas as pd

    # Explicit 2-component forest: A = {0,1,2}, B = {10,11,12}.
    nodes = pd.DataFrame(
        {
            "node_id": [0, 1, 2, 10, 11, 12],
            "parent_id": [1, 2, -1, 11, 12, -1],
            "x": [0, 1, 2, 10, 11, 12],
            "y": 0,
            "z": 0,
        }
    )
    forest = navis.TreeNeuron(nodes)

    # Targets only in component B -> all of component A is unreachable.
    nearest, dist = graph._geodesic_nearest(
        forest, targets=[10, 12], query=[0, 1, 2]
    )
    assert np.all(nearest == -1)
    assert np.all(np.isinf(dist))

    # Sanity: within-component queries DO resolve.
    nearest, dist = graph._geodesic_nearest(forest, targets=[2], query=[0, 1])
    assert np.all(nearest == 2)
    assert np.allclose(dist, [2.0, 1.0])


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("weight", [None, "weight"])
def test_geodesic_nearest_matches_ground_truth(neuron, directed, weight):
    """fastcore + scipy fallback must both match `geodesic_matrix` ground truth."""
    n = neuron
    ids = n.nodes.node_id.values
    rng = np.random.default_rng(0)
    miss = rng.choice(ids, size=500, replace=False)
    labeled = ids[~np.isin(ids, miss)]

    # Ground truth: min over target columns of geodesic_matrix(from_=query).
    M = gu.geodesic_matrix(n, directed=directed, weight=weight, from_=miss)
    gt = M.loc[:, labeled].values.min(axis=1)
    gt = np.array([dict(zip(M.index.values, gt))[m] for m in miss])

    def run():
        near, dist = graph._geodesic_nearest(
            n, targets=labeled, query=miss, weight=weight, directed=directed
        )
        dist = dist.copy()
        dist[near < 0] = np.inf
        return dist

    # fastcore path (if installed)
    dist_fc = run()
    # forced scipy fallback
    orig = gu.utils.fastcore
    try:
        gu.utils.fastcore = None
        dist_sp = run()
    finally:
        gu.utils.fastcore = orig

    for dist in (dist_fc, dist_sp):
        assert np.array_equal(np.isinf(gt), np.isinf(dist))
        fin = np.isfinite(gt)
        assert np.allclose(gt[fin], dist[fin], rtol=1e-4)


def test_split_axon_dendrite_runs(neuron):
    """End-to-end: split must still produce the expected compartments."""
    n = neuron
    labeled = navis.split_axon_dendrite(n, label_only=True)
    comps = set(labeled.nodes.compartment.dropna().unique())
    assert {"axon", "dendrite"} <= comps
    # No node left unlabeled (example neuron is fully connected).
    assert labeled.nodes.compartment.notna().all()
