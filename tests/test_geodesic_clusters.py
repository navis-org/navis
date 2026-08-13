"""Tests for `navis.graph.geodesic_clusters`.

Runs on `navis_fastcore.geodesic_clusters`. The greedy carve-out is
deterministic, so these pin what the partition must *be* - every node within
`max_dist` of its cluster's seed, every cluster connected - rather than
comparing against a second implementation.
"""

import numpy as np
import pytest

import navis
from navis.graph.graph_utils import geodesic_clusters


@pytest.fixture(scope="module")
def neuron():
    return navis.example_neurons(1, kind="skeleton")


@pytest.fixture(scope="module")
def mesh():
    return navis.example_neurons(1, kind="mesh")


# --------------------------------------------------------------------- shape


@pytest.mark.parametrize("max_dist", [1000, 5000, 20000])
def test_every_node_is_labelled(neuron, max_dist):
    labels = geodesic_clusters(neuron, max_dist)

    assert len(labels) == neuron.n_nodes
    assert (labels >= 0).all()
    # Contiguous in [0, n_clusters)
    assert set(np.unique(labels).tolist()) == set(range(labels.max() + 1))


def test_larger_radius_gives_fewer_clusters(neuron):
    counts = [
        geodesic_clusters(neuron, d).max() + 1 for d in (1000, 5000, 20000)
    ]
    assert counts == sorted(counts, reverse=True), counts


def test_zero_radius_puts_every_node_in_its_own_cluster(neuron):
    labels = geodesic_clusters(neuron, 0)
    assert labels.max() + 1 == neuron.n_nodes


def test_huge_radius_gives_one_cluster_per_component(neuron):
    labels = geodesic_clusters(neuron, 1e9)
    assert labels.max() + 1 == neuron.n_components


# ---------------------------------------------------------------- invariants


@pytest.mark.parametrize("max_dist", [2000, 10000])
def test_clusters_are_connected_by_default(neuron, max_dist):
    """`connected=True` must leave every cluster a connected subgraph."""
    labels = geodesic_clusters(neuron, max_dist)
    ids = neuron.nodes.node_id.values

    for c in range(labels.max() + 1):
        members = set(ids[labels == c].tolist())
        comps = navis.graph.graph_utils._component_ids(neuron, mask=members)
        assert len(comps) == 1, f"cluster {c} is not connected ({len(comps)} parts)"


@pytest.mark.parametrize("max_dist", [2000, 10000])
def test_raw_clusters_are_frequently_disconnected(neuron, max_dist):
    """`connected=False` exposes the raw greedy assignment, which is not.

    A cluster is its seed's ball *minus* whatever earlier clusters claimed, and
    that subtraction routinely splits it - which is exactly why `connected`
    defaults to True. Pinned here so the default is not "fixed" away as
    redundant.
    """
    labels = geodesic_clusters(neuron, max_dist, connected=False)
    ids = neuron.nodes.node_id.values

    n_disconnected = sum(
        len(navis.graph.graph_utils._component_ids(neuron, mask=ids[labels == c]))
        > 1
        for c in range(labels.max() + 1)
    )
    assert n_disconnected > 0, "expected the raw assignment to disconnect clusters"


def test_connected_only_subdivides(neuron):
    """Splitting must refine the raw partition, never merge across it."""
    raw = geodesic_clusters(neuron, 5000, connected=False)
    split = geodesic_clusters(neuron, 5000, connected=True)

    assert split.max() >= raw.max()
    # Every split cluster lies wholly inside one raw cluster
    for c in range(split.max() + 1):
        assert len(np.unique(raw[split == c])) == 1


@pytest.mark.parametrize("max_dist", [1000, 2000, 10000])
def test_clusters_are_balls_of_bounded_radius(neuron, max_dist):
    """Every member is within `max_dist` of *some* member (its seed).

    The documented guarantee is the true geodesic distance from the seed, not
    the length of the walk that reached it. We do not know which member the
    seed was, so check the weaker but sufficient property that some member is
    within `max_dist` of all the others.
    """
    labels = geodesic_clusters(neuron, max_dist)
    ids = neuron.nodes.node_id.values

    # Checking every cluster is expensive; a sample is enough to catch a
    # systematically wrong radius.
    rng = np.random.default_rng(0)
    for c in rng.choice(labels.max() + 1, size=min(5, labels.max() + 1),
                        replace=False):
        members = ids[labels == c]
        if len(members) == 1:
            continue
        dists = navis.geodesic_matrix(neuron, from_=members, directed=False)
        sub = dists.loc[members, members].values
        # At least one member reaches all the others within max_dist
        assert (sub.max(axis=1) <= max_dist + 1e-6).any(), (
            f"cluster {c} has no valid seed: min radius {sub.max(axis=1).min()}"
        )


def test_seeds_are_honoured(neuron):
    """A supplied seed grows the first cluster.

    Checked on the raw assignment: `connected=True` renumbers labels when it
    splits, so cluster 0 is no longer meaningfully "the first".
    """
    ids = neuron.nodes.node_id.values
    seed = ids[1000]

    labels = geodesic_clusters(neuron, 5000, seeds=[seed], connected=False)

    assert labels[np.flatnonzero(ids == seed)[0]] == 0


def test_seeds_change_the_partition(neuron):
    """Seeding somewhere else must actually produce a different partition."""
    ids = neuron.nodes.node_id.values

    a = geodesic_clusters(neuron, 5000, connected=False)
    b = geodesic_clusters(neuron, 5000, seeds=[ids[2000]], connected=False)

    assert not np.array_equal(a, b)


def test_unknown_seed_raises(neuron):
    with pytest.raises(ValueError, match="not part of this neuron"):
        geodesic_clusters(neuron, 5000, seeds=[-12345])


@pytest.mark.parametrize("bad", [-1, np.inf, np.nan])
def test_bad_max_dist_raises(neuron, bad):
    with pytest.raises(ValueError, match="finite and non-negative"):
        geodesic_clusters(neuron, bad)


def test_rejects_wrong_neuron_type():
    dp = navis.make_dotprops(navis.example_neurons(1, kind="skeleton"), k=5)
    with pytest.raises(TypeError, match="Skeleton or Mesh"):
        geodesic_clusters(dp, 1000)


# ------------------------------------------------------------------- weights


def test_hop_count_mode(neuron):
    """`weight=None` measures in hops, so clusters are ~2*max_dist+1 nodes."""
    labels = geodesic_clusters(neuron, 5, weight=None)
    assert (labels >= 0).all()
    # Every cluster is at most a ball of 5 hops -> bounded size
    sizes = np.bincount(labels)
    assert sizes.max() <= neuron.n_nodes


def test_weighted_and_hops_differ(neuron):
    a = geodesic_clusters(neuron, 10, weight=None)
    b = geodesic_clusters(neuron, 10, weight="weight")
    # 10 hops covers far more ground than 10 nm
    assert a.max() < b.max()


# ---------------------------------------------------------------- mesh input


def test_mesh_clusters(mesh):
    labels = geodesic_clusters(mesh, 2000)
    assert len(labels) == len(mesh.vertices)
    assert (labels >= 0).all()
    assert set(np.unique(labels).tolist()) == set(range(labels.max() + 1))


# --------------------------------------------------------------- downsampling


def test_centroids_lie_inside_their_cluster(neuron):
    """A connected cluster's centroid must sit near its own members.

    This is the property that makes centroid-collapse meaningful, and the one
    that `connected=False` breaks - so it is asserted for the default only.
    Note we deliberately do *not* assert that centroids are spaced by anything
    like `max_dist`: the greedy carve-out leaves many small fragments, so they
    are not (see this module's docstring and the function's own warning).
    """
    max_dist = 5000
    labels = geodesic_clusters(neuron, max_dist)
    co = neuron.nodes[["x", "y", "z"]].values.astype(float)

    assert labels.max() + 1 < neuron.n_nodes

    for c in range(labels.max() + 1):
        pts = co[labels == c]
        centroid = pts.mean(axis=0)
        # Every member is within max_dist of the centroid (a connected subset of
        # a ball of radius max_dist cannot spread further than its diameter).
        assert np.linalg.norm(pts - centroid, axis=1).max() <= 2 * max_dist + 1e-6
