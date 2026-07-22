"""Tests for the angle morphometrics in `navis.morpho.angles`.

Correctness is pinned against a small hand-built neuron whose geometry has
angles we can work out by hand (90-degree bifurcations, a straight path, a
90-degree bend, 45-degree root angles). The example (insect) neurons are only
used for smoke-testing shapes, ranges and the NeuronList/MeshNeuron plumbing -
angle *values* on those are not biologically meaningful.
"""

import navis
import numpy as np
import pandas as pd

import pytest


@pytest.fixture
def toy():
    """A neuron with analytically known angles.

    Layout (all in the z=0 plane)::

              5 (10,30)   6 (-10,30)
               \\         /
                \\       /
                 4 (0,20)          <- branch point, 90 deg between children
                 |
                 2 (0,10)          <- slab, straight -> path angle 0
                 |
        3 (10,0) 1 (0,0)           <- root with two stems 2 & 3 -> 90 deg apart
                \\|
                 (root at node 1)
    """
    nodes = pd.DataFrame(
        {
            "node_id":   [1, 2, 3, 4, 5, 6],
            "parent_id": [-1, 1, 1, 2, 4, 4],
            "x": [0.0, 0.0, 10.0, 0.0, 10.0, -10.0],
            "y": [0.0, 10.0, 0.0, 20.0, 30.0, 30.0],
            "z": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    n = navis.TreeNeuron(nodes)
    n.soma = 1
    return n


def test_branch_angles_toy(toy):
    ba = navis.branch_angles(toy)
    assert list(ba.columns) == ["node_id", "branch_angle"]
    # Only the (non-root) branch point at node 4; root (node 1) is excluded
    assert list(ba.node_id) == [4]
    assert np.isclose(ba.branch_angle.iloc[0], 90.0, atol=1e-6)


def test_path_angles_toy(toy):
    pa = navis.path_angles(toy)
    assert list(pa.columns) == ["node_id", "path_angle"]
    # Node 2 is the only slab node; the path continues straight -> 0 degrees
    assert list(pa.node_id) == [2]
    assert np.isclose(pa.path_angle.iloc[0], 0.0, atol=1e-6)


def test_root_angles_toy(toy):
    ra = navis.root_angles(toy).set_index("node_id").root_angle
    assert list(navis.root_angles(toy).columns) == ["node_id", "root_angle"]
    # Stems (edges leaving the root) have no reference direction -> NaN
    assert np.isnan(ra.loc[2]) and np.isnan(ra.loc[3])
    # Edge 2->4 points the same way as root->2 -> 0 degrees
    assert np.isclose(ra.loc[4], 0.0, atol=1e-6)
    # Edges 4->5 and 4->6 are 45 degrees off the root->4 direction
    assert np.isclose(ra.loc[5], 45.0, atol=1e-6)
    assert np.isclose(ra.loc[6], 45.0, atol=1e-6)


def test_soma_exit_angles_toy(toy):
    sa = navis.soma_exit_angles(toy)
    assert list(sa.columns) == ["root_id", "soma_exit_angle"]
    # Two stems (nodes 2 & 3) -> a single pair at 90 degrees
    assert len(sa) == 1
    assert list(sa.root_id) == [1]
    assert np.isclose(sa.soma_exit_angle.iloc[0], 90.0, atol=1e-6)


def test_degrees_vs_radians(toy):
    deg = navis.branch_angles(toy, degrees=True).branch_angle.iloc[0]
    rad = navis.branch_angles(toy, degrees=False).branch_angle.iloc[0]
    assert np.isclose(deg, 90.0, atol=1e-6)
    assert np.isclose(rad, np.pi / 2, atol=1e-6)


@pytest.mark.parametrize(
    "func,value_col",
    [
        (navis.branch_angles, "branch_angle"),
        (navis.path_angles, "path_angle"),
        (navis.root_angles, "root_angle"),
        (navis.soma_exit_angles, "soma_exit_angle"),
    ],
    ids=["branch", "path", "root", "soma_exit"],
)
def test_example_neuron_ranges(func, value_col):
    n = navis.example_neurons(1, kind="skeleton")
    n.reroot(n.soma, inplace=True)
    res = func(n)
    assert isinstance(res, pd.DataFrame)
    assert value_col in res.columns
    vals = res[value_col].dropna()
    # Angles must sit in [0, 180] degrees
    assert (vals >= 0).all() and (vals <= 180).all()


def test_meshneuron_skeletonized():
    """MeshNeuron input should be skeletonized transparently."""
    m = navis.example_neurons(1, kind="mesh")
    res = navis.branch_angles(m)
    assert isinstance(res, pd.DataFrame)
    assert list(res.columns) == ["node_id", "branch_angle"]
    assert len(res) > 0


def test_neuronlist_mapping():
    """NeuronList input yields one concatenated frame with a `neuron` column."""
    nl = navis.example_neurons(2, kind="skeleton")
    res = navis.branch_angles(nl)
    assert isinstance(res, pd.DataFrame)
    assert res.columns[0] == "neuron"
    assert res.neuron.nunique() == 2


def test_multi_root(toy):
    """Fragmented neurons: each component is referenced to its own root."""
    # Second disconnected component, offset in x, with its own root (node 11)
    frag = pd.DataFrame(
        {
            "node_id":   [11, 12, 13],
            "parent_id": [-1, 11, 11],
            "x": [100.0, 100.0, 110.0],
            "y": [0.0, 10.0, 0.0],
            "z": [0.0, 0.0, 0.0],
        }
    )
    combined = pd.concat([toy.nodes[["node_id", "parent_id", "x", "y", "z"]], frag])
    n = navis.TreeNeuron(combined)
    assert len(n.root) == 2

    # soma_exit_angles should report a pair for each root's stems
    sa = navis.soma_exit_angles(n)
    assert set(sa.root_id) == {1, 11}

    # root_angles must not error and each stem edge stays NaN (4 stems total)
    ra = navis.root_angles(n)
    assert ra.root_angle.isna().sum() == 4
