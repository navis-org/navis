"""Tests for `navis.patch_caveclient`.

`caveclient` is an optional dependency and the skeleton service needs both a
network connection and an auth token, so everything here runs against a stub
client that returns the payloads the real service does.

Note this module must stay free of doctests - pytest runs with
`--doctest-modules`.
"""

import sys
import types

import numpy as np
import pandas as pd
import pytest

import navis


# --- the two payload shapes the skeleton service returns --------------------
#
#        3 (soma)
#        |
#        0
#       / \
#      1   2
#
# `output_format="swc"` numbers the soma first and is in microns;
# `output_format="dict"` keeps meshparty's vertex order (soma last here, as in
# the real payloads) and is in nanometres.

SWC = pd.DataFrame(
    {
        "id": [0.0, 1.0, 2.0, 3.0],
        "type": [1.0, 3.0, 3.0, 2.0],
        "x": [0.0, 0.0, 1.0, -1.0],
        "y": [0.0, 1.0, 2.0, 2.0],
        "z": [0.0, 0.0, 0.0, 0.0],
        "radius": [5.0, 0.5, 0.4, 0.3],
        "parent": [-1.0, 0.0, 1.0, 1.0],
    }
)

DICT = {
    "meta": {"root_id": 12345, "soma_radius": 5000},
    "vertices": np.array(
        [[0.0, 1000.0, 0.0], [1000.0, 2000.0, 0.0], [-1000.0, 2000.0, 0.0], [0.0, 0.0, 0.0]]
    ),
    "edges": np.array([[0, 3], [1, 0], [2, 0]]),
    "root": 3,
    "radius": np.array([500.0, 400.0, 300.0, 5000.0]),
    "compartment": np.array([3, 3, 2, 1]),
}


@pytest.fixture
def cave(monkeypatch):
    """A stub `caveclient` whose skeleton service returns the payloads above."""

    class SkeletonClient:
        def get_skeleton(self, root_id, output_format="dict", **kwargs):
            return SWC.copy() if output_format == "swc" else dict(DICT)

        def get_bulk_skeletons(self, root_ids, output_format="dict", **kwargs):
            # the real service keys these by root ID *as a string*
            return {
                str(r): (SWC.copy() if output_format == "swc" else dict(DICT))
                for r in root_ids
            }

        def fetch_skeletons(self, root_ids, output_format="dict", **kwargs):
            return self.get_bulk_skeletons(root_ids, output_format=output_format)

    pkg = types.ModuleType("caveclient")
    mod = types.ModuleType("caveclient.skeletonservice")
    mod.SkeletonClient = SkeletonClient
    pkg.skeletonservice = mod

    # Shadow the real caveclient (if installed) so we never patch it for real
    monkeypatch.setitem(sys.modules, "caveclient", pkg)
    monkeypatch.setitem(sys.modules, "caveclient.skeletonservice", mod)

    navis.patch_caveclient()
    return SkeletonClient()


def test_patch_adds_methods(cave):
    for name in ("get_skeleton", "get_bulk_skeletons", "fetch_skeletons"):
        assert hasattr(cave, f"{name}_navis")


def test_no_kwarg_returns_raw(cave):
    """Without `as_navis` the original return value must come back untouched."""
    assert isinstance(cave.get_skeleton(1, output_format="swc"), pd.DataFrame)
    assert isinstance(cave.get_skeleton(1), dict)
    assert isinstance(cave.get_bulk_skeletons([1, 2]), dict)


@pytest.mark.parametrize("via_kwarg", [True, False])
def test_swc_format(cave, via_kwarg):
    if via_kwarg:
        n = cave.get_skeleton(12345, output_format="swc", as_navis=True)
    else:
        n = cave.get_skeleton_navis(12345, output_format="swc")

    assert isinstance(n, navis.Skeleton)
    assert n.id == 12345
    assert str(n.units) == "1 micrometer"
    assert n.n_nodes == 4
    # SWC `type` becomes `label`, which is what ivscc_features & co. read
    assert n.nodes.label.tolist() == [1, 3, 3, 2]
    assert n.nodes.node_id.dtype.kind in "iu"
    assert n.root[0] == 0
    assert n.soma == 0                      # found via `label == 1`


@pytest.mark.parametrize("via_kwarg", [True, False])
def test_dict_format(cave, via_kwarg):
    if via_kwarg:
        n = cave.get_skeleton(12345, as_navis=True)
    else:
        n = cave.get_skeleton_navis(12345)

    assert isinstance(n, navis.Skeleton)
    assert n.id == 12345
    assert str(n.units) == "1 nanometer"
    assert n.n_nodes == 4
    # rooted where the payload says, not where `edges2neuron` happened to start
    assert n.root[0] == DICT["root"]
    assert n.soma == DICT["root"]

    # per-vertex arrays must follow node ID, not node-table position
    nodes = n.nodes.set_index("node_id")
    assert nodes.loc[DICT["root"], "label"] == 1
    assert nodes.loc[DICT["root"], "radius"] == 5000.0
    np.testing.assert_array_equal(
        nodes.loc[np.arange(4), "label"].values, DICT["compartment"]
    )
    np.testing.assert_array_equal(
        nodes.loc[np.arange(4), "radius"].values, DICT["radius"]
    )


def test_formats_agree(cave):
    """The two payloads describe the same neuron in different units."""
    nm = cave.get_skeleton_navis(12345)
    um = cave.get_skeleton_navis(12345, output_format="swc")

    assert nm.n_nodes == um.n_nodes
    assert nm.n_leafs == um.n_leafs
    assert nm.n_branches == um.n_branches
    assert nm.cable_length / 1000 == pytest.approx(um.cable_length, rel=1e-5)
    np.testing.assert_allclose(nm.soma_pos[0] / 1000, um.soma_pos[0], atol=1e-5)


@pytest.mark.parametrize("method", ["get_bulk_skeletons", "fetch_skeletons"])
def test_bulk(cave, method):
    ids = [12345, 67890]
    nl = getattr(cave, f"{method}_navis")(ids, output_format="swc")

    assert isinstance(nl, navis.NeuronList)
    assert len(nl) == 2
    # keys come back as strings but should end up as integer neuron IDs
    assert [n.id for n in nl] == ids
    assert all(isinstance(n.id, int) for n in nl)
    assert nl.idx[12345].n_nodes == 4

    assert isinstance(getattr(cave, method)(ids, as_navis=True), navis.NeuronList)


def test_bulk_truncation_warns(cave, caplog):
    """The bulk endpoints cap what they return - that must not pass unnoticed."""
    # return one skeleton for the three that were asked for
    type(cave).get_bulk_skeletons = lambda self, root_ids, **kw: {
        str(root_ids[0]): SWC.copy()
    }
    navis.patch_caveclient()

    with caplog.at_level("WARNING"):
        nl = cave.get_bulk_skeletons_navis([1, 2, 3])

    assert len(nl) == 1
    assert "1 skeletons for 3 root IDs" in caplog.text


def test_patch_is_idempotent(cave):
    """Patching twice must not wrap the wrapper."""
    navis.patch_caveclient()
    navis.patch_caveclient()

    assert isinstance(cave.get_skeleton(1, output_format="swc"), pd.DataFrame)
    assert isinstance(cave.get_skeleton_navis(1, output_format="swc"), navis.Skeleton)


def test_unknown_payload_passes_through(cave, monkeypatch):
    """Anything we can't convert comes back as-is rather than blowing up."""
    monkeypatch.setattr(
        type(cave), "get_skeleton", lambda self, root_id, **kw: "not a skeleton"
    )
    navis.patch_caveclient()

    assert cave.get_skeleton(1, as_navis=True) == "not a skeleton"


def test_missing_caveclient_is_a_noop(monkeypatch):
    """No caveclient installed -> log and return, don't raise."""
    monkeypatch.setitem(sys.modules, "caveclient", None)
    monkeypatch.setitem(sys.modules, "caveclient.skeletonservice", None)

    navis.patch_caveclient()
