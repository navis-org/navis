"""Tests for the IVSCC features in `navis.morpho.ivscc`.

Correctness is pinned against a small hand-built neuron whose geometry we can
work out on paper. The example (insect) neurons are only used to smoke-test the
plumbing and to cross-check features against navis' own implementations of the
same quantity - IVSCC feature *values* on those are not meaningful.
"""

import navis
import numpy as np
import pandas as pd

import pytest

from navis.morpho.ivscc import (
    BasicFeatures,
    CompartmentNotFoundError,
    NeuronContext,
    OverlapFeatures,
)


def _skeleton(rows, **kwargs):
    """Build a Skeleton from `(node_id, parent_id, x, y, z, label, radius)` rows."""
    return navis.Skeleton(
        pd.DataFrame(
            rows, columns=["node_id", "parent_id", "x", "y", "z", "label", "radius"]
        ),
        **kwargs,
    )


@pytest.fixture
def toy():
    """A labelled neuron with analytically known features.

    Layout (all in the z=0 plane; labels are SWC compartment IDs)::

           4 (5,30)   5 (-5,30)      apical dendrite (4)
                \\     /
                 3 (0,20)            <- bifurcation, 53.13 deg
                 |
                 2 (0,10)
                 |
                 1 (0,0)             <- soma (1), radius 5
                / \\
        6 (0,-10)  9 (2,-12)         <- basal stem / axon stem
          /   \\        \\
    7 (8,-20)  8 (-8,-20)  10 (2,-40)
        basal dendrite (3)      axon (2)

    Note that node 6 bifurcates immediately - it is the *root* of the basal
    dendrite once that compartment is subset out, which is exactly the case
    that used to go missing.
    """
    return _skeleton(
        [
            (1, -1, 0, 0, 0, 1, 5),
            (2, 1, 0, 10, 0, 4, 1),
            (3, 2, 0, 20, 0, 4, 1),
            (4, 3, 5, 30, 0, 4, 1),
            (5, 3, -5, 30, 0, 4, 1),
            (6, 1, 0, -10, 0, 3, 1),
            (7, 6, 8, -20, 0, 3, 1),
            (8, 6, -8, -20, 0, 3, 1),
            (9, 1, 2, -12, 0, 2, 1),
            (10, 9, 2, -40, 0, 2, 1),
        ],
        id="toy",
        soma=1,
    )


@pytest.fixture
def labelled_example():
    """An example neuron split into a fake apical/basal dendrite at the soma."""
    n = navis.example_neurons(1, kind="skeleton").copy()
    soma_y = n.soma_pos[0][1]
    n.nodes["label"] = np.where(n.nodes.y.values > soma_y, 4, 3)
    n.nodes.loc[n.nodes.node_id == n.soma, "label"] = 1
    return n


def test_result_layout(toy):
    """One row per neuron, one column per feature."""
    res = navis.ivscc_features(toy, progress=False)

    assert isinstance(res, pd.DataFrame)
    assert res.index.tolist() == ["toy"]
    assert res.index.name == "id"
    assert "axon_total_length" in res.columns

    # Duplicate IDs must not collapse into a single row
    res = navis.ivscc_features(navis.NeuronList([toy, toy]), progress=False)
    assert res.index.tolist() == ["toy", "toy"]


def test_max_euclidean_distance(toy):
    """Must be the *max* distance from the soma, not the sum of all distances."""
    res = navis.ivscc_features(toy, progress=False).iloc[0]

    axon = toy.nodes[toy.nodes.label == 2][["x", "y", "z"]].values
    expected = np.linalg.norm(axon - toy.soma_pos[0], axis=1).max()

    assert np.isclose(res.axon_max_euclidean_distance, expected)
    # The path to the farthest node can't be shorter than the straight line
    assert res.axon_max_path_length >= res.axon_max_euclidean_distance


def test_num_stems(toy):
    """Stems have to be counted on the full neuron - the subset loses the soma."""
    res = navis.ivscc_features(toy, progress=False).iloc[0]

    assert res.axon_num_stems == 1
    assert res.basal_dendrite_num_stems == 1
    assert res.apical_dendrite_num_stems == 1
    # Whole-cell count across all compartments
    assert res.num_stems == 3


def test_branch_order(toy):
    """Branch order counts bifurcations along a path, not branch points in total."""
    res = navis.ivscc_features(toy, progress=False).iloc[0]

    assert res.axon_max_branch_order == 1  # unbranched
    assert res.apical_dendrite_max_branch_order == 2
    # Node 6 is the root of the basal subset but still a genuine bifurcation
    assert res.basal_dendrite_max_branch_order == 2
    assert res.basal_dendrite_num_branch_points == 1


def test_branch_order_scales_with_depth():
    """A deep binary tree must not report its total number of branch points."""
    rows, next_id = [(1, -1, 0.0, 0.0, 0.0, 1, 5)], 2
    frontier, depth = [1], 4
    for level in range(depth):
        new_frontier = []
        for parent in frontier:
            for side in (-1, 1):
                rows.append(
                    (next_id, parent, side * (depth - level), (level + 1) * 10.0, 0.0, 4, 1)
                )
                new_frontier.append(next_id)
                next_id += 1
        frontier = new_frontier

    n = _skeleton(rows, id="tree", soma=1)
    res = navis.ivscc_features(n, progress=False).iloc[0]

    # The soma is not part of the compartment, so the dendrite comes out as two
    # trees with 2 + 4 + ... + 2**(depth - 1) branch points between them. Only
    # `depth` of those sit on any one root-to-tip path.
    assert res.apical_dendrite_num_branch_points == 2**depth - 2
    assert res.apical_dendrite_num_tips == 2**depth
    assert res.apical_dendrite_max_branch_order == depth


def test_contraction_is_not_tortuosity(toy):
    """Contraction is R/L and therefore bounded by 1."""
    res = navis.ivscc_features(toy, progress=False).iloc[0]

    # Every segment of the toy neuron is a straight line
    for comp in ("axon", "basal_dendrite", "apical_dendrite"):
        assert np.isclose(res[f"{comp}_mean_contraction"], 1.0)


def test_contraction_matches_segment_analysis(labelled_example):
    """Cross-check against navis' own per-segment tortuosity."""
    n = labelled_example.reroot(labelled_example.soma)
    res = navis.ivscc_features(n, features=[BasicFeatures], progress=False).iloc[0]

    expected = (1 / navis.segment_analysis(n).tortuosity).mean()

    assert np.isclose(res.mean_contraction, expected)
    assert 0 < res.mean_contraction <= 1


def test_exit_distance_and_theta_agree(toy):
    """Both must describe the *same* root - the one closest to the soma."""
    # Axon stem is node 9 at (2, -12); soma radius is 5
    res = navis.ivscc_features(toy, progress=False).iloc[0]

    assert np.isclose(res.axon_exit_distance, np.sqrt(2**2 + 12**2) - 5)
    assert np.isclose(res.axon_exit_theta, np.arctan2(-12, 2))


def test_exit_features_pick_closest_root():
    """A fragmented compartment must not mix up which root it reports."""
    n = _skeleton(
        [
            (1, -1, 0, 0, 0, 1, 5),
            (2, 1, 0, 10, 0, 4, 1),
            (3, 1, 5, -10, 0, 2, 1),  # attached axon stem
            (4, 3, 5, -20, 0, 2, 1),
            (5, -1, 100, 100, 0, 2, 1),  # far-away axon fragment
            (6, 5, 100, 110, 0, 2, 1),
        ],
        id="frag",
        soma=1,
    )
    res = navis.ivscc_features(n, progress=False).iloc[0]

    assert np.isclose(res.axon_exit_distance, np.sqrt(5**2 + 10**2) - 5)
    assert np.isclose(res.axon_exit_theta, np.arctan2(-10, 5))
    # Only the attached fragment sprouts from the soma
    assert res.axon_num_stems == 1


def test_bifurcation_angles(toy):
    """Local and remote angles; both are 53.13 deg for the (straight) apical fork."""
    res = navis.ivscc_features(toy, progress=False).iloc[0]

    expected = np.degrees(np.arccos(np.dot([5, 10], [-5, 10]) / (np.sqrt(125) ** 2)))
    assert np.isclose(res.apical_dendrite_bifurcation_angle_local, expected)
    assert np.isclose(res.apical_dendrite_bifurcation_angle_remote, expected)

    # Unbranched compartment has no bifurcation at all
    assert np.isnan(res.axon_bifurcation_angle_local)


def test_local_angles_match_branch_angles(labelled_example):
    """Cross-check against `navis.branch_angles` on the soma-rooted neuron."""
    n = labelled_example.reroot(labelled_example.soma)
    res = navis.ivscc_features(n, features=[BasicFeatures], progress=False).iloc[0]

    assert np.isclose(
        res.bifurcation_angle_local, navis.branch_angles(n).branch_angle.mean()
    )


def test_radius_features(toy, labelled_example):
    """Surface/volume must match the Skeleton properties for the whole neuron."""
    res = navis.ivscc_features(
        labelled_example, features=[BasicFeatures], progress=False
    ).iloc[0]

    assert np.isclose(res.total_surface, float(labelled_example.surface_area), rtol=1e-5)
    assert np.isclose(res.total_volume, float(labelled_example.volume), rtol=1e-5)
    assert np.isclose(res.mean_diameter, labelled_example.nodes.radius.mean() * 2)

    # All radii are 1 in the toy neuron, so daughters and parents match
    toy_res = navis.ivscc_features(toy, progress=False).iloc[0]
    assert np.isclose(toy_res.apical_dendrite_parent_daughter_ratio, 1.0)
    assert np.isclose(toy_res.soma_surface, 4 * np.pi * 5**2)


def test_no_usable_radii(toy):
    """navis fills in zero radii - those must not be reported as real measurements."""
    nodes = toy.nodes.drop(columns=["radius"])
    n = navis.Skeleton(nodes, id="norad", soma=1)

    res = navis.ivscc_features(n, progress=False)

    for col in ("axon_total_surface", "axon_total_volume", "axon_mean_diameter"):
        assert col not in res.columns
    assert np.isnan(res.soma_surface.iloc[0])


def test_extent_covers_all_three_axes(toy):
    res = navis.ivscc_features(toy, progress=False).iloc[0]

    assert np.isclose(res.apical_dendrite_extent_x, 10)
    assert np.isclose(res.apical_dendrite_extent_y, 20)
    assert np.isclose(res.apical_dendrite_extent_z, 0)


def test_somaless_neuron(toy):
    """Must not blow up - soma-dependent features are simply left out."""
    n = navis.Skeleton(toy.nodes.copy(), id="nosoma", soma=None)

    res = navis.ivscc_features(n, progress=False)
    assert len(res) == 1
    assert "axon_total_length" in res.columns
    assert "axon_exit_distance" not in res.columns

    # `BasicFeatures` on its own used to return None here and raise a TypeError
    res = navis.ivscc_features(n, features=[BasicFeatures], progress=False)
    assert len(res) == 1
    assert "max_euclidean_distance" not in res.columns


@pytest.mark.parametrize("missing", ["ignore", "skip", "raise"])
def test_missing_compartment(toy, missing):
    n = navis.Skeleton(toy.nodes[toy.nodes.label != 2].copy(), id="noaxon", soma=1)

    if missing == "raise":
        with pytest.raises(CompartmentNotFoundError):
            navis.ivscc_features(n, missing_compartments=missing, progress=False)
        return

    res = navis.ivscc_features(n, missing_compartments=missing, progress=False)

    if missing == "skip":
        assert len(res) == 0
    else:
        assert len(res) == 1
        assert not [c for c in res.columns if c.startswith("axon_")]


@pytest.mark.parametrize("missing", ["ignore", "skip", "raise"])
def test_missing_label_column(toy, missing):
    """A neuron without labels is handled by `missing_compartments`, not a crash."""
    n = navis.Skeleton(toy.nodes.drop(columns=["label"]), id="nolabel", soma=1)

    if missing == "raise":
        with pytest.raises(CompartmentNotFoundError):
            navis.ivscc_features(n, missing_compartments=missing, progress=False)
        return

    res = navis.ivscc_features(n, missing_compartments=missing, progress=False)

    assert len(res) == (0 if missing == "skip" else 1)
    if missing == "ignore":
        # Whole-cell features still work
        assert np.isclose(res.soma_radius.iloc[0], 5)


def test_labels_may_be_names_or_ids(toy):
    """SWC label IDs and compartment names must give the same result."""
    named = toy.nodes.copy()
    named["label"] = named.label.map(
        {1: "soma", 2: "axon", 3: "basal_dendrite", 4: "apical_dendrite"}
    )
    n = navis.Skeleton(named, id="toy", soma=1)

    pd.testing.assert_frame_equal(
        navis.ivscc_features(toy, progress=False),
        navis.ivscc_features(n, progress=False),
    )


def test_rerooting_is_transparent(toy):
    """Results must not depend on where the neuron happened to be rooted."""
    rerooted = toy.reroot(10)

    pd.testing.assert_frame_equal(
        navis.ivscc_features(toy, progress=False),
        navis.ivscc_features(rerooted, progress=False),
    )


def test_overlap_features(toy):
    res = navis.ivscc_features(toy, features=[OverlapFeatures], progress=False).iloc[0]

    # The apical dendrite sits entirely above both other compartments
    assert res.apical_dendrite_frac_above_axon == 1
    assert res.apical_dendrite_frac_below_axon == 0
    assert res.axon_frac_below_apical_dendrite == 1

    # The three fractions partition the nodes
    assert np.isclose(
        res.axon_frac_above_basal_dendrite
        + res.axon_frac_intersect_basal_dendrite
        + res.axon_frac_below_basal_dendrite,
        1,
    )

    # EMD is symmetric and recorded only once per pair
    assert "axon_emd_with_apical_dendrite" in res.index
    assert "apical_dendrite_emd_with_axon" not in res.index


def test_invalid_arguments(toy):
    with pytest.raises(ValueError):
        navis.ivscc_features(toy, missing_compartments="nonsense", progress=False)

    with pytest.raises(ValueError):
        navis.ivscc_features(navis.example_neurons(1, kind="mesh"), progress=False)


def test_neuronlist(labelled_example, toy):
    """Neurons with different compartments line up into one (padded) table."""
    nl = navis.NeuronList([labelled_example, toy])

    res = navis.ivscc_features(nl, progress=False)

    assert len(res) == 2
    # The example neuron has no axon, the toy neuron does
    assert res.axon_total_length.isna().sum() == 1


def test_context_is_shared(toy):
    """The expensive root-distance pass must only run once per neuron."""
    ctx = NeuronContext(toy)
    calls = []

    original = navis.utils.fastcore.dag.dist_to_root

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    navis.utils.fastcore.dag.dist_to_root = counting
    try:
        assert ctx.dist_to_root is ctx.dist_to_root
        assert len(calls) == 1
    finally:
        navis.utils.fastcore.dag.dist_to_root = original

    # Rooted at the soma, so the soma is at distance 0 from itself
    assert ctx.dist_to_root.loc[toy.soma] == 0
