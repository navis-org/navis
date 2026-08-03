"""Tests for masking - temporarily restricting a neuron to part of itself.

Masking is a thin layer over `subset_neuron(track=True)` + `merge_subset`, so
most of the correctness lives in `test_schema.py`. What is tested here is the
layer itself: that a mask is undone exactly, that edits made through one are
carried back, that masks nest, and that nothing is left half-masked when
something goes wrong.

"""

import logging

import numpy as np
import pytest

import navis
from navis.core.masking import MaskingError
from navis.core.schema import MergeError


@pytest.fixture
def skeleton():
    n = navis.example_neurons(1, kind="skeleton")
    navis.split_axon_dendrite(n, label_only=True)
    n.tags = {"soma": [int(n.soma)]}
    return n


@pytest.fixture
def mesh():
    return navis.example_neurons(1, kind="mesh")


def axon(n):
    return n.nodes.compartment == "axon"


# ---------------------------------------------------------------------------
# The mask itself
# ---------------------------------------------------------------------------


def test_mask_restricts_in_place(skeleton):
    """Inside the mask the neuron *is* the masked region - same object."""
    full = skeleton.n_nodes
    full_cable = float(skeleton.cable_length)
    held = skeleton  # a reference someone else might be holding

    with navis.masked(skeleton, axon):
        assert skeleton.is_masked
        assert skeleton.n_nodes < full
        assert held.n_nodes == skeleton.n_nodes  # the reference sees it too
        assert float(skeleton.cable_length) < full_cable

    assert not skeleton.is_masked
    assert skeleton.n_nodes == full
    assert float(skeleton.cable_length) == full_cable


def test_unmask_restores_exactly(skeleton):
    before = (
        skeleton.n_nodes,
        skeleton.n_connectors,
        float(skeleton.cable_length),
        len(skeleton.root),
        skeleton.soma,
        dict(skeleton.tags),
    )
    with navis.masked(skeleton, axon):
        navis.prune_twigs(skeleton, 5000, inplace=True)  # discarded on exit

    assert (
        skeleton.n_nodes,
        skeleton.n_connectors,
        float(skeleton.cable_length),
        len(skeleton.root),
        skeleton.soma,
        dict(skeleton.tags),
    ) == before


def test_edits_carry_back_when_not_resetting(skeleton):
    full = skeleton.n_nodes
    with navis.masked(skeleton, axon):
        masked_nodes = skeleton.n_nodes

    with navis.masked(skeleton, axon, reset=False):
        navis.prune_twigs(skeleton, 5000, inplace=True)
        edited = skeleton.n_nodes

    assert skeleton.n_nodes == full - masked_nodes + edited
    # The rest of the neuron is intact and nothing dangles
    assert len(skeleton.root) == 1
    dangling = skeleton.nodes.parent_id[
        (skeleton.nodes.parent_id >= 0)
        & ~skeleton.nodes.parent_id.isin(skeleton.nodes.node_id)
    ]
    assert not len(dangling)
    assert skeleton.connectors.node_id.isin(skeleton.nodes.node_id).all()


def test_mesh_edits_carry_back(mesh):
    """A structural edit inside a mesh mask must survive unmasking.

    This is the case that needs provenance to be carried through the inner
    selection *and* the original selection's footprint to be remembered
    separately - going by the surviving elements alone would quietly restore
    the ones the edit deleted.
    """
    full = mesh.n_vertices
    half = np.where(np.arange(mesh.n_vertices) < mesh.n_vertices // 2)[0]

    with navis.masked(mesh, half, reset=False):
        masked_verts = mesh.n_vertices
        navis.subset_neuron(mesh, np.arange(mesh.n_vertices // 2), inplace=True)
        edited = mesh.n_vertices

    assert edited < masked_verts
    assert mesh.n_vertices == full - masked_verts + edited
    assert mesh.faces.max() < mesh.n_vertices and mesh.faces.min() >= 0


def test_mesh_mask_round_trip(mesh):
    before = (mesh.n_vertices, mesh.n_faces)
    with navis.masked(mesh, np.arange(mesh.n_vertices // 2)):
        assert mesh.n_vertices < before[0]
    assert (mesh.n_vertices, mesh.n_faces) == before


def test_dotprops_mask_round_trip():
    dp = navis.make_dotprops(navis.example_neurons(1), k=5)
    before = dp.n_points
    with navis.masked(dp, np.arange(1000)):
        assert dp.n_points == 1000
    assert dp.n_points == before


# ---------------------------------------------------------------------------
# Nesting
# ---------------------------------------------------------------------------


def test_masks_nest(skeleton):
    full = skeleton.n_nodes

    skeleton.mask(axon(skeleton), inplace=True)
    outer = skeleton.n_nodes
    skeleton.mask(skeleton.nodes.node_id.values[:100], inplace=True)

    assert skeleton.n_nodes == 100
    assert len(skeleton._mask_stack) == 2

    skeleton.unmask()
    assert skeleton.n_nodes == outer
    skeleton.unmask()
    assert skeleton.n_nodes == full
    assert not skeleton.is_masked


def test_nested_context_managers(skeleton):
    full = skeleton.n_nodes
    with navis.masked(skeleton, axon):
        outer = skeleton.n_nodes
        with navis.masked(skeleton, skeleton.nodes.node_id.values[:50]):
            assert skeleton.n_nodes == 50
        assert skeleton.n_nodes == outer
    assert skeleton.n_nodes == full


def test_nested_edits_carry_back(skeleton):
    """An inner mask must be mergeable without knowing an outer one exists.

    The provenance an inner selection records describes the *outer* neuron, and
    is only usable if it says so - going by the masked neuron's `.id` alone
    cannot tell the two apart, because masking in place keeps the id.
    """
    skeleton.mask(skeleton.nodes.node_id.values[:2000], inplace=True)
    outer = skeleton.n_nodes

    skeleton.mask(skeleton.nodes.node_id.values[:500], inplace=True)
    navis.prune_twigs(skeleton, 5000, inplace=True)
    inner = skeleton.n_nodes
    assert inner < 500

    skeleton.unmask(reset=False)
    assert skeleton.n_nodes == outer - 500 + inner
    assert skeleton.is_masked, "only the inner mask should have been popped"

    skeleton.unmask(reset=False)
    assert not skeleton.is_masked
    assert not len(
        skeleton.nodes.parent_id[
            (skeleton.nodes.parent_id >= 0)
            & ~skeleton.nodes.parent_id.isin(skeleton.nodes.node_id)
        ]
    )


def test_nested_edits_carry_back_positional(mesh):
    """Same, for an axis where provenance *is* the elements' identity.

    Each merge has to compose the provenance it rebuilds onto the one the
    parent already had, or the outer unmask finds a parent whose provenance no
    longer describes its own elements and (rightly) refuses.
    """
    mesh.mask(np.arange(4000), inplace=True)
    outer = mesh.n_vertices

    mesh.mask(np.arange(1000), inplace=True)
    covered = len(mesh._prov.covered["vertices"])
    navis.subset_neuron(mesh, np.arange(400), inplace=True)
    inner = mesh.n_vertices
    assert inner < covered

    mesh.unmask(reset=False)
    assert mesh.n_vertices == outer - covered + inner
    assert mesh.is_masked

    mesh.unmask(reset=False)
    assert not mesh.is_masked
    assert mesh.faces.max() < mesh.n_vertices and mesh.faces.min() >= 0


def test_copy_does_not_share_the_mask_stack(skeleton):
    skeleton.mask(axon(skeleton), inplace=True)
    twin = skeleton.copy()

    twin.unmask()

    assert not twin.is_masked
    assert skeleton.is_masked, "unmasking a copy must not unmask the original"


def test_copy_does_not_share_the_snapshots(skeleton):
    """Copying the stack is not enough - the frames in it have to be copied too.

    Otherwise the copy adopts a snapshot the original is still holding, and from
    then on the two share their data tables.
    """
    skeleton.mask(axon(skeleton), inplace=True)
    twin = skeleton.copy()

    twin.unmask()
    twin.nodes.loc[0, "x"] = 999999.0

    skeleton.unmask()
    assert skeleton.nodes.loc[0, "x"] != 999999.0


# ---------------------------------------------------------------------------
# Masks that cut across branches
# ---------------------------------------------------------------------------


def cuts_branches(n):
    """A mask that slices across the arbour rather than following it."""
    geo = navis.graph.geodesic_matrix(n, from_=[n.root[0]]).values[0]
    return geo < np.percentile(geo, 60)


def warnings_from(caplog):
    return [r.message for r in caplog.records if r.levelno >= logging.WARNING]


def test_warns_when_the_mask_cuts_across_branches(skeleton, caplog):
    """Those nodes look like the ends of the arbour to everything downstream."""
    with caplog.at_level(logging.WARNING, logger="navis"):
        with navis.masked(skeleton, cuts_branches(skeleton)):
            pass

    assert any("cut across" in m for m in warnings_from(caplog))


def test_no_warning_when_the_mask_keeps_whole_subtrees(skeleton, caplog):
    """A compartment ends where the neuron does, so nothing is misleading."""
    with caplog.at_level(logging.WARNING, logger="navis"):
        with navis.masked(skeleton, axon):
            pass

    assert not any("cut across" in m for m in warnings_from(caplog))


@pytest.mark.parametrize("via", ["context_manager", "method"])
def test_cut_warning_can_be_silenced(skeleton, caplog, via):
    mask = cuts_branches(skeleton)
    with caplog.at_level(logging.WARNING, logger="navis"):
        if via == "context_manager":
            with navis.masked(skeleton, mask, warn_cut=False):
                pass
        else:
            skeleton.mask(mask, inplace=True, warn_cut=False)

    assert not any("cut across" in m for m in warnings_from(caplog))


def test_neuronlist_cut_warning_is_emitted_once(caplog):
    """One warning for the list - not one per neuron."""
    nl = navis.example_neurons(3)
    with caplog.at_level(logging.WARNING, logger="navis"):
        with navis.masked(nl, lambda n: n.nodes.y > 35_000):
            pass

    cut = [m for m in warnings_from(caplog) if "cut across" in m]
    assert len(cut) == 1
    assert "3 of 3 neurons" in cut[0]


def test_warns_when_merging_leaves_the_neuron_in_pieces(skeleton, caplog):
    """The sharp warning: fires only once something actually came apart."""
    with caplog.at_level(logging.WARNING, logger="navis"):
        with navis.masked(skeleton, cuts_branches(skeleton), reset=False):
            navis.prune_twigs(skeleton, 10_000, inplace=True)

    assert len(skeleton.root) > 1
    assert any("more piece(s)" in m for m in warnings_from(caplog))


def test_no_severed_warning_when_resetting(skeleton, caplog):
    """Nothing was folded back, so nothing can have come apart."""
    with caplog.at_level(logging.WARNING, logger="navis"):
        with navis.masked(skeleton, cuts_branches(skeleton), reset=True):
            navis.prune_twigs(skeleton, 10_000, inplace=True)

    assert len(skeleton.root) == 1
    assert not any("more piece(s)" in m for m in warnings_from(caplog))


def test_no_severed_warning_when_the_merge_is_clean(skeleton, caplog):
    """Pruning within a compartment removes real twigs and severs nothing."""
    with caplog.at_level(logging.WARNING, logger="navis"):
        with navis.masked(skeleton, axon, reset=False):
            navis.prune_twigs(skeleton, 5000, inplace=True)

    assert len(skeleton.root) == 1
    assert not any("more piece(s)" in m for m in warnings_from(caplog))


def test_severed_warning_can_be_silenced(skeleton, caplog):
    mask = cuts_branches(skeleton)
    with caplog.at_level(logging.WARNING, logger="navis"):
        skeleton.mask(mask, inplace=True, warn_cut=False)
        navis.prune_twigs(skeleton, 10_000, inplace=True)
        skeleton.unmask(reset=False, warn_cut=False)

    assert len(skeleton.root) > 1
    assert not warnings_from(caplog)


@pytest.mark.parametrize("kind", ["mesh", "dotprops"])
def test_no_cut_warning_for_types_without_terminals(caplog, kind):
    n = navis.example_neurons(1, kind="mesh")
    if kind == "dotprops":
        n = navis.make_dotprops(navis.example_neurons(1, kind="skeleton"), k=5)

    with caplog.at_level(logging.WARNING, logger="navis"):
        with navis.masked(n, np.arange(1000)):
            pass

    assert not any("cut across" in m for m in warnings_from(caplog))


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_mask_unwinds_if_the_block_raises(skeleton):
    full = skeleton.n_nodes
    with pytest.raises(RuntimeError):
        with navis.masked(skeleton, axon):
            raise RuntimeError("boom")

    assert not skeleton.is_masked
    assert skeleton.n_nodes == full


def test_block_that_raises_resets_even_when_asked_not_to(skeleton):
    """`reset=False` promises to keep the edit, not half of one."""
    before = skeleton.n_nodes
    with pytest.raises(RuntimeError):
        with navis.masked(skeleton, axon(skeleton), reset=False):
            navis.prune_twigs(skeleton, 5000, inplace=True)
            raise RuntimeError("boom")

    assert not skeleton.is_masked
    assert skeleton.n_nodes == before


def test_unwinding_continues_past_a_neuron_that_cannot_be_merged():
    """One neuron's bad merge must not leave the others masked."""
    nl = navis.example_neurons(3)

    with pytest.raises(MaskingError, match="could not be unmasked"):
        with navis.masked(nl, lambda n: n.nodes.node_id.values[:500], reset=False):
            navis.prune_twigs(nl, 5000, inplace=True)
            nl[1]._prov = None  # only the middle one is unmergeable

    assert not any(n.is_masked for n in nl)
    # The one that failed was restored; the others kept their edits
    assert nl[1].n_nodes == navis.example_neurons(3)[1].n_nodes
    assert nl[0].n_nodes < navis.example_neurons(3)[0].n_nodes


def test_a_failed_unwind_does_not_bury_the_block_s_own_error():
    """Whatever went wrong in the caller's block is the more useful error."""
    nl = navis.example_neurons(2)

    with pytest.raises(ValueError, match="real problem"):
        with navis.masked(nl, lambda n: n.nodes.node_id.values[:500]):
            nl[0]._mask_stack = []  # sabotage the unwind
            raise ValueError("real problem")


def test_neuronlist_unwinds_if_one_neuron_cannot_be_masked():
    """A mask that fails part way must not leave the earlier neurons masked."""
    nl = navis.example_neurons(3)

    def picky(n):
        if n.id == nl[2].id:
            raise ValueError("nope")
        return n.nodes.node_id.values[:100]

    with pytest.raises(ValueError):
        with navis.masked(nl, picky):
            pass

    assert not any(n.is_masked for n in nl)


def test_masking_an_unsupported_type_says_so():
    """The error must be about masking, not leak `subset_neuron`'s type check."""
    vx = navis.voxelize(navis.example_neurons(1), pitch="2 microns")
    with pytest.raises(MaskingError, match="cannot be masked"):
        vx.mask(np.zeros(vx.shape, dtype=bool))


def test_unmask_without_mask_raises(skeleton):
    with pytest.raises(MaskingError, match="not masked"):
        skeleton.unmask()


def test_apply_mask_makes_it_permanent(skeleton):
    skeleton.mask(skeleton.nodes.node_id.values[:500], inplace=True)
    skeleton.apply_mask(inplace=True)

    assert skeleton.n_nodes == 500
    assert not skeleton.is_masked
    assert not hasattr(skeleton, "_prov")
    with pytest.raises(MaskingError):
        skeleton.unmask()


def test_apply_mask_not_inplace_leaves_original_masked(skeleton):
    skeleton.mask(skeleton.nodes.node_id.values[:500], inplace=True)
    applied = skeleton.apply_mask()

    assert not applied.is_masked
    assert skeleton.is_masked


def test_reroot_through_a_mask_is_not_undone(skeleton):
    """Merging heals the cut the *selection* made - never an edit of the caller's.

    Rerooting is the sharp case: every node along the path to the old root has
    its parent flipped, so the top one looks exactly like a node the selection
    cut loose. Re-attaching it puts the link back the other way round too, which
    closes a cycle and leaves the neuron with no root at all.
    """
    skeleton.mask(skeleton.nodes.node_id.values[:2000], inplace=True)
    new_root = int(skeleton.nodes.node_id.values[50])
    skeleton.reroot(new_root, inplace=True)

    skeleton.unmask(reset=False)

    assert skeleton.root.tolist() == [new_root]
    # A cycle would show up as nodes that no walk up the parents can reach
    seen, node = set(), new_root
    parents = dict(skeleton.nodes[["node_id", "parent_id"]].values)
    while node != -1:
        assert node not in seen, "parent chain loops back on itself"
        seen.add(node)
        node = parents[node]


def test_deliberate_break_through_a_mask_stands(skeleton):
    """Cutting inside a mask is an edit, not damage to be repaired."""
    skeleton.mask(skeleton.nodes.node_id.values[:2000], inplace=True)
    victim = int(skeleton.nodes.node_id.values[500])
    skeleton.nodes.loc[skeleton.nodes.node_id == victim, "parent_id"] = -1

    skeleton.unmask(reset=False)

    assert victim in skeleton.root.tolist()


def test_unmask_refuses_to_guess_when_provenance_is_lost(mesh):
    """A mesh restructured behind the schema's back cannot be merged back."""
    mesh.mask(np.arange(3000), inplace=True)
    mesh._vertices = mesh._vertices[:100]  # nothing maintained provenance here

    with pytest.raises(MergeError, match="restructured"):
        mesh.unmask(reset=False)


def test_reset_works_even_when_merge_would_not(mesh):
    """`reset=True` never needs provenance, so it is always available."""
    before = mesh.n_vertices
    mesh.mask(np.arange(3000), inplace=True)
    mesh._vertices = mesh._vertices[:100]

    mesh.unmask(reset=True)
    assert mesh.n_vertices == before


# ---------------------------------------------------------------------------
# NeuronLists
# ---------------------------------------------------------------------------


def test_masked_neuronlist():
    nl = navis.example_neurons(3)
    navis.split_axon_dendrite(nl, label_only=True)
    before = nl.n_nodes.copy()

    with navis.masked(nl, axon):
        assert all(n.is_masked for n in nl)
        assert (nl.n_nodes < before).all()

    assert not any(n.is_masked for n in nl)
    assert (nl.n_nodes == before).all()


def test_masked_neuronlist_with_dict():
    nl = navis.example_neurons(2)
    masks = {n.id: n.nodes.node_id.values[:100] for n in nl}

    with navis.masked(nl, masks):
        assert (nl.n_nodes == 100).all()
    assert (nl.n_nodes > 100).all()


def test_masked_neuronlist_missing_entry_raises():
    nl = navis.example_neurons(2)
    with pytest.raises(MaskingError, match="No mask given"):
        with navis.masked(nl, {nl[0].id: nl[0].nodes.node_id.values[:10]}):
            pass
    assert not any(n.is_masked for n in nl)
