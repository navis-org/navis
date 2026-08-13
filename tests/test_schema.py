"""Tests for the element-axis schema (`navis.core.schema`).

Two kinds of test live here:

- `test_schema_is_complete` guards the schema itself. Missing a field in an
  `AXES` declaration means that field silently survives a subset unfiltered, so
  the schema being complete is a property worth checking mechanically rather
  than by eye.
- the rest exercise the primitives (`resolve_selection`, `Survivors`,
  `repair_refs`, `record_provenance`) directly, plus the behaviours they fixed
  when `subset_neuron` moved onto them.

"""

import numpy as np
import pandas as pd
import pytest

import navis
from navis.core import schema


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def skeleton():
    n = navis.example_neurons(1, kind="skeleton")
    n.tags = {"soma": [int(n.soma)], "ends": n.leafs.node_id.values[:5].tolist()}
    return n


@pytest.fixture
def mesh():
    return navis.example_neurons(1, kind="mesh")


@pytest.fixture
def dotprops(skeleton):
    dp = navis.make_dotprops(skeleton, k=5)
    dp.soma = 100
    return dp


# ---------------------------------------------------------------------------
# The schema itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["skeleton", "mesh", "dotprops"])
def test_schema_is_complete(kind, request):
    """Every array-like field must be declared, temporary, or exempt.

    This is the test that turns "did I remember every corner case?" into CI. It
    is deliberately mechanical: it walks a populated neuron's `__dict__` and
    insists that anything holding bulk data has been accounted for.
    """
    n = request.getfixturevalue(kind)
    # Give it connectors so they show up in `__dict__`
    if not n.has_connectors:
        n.connectors = navis.example_neurons(1, kind="skeleton").connectors
    # ... and, for a mesh, its skeleton, so the link bookkeeping shows up too
    if isinstance(n, navis.Mesh):
        _ = n.skeleton

    declared = set()
    for axis in schema.declared_axes(n).values():
        declared.update(axis.data)
        declared.update(ref.attr for ref in axis.refs)
    # A link's mapping is aligned data of its source axis; it just happens to
    # live behind a path rather than a plain name.
    declared.update(link.mapping.split(".")[0] for link in schema.declared_links(n))

    accounted = declared | set(n.TEMP_ATTR) | set(n.AXIS_INDEPENDENT)

    undeclared = [
        key
        for key, value in n.__dict__.items()
        if isinstance(value, (np.ndarray, pd.DataFrame, dict))
        and key not in accounted
        # Per-instance copies of class config, and the schema's own bookkeeping
        and key not in ("SUMMARY_PROPS", "TEMP_ATTR", "_link_state", "_axes")
    ]

    assert not undeclared, (
        f"{type(n).__name__} has array-like attributes that no axis claims: "
        f"{undeclared}. Either add them to an axis in `AXES`, to `TEMP_ATTR` if "
        f"they are derived, or to `AXIS_INDEPENDENT` if they genuinely do not "
        f"align to any axis."
    )


@pytest.mark.parametrize("kind", ["skeleton", "mesh", "dotprops"])
def test_schema_declarations_are_resolvable(kind, request):
    """Declared attributes must actually exist (catches typos and renames)."""
    n = request.getfixturevalue(kind)

    for name, axis in schema.declared_axes(n).items():
        assert axis.name == name, "AXES key and Axis.name disagree"

        table = getattr(n, axis.data[0], None)
        if table is None:
            # Legitimately empty - a neuron need not have connectors - but then
            # the axis must read as empty rather than blow up
            assert schema.axis_length(n, axis) == 0
            continue

        if not axis.positional:
            assert axis.ids in table.columns, (
                f"axis '{axis.name}' declares ids column '{axis.ids}' which is "
                f"not in {axis.data[0]}"
            )


# ---------------------------------------------------------------------------
# resolve_selection
# ---------------------------------------------------------------------------


def test_resolve_selection_id_axis(skeleton):
    axis = schema.get_axis(skeleton, "nodes")
    ids = skeleton.nodes.node_id.values

    # By ID
    keep = schema.resolve_selection(skeleton, axis, ids[:10])
    assert keep.sum() == 10
    assert (skeleton.nodes.node_id.values[keep] == ids[:10]).all()

    # By boolean mask
    mask = np.zeros(len(ids), dtype=bool)
    mask[:10] = True
    assert (schema.resolve_selection(skeleton, axis, mask) == mask).all()

    # By DataFrame
    keep = schema.resolve_selection(skeleton, axis, skeleton.nodes.iloc[:10])
    assert keep.sum() == 10

    # By set - order must not matter
    keep = schema.resolve_selection(skeleton, axis, set(ids[:10].tolist()))
    assert keep.sum() == 10

    # Empty means keep nothing
    assert schema.resolve_selection(skeleton, axis, []).sum() == 0


def test_resolve_selection_positional_axis(dotprops):
    axis = schema.get_axis(dotprops, "points")

    keep = schema.resolve_selection(dotprops, axis, [3, 1, 2])
    assert keep.sum() == 3
    # Indices, so position - not order - is what counts
    assert np.where(keep)[0].tolist() == [1, 2, 3]


def test_resolve_selection_rejects_wrong_length_mask(skeleton):
    axis = schema.get_axis(skeleton, "nodes")
    with pytest.raises(ValueError, match="expected"):
        schema.resolve_selection(skeleton, axis, np.zeros(5, dtype=bool))


def test_get_axis_rejects_unknown_axis(skeleton):
    with pytest.raises(KeyError, match="no \"vertices\" axis"):
        schema.get_axis(skeleton, "vertices")


# ---------------------------------------------------------------------------
# Survivors
# ---------------------------------------------------------------------------


def test_survivors_from_mask():
    keep = np.array([True, False, True, True])
    s = schema.Survivors.from_mask(keep)
    assert s.kept.tolist() == [0, 2, 3]
    # old index -> new index, -1 where dropped
    assert s.old2new.tolist() == [0, schema.DROPPED, 1, 2]


def test_survivors_from_kept_respects_order():
    """`submesh` returns survivors in *its* order, which must be honoured."""
    s = schema.Survivors.from_kept(5, [4, 0, 2])
    assert s.old2new[4] == 0 and s.old2new[0] == 1 and s.old2new[2] == 2
    assert s.old2new[1] == schema.DROPPED and s.old2new[3] == schema.DROPPED


# ---------------------------------------------------------------------------
# repair_refs
# ---------------------------------------------------------------------------


def test_repair_column_drops_dangling(skeleton):
    axis = schema.get_axis(skeleton, "nodes")
    keep = np.zeros(skeleton.n_nodes, dtype=bool)
    keep[:2000] = True
    kept_ids = skeleton.nodes.node_id.values[keep]

    n_before = len(skeleton.connectors)
    schema.apply_selection(skeleton, axis, keep)

    assert len(skeleton.connectors) < n_before
    assert skeleton.connectors.node_id.isin(kept_ids).all()


def test_repair_column_null_makes_new_roots(skeleton):
    """A node whose parent was dropped becomes a root, not a dangling pointer."""
    axis = schema.get_axis(skeleton, "nodes")
    keep = np.zeros(skeleton.n_nodes, dtype=bool)
    keep[:2000] = True
    kept_ids = skeleton.nodes.node_id.values[keep]

    schema.apply_selection(skeleton, axis, keep)

    parents = skeleton.nodes.parent_id.values
    dangling = parents[(parents >= 0) & ~np.isin(parents, kept_ids)]
    assert len(dangling) == 0
    # No rows were dropped - only re-pointed
    assert len(skeleton.nodes) == 2000


def test_repair_column_preserves_column_order_and_dtype(skeleton):
    before_cols = list(skeleton.nodes.columns)
    before_dtypes = skeleton.nodes.dtypes.to_dict()

    sub = navis.subset_neuron(skeleton, skeleton.nodes.node_id.values[:2000])

    assert list(sub.nodes.columns) == before_cols
    assert sub.nodes.dtypes.to_dict() == before_dtypes


def test_repair_id_lists_drops_empty_keys(skeleton):
    """Tags pointing only at dropped nodes go away entirely."""
    ends = skeleton.tags["ends"]
    # Keep everything except the nodes the "ends" tag points at
    keep = ~skeleton.nodes.node_id.isin(ends).values

    axis = schema.get_axis(skeleton, "nodes")
    schema.apply_selection(skeleton, axis, keep)

    assert "ends" not in skeleton.tags
    assert "soma" in skeleton.tags


def test_repair_index_array_drops_incomplete_rows(mesh):
    """A face survives only if all three of its corners do."""
    sub = navis.subset_neuron(mesh, np.arange(5000))

    assert len(sub.faces)
    assert sub.faces.max() < len(sub.vertices)
    assert sub.faces.min() >= 0


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_record_provenance_id_axis(skeleton):
    axis = schema.get_axis(skeleton, "nodes")
    keep = np.zeros(skeleton.n_nodes, dtype=bool)
    keep[:10] = True
    kept_ids = skeleton.nodes.node_id.values[keep]

    child = skeleton.copy()
    survivors = schema.apply_selection(child, axis, keep)
    schema.record_provenance(
        child, skeleton.id, skeleton.core_md5, axis, survivors
    )

    assert child._prov.parent_id == skeleton.id
    # For an id-bearing axis, provenance *is* the surviving IDs
    assert child._prov.origin["nodes"].tolist() == kept_ids.tolist()


def test_record_provenance_positional_axis(dotprops):
    axis = schema.get_axis(dotprops, "points")
    keep = np.zeros(dotprops.n_points, dtype=bool)
    keep[[2, 7, 9]] = True

    child = dotprops.copy()
    survivors = schema.apply_selection(child, axis, keep)
    schema.record_provenance(
        child, dotprops.id, dotprops.core_md5, axis, survivors
    )

    # Child point i came from parent point origin[i]
    origin = child._prov.origin["points"]
    assert origin.tolist() == [2, 7, 9]
    assert np.allclose(child.points, dotprops.points[origin])


# ---------------------------------------------------------------------------
# Behaviours the schema fixed
# ---------------------------------------------------------------------------


def test_subset_remaps_dotprops_soma(dotprops):
    """Dotprops soma is a point *index* and has to be remapped, not left alone."""
    sub = navis.subset_neuron(dotprops, np.arange(50, 200))
    # Point 100 is now point 50
    assert sub.soma == 50
    assert np.allclose(sub.points[sub.soma], dotprops.points[100])


def test_subset_drops_dotprops_soma_when_outside(dotprops):
    sub = navis.subset_neuron(dotprops, np.arange(200, 400))
    assert sub.soma is None


def test_subset_keeps_soma_detection_callable():
    """A callable soma is a *rule*, not a reference - it survives a subset."""
    n = navis.example_neurons(1, kind="skeleton")
    # Rebuild from the node table so `_soma` is the default `find_soma` callable
    sk = navis.Skeleton(n.nodes.copy(), units=n.units)
    assert callable(sk._soma)
    assert sk.soma is not None

    sub = navis.subset_neuron(sk, sk.nodes.node_id.values)

    assert callable(sub._soma), "subsetting must not clobber the soma finder"
    assert sub.soma == sk.soma


def test_subset_mesh_connectors_with_unsorted_indices(mesh, skeleton):
    """Connector vertex indices must be remapped via the *actual* survivors."""
    mesh.connectors = skeleton.connectors.copy()
    mesh.connectors["vertex_id"] = mesh.snap(
        mesh.connectors[["x", "y", "z"]].values
    )[0]

    # Deliberately unsorted - `submesh` returns vertices in sorted order, so a
    # remap built from the requested order would be wrong
    idx = np.random.default_rng(0).permutation(np.arange(4000))
    sub = navis.subset_neuron(mesh, idx)

    assert len(sub.connectors)
    assert sub.connectors.vertex_id.max() < len(sub.vertices)
    assert sub.connectors.vertex_id.min() >= 0

    # Coordinates are the ground truth: each surviving connector must still
    # point at the same *place* it did before, under its new index. Join on
    # `connector_id` so the check does not lean on row order.
    joined = sub.connectors.merge(
        mesh.connectors[["connector_id", "vertex_id"]],
        on="connector_id",
        suffixes=("", "_old"),
    )
    assert len(joined) == len(sub.connectors)
    assert np.allclose(
        sub.vertices[joined.vertex_id.values],
        mesh.vertices[joined.vertex_id_old.values],
    )


def test_prune_by_strahler_repairs_tags_and_soma(skeleton):
    """`prune_by_strahler` used to drop nodes without repairing tags or soma."""
    navis.strahler_index(skeleton)
    twigs = skeleton.nodes.node_id.values[skeleton.nodes.strahler_index == 1]
    skeleton.tags = {"twig": twigs[:5].tolist(), "keep": [int(skeleton.soma)]}
    skeleton._soma = int(twigs[0])

    pruned = navis.prune_by_strahler(skeleton, to_prune=1, inplace=False)

    # The tag pointed only at pruned nodes, so it should be gone entirely
    assert "twig" not in pruned.tags
    assert "keep" in pruned.tags
    # Ditto the soma, which sat on a pruned node
    assert pruned._soma is None
    # And nothing may be left pointing at a node that no longer exists
    assert pruned.connectors.node_id.isin(pruned.nodes.node_id).all()


def test_mask_accepts_bool_and_ids_alike():
    """The three masked entry points resolve both spellings identically."""
    n = navis.example_neurons(1, kind="skeleton")
    rng = np.random.default_rng(0)
    mask = rng.random(n.n_nodes) < 0.7
    ids = n.nodes.node_id.values[mask]

    for kwargs in ({}, {"recursive": True}):
        by_mask = navis.prune_twigs(n, min_length=2000, mask=mask, **kwargs)
        by_ids = navis.prune_twigs(n, min_length=2000, mask=ids, **kwargs)
        assert by_mask.n_nodes == by_ids.n_nodes
        assert by_mask.cable_length == by_ids.cable_length


def test_selection_preserves_element_order(skeleton, mesh, dotprops):
    """Subsetting must not reorder what it keeps - only merging does that.

    Everything on the selection path is boolean masking, so surviving elements
    keep their relative order. Plenty of downstream code indexes into
    `.nodes`/`.vertices` positionally, so this is a contract, not an accident.
    """
    ids = skeleton.nodes.node_id.values
    picked = ids[[900, 100, 500, 5]]  # deliberately out of table order
    sub = navis.subset_neuron(skeleton, picked)
    assert sub.nodes.node_id.tolist() == sorted(picked.tolist(), key=list(ids).index)

    keep = np.zeros(skeleton.n_nodes, dtype=bool)
    keep[[5, 100, 500, 900]] = True
    sub = navis.subset_neuron(skeleton, keep)
    assert np.array_equal(sub.nodes.node_id.values, ids[keep])
    assert np.array_equal(
        sub.nodes[["x", "y", "z"]].values, skeleton.nodes[["x", "y", "z"]].values[keep]
    )

    pts = np.array([7, 2, 9, 4])
    d_sub = navis.subset_neuron(dotprops, pts)
    assert np.allclose(d_sub.points, dotprops.points[np.sort(pts)])

    # Meshes go through `submesh`. Provenance says exactly where each surviving
    # vertex came from, so "order preserved" is testable directly: the original
    # indices must come back strictly increasing.
    m_sub = navis.subset_neuron(mesh, np.arange(2000), track=True)
    origin = m_sub._prov.origin["vertices"]
    assert (np.diff(origin) > 0).all()
    assert np.allclose(m_sub.vertices, mesh.vertices[origin])


# ---------------------------------------------------------------------------
# Provenance tracking and merging
# ---------------------------------------------------------------------------


def test_subset_does_not_track_by_default(skeleton):
    sub = navis.subset_neuron(skeleton, skeleton.nodes.node_id.values[:100])
    assert not hasattr(sub, "_prov")


def test_tracked_subset_records_parent_and_epoch(skeleton):
    sub = navis.subset_neuron(
        skeleton, skeleton.nodes.node_id.values[:100], track=True
    )
    assert sub._prov.parent_id == skeleton.id
    assert sub._prov.parent_epoch == skeleton.core_md5


def test_merge_round_trip_skeleton(skeleton):
    """An untouched subset must put the neuron back exactly as it was."""
    navis.split_axon_dendrite(skeleton, label_only=True)
    axon = navis.subset_neuron(
        skeleton, skeleton.nodes.compartment == "axon", track=True
    )
    back = navis.merge_subset(skeleton, axon)

    assert back.n_nodes == skeleton.n_nodes
    assert len(back.root) == len(skeleton.root)
    assert float(back.cable_length) == pytest.approx(float(skeleton.cable_length))
    assert back.n_connectors == skeleton.n_connectors
    assert set(back.tags) == set(skeleton.tags)
    assert back.soma == skeleton.soma


def test_merge_round_trip_mesh(mesh):
    """Faces straddling the selection boundary must survive the round trip.

    The subset never contained them - only the parent did - so they can only
    come back because merge remaps the *parent's* faces rather than taking the
    child's.
    """
    half = np.where(np.arange(mesh.n_vertices) < mesh.n_vertices // 2)[0]
    sub = navis.subset_neuron(mesh, half, track=True)
    assert sub.n_faces < mesh.n_faces  # boundary faces were cut

    back = navis.merge_subset(mesh, sub)

    assert back.n_vertices == mesh.n_vertices
    assert back.n_faces == mesh.n_faces
    assert back.faces.max() < back.n_vertices and back.faces.min() >= 0
    assert np.allclose(np.sort(back.vertices, axis=0), np.sort(mesh.vertices, axis=0))


def test_merge_round_trip_dotprops(dotprops):
    sub = navis.subset_neuron(dotprops, np.arange(50, 200), track=True)
    back = navis.merge_subset(dotprops, sub)

    assert back.n_points == dotprops.n_points
    assert np.allclose(np.sort(back.points, axis=0), np.sort(dotprops.points, axis=0))
    # Element *order* is not preserved by a merge (retained elements come
    # first), but identity is: the soma still sits on the point it did
    assert np.allclose(back.points[back.soma], dotprops.points[100])
    assert np.allclose(back.vect[back.soma], dotprops.vect[100])


def test_merge_carries_edits_back(skeleton):
    navis.split_axon_dendrite(skeleton, label_only=True)
    axon = navis.subset_neuron(
        skeleton, skeleton.nodes.compartment == "axon", track=True
    )
    pruned = navis.prune_twigs(axon, 5000)
    assert pruned.n_nodes < axon.n_nodes

    merged = navis.merge_subset(skeleton, pruned)

    assert merged.n_nodes == skeleton.n_nodes - axon.n_nodes + pruned.n_nodes
    # The rest of the neuron is untouched and nothing is left dangling
    assert len(merged.root) == len(skeleton.root)
    dangling = merged.nodes.parent_id[
        (merged.nodes.parent_id >= 0)
        & ~merged.nodes.parent_id.isin(merged.nodes.node_id)
    ]
    assert not len(dangling)
    assert merged.connectors.node_id.isin(merged.nodes.node_id).all()


def test_merge_heals_the_cut_but_not_a_deliberate_break(skeleton):
    """Selecting severs the border element from its parent; merging reconnects it.

    But where the subset *deleted* the old parent, the break is the edit and
    must stand.
    """
    ids = skeleton.nodes.node_id.values
    # A subtree that is not the root, so its top node loses its parent
    distal = navis.graph.geodesic_matrix(skeleton, from_=[ids[0]])
    far = skeleton.nodes.node_id.values[np.argsort(distal.values[0])[-500:]]

    sub = navis.subset_neuron(skeleton, far, track=True)
    assert len(sub.root) >= 1  # cut loose by the selection

    back = navis.merge_subset(skeleton, sub)
    assert len(back.root) == len(skeleton.root), "the cut should have healed"

    # Now delete the subset's own root; its children must stay roots
    sub2 = navis.subset_neuron(skeleton, far, track=True)
    keep = ~np.isin(sub2.nodes.node_id.values, sub2.root)
    sub2 = navis.subset_neuron(sub2, sub2.nodes.node_id.values[keep])
    sub2._prov = navis.subset_neuron(skeleton, far, track=True)._prov
    merged = navis.merge_subset(skeleton, sub2)
    assert len(merged.root) > len(skeleton.root)


def test_merge_inplace(skeleton):
    sub = navis.subset_neuron(skeleton, skeleton.nodes.node_id.values[:2000], track=True)
    before = skeleton.n_nodes
    out = navis.merge_subset(skeleton, sub, inplace=True)
    assert out is skeleton
    assert skeleton.n_nodes == before


@pytest.mark.parametrize(
    "break_it, match",
    [
        ("untracked", "no provenance"),
        ("other_parent", "selected from"),
        ("parent_moved", "modified since"),
    ],
)
def test_merge_refuses_when_it_cannot_be_sure(skeleton, break_it, match):
    """Refusing is the point: a wrong merge is worse than no merge."""
    sub = navis.subset_neuron(skeleton, skeleton.nodes.node_id.values[:2000], track=True)
    parent = skeleton

    if break_it == "untracked":
        sub = navis.subset_neuron(skeleton, skeleton.nodes.node_id.values[:2000])
    elif break_it == "other_parent":
        parent = navis.example_neurons(1, kind="skeleton")
        parent.id = "some-other-neuron"
    elif break_it == "parent_moved":
        parent = skeleton.copy()
        navis.prune_twigs(parent, 5000, inplace=True)

    with pytest.raises(schema.MergeError, match=match):
        navis.merge_subset(parent, sub)


def test_merge_refuses_a_selection_of_a_selection(mesh):
    """A subset of a subset describes its *immediate* parent, and must say so.

    Both neurons carry the same `.id` - subsetting does not mint a new one - so
    the epoch is the only thing that can tell the grandparent it is not the
    parent. Without it this merges vertex indices that mean something else
    entirely, which is the one outcome the design is meant to rule out.
    """
    sub = navis.subset_neuron(mesh, np.arange(4000), track=True)
    sub2 = navis.subset_neuron(sub, np.arange(1000, 2000), track=True)

    with pytest.raises(schema.MergeError, match="modified since"):
        navis.merge_subset(mesh, sub2)

    # ... but it is still a valid selection of the neuron it did come from
    back = navis.merge_subset(sub, sub2)
    assert back.n_vertices == sub.n_vertices


def test_merge_refuses_misaligned_positional_provenance(mesh):
    """A positional subset restructured behind our back cannot be traced back."""
    sub = navis.subset_neuron(mesh, np.arange(3000), track=True)
    sub._vertices = sub._vertices[:100]

    with pytest.raises(schema.MergeError, match="restructured"):
        navis.merge_subset(mesh, sub)


def test_merge_refuses_id_collisions(skeleton):
    """A subset that invented an ID already in use outside it must not merge."""
    ids = skeleton.nodes.node_id.values
    sub = navis.subset_neuron(skeleton, ids[:2000], track=True)
    # Rename one of the subset's nodes onto a node that was *not* selected
    nodes = sub.nodes.copy()
    nodes.loc[nodes.index[0], "node_id"] = ids[-1]
    sub._nodes = nodes

    with pytest.raises(schema.MergeError, match="duplicate IDs"):
        navis.merge_subset(skeleton, sub)


def test_subset_keep_disc_cn_leaves_connectors_alone(skeleton):
    n_before = len(skeleton.connectors)
    sub = navis.subset_neuron(
        skeleton, skeleton.nodes.node_id.values[:2000], keep_disc_cn=True
    )
    assert len(sub.connectors) == n_before
