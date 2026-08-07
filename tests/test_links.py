"""Tests for links between representations (`navis.core.schema`).

A link is an array aligned to one axis whose values name elements of another,
so the tests come in three groups:

- that a selection *carries* it - the mesh/skeleton cascade, which is what stops
  a masked mesh from throwing its skeleton away and re-skeletonizing the
  remainder into a different set of nodes;
- that it goes *stale* whenever we cannot vouch for it, since the whole point of
  keeping a skeleton is that its node IDs still mean something;
- that mappings *compose*, so a correspondence nobody declared directly is still
  available.

"""

import numpy as np
import pandas as pd
import pytest

import navis
from navis.core import schema


@pytest.fixture(scope="module")
def _skeletonized():
    """Skeletonize once for the module - it costs ~200x what copying does."""
    n = navis.example_neurons(1, kind="mesh")
    _ = n.skeleton
    return n


@pytest.fixture
def mesh(_skeletonized):
    """A mesh with its skeleton (and therefore its vertex map) attached."""
    return _skeletonized.copy()


@pytest.fixture
def skeleton():
    return navis.example_neurons(1, kind="skeleton")


@pytest.fixture
def dotprops(skeleton):
    dp = navis.make_dotprops(skeleton, k=5)
    dp.connectors = skeleton.connectors.copy()
    return dp


@pytest.fixture
def link(mesh):
    return schema.get_link(mesh, "skeleton")


def half(mesh):
    """Vertex indices for one half of a mesh - a mask that cuts across it."""
    return np.where(mesh.vertices[:, 0] > np.median(mesh.vertices[:, 0]))[0]


def still_nearby(mesh):
    """How far something may move and still be where it was, roughly.

    A tenth of the neuron's own extent: loose enough that decimation moving a
    vertex to the quadric-optimal point never trips it, tight enough that a
    reference landing on an unrelated branch always does.
    """
    return np.ptp(mesh.vertices, axis=0).max() / 10


# ---------------------------------------------------------------------------
# The declaration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["skeleton", "mesh", "dotprops"])
def test_declared_links_are_resolvable(kind, request):
    """Every declared link must point at things that exist (catches renames)."""
    n = request.getfixturevalue(kind)
    axes = schema.declared_axes(n)

    for lk in schema.declared_links(n):
        assert lk.source in axes, f"link '{lk.key}' starts from no axis"

        target = schema.link_target(n, lk)
        assert target is not None, f"link '{lk.key}' has nothing attached"
        assert lk.target_axis in schema.declared_axes(target), (
            f"link '{lk.key}' names axis '{lk.target_axis}' which "
            f"{type(target).__name__} does not have"
        )
        # The mapping itself may legitimately not be there yet - a mesh's
        # connectors only learn which vertex they sit on when asked - but the
        # thing it hangs off has to be resolvable.
        assert schema._read_path(n, lk.mapping) is not None, (
            f"link '{lk.key}' declares mapping '{lk.mapping}' which does not exist"
        )


def test_links_reject_unknown_cascade():
    with pytest.raises(ValueError, match="Unknown cascade"):
        schema.Link(name="x", source="vertices", mapping="_m", target_axis="nodes",
                    cascade="teleport")


def test_links_reject_duplicates():
    lk = schema.Link(name="x", source="vertices", mapping="_m", target_axis="nodes")
    with pytest.raises(ValueError, match="Duplicate link"):
        schema.links(lk, lk)


def test_subsetting_a_skeleton_leaves_its_vertex_map_alone(mesh):
    """It is aligned to the *mesh's* vertices, so the skeleton must not subset it."""
    skeleton = mesh.skeleton
    before = skeleton.vertex_map.copy()

    navis.subset_neuron(
        skeleton, skeleton.nodes.node_id.values[:100], inplace=True
    )

    assert np.array_equal(skeleton.vertex_map, before)


def test_collapse_nodes_maintains_the_vertex_map(mesh):
    """The branch that does this guarded on a name that never existed."""
    skeleton = mesh.skeleton
    which = skeleton.nodes.node_id.values[1:4]

    navis.graph.graph_utils.collapse_nodes(skeleton, which, inplace=True)

    assert len(skeleton.vertex_map) == len(mesh.vertices)
    assert not np.isin(skeleton.vertex_map, which[1:]).any()
    assert np.isin(skeleton.vertex_map, skeleton.nodes.node_id.values).all()


def test_vertex_map_missing_says_where_it_comes_from():
    sk = navis.example_neurons(1, kind="skeleton")
    with pytest.raises(AttributeError, match="mesh2skeleton"):
        sk.vertex_map


# ---------------------------------------------------------------------------
# Carrying a link through a selection
# ---------------------------------------------------------------------------


def test_subset_carries_the_skeleton(mesh):
    """The point of the whole exercise: node IDs survive a subset."""
    before = mesh.skeleton.nodes.node_id.values

    sub = navis.subset_neuron(mesh, half(mesh))

    assert sub.skeleton.n_nodes < len(before)
    assert np.isin(sub.skeleton.nodes.node_id.values, before).all(), (
        "the skeleton was regenerated rather than carried - node IDs that never "
        "existed before have appeared"
    )


def test_subset_keeps_the_vertex_map_aligned_and_valid(mesh):
    sub = navis.subset_neuron(mesh, half(mesh))
    vertex_map = sub.skeleton.vertex_map

    assert len(vertex_map) == len(sub.vertices)
    live = vertex_map[vertex_map != schema.DROPPED]
    assert np.isin(live, sub.skeleton.nodes.node_id.values).all()


def test_subset_keeps_only_nodes_that_still_have_vertices(mesh):
    keep = half(mesh)
    expected = set(np.unique(mesh.skeleton.vertex_map[keep]))

    sub = navis.subset_neuron(mesh, keep)

    # `submesh` drops degenerate vertices, so the skeleton may lose a node the
    # requested mask would have kept - but never gain one.
    assert set(sub.skeleton.nodes.node_id.values) <= expected


def test_subset_does_not_reskeletonize(mesh, monkeypatch):
    """Carrying the skeleton has to actually replace the expensive path."""
    def boom(*args, **kwargs):
        raise AssertionError("skeletonize() was called - the link was not carried")

    monkeypatch.setattr(navis.Mesh, "skeletonize", boom)

    sub = navis.subset_neuron(mesh, half(mesh))
    assert sub.skeleton.n_nodes > 0


def test_cascade_reclassifies_nodes(mesh):
    """Roots, branches and leaves move - and `type` is data, not a cache."""
    sub = navis.subset_neuron(mesh, half(mesh))
    carried = sub.skeleton.nodes["type"].value_counts().to_dict()

    fresh = sub.skeleton.copy()
    navis.graph.classify_nodes(fresh, inplace=True)

    assert carried == fresh.nodes["type"].value_counts().to_dict()


def test_subset_to_nothing(mesh):
    sub = navis.subset_neuron(mesh, np.array([], dtype=int))
    assert len(sub.vertices) == 0
    assert sub.skeleton.n_nodes == 0


def test_mask_carries_and_unmask_restores(mesh):
    before = mesh.skeleton.nodes.node_id.values

    with navis.masked(mesh, half(mesh)):
        assert mesh.skeleton.n_nodes < len(before)
        assert np.isin(mesh.skeleton.nodes.node_id.values, before).all()

    assert mesh.skeleton.n_nodes == len(before)


def test_prune_twigs_on_a_mesh_keeps_node_identity(mesh):
    """`meshneuron_skeleton` subsets the mesh, so it rides on the same cascade."""
    before = mesh.skeleton.nodes.node_id.values

    pruned = navis.prune_twigs(mesh, 5000)

    assert len(pruned.vertices) < len(mesh.vertices)
    assert np.isin(pruned.skeleton.nodes.node_id.values, before).all()


# ---------------------------------------------------------------------------
# Going stale
# ---------------------------------------------------------------------------


def test_unmanaged_vertex_change_invalidates(mesh, link):
    before = mesh.skeleton

    mesh.vertices = mesh.vertices * 2

    assert not schema.target_is_current(mesh, link)
    assert mesh.skeleton is not before


def test_hand_attached_skeleton_survives_until_the_mesh_changes(mesh):
    """No vertex map to carry, but it is still this mesh's skeleton for now."""
    hand = navis.example_neurons(1, kind="skeleton")
    mesh.skeleton = hand

    assert mesh.skeleton is hand

    mesh.vertices = mesh.vertices + 1
    assert mesh.skeleton is not hand


def test_editing_the_skeleton_underneath_invalidates(mesh, link):
    """The far end counts too - a link is only as good as both of its axes."""
    navis.subset_neuron(mesh.skeleton, mesh.skeleton.nodes.node_id.values[:100],
                        inplace=True)

    assert not schema.target_is_current(mesh, link)


def test_moving_the_skeleton_does_not_invalidate(mesh, link):
    """Links store node IDs, so rerooting or moving nodes leaves them valid."""
    mesh.skeleton.reroot(mesh.skeleton.leafs.node_id.values[0], inplace=True)

    assert schema.target_is_current(mesh, link)
    assert schema.mapping_is_current(mesh, link)


def test_merge_subset_leaves_the_link_dead(mesh, link):
    """We do not know how to merge two skeletons, so we must not pretend to."""
    sub = navis.subset_neuron(mesh, half(mesh), track=True)
    merged = navis.merge_subset(mesh, sub)

    assert not schema.target_is_current(merged, link)


def test_unstamped_link_reads_as_stale(mesh, link):
    """The failure mode of forgetting `refresh_links` must be a wasted rebuild."""
    axis = schema.get_axis(mesh, "vertices")
    keep = np.ones(len(mesh.vertices), dtype=bool)
    schema.follow_links(
        mesh,
        axis,
        schema.snapshot_links(mesh, axis),
        keep,
        schema.Survivors.from_mask(keep),
    )

    assert not schema.target_is_current(mesh, link)


def test_apply_selection_makes_a_pre_subsetting_caller_say(mesh):
    """Forgetting to snapshot looks exactly like having nothing to snapshot."""
    axis = schema.get_axis(mesh, "vertices")
    survivors = schema.Survivors.from_kept(len(mesh.vertices), [0, 1, 2])

    with pytest.raises(TypeError, match="must also be given `links`"):
        schema.apply_selection(mesh, axis, survivors=survivors)


def test_copies_do_not_share_link_state(mesh):
    sub = navis.subset_neuron(mesh, half(mesh))
    carried = sub.skeleton

    copy = sub.copy()
    copy.vertices = copy.vertices * 2

    assert copy.skeleton is not carried
    assert sub.skeleton is carried


def test_link_survives_pickling(mesh):
    import pickle

    sub = navis.subset_neuron(mesh, half(mesh))
    restored = pickle.loads(pickle.dumps(sub))

    assert restored.skeleton.n_nodes == sub.skeleton.n_nodes
    assert np.array_equal(restored.skeleton.vertex_map, sub.skeleton.vertex_map)


# ---------------------------------------------------------------------------
# Mappings
# ---------------------------------------------------------------------------


def test_get_mapping_is_the_vertex_map(mesh):
    assert np.array_equal(
        schema.get_mapping(mesh, "vertices", "skeleton"), mesh.skeleton.vertex_map
    )


def test_get_mapping_to_self_is_identity(mesh):
    assert np.array_equal(
        schema.get_mapping(mesh, "vertices", "vertices"), np.arange(len(mesh.vertices))
    )


def test_get_mapping_refuses_a_stale_link(mesh):
    mesh.vertices = mesh.vertices * 2
    with pytest.raises(schema.MappingError, match="out of date"):
        schema.get_mapping(mesh, "vertices", "skeleton")


def test_select_across_goes_backwards(mesh):
    nodes = mesh.skeleton.nodes.node_id.values[:200]

    keep = schema.select_across(mesh, "vertices", "skeleton", nodes)

    assert keep.dtype == bool and len(keep) == len(mesh.vertices)
    assert np.array_equal(keep, np.isin(mesh.skeleton.vertex_map, nodes))


def test_link_path_refuses_to_run_backwards(mesh):
    """One vertex has one node; one node has many vertices. Not the same question."""
    with pytest.raises(KeyError, match="only compose forwards"):
        schema.link_path(mesh, "skeleton", "vertices")


def test_link_path_reports_unknown_endpoints(mesh):
    with pytest.raises(KeyError, match="No links lead"):
        schema.link_path(mesh, "vertices", "voxels")


def test_get_link_reports_what_is_there(mesh):
    with pytest.raises(KeyError, match="has no link"):
        schema.get_link(mesh, "voxels")


# ---------------------------------------------------------------------------
# Connectors
#
# Connectors are the first thing besides the vertex map to be declared as a
# link: elements of their own, sitting on a node or a vertex. That buys three
# things the old bare reference could not - anything aligned to them is carried,
# the correspondence composes onto the skeleton, and the same declaration serves
# every neuron type.
# ---------------------------------------------------------------------------


def test_connectors_are_one_axis_for_every_type(skeleton, mesh, dotprops):
    for n in (skeleton, mesh, dotprops):
        axis = schema.get_axis(n, "connectors")
        assert schema.axis_length(n, axis) == n.n_connectors
        # One shared declaration, not one per class
        assert axis is schema.CONNECTOR_AXIS


def test_connectors_work_on_a_type_with_no_schema(skeleton):
    """`Voxels` declares no axes, so its connectors axis has to be made for it.

    `attach` no longer conjures an axis nobody declared, so the setter says it.
    """
    voxels = navis.Voxels(np.random.rand(4, 4, 4))

    voxels.connectors = skeleton.connectors

    assert voxels.n_connectors == skeleton.n_connectors
    assert "connectors" in schema.declared_axes(voxels)
    # ... on the instance only: an empty `AXES` is still what says this type
    # cannot be subset or masked element-wise
    assert navis.Voxels.AXES == {}


def test_connectors_are_not_given_an_invented_identity(mesh):
    """navis must not write columns into the table it was handed."""
    table = mesh.connectors[["x", "y", "z"]].copy()
    mesh.connectors = table

    assert list(mesh.connectors.columns) == ["x", "y", "z"]


def test_subset_drops_connectors_whose_node_is_gone(skeleton):
    keep = skeleton.nodes.node_id.values[:500]

    sub = navis.subset_neuron(skeleton, keep)

    assert sub.n_connectors < skeleton.n_connectors
    assert sub.connectors.node_id.isin(sub.nodes.node_id).all()


def test_keep_disc_cn_still_keeps_them(skeleton):
    sub = navis.subset_neuron(
        skeleton, skeleton.nodes.node_id.values[:500], keep_disc_cn=True
    )

    assert sub.n_connectors == skeleton.n_connectors


def test_keep_disc_cn_reaches_the_link_cascade(mesh):
    """`skip` has to reach the links, not just the refs.

    Only shows up on a *second* subset: the first is what creates the
    `vertex_id` column, and until it exists there is no link to skip.
    """
    once = navis.subset_neuron(mesh, half(mesh))
    assert "vertex_id" in once.connectors.columns

    twice = navis.subset_neuron(once, half(once), keep_disc_cn=True)

    assert twice.n_connectors == once.n_connectors


def test_attaching_a_link_shadows_the_class_declaration(skeleton):
    """Two links with one key would share one slot of the bookkeeping."""
    skeleton.attach_link(
        "nodes", "_connectors.node_id", source="connectors",
        target_axis="nodes", dangling="blank",
    )

    keys = [lk.key for lk in schema.declared_links(skeleton)]
    assert keys.count("connectors->nodes") == 1
    assert schema.get_link(skeleton, "nodes").dangling == "blank"


def test_a_users_link_cannot_hijack_the_skeleton_link(mesh):
    """`Mesh.skeleton` must not be at the mercy of what a user attaches."""
    mesh.attach("mito", pd.DataFrame({"vertex_id": [0, 1, 2]}), ids="vertex_id")
    mesh.attach_link(
        "skeleton", "mito.vertex_id", source="mito", target_axis="nodes",
        target="_skeleton", cascade="keep",
    )

    assert mesh.skeleton.n_nodes > 0


def test_detach_removes_an_axis_this_neuron_alone_declared(skeleton):
    skeleton.attach(
        "mito",
        pd.DataFrame({"mito_id": np.arange(5),
                      "node_id": skeleton.nodes.node_id.values[:5]}),
        ids="mito_id",
    )
    skeleton.attach_link(
        "nodes", "mito", column="node_id", source="mito", target_axis="nodes",
        cascade="keep",
    )

    skeleton.detach("mito")

    assert not hasattr(skeleton, "mito")
    assert "mito" not in schema.declared_axes(skeleton)
    assert not [lk for lk in schema.declared_links(skeleton) if lk.source == "mito"]


def test_detach_only_empties_an_axis_the_class_declared(skeleton):
    """A neuron with no connectors is still one that *can* have them."""
    skeleton.detach("_connectors")

    assert skeleton.connectors is None
    assert "connectors" in schema.declared_axes(skeleton)


def test_subsetting_connectors_leaves_the_nodes_alone(skeleton):
    """`cascade="keep"`: connectors depend on nodes, not the other way round."""
    axis = schema.get_axis(skeleton, "connectors")
    keep = np.zeros(skeleton.n_connectors, dtype=bool)
    keep[:10] = True
    n_nodes = skeleton.n_nodes

    schema.apply_selection(skeleton, axis, keep)

    assert skeleton.n_connectors == 10
    assert skeleton.n_nodes == n_nodes


def test_connectors_compose_onto_the_skeleton(mesh):
    """connectors -> vertices -> nodes, without anyone declaring the shortcut."""
    # The vertex hop only exists once something has asked for it
    mesh.connectors["vertex_id"] = mesh.snap(
        mesh.connectors[["x", "y", "z"]].values
    )[0]
    schema.stamp_links(mesh, "connectors")

    nodes = schema.get_mapping(mesh, "connectors", "skeleton")

    assert len(nodes) == mesh.n_connectors
    expected = mesh.skeleton.vertex_map[mesh.connectors.vertex_id.values]
    assert np.array_equal(nodes, expected)


def test_mesh_subset_keeps_connectors_on_live_vertices(mesh):
    sub = navis.subset_neuron(mesh, half(mesh))

    assert sub.n_connectors < mesh.n_connectors
    assert (sub.connectors.vertex_id < len(sub.vertices)).all()
    assert (sub.connectors.vertex_id >= 0).all()


def test_merge_repairs_connectors(skeleton):
    """A merge rebuilds the axis as thoroughly as a selection does."""
    axon = skeleton.nodes.node_id.values[:2000]
    sub = navis.subset_neuron(skeleton, axon, track=True)
    sub = navis.subset_neuron(sub, sub.nodes.node_id.values[:500])

    merged = navis.merge_subset(skeleton, sub)

    assert merged.connectors.node_id.isin(merged.nodes.node_id).all()


# ---------------------------------------------------------------------------
# Data a user brought along
# ---------------------------------------------------------------------------


def test_attach_carries_data_through_a_subset(mesh):
    mesh.attach("compartment", np.arange(len(mesh.vertices)), axis="vertices")

    keep = half(mesh)
    sub = navis.subset_neuron(mesh, keep)

    assert len(sub.compartment) == len(sub.vertices)
    # The values are the original indices, so they say where they came from
    assert set(sub.compartment) <= set(keep)


def test_attach_carries_data_through_a_mask(skeleton):
    skeleton.attach("score", np.arange(skeleton.n_nodes), axis="nodes")

    with navis.masked(skeleton, skeleton.nodes.node_id.values[:100]):
        assert len(skeleton.score) == skeleton.n_nodes

    assert len(skeleton.score) == skeleton.n_nodes


def test_attach_follows_connectors_out(skeleton):
    """Per-connector data has to go when the connectors do."""
    skeleton.attach("weight", np.arange(skeleton.n_connectors), axis="connectors")

    sub = navis.subset_neuron(skeleton, skeleton.nodes.node_id.values[:500])

    assert len(sub.weight) == sub.n_connectors


def test_attach_rejects_the_wrong_length(mesh):
    with pytest.raises(ValueError, match="entries but axis"):
        mesh.attach("compartment", np.arange(5), axis="vertices")


def test_attach_a_new_axis(skeleton):
    """Data that brings its own elements becomes an axis in its own right."""
    table = pd.DataFrame(
        {"mito_id": np.arange(20), "node_id": skeleton.nodes.node_id.values[:20]}
    )
    skeleton.attach("mito", table, ids="mito_id")
    skeleton.attach_link(
        "nodes", "mito", column="node_id", source="mito", target_axis="nodes",
        cascade="keep",
    )

    sub = navis.subset_neuron(skeleton, skeleton.nodes.node_id.values[:10])

    assert len(sub.mito) == 10
    assert sub.mito.node_id.isin(sub.nodes.node_id).all()


def test_attach_link_over_a_standalone_array(skeleton):
    """The mapping need not be a column - an array aligned to the axis will do."""
    skeleton.attach("mito", pd.DataFrame({"mito_id": np.arange(20)}), ids="mito_id")
    skeleton.attach("on_node", skeleton.nodes.node_id.values[:20], axis="mito")
    skeleton.attach_link(
        "nodes", "on_node", source="mito", target_axis="nodes", cascade="keep"
    )

    assert np.array_equal(
        schema.get_mapping(skeleton, "mito", "nodes"),
        skeleton.nodes.node_id.values[:20],
    )


def test_detach(mesh):
    mesh.attach("compartment", np.arange(len(mesh.vertices)), axis="vertices")
    mesh.detach("compartment")

    assert not hasattr(mesh, "compartment")
    assert "compartment" not in schema.get_axis(mesh, "vertices").data


def test_attached_declarations_do_not_leak_between_copies(mesh):
    mesh.attach("compartment", np.arange(len(mesh.vertices)), axis="vertices")

    other = navis.example_neurons(1, kind="mesh")

    assert "compartment" not in schema.get_axis(other, "vertices").data
    assert "compartment" not in navis.Mesh.AXES["vertices"].data


def test_connectors_setter_goes_through_attach(skeleton):
    """The setter is meant to be a thin wrapper, not a second implementation."""
    skeleton.connectors = skeleton.connectors.iloc[:100].copy()

    assert skeleton.n_connectors == 100
    assert schema.axis_length(skeleton, schema.get_axis(skeleton, "connectors")) == 100
    # ...and the link to the nodes is still good, so it composes
    assert len(schema.get_mapping(skeleton, "connectors", "nodes")) == 100


def test_replacing_an_axis_orphans_what_described_its_elements(skeleton):
    """Assigning is not selecting: nothing can say where the old elements went."""
    skeleton.attach("weight", np.arange(skeleton.n_connectors), axis="connectors")

    skeleton.connectors = skeleton.connectors.iloc[:100].copy()

    assert not hasattr(skeleton, "weight")
    assert "weight" not in schema.get_axis(skeleton, "connectors").data


def _mito(skeleton, n=20):
    return pd.DataFrame({"mito_id": np.arange(n)})


def test_replacing_an_id_axis_with_a_subset_of_itself_carries_the_data(skeleton):
    """An id-bearing axis says exactly which elements survived, so use that."""
    skeleton.attach("mito", _mito(skeleton), ids="mito_id")
    skeleton.attach("mito_volume", np.arange(20) * 2.0, axis="mito")

    skeleton.attach("mito", _mito(skeleton, 20).iloc[:5], ids="mito_id")

    assert np.array_equal(skeleton.mito_volume, np.arange(5) * 2.0)


def test_replacing_an_id_axis_with_different_elements_of_the_same_count(skeleton):
    """Same length is not the same elements - the IDs are what say so."""
    skeleton.attach("mito", _mito(skeleton), ids="mito_id")
    skeleton.attach("mito_volume", np.arange(20) * 2.0, axis="mito")

    replacement = _mito(skeleton)
    replacement["mito_id"] += 10**9
    skeleton.attach("mito", replacement, ids="mito_id")

    assert not hasattr(skeleton, "mito_volume")


def test_replacing_an_axis_in_place_keeps_what_described_it(skeleton):
    """Same elements, so there is nothing to orphan."""
    skeleton.attach("weight", np.arange(skeleton.n_connectors), axis="connectors")

    skeleton.connectors = skeleton.connectors.copy()

    assert len(skeleton.weight) == skeleton.n_connectors


def test_attach_refuses_to_shadow_a_computed_property(skeleton):
    with pytest.raises(AttributeError, match="already"):
        skeleton.attach("cable_length", np.arange(skeleton.n_nodes), axis="nodes")


def test_attach_refuses_to_shadow_a_writable_property(skeleton):
    """A settable property has its own validation and its own store."""
    with pytest.raises(AttributeError, match="already"):
        skeleton.attach("label", np.arange(skeleton.n_nodes), axis="nodes")


@pytest.mark.parametrize("name", ["plot3d", "attach", "copy", "AXES"])
def test_attach_refuses_to_shadow_anything_the_class_defines(skeleton, name):
    """Attaching sets an instance attribute, which shadows a method too."""
    with pytest.raises(AttributeError, match="already"):
        skeleton.attach(name, np.arange(skeleton.n_nodes), axis="nodes")


def test_attach_refuses_an_axis_that_is_not_there(skeleton):
    """The typo case: declaring it would leave data nothing ever carries."""
    with pytest.raises(KeyError, match='no "node" axis'):
        skeleton.attach("score", np.arange(skeleton.n_nodes), axis="node")

    assert "node" not in schema.declared_axes(skeleton)
    assert not hasattr(skeleton, "score")


def test_attach_still_makes_an_axis_for_data_with_its_own_elements(skeleton):
    """The one case that legitimately declares one - spelled by omitting `axis`."""
    mito = pd.DataFrame({"mito_id": np.arange(5), "volume": np.arange(5.0)})

    skeleton.attach("mito", mito, ids="mito_id")

    assert "mito" in schema.declared_axes(skeleton)


def test_shadowing_a_class_link_with_other_values_warns(skeleton, caplog):
    """Silent otherwise, and it stops the built-in being maintained at all."""
    partners = skeleton.nodes.node_id.values[: skeleton.n_connectors]
    skeleton.attach("partner_node", partners, axis="connectors")

    skeleton.attach_link(
        "nodes", "partner_node", source="connectors", target_axis="nodes",
        cascade="keep", dangling="blank",
    )

    assert 'Link "connectors->nodes" replaces' in caplog.text


def test_shadowing_a_class_link_to_change_its_policy_is_silent(skeleton, caplog):
    """Same values, different policy - the documented way to override one."""
    skeleton.attach_link(
        "nodes", "_connectors", column="node_id", source="connectors",
        target_axis="nodes", cascade="keep", dangling="blank",
    )

    assert "replaces" not in caplog.text
    assert schema.get_link(skeleton, "nodes").dangling == "blank"


def test_get_mapping_tells_a_missing_mapping_from_a_stale_one(mesh):
    """"Build it" and "rebuild it" are different instructions."""
    # Nothing has said which vertex each connector sits on
    with pytest.raises(schema.MappingError, match='no "connectors->vertices"'):
        schema.get_mapping(mesh, "connectors", "vertices")

    mesh.connectors["vertex_id"] = mesh.snap(
        mesh.connectors[["x", "y", "z"]].values
    )[0]
    schema.stamp_links(mesh, "connectors")
    assert len(schema.get_mapping(mesh, "connectors", "vertices"))

    # ... and now it is there, but describes vertices that have moved
    mesh.vertices = mesh.vertices * 2
    with pytest.raises(schema.MappingError, match="out of date"):
        schema.get_mapping(mesh, "connectors", "vertices")


def test_the_backwards_error_names_the_call_that_works(mesh):
    """Swapping the arguments lands you back here, so spell the call out."""
    with pytest.raises(KeyError) as excinfo:
        schema.get_mapping(mesh, "skeleton", "vertices")

    assert 'neuron.select_across("vertices", "skeleton", selection)' in str(
        excinfo.value
    )


def test_a_second_link_out_of_an_axis_survives_the_first_ones_drop(skeleton):
    """`dangling="drop"` re-selects the source axis under the other links' feet.

    The connector table's own `node_id` link drops connectors whose node went;
    a user's second link out of the same connectors has a snapshot that predates
    that, and must not write it back.
    """
    rng = np.random.default_rng(0)
    # A reference that is *not* the node the connector sits on, so that which
    # connectors survive says nothing about which of these values do
    partners = rng.choice(skeleton.nodes.node_id.values, skeleton.n_connectors)
    skeleton.attach("partner_node", partners, axis="connectors")
    skeleton.attach_link(
        "partner", "partner_node", source="connectors", target_axis="nodes",
        cascade="keep", dangling="blank",
    )

    sub = navis.subset_neuron(skeleton, skeleton.nodes.node_id.values[:500])

    assert len(sub.partner_node) == sub.n_connectors
    live = sub.partner_node[sub.partner_node != schema.DROPPED]
    assert len(live)                                   # not vacuously true
    assert np.isin(live, sub.nodes.node_id.values).all()


# ---------------------------------------------------------------------------
# Seeing what is attached, and paying for it
# ---------------------------------------------------------------------------


def test_attached_is_empty_for_a_neuron_nobody_touched(skeleton, mesh):
    """A class' own axes are a property of the type, not news about a neuron."""
    for n in (skeleton, mesh):
        assert n.attached().empty
        assert list(n.attached().columns) == ["name", "kind", "axis", "names", "shape"]


def test_attached_reports_each_kind(skeleton):
    mito = pd.DataFrame({"mito_id": np.arange(30), "volume": np.arange(30.0)})
    skeleton.attach("embedding", np.zeros((skeleton.n_nodes, 8)), axis="nodes")
    skeleton.attach("mito", mito, ids="mito_id")
    skeleton.attach("mito_of_node", np.zeros(skeleton.n_nodes, dtype=int), axis="nodes")
    skeleton.attach_link(
        "mito", "mito_of_node", source="nodes", target_axis="mito", dangling="blank"
    )

    rows = skeleton.attached().set_index("name")

    assert rows.loc["embedding", "kind"] == "aligned"
    assert rows.loc["embedding", "axis"] == "nodes"
    assert rows.loc["embedding", "shape"] == (skeleton.n_nodes, 8)
    # The table brought elements of its own, so it *is* an axis
    assert rows.loc["mito", "kind"] == "axis"
    assert rows.loc["mito", "shape"] == (30, 2)
    assert rows.loc["nodes->mito", "kind"] == "link"
    assert rows.loc["nodes->mito", "names"] == "mito"


def test_attached_says_when_a_link_has_nothing_to_read(mesh):
    """The mesh's connectors have no `vertex_id` until something works it out."""
    rows = mesh.attached()
    assert rows.empty                       # built-in links are not attachments

    mesh.attach("depth", np.arange(mesh.n_vertices), axis="vertices")
    mesh.attach_link(
        "far", "vertex_id", source="connectors", target_axis="vertices",
        cascade="keep", dangling="blank",
    )

    link = mesh.attached().set_index("name").loc["connectors->far"]
    assert link["shape"] is None


def test_attached_follows_a_selection(skeleton):
    skeleton.attach("score", np.arange(skeleton.n_nodes), axis="nodes")

    sub = navis.subset_neuron(skeleton, skeleton.nodes.node_id.values[:100])

    assert sub.attached().set_index("name").loc["score", "shape"] == (100,)


def test_detach_removes_the_row(skeleton):
    skeleton.attach("score", np.arange(skeleton.n_nodes), axis="nodes")
    skeleton.detach("score")

    assert skeleton.attached().empty


def test_neuronlist_attached_counts_neurons(skeleton):
    """Attached data is per neuron, so a list can be ragged - say so."""
    nl = navis.NeuronList([skeleton.copy() for _ in range(3)])
    for n in nl[:2]:
        n.attach("score", np.arange(n.n_nodes), axis="nodes")
    nl[2].attach("conf", np.arange(nl[2].n_connectors), axis="connectors")

    rows = nl.attached().set_index("name")

    assert rows.loc["score", "neurons"] == 2
    assert rows.loc["conf", "neurons"] == 1
    assert "shape" not in rows.columns


def test_neuronlist_attached_is_empty_when_nothing_is(skeleton):
    nl = navis.NeuronList([skeleton.copy() for _ in range(2)])

    assert nl.attached().empty
    assert "neurons" in nl.attached().columns


def test_memory_usage_counts_attached_data(skeleton):
    """The cache is the trap: nothing else invalidates it when you attach."""
    before = skeleton.memory_usage()
    big = np.zeros((skeleton.n_nodes, 100))

    skeleton.attach("big", big, axis="nodes")

    assert skeleton.memory_usage() >= before + big.nbytes
    skeleton.detach("big")
    assert skeleton.memory_usage() == before


def test_memory_usage_counts_a_kept_skeleton(mesh):
    """A mesh keeping its skeleton is the whole point of the link - price it."""
    bare = navis.example_neurons(1, kind="mesh")
    bare._connectors = mesh._connectors

    assert mesh.memory_usage() > bare.memory_usage()
    assert mesh.memory_usage() >= bare.memory_usage() + mesh.skeleton.memory_usage()


def test_memory_usage_counts_the_mask_snapshot(skeleton):
    """A masked neuron holds a whole copy of itself to be restored from."""
    whole = skeleton.memory_usage()

    skeleton.mask(skeleton.nodes.node_id.values[:100], inplace=True, warn_cut=False)

    # Smaller neuron, but the snapshot it will be restored from is still here
    assert skeleton.memory_usage() > whole

    skeleton.unmask()
    assert skeleton.memory_usage() == whole


# ---------------------------------------------------------------------------
# Asking across links, as methods
# ---------------------------------------------------------------------------


def test_get_mapping_method_matches_the_function(mesh):
    assert np.array_equal(
        mesh.get_mapping("vertices", "skeleton"),
        schema.get_mapping(mesh, "vertices", "skeleton"),
    )


def test_select_across_method_matches_the_function(mesh):
    nodes = mesh.skeleton.nodes.node_id.values[:50]

    assert np.array_equal(
        mesh.select_across("vertices", "skeleton", nodes),
        schema.select_across(mesh, "vertices", "skeleton", nodes),
    )


def test_get_mapping_method_raises_the_same_way(mesh):
    """The errors are the useful part, so they must survive the wrapper."""
    with pytest.raises(KeyError) as excinfo:
        mesh.get_mapping("skeleton", "vertices")

    assert 'neuron.select_across("vertices", "skeleton", selection)' in str(
        excinfo.value
    )


# ---------------------------------------------------------------------------
# Rebuilding
#
# A rebuild replaces an axis' elements rather than taking some away, so it has
# to say two separate things: where a *reference* should now point, and which of
# the new elements *is* an old one. The second may never be read off the first.
# ---------------------------------------------------------------------------


def test_a_rebuild_carries_data_that_asked_to_be(skeleton):
    """`downsample_neuron` only thins slabs, so identity is real and claimable."""
    skeleton.attach(
        "score", np.arange(skeleton.n_nodes, dtype=float), axis="nodes",
        on_rebuild="carry",
    )

    ds = navis.downsample_neuron(skeleton, 10)

    assert len(ds.score) == ds.n_nodes
    # ... and each value still describes the node it is sitting on
    was = dict(zip(skeleton.nodes.node_id.values, skeleton.score))
    assert np.array_equal(ds.score, ds.nodes.node_id.map(was).values)


def test_a_rebuild_drops_data_by_default(skeleton, caplog):
    skeleton.attach("score", np.arange(skeleton.n_nodes), axis="nodes")

    ds = navis.downsample_neuron(skeleton, 10)

    assert not hasattr(ds, "score")
    assert "dropped score" in caplog.text


def test_a_rebuild_leaves_the_input_alone(skeleton):
    """The data is moved aside for the call - `inplace=False` must not lose it."""
    skeleton.attach("score", np.arange(skeleton.n_nodes), axis="nodes")

    navis.downsample_neuron(skeleton, 10, inplace=False)

    assert len(skeleton.score) == skeleton.n_nodes


def test_a_rebuild_that_claims_no_identity_drops_even_carry(skeleton, caplog):
    """Resampling re-samples: a reused ID is not thereby the same point."""
    skeleton.attach(
        "score", np.arange(skeleton.n_nodes, dtype=float), axis="nodes",
        on_rebuild="carry",
    )

    rs = navis.resample_skeleton(skeleton, resample_to=1000)

    assert not hasattr(rs, "score")
    assert "did not record where the old elements went" in caplog.text


@pytest.mark.parametrize(
    "fn,kwargs",
    [
        (navis.downsample_neuron, dict(downsampling_factor=10)),
        (navis.resample_skeleton, dict(resample_to=1000)),
    ],
)
def test_a_rebuild_snaps_what_names_its_elements(skeleton, fn, kwargs):
    """Connectors, tags and the soma go through the generic path now."""
    skeleton.tags = {"demo": skeleton.nodes.node_id.values[:200].tolist()}

    res = fn(skeleton, **kwargs)

    live = res.nodes.node_id.values
    assert res.n_connectors == skeleton.n_connectors      # moved, never dropped
    assert np.isin(res.connectors.node_id.values, live).all()
    assert len(res.tags["demo"]) == 200
    assert np.isin(res.tags["demo"], live).all()
    assert np.isin(res.soma, live).all()


def test_an_attached_link_can_ask_to_be_snapped(skeleton):
    """The connector table's `node_id` is not the only thing that names nodes."""
    partners = skeleton.nodes.node_id.values[: skeleton.n_connectors]
    skeleton.attach("partner_node", partners, axis="connectors")
    skeleton.attach_link(
        "partner", "partner_node", source="connectors", target_axis="nodes",
        cascade="keep", dangling="blank", on_rebuild="snap",
    )

    ds = navis.downsample_neuron(skeleton, 10)

    live = ds.partner_node[ds.partner_node != schema.DROPPED]
    assert len(live) == len(ds.partner_node)              # nothing left stranded
    assert np.isin(live, ds.nodes.node_id.values).all()


def test_an_attached_link_is_not_snapped_unless_it_asks(skeleton):
    """The default is the selection answer: the element it named is gone."""
    partners = skeleton.nodes.node_id.values[: skeleton.n_connectors]
    skeleton.attach("partner_node", partners, axis="connectors")
    skeleton.attach_link(
        "partner", "partner_node", source="connectors", target_axis="nodes",
        cascade="keep", dangling="blank",
    )

    ds = navis.downsample_neuron(skeleton, 10)

    assert (ds.partner_node == schema.DROPPED).any()
    live = ds.partner_node[ds.partner_node != schema.DROPPED]
    assert np.isin(live, ds.nodes.node_id.values).all()


def test_rebuilds_insists_on_being_told(skeleton):
    """Saying nothing must not be spelled the same way as forgetting to."""

    @navis.utils.rebuilds("nodes")
    def forgetful(x):
        return x

    with pytest.raises(TypeError, match="must return"):
        forgetful(skeleton)


def test_a_rebuild_with_nothing_to_say_repairs_like_a_selection(skeleton):
    """`Rebuild()` means "I cannot say", not "nothing moved"."""

    @navis.utils.rebuilds("nodes")
    def halve(x):
        x.nodes = x.nodes.iloc[:500].copy()
        return x, schema.Rebuild()

    res = halve(skeleton)

    assert np.isin(res.connectors.node_id.values, res.nodes.node_id.values).all()
    assert res.n_connectors < 2705


# ---------------------------------------------------------------------------
# The setters, which is where a rebuild lands when nobody says anything
# ---------------------------------------------------------------------------


def test_assigning_nodes_drops_attached_data(skeleton, caplog):
    """It replaces the elements; nothing can say where the old ones went."""
    skeleton.attach("score", np.arange(skeleton.n_nodes), axis="nodes")

    fresh = skeleton.nodes.copy()
    fresh["node_id"] = fresh.node_id + 10**9
    skeleton.nodes = fresh

    assert not hasattr(skeleton, "score")
    assert "dropped score" in caplog.text


def test_assigning_a_subset_of_the_nodes_carries_it(skeleton):
    """Same IDs, so this is a selection written as an assignment."""
    skeleton.attach("score", np.arange(skeleton.n_nodes), axis="nodes")

    skeleton.nodes = skeleton.nodes.iloc[:100].copy()

    assert np.array_equal(skeleton.score, np.arange(100))


def test_assigning_points_does_not_orphan_a_dotprops_own_vectors(dotprops):
    """`_vect`/`_alpha` are aligned to the points and are the class' business."""
    dotprops.recalculate_tangents(k=5, inplace=True)

    dotprops.points = dotprops.points[:100]

    assert dotprops._vect is not None


def test_subsetting_a_mesh_still_carries_attached_data(mesh):
    """`_subset_meshneuron` assigns `.vertices` mid-selection - see `replacing`."""
    mesh.attach("depth", np.arange(mesh.n_vertices), axis="vertices")

    sub = navis.subset_neuron(mesh, half(mesh))

    assert len(sub.depth) == sub.n_vertices


def test_downsampling_dotprops_is_a_selection(dotprops):
    """Thinning points takes them away, so everything follows by construction."""
    dotprops.recalculate_tangents(k=5, inplace=True)
    dotprops.attach("score", np.arange(len(dotprops.points)), axis="points")

    ds = navis.downsample_neuron(dotprops, 10)

    assert len(ds.score) == len(ds.points) < len(dotprops.points)
    assert len(ds.vect) == len(ds.points)
    # ... which is what says these are the points they always were
    assert np.array_equal(ds.score, np.arange(0, len(dotprops.points), 10))


def test_simplify_mesh_keeps_connectors_on_the_surface(mesh):
    """Decimation replaces the vertices, so an old index names nothing."""
    cn = mesh.connectors.copy()
    cn["vertex_id"] = mesh.snap(cn[["x", "y", "z"]].values)[0]
    mesh.connectors = cn
    was = mesh.vertices[cn.vertex_id.values]

    simple = navis.simplify_mesh(mesh, F=0.2)

    vid = simple.connectors.vertex_id.values
    assert simple.n_connectors == mesh.n_connectors        # moved, not dropped
    assert (vid < simple.n_vertices).all() and (vid >= 0).all()
    # ... and to somewhere that is still the same part of the surface
    moved = np.linalg.norm(simple.vertices[vid] - was, axis=1)
    assert moved.max() < still_nearby(mesh)


def test_simplify_mesh_carries_per_vertex_data(mesh):
    """Decimation merges vertices, and says which - so values come along.

    `depth` is each vertex's own index, so the values that come through say
    which old vertex each new one took them from.
    """
    mesh.attach(
        "depth", np.arange(mesh.n_vertices, dtype=float), axis="vertices",
        on_rebuild="carry",
    )

    simple = navis.simplify_mesh(mesh, F=0.2)

    assert len(simple.depth) == simple.n_vertices
    # Every value is one an old vertex actually had, rather than an average of
    # the group or an index into the new vertices
    assert set(simple.depth.tolist()) <= set(mesh.depth.tolist())
    # ... and from a vertex that became this one, which - since a merged vertex
    # sits where the ones it swallowed were - means from somewhere nearby
    was = mesh.vertices[simple.depth.astype(int)]
    moved = np.linalg.norm(simple.vertices - was, axis=1)
    assert moved.max() < still_nearby(mesh)


def test_simplify_mesh_still_drops_data_that_did_not_ask_to_be_carried(mesh, caplog):
    """`on_rebuild="carry"` is opt-in, and silence still means drop."""
    mesh.attach("depth", np.arange(mesh.n_vertices, dtype=float), axis="vertices")

    simple = navis.simplify_mesh(mesh, F=0.2)

    assert not hasattr(simple, "depth")
    assert "dropped depth" in caplog.text


def test_simplify_mesh_keeps_the_skeleton_correspondence(mesh, monkeypatch):
    """The arbour is where it was, so the vertex map follows the vertices."""
    before = mesh.skeleton.nodes.node_id.values

    def boom(*args, **kwargs):
        raise AssertionError("skeletonize() was called - the link was not carried")

    monkeypatch.setattr(navis.Mesh, "skeletonize", boom)
    simple = navis.simplify_mesh(mesh, F=0.2)

    # Not re-skeletonized, and not thinned out either - the rebuild happened on
    # the mesh, and a skeleton is entitled to nodes no vertex speaks for
    assert np.array_equal(simple.skeleton.nodes.node_id.values, before)
    vmap = schema.get_mapping(simple, "vertices", "skeleton")
    assert len(vmap) == simple.n_vertices
    assert np.isin(vmap[vmap >= 0], before).all()


def test_simplify_mesh_puts_each_vertex_on_the_node_it_belonged_to(mesh):
    """A merged vertex takes the node of a vertex that became it."""
    skel = mesh.skeleton
    before = schema.get_mapping(mesh, "vertices", "skeleton")

    simple = navis.simplify_mesh(mesh, F=0.2)

    after = schema.get_mapping(simple, "vertices", "skeleton")
    ok = after >= 0
    # The node a new vertex names sits about where the new vertex does - which
    # it would not if the map had simply been left at the old indices
    pos = skel.nodes.set_index("node_id").loc[after[ok], ["x", "y", "z"]].values
    dist = np.linalg.norm(simple.vertices[ok] - pos, axis=1)
    was = np.linalg.norm(
        mesh.vertices[before >= 0]
        - skel.nodes.set_index("node_id")
        .loc[before[before >= 0], ["x", "y", "z"]].values,
        axis=1,
    )
    assert np.median(dist) < np.median(was) * 2


def test_simplify_mesh_leaves_the_faces_alone(mesh):
    """A face *is* three vertex indices, so the backend already rebuilt them."""
    simple = navis.simplify_mesh(mesh, F=0.2)

    assert len(simple.faces)
    assert simple.faces.max() < simple.n_vertices


def test_simplify_mesh_keeps_extra_edges_on_the_surface(mesh):
    """A bridge names a place on the surface, so it follows its endpoints.

    Note the `.vertices` setter drops extra edges outright when the count
    changes, which decimation always does - so this only works because
    `simplify_mesh` takes them out of its way first.
    """
    n = mesh.n_vertices
    mesh.extra_edges = [[0, n // 2], [1, n - 1]]
    was = mesh.vertices[mesh.extra_edges].reshape(-1, 3)

    simple = navis.simplify_mesh(mesh, F=0.2)

    assert simple.n_extra_edges == 2                    # moved, not dropped
    assert (simple.extra_edges >= 0).all()
    assert (simple.extra_edges < simple.n_vertices).all()
    # ... and every end is still on the part of the surface it was on
    got = simple.vertices[simple.extra_edges].reshape(-1, 3)
    nearest = np.linalg.norm(got[:, None, :] - was[None, :, :], axis=-1).min(axis=1)
    assert nearest.max() < still_nearby(mesh)


def test_a_positional_rebuild_drops_what_it_did_not_move(mesh):
    """An index is not a name: vertex 3 of the rebuilt mesh is not the old one.

    The count is deliberately unchanged, so nothing along the way drops anything
    and the reference is left having to justify itself. A bounds check would say
    yes to every one of them.
    """
    mesh.extra_edges = [[0, 1], [2, 3]]

    @navis.utils.rebuilds("vertices")
    def reverse(x):
        # Same number of vertices, different vertices - and the rebuild says it
        # cannot tell us where any of them went
        x.vertices = x.vertices[::-1].copy()
        return x, schema.Rebuild()

    res = reverse(mesh.copy())

    assert res.n_vertices == mesh.n_vertices
    assert res.n_extra_edges == 0


def test_a_positional_rebuild_keeps_what_it_moved(mesh):
    """The other half: `snap` may be partial, and what is in it survives."""
    mesh.extra_edges = [[0, 1], [2, 3]]
    n = mesh.n_vertices

    @navis.utils.rebuilds("vertices")
    def reverse(x):
        x.vertices = x.vertices[::-1].copy()
        # Only the first edge's ends are accounted for
        return x, schema.Rebuild(snap=([0, 1], [n - 1, n - 2]))

    res = reverse(mesh.copy())

    assert res.extra_edges.tolist() == [[n - 2, n - 1]]


def _fold_in_pairs(n):
    """A rebuild that merges each pair of vertices: 0 and 1 become 0, and so on."""

    @navis.utils.rebuilds("vertices")
    def fold(x, merged=None):
        # Same dance as `simplify_mesh`: the setter drops extra edges outright
        # when the count changes, so they go out of its way and come back
        # un-remapped for the rebuild to repair
        edges, x._extra_edges = getattr(x, "_extra_edges", None), None
        x.vertices = x.vertices[::2].copy()
        x._extra_edges = edges
        return x, schema.Rebuild(
            merged=np.arange(n) // 2 if merged is None else merged
        )

    return fold


def test_a_merging_rebuild_says_where_everything_went(mesh):
    """`merged` is complete, so it is `snap` as well - one answer, not two."""
    n = mesh.n_vertices
    mesh.extra_edges = [[0, 2], [4, 6]]

    res = _fold_in_pairs(n)(mesh.copy())

    # Both edges survive, each end following the vertex it was folded into
    assert res.extra_edges.tolist() == [[0, 1], [2, 3]]


def test_a_merging_rebuild_carries_from_the_first_of_the_group(mesh):
    """Several old elements, one new one - so one of them has to speak."""
    n = mesh.n_vertices
    mesh.attach("lab", np.arange(n), axis="vertices", on_rebuild="carry")

    res = _fold_in_pairs(n)(mesh.copy())

    # Vertices 0 and 1 became vertex 0, and it is 0's value that comes through
    assert res.lab.tolist() == np.arange(0, n, 2).tolist()


def test_a_merging_rebuild_that_leaves_a_new_element_unaccounted_for(mesh, caplog):
    """Carrying needs every new element to have come from somewhere."""
    n = mesh.n_vertices
    mesh.attach("lab", np.arange(n), axis="vertices", on_rebuild="carry")
    merged = np.arange(n) // 2
    merged[merged == 3] = schema.DROPPED       # nothing became vertex 3

    res = _fold_in_pairs(n)(mesh.copy(), merged=merged)

    assert not hasattr(res, "lab")
    assert "have no old element behind them" in caplog.text


def test_simplify_mesh_leaves_the_input_alone(mesh):
    """`inplace=False` must not write the new vertex map onto the original."""
    before = mesh.skeleton.vertex_map.copy()

    simple = navis.simplify_mesh(mesh, F=0.2)

    assert np.array_equal(mesh.skeleton.vertex_map, before)
    assert len(simple.skeleton.vertex_map) == simple.n_vertices


def test_simplify_mesh_still_takes_a_volume():
    """`Volume` and bare trimeshes have no schema to carry."""
    m = navis.example_neurons(1, kind="mesh")
    vol = navis.Volume(m.vertices, m.faces)

    simple = navis.simplify_mesh(vol, F=0.2)

    assert isinstance(simple, navis.Volume)
    assert len(simple.vertices) < len(m.vertices)


def test_simplify_mesh_rejects_a_stale_backend(mesh):
    """The argument is gone; passing it should say so rather than be ignored."""
    with pytest.warns(DeprecationWarning, match="navis-fastcore"):
        navis.simplify_mesh(mesh, F=0.2, backend="pyfqmr")


def test_insert_nodes_does_not_leave_attached_data_behind(skeleton):
    """Inserting nodes adds elements, so aligned data cannot simply stay put."""
    skeleton.attach("lab", np.arange(skeleton.n_nodes), axis="nodes")
    edges = skeleton.nodes[skeleton.nodes.parent_id >= 0]
    where = list(zip(edges.node_id.values[:3], edges.parent_id.values[:3]))

    grown = navis.insert_nodes(skeleton, where=where)

    assert grown.n_nodes == skeleton.n_nodes + 3
    # Dropped rather than left at the old length, describing nodes that moved
    # under it - there is no label to give a node that was not there before.
    assert not hasattr(grown, "lab")
    # ... and the neuron it was called on keeps what it always had
    assert len(skeleton.lab) == skeleton.n_nodes


def test_insert_nodes_keeps_every_reference(skeleton):
    """No old node went anywhere, so nothing that named one should be dropped."""
    edges = skeleton.nodes[skeleton.nodes.parent_id >= 0]
    where = list(zip(edges.node_id.values[:3], edges.parent_id.values[:3]))

    grown = navis.insert_nodes(skeleton, where=where)

    assert len(grown.connectors) == len(skeleton.connectors)
    assert grown.connectors.node_id.isin(grown.nodes.node_id).all()


def test_insert_nodes_cannot_carry_what_it_did_not_make(skeleton):
    """`on_rebuild="carry"` still needs every new element to be an old one."""
    skeleton.attach(
        "lab", np.arange(skeleton.n_nodes), axis="nodes", on_rebuild="carry"
    )
    edges = skeleton.nodes[skeleton.nodes.parent_id >= 0]
    where = list(zip(edges.node_id.values[:3], edges.parent_id.values[:3]))

    grown = navis.insert_nodes(skeleton, where=where)

    assert not hasattr(grown, "lab")


def test_merge_duplicate_nodes_carries_attached_data(skeleton):
    """Folding a duplicate into its twin takes elements away - a selection."""
    skeleton.nodes.loc[1, ["x", "y", "z"]] = skeleton.nodes.loc[
        0, ["x", "y", "z"]
    ].values
    skeleton.attach("lab", np.arange(skeleton.n_nodes), axis="nodes")

    fixed = navis.graph.clinic.merge_duplicate_nodes(skeleton)

    assert fixed.n_nodes == skeleton.n_nodes - 1
    assert len(fixed.lab) == fixed.n_nodes


def test_merge_duplicate_nodes_leaves_no_dangling_connector(skeleton):
    """A connector on the collapsed duplicate has to go somewhere - not nowhere."""
    skeleton.nodes.loc[1, ["x", "y", "z"]] = skeleton.nodes.loc[
        0, ["x", "y", "z"]
    ].values
    collapsed = int(skeleton.nodes.node_id.iloc[1])
    skeleton.connectors.loc[skeleton.connectors.index[0], "node_id"] = collapsed

    fixed = navis.graph.clinic.merge_duplicate_nodes(skeleton)

    assert fixed.connectors.node_id.isin(fixed.nodes.node_id).all()


def test_merge_duplicate_nodes_moves_what_sat_on_the_duplicate(skeleton):
    """"Somewhere" is not good enough: the folded node is a place, not a loss.

    Nothing about the neuron went away - the duplicate is the very same point in
    space as the node it was folded into - so dropping a connector, tag or soma
    that sat on it would be deleting data to tidy up a table.
    """
    skeleton.nodes.loc[1, ["x", "y", "z"]] = skeleton.nodes.loc[
        0, ["x", "y", "z"]
    ].values
    collapsed = int(skeleton.nodes.node_id.iloc[1])
    was = skeleton.nodes.loc[1, ["x", "y", "z"]].values.astype(float)

    skeleton.connectors.loc[skeleton.connectors.index[0], "node_id"] = collapsed
    skeleton.tags = {"here": [collapsed]}
    skeleton.soma = collapsed
    n_nodes, n_connectors = skeleton.n_nodes, skeleton.n_connectors

    fixed = navis.graph.clinic.merge_duplicate_nodes(skeleton)

    assert fixed.n_nodes == n_nodes - 1            # one of the two really was folded
    assert fixed.n_connectors == n_connectors      # moved, not dropped
    assert len(fixed.tags["here"]) == 1
    assert fixed.soma is not None

    # ... and all three onto a node standing exactly where they were
    at = fixed.nodes.set_index("node_id")[["x", "y", "z"]]
    for node in (fixed.connectors.node_id.iloc[0], fixed.tags["here"][0],
                 int(fixed.soma)):
        assert np.allclose(at.loc[node].values.astype(float), was)


#: Modules outside `navis.core` that write an axis' primary attribute directly,
#: and why each is allowed to. Everything else has to go through the setter (so
#: `_replacing` gets a say) or through the schema.
_DIRECT_AXIS_WRITES = {
    # Inside `@rebuilds`, which stands `_replacing` down and puts the data back
    "navis/graph/graph_utils.py": 1,
    # Same elements in the same order, only the dtype changes
    "navis/interfaces/neuprint.py": 1,
    # Building a neuron from scratch; there is nothing attached yet
    "navis/io/json_io.py": 2,
    # `stitch_skeletons`/`combine_neurons`, which call `_replacing` first
    "navis/morpho/manipulation.py": 3,
}


def test_no_new_direct_writes_to_an_axis():
    """Assigning to `_nodes`/`_vertices`/`_points` skips everything.

    The data attached to the old elements is then left exactly where it was - at
    the old length, describing elements that are gone, and indexing cleanly
    enough that nothing ever complains. That is silent, so a new site should be
    a failing test rather than something noticed a release later.
    """
    import pathlib
    import re

    pattern = re.compile(r"^\s*[A-Za-z_][\w.]*\._(?:nodes|vertices|points)\s*=(?!=)")
    root = pathlib.Path(navis.__file__).parent

    found = {}
    for path in sorted(root.rglob("*.py")):
        if path.relative_to(root).parts[0] == "core":
            continue
        hits = sum(bool(pattern.match(ln)) for ln in path.read_text().splitlines())
        if hits:
            found[f"navis/{path.relative_to(root)}"] = hits

    assert found == _DIRECT_AXIS_WRITES, (
        "Direct writes to an axis' elements changed. If you added one, route it "
        "through the public setter or `schema.apply_selection`/`@utils.rebuilds` "
        "instead - or add it to `_DIRECT_AXIS_WRITES` with a reason."
    )
