"""Tests for the neuron methods that wrap a top-level function.

These wrappers are thin by design, so what is worth pinning down is not what
they compute - the base function's own tests cover that - but that they exist
on the right classes, agree with the function they wrap, and honour `inplace`.

The last two are what actually went wrong before: `drop_fluff` was a method on
`Dotprops` alone despite the function taking all four neuron types, and half the
morphology functions had no method at all.
"""

import numpy as np
import pytest

import navis


@pytest.fixture(scope="module")
def skeleton():
    return navis.example_neurons(1, kind="skeleton")


@pytest.fixture(scope="module")
def mesh():
    return navis.example_neurons(1, kind="mesh")


@pytest.fixture(scope="module")
def dotprops(skeleton):
    return navis.make_dotprops(skeleton, k=5)


@pytest.fixture(scope="module")
def voxels(mesh):
    return navis.voxelize(mesh, pitch="1 micron")


@pytest.fixture
def fragmented(skeleton):
    """A skeleton in three pieces, by orphaning two nodes."""
    n = skeleton.copy()
    ids = n.nodes.node_id.values
    n.nodes.loc[n.nodes.node_id.isin([ids[500], ids[1500]]), "parent_id"] = -1
    n._clear_temp_attr()
    return n


# ---------------------------------------------------------------------------
# Presence: the method exists wherever the function applies
# ---------------------------------------------------------------------------

#: (function, method, class it must live on)
WRAPPERS = [
    ("subset_neuron", "subset", "Skeleton"),
    ("subset_neuron", "subset", "Mesh"),
    ("subset_neuron", "subset", "Dotprops"),
    ("drop_fluff", "drop_fluff", "Skeleton"),
    ("drop_fluff", "drop_fluff", "Mesh"),
    ("drop_fluff", "drop_fluff", "Dotprops"),
    ("drop_fluff", "drop_fluff", "Voxels"),
    ("split_axon_dendrite", "split_axon_dendrite", "Skeleton"),
    ("heal_skeleton", "heal", "Skeleton"),
    ("heal_mesh", "heal", "Mesh"),
    ("smooth_skeleton", "smooth", "Skeleton"),
    ("smooth_mesh", "smooth", "Mesh"),
    ("smooth_voxels", "smooth", "Voxels"),
    ("despike_skeleton", "despike", "Skeleton"),
    ("cut_skeleton", "cut", "Skeleton"),
    ("connected_components", "connected_components", "Skeleton"),
]


@pytest.mark.parametrize("func,method,cls_name", WRAPPERS)
def test_method_exists(func, method, cls_name):
    assert hasattr(navis, func), f"navis.{func} is gone"
    assert callable(
        getattr(getattr(navis, cls_name), method, None)
    ), f"{cls_name}.{method}() is missing"


@pytest.mark.parametrize(
    "method", ["drop_fluff", "subset", "connected_components", "split_axon_dendrite"]
)
def test_shared_methods_are_on_the_base_class(method):
    """Not re-implemented per type - that is how `drop_fluff` drifted before."""
    assert hasattr(navis.BaseNeuron, method)
    for cls in (navis.Skeleton, navis.Mesh, navis.Dotprops, navis.Voxels):
        assert getattr(cls, method) is getattr(navis.BaseNeuron, method)


def test_methods_drop_the_type_suffix():
    """`heal_skeleton` -> `.heal()`, not `.heal_skeleton()`."""
    for cls, bad in [
        (navis.Skeleton, ["heal_skeleton", "smooth_skeleton", "despike_skeleton",
                          "cut_skeleton", "subset_neuron"]),
        (navis.Mesh, ["heal_mesh", "smooth_mesh"]),
        (navis.Voxels, ["smooth_voxels"]),
    ]:
        for name in bad:
            assert not hasattr(cls, name), f"{cls.__name__}.{name} kept the suffix"


# ---------------------------------------------------------------------------
# The method agrees with the function it wraps
# ---------------------------------------------------------------------------


def test_drop_fluff_matches_function(mesh):
    assert (
        mesh.drop_fluff(min_size=100).n_vertices
        == navis.drop_fluff(mesh, min_size=100).n_vertices
    )


def test_drop_fluff_on_every_type(skeleton, mesh, dotprops, voxels):
    """The bug this closes: only Dotprops used to have the method."""
    assert mesh.drop_fluff().n_vertices < mesh.n_vertices
    assert dotprops.drop_fluff(epsilon=2000).n_points <= dotprops.n_points
    assert voxels.drop_fluff().nnz <= voxels.nnz
    assert skeleton.drop_fluff().n_nodes <= skeleton.n_nodes


def test_subset_matches_function(skeleton):
    keep = skeleton.nodes.node_id.values[:100]

    assert np.array_equal(
        skeleton.subset(keep).nodes.node_id.values,
        navis.subset_neuron(skeleton, keep).nodes.node_id.values,
    )


def test_heal_matches_function(fragmented, mesh):
    assert fragmented.heal().n_components == 1
    assert mesh.heal().n_components == navis.heal_mesh(mesh).n_components


def test_smooth_matches_function(skeleton, mesh):
    assert skeleton.smooth(window=5).cable_length == pytest.approx(
        navis.smooth_skeleton(skeleton, window=5).cable_length
    )
    assert mesh.smooth(iterations=2).n_vertices == mesh.n_vertices


def test_cut_matches_function(skeleton):
    where = skeleton.nodes.node_id.values[500]

    assert len(skeleton.cut(where)) == len(navis.cut_skeleton(skeleton, where))
    assert len(skeleton.cut(where, ret="distal")) == 1


def test_split_axon_dendrite_matches_function(skeleton):
    assert len(skeleton.split_axon_dendrite()) == len(
        navis.split_axon_dendrite(skeleton)
    )


def test_kwargs_reach_the_base_function(mesh, skeleton):
    """The wrappers are `**kwargs` passthroughs - a typo must not pass silently."""
    assert mesh.drop_fluff(connectivity="face").n_vertices == (
        navis.drop_fluff(mesh, connectivity="face").n_vertices
    )
    with pytest.raises(TypeError):
        skeleton.despike(definitely_not_a_parameter=1)


# ---------------------------------------------------------------------------
# `inplace`
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("drop_fluff", {}),
        ("subset", {}),
        ("heal", {}),
        ("smooth", {"window": 3}),
        ("despike", {}),
    ],
)
def test_inplace_returns_none_and_mutates(fragmented, method, kwargs):
    if method == "subset":
        kwargs = {"subset": fragmented.nodes.node_id.values[:200]}

    before = fragmented.n_nodes
    out = getattr(fragmented, method)(inplace=True, **kwargs)

    assert out is None, f"{method}(inplace=True) should return None"
    if method == "subset":
        assert fragmented.n_nodes == 200 != before


@pytest.mark.parametrize(
    "kind,method,kwargs",
    [
        ("skeleton", "heal", {}),
        ("mesh", "heal", {}),
        ("skeleton", "smooth", {"window": 3}),
        ("mesh", "smooth", {"iterations": 1}),
        ("voxels", "smooth", {"sigma": 1}),
        ("mesh", "drop_fluff", {}),
    ],
)
def test_same_method_name_same_inplace_contract(kind, method, kwargs, request):
    """`Mesh.heal(inplace=True)` used to hand back the neuron where
    `Skeleton.heal(inplace=True)` returned None - one name, two contracts."""
    x = request.getfixturevalue(kind).copy()

    assert getattr(x, method)(inplace=True, **kwargs) is None


@pytest.mark.parametrize("method", ["drop_fluff", "heal", "despike"])
def test_not_inplace_leaves_original_alone(fragmented, method):
    before = fragmented.n_nodes

    out = getattr(fragmented, method)()

    assert out is not fragmented
    assert fragmented.n_nodes == before


# ---------------------------------------------------------------------------
# Parameter naming conventions
# ---------------------------------------------------------------------------
#
# These are API-shape tests rather than behaviour tests: they exist so that a
# new function cannot quietly reintroduce a fifth spelling of "cap the distance
# here", or use `min_size` for something measured in nanometres.

import inspect


def _public_functions():
    return {
        n: getattr(navis, n)
        for n in dir(navis)
        if not n.startswith("_") and inspect.isfunction(getattr(navis, n))
    }


#: `limit` survives only where it never meant a distance.
LIMIT_IS_NOT_A_DISTANCE = {
    "guess_radius",  # count of consecutive missing radii
    "read_h5", "read_json", "read_mesh", "read_nml", "read_nmx", "read_nrrd",
    "read_parquet", "read_precomputed", "read_rda", "read_swc", "read_tiff",
}


@pytest.mark.parametrize("banned", ["limit_dist", "dist"])
def test_no_other_spelling_of_max_dist(banned):
    offenders = [
        n for n, f in _public_functions().items()
        if banned in inspect.signature(f).parameters
    ]
    assert not offenders, f"`{banned}` should be `max_dist`: {offenders}"


def test_limit_only_survives_where_it_is_not_a_distance():
    offenders = [
        n for n, f in _public_functions().items()
        if "limit" in inspect.signature(f).parameters
        and n not in LIMIT_IS_NOT_A_DISTANCE
    ]
    assert not offenders, f"`limit` used for a distance: {offenders}"


def test_size_is_a_count_and_length_is_a_distance():
    """`min_size` counts elements; anything measured in space is `min_length`."""
    fns = _public_functions()

    # Nothing should take a bare `size` any more
    assert not [n for n, f in fns.items() if "size" in inspect.signature(f).parameters]

    # A `min_size` must not advertise unit strings - that would make it a length
    for n, f in fns.items():
        if "min_size" not in inspect.signature(f).parameters:
            continue
        doc = (f.__doc__ or "")
        block = doc.split("min_size", 1)[1][:400] if "min_size" in doc else ""
        assert "micron" not in block, f"{n}.min_size looks like a length"


@pytest.mark.parametrize(
    "func,kwargs",
    [
        ("prune_twigs", {"min_length": "5 microns"}),
        ("geodesic_matrix", {"max_dist": "10 microns"}),
        ("split_neurites", {"n": 3, "min_length": "10 microns"}),
    ],
)
def test_unit_strings_are_accepted(skeleton, func, kwargs):
    """A distance/length argument reads `"5 microns"` off the neuron's units."""
    out = getattr(navis, func)(skeleton, **kwargs)

    assert out is not None


def test_unit_string_matches_the_explicit_number(skeleton):
    """A unit string resolves to the neuron's own coordinate space.

    N.B. the example neuron is in 8 nm voxels, so `"5 microns"` is 625 - not
    5000. That is exactly why going through `map_units` matters.
    """
    in_units = float(skeleton.map_units("5 microns"))
    assert in_units == 625

    assert (
        navis.prune_twigs(skeleton, min_length="5 microns").n_nodes
        == navis.prune_twigs(skeleton, min_length=in_units).n_nodes
    )
