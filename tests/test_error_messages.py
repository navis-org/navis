"""Tests for error messages that tell the caller how to fix the problem.

These pin *repair signals*: each message names the way out, not just the
symptom. Three of them exist because the underlying failure used to surface as
a raw third-party exception (pint, pandas, scipy) that mentioned neither navis
nor the parameter at fault - so the assertions here are as much about the error
no longer escaping as about its wording.
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
def voxels(skeleton):
    return navis.voxelize(skeleton, pitch="1 micron")


def test_convert_units_without_units(skeleton):
    """`units = None` is dimensionless, not absent - it used to slip through.

    The isinstance guard passed and pint raised DimensionalityError from inside
    `.to()`, which never mentioned `.units`.
    """
    n = skeleton.copy()
    n.units = None

    with pytest.raises(ValueError) as exc:
        n.convert_units("um")

    msg = str(exc.value)
    assert "no units set" in msg
    assert ".units = " in msg, "should show how to set units"
    assert "DimensionalityError" not in msg


def test_convert_units_still_works(skeleton):
    """The stricter guard must not reject neurons that do have units."""
    assert skeleton.units.dimensionless is False
    assert navis.example_neurons(1).convert_units("um").units.dimensionless is False


def test_reroot_skeleton_unknown_node_id(skeleton):
    """Used to escape as a bare `KeyError: np.int64(...)` from the igraph path."""
    with pytest.raises(ValueError) as exc:
        navis.reroot_skeleton(skeleton, 99999999)

    msg = str(exc.value)
    assert "99999999" in msg
    assert "node ID" in msg
    assert "find_soma" in msg, "should point at a way to get a valid root"


def test_reroot_skeleton_valid_node_id_still_works(skeleton):
    """The new up-front validation must not reject legitimate roots."""
    node_id = skeleton.nodes.node_id.values[100]
    assert navis.reroot_skeleton(skeleton, node_id).root[0] == node_id


def test_resample_skeleton_unknown_method(skeleton):
    """Used to surface scipy's "Use fitpack routines for other types"."""
    with pytest.raises(ValueError) as exc:
        navis.resample_skeleton(skeleton, 1000, method="bogus")

    msg = str(exc.value)
    assert "bogus" in msg
    assert "linear" in msg and "cubic" in msg, "should list the valid methods"
    assert "fitpack" not in msg


@pytest.mark.parametrize("method", ["linear", "cubic", 2])
def test_resample_skeleton_accepts_valid_methods(skeleton, method):
    """Including the int spline order, which is not in the string list."""
    assert len(navis.resample_skeleton(skeleton, 2000, method=method).nodes) > 0


def test_heal_skeleton_unknown_method(skeleton):
    with pytest.raises(ValueError) as exc:
        navis.heal_skeleton(skeleton, method="bogus")

    msg = str(exc.value)
    assert '"bogus"' in msg, "should echo what was passed, not an upper-cased form"
    assert "BOGUS" not in msg
    assert "LEAFS" in msg and "ALL" in msg


@pytest.mark.parametrize("method", ["LEAFS", "leafs", "ALL"])
def test_heal_skeleton_method_stays_case_insensitive(skeleton, method):
    assert navis.heal_skeleton(skeleton, method=method) is not None


@pytest.mark.parametrize("kind", ["dotprops", "voxels", "mesh"])
def test_cable_length_names_the_type_that_has_it(request, kind):
    """`Attribute "cable_length" not found` was a dead end for all three."""
    neuron = request.getfixturevalue(kind)

    with pytest.raises(AttributeError) as exc:
        neuron.cable_length

    msg = str(exc.value)
    assert type(neuron).__name__ in msg
    assert "Skeleton" in msg, "should say which type does have it"
    assert "navis.skeletonize(x)" in msg, "should give the conversion"


def test_genuinely_unknown_attribute_stays_plain(dotprops):
    """No neuron type has this, so there is nothing to suggest."""
    with pytest.raises(AttributeError) as exc:
        dotprops.not_a_real_attribute

    msg = str(exc.value)
    assert "not_a_real_attribute" in msg
    assert "convert" not in msg


def test_private_attribute_miss_stays_plain(dotprops):
    """The dunder/private path is hot (copy, pickle, hasattr) and stays cheap."""
    with pytest.raises(AttributeError) as exc:
        dotprops._not_a_real_attribute

    assert "convert" not in str(exc.value)


def test_hasattr_still_works(dotprops, skeleton):
    """The nicer message must not turn a missing attribute into a raise."""
    assert hasattr(dotprops, "cable_length") is False
    assert hasattr(skeleton, "cable_length") is True
    # `has_*` / `n_*` are resolved through the same __getattr__ path.
    assert dotprops.has_connectors is False
    assert skeleton.n_nodes == len(skeleton.nodes)


def test_mesh_soma_keeps_its_own_message(mesh):
    """`Mesh.soma` raises AttributeError, so Python falls back to __getattr__.

    That used to swallow the property's own "use `.soma_pos`" hint and replace
    it with a generic "convert first" suggestion.
    """
    with pytest.raises(AttributeError) as exc:
        mesh.soma

    assert ".soma_pos" in str(exc.value), "should point at the mesh equivalent"


def test_missing_attr_msg_skips_not_implemented_stubs(mesh):
    """Only advertise types that actually implement the attribute.

    `hasattr(Voxels, "soma")` is True - but only because `Voxels` inherits
    `BaseNeuron`'s placeholder, which raises NotImplementedError. Converting to
    one would not get you a soma.
    """
    with pytest.raises(NotImplementedError):
        navis.voxelize(mesh, pitch="1 micron").soma

    msg = mesh._missing_attr_msg("soma")
    assert "Voxels" not in msg
    # The types that do implement it are still named.
    assert "Skeleton" in msg and "Dotprops" in msg


def test_metaclass_attributes_are_not_advertised(skeleton):
    """`hasattr(Mesh, "mro")` is True - but via the metaclass, not the neuron.

    Looking the key up on the class dicts (rather than with `getattr`) keeps
    `type`'s own attributes out of both the suggestion and the lookup.
    """
    with pytest.raises(AttributeError) as exc:
        skeleton.mro

    msg = str(exc.value)
    assert "convert" not in msg
    assert "Mesh" not in msg


@pytest.mark.parametrize("kind", ["mesh", "voxels"])
def test_has_soma_answers_instead_of_raising(request, kind):
    """`hasattr` only swallows AttributeError, so `Voxels.has_soma` exploded."""
    neuron = request.getfixturevalue(kind)

    assert neuron.has_soma is False
    assert neuron.n_soma is None


def test_neurons_still_copy_and_pickle(dotprops):
    """Guards against the error path interfering with dunder probing."""
    import pickle

    assert len(pickle.loads(pickle.dumps(dotprops)).points) == len(dotprops.points)
    assert np.array_equal(dotprops.copy().points, dotprops.points)
