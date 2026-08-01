"""Tests for the pre-2.0 neuron class names.

navis 2.0 renamed `TreeNeuron`/`MeshNeuron`/`VoxelNeuron` to `Skeleton`/`Mesh`/
`Voxels`. The old names have to keep working for a while, and the point of these
tests is that they keep working *as aliases* - downstream code does
`isinstance(x, navis.TreeNeuron)` and `class CatmaidNeuron(navis.TreeNeuron)`,
neither of which survives a shim that hands back a different object.
"""

import contextlib
import pickle
import warnings

import navis
import navis.core
import navis.core.mesh
import navis.core.neurons
import navis.core.skeleton
import navis.core.voxel
import pytest

from navis._deprecated import (
    DEPRECATED_NEURON_CLASSES,
    reset_deprecation_warnings,
)

RENAMES = sorted(DEPRECATED_NEURON_CLASSES.items())


@contextlib.contextmanager
def no_deprecation_warning():
    """Turn any DeprecationWarning in the block into an error."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        yield


@pytest.fixture(autouse=True)
def fresh_warnings():
    """Warn-once state is global; each test needs it clean to observe a warning."""
    reset_deprecation_warnings()


def test_rename_table():
    assert DEPRECATED_NEURON_CLASSES == {
        "TreeNeuron": "Skeleton",
        "MeshNeuron": "Mesh",
        "VoxelNeuron": "Voxels",
    }


@pytest.mark.parametrize("old,new", RENAMES)
def test_old_name_is_the_new_class(old, new):
    # The match asserts the warning names *both* spellings - `match=new` alone
    # would be satisfied by "MeshNeuron" containing "Mesh".
    pattern = rf"`navis\.{old}`.*`navis\.{new}`"
    with pytest.warns(DeprecationWarning, match=pattern):
        assert getattr(navis, old) is getattr(navis, new)


@pytest.mark.parametrize("old,new", RENAMES)
def test_warns_only_once_per_session(old, new):
    with pytest.warns(DeprecationWarning):
        getattr(navis, old)

    # A loop over a NeuronList mustn't produce one warning per neuron
    with no_deprecation_warning():
        for _ in range(3):
            getattr(navis, old)


def test_error_filter_is_not_swallowed():
    """`-W error::DeprecationWarning` must keep raising, not just the first time."""
    for _ in range(3):
        with pytest.raises(DeprecationWarning):
            with no_deprecation_warning():
                navis.TreeNeuron


def test_new_names_do_not_warn():
    with no_deprecation_warning():
        assert navis.Skeleton is not None
        assert navis.Mesh is not None
        assert navis.Voxels is not None
        assert navis.Dotprops is not None


def test_unknown_attribute_still_raises():
    with pytest.raises(AttributeError):
        navis.DefinitelyNotANavisThing

    # `hasattr` and friends rely on the AttributeError above
    assert not hasattr(navis, "DefinitelyNotANavisThing")


def test_isinstance_and_subclassing():
    with pytest.warns(DeprecationWarning):
        TreeNeuron = navis.TreeNeuron

    n = navis.example_neurons(1, kind="skeleton")
    assert isinstance(n, TreeNeuron)

    # pymaid & co subclass the neuron classes
    class CustomNeuron(TreeNeuron):
        pass

    assert issubclass(CustomNeuron, navis.Skeleton)


@pytest.mark.parametrize("old,new", RENAMES)
def test_navis_core_resolves_quietly(old, new):
    """`navis.core` is on the hot path for navis' own `core.X` lookups.

    A module `__getattr__` there costs every one of them the `LOAD_ATTR_MODULE`
    bytecode specialization, so the old names are plain aliases and only the
    top-level `navis` namespace warns.
    """
    with no_deprecation_warning():
        assert getattr(navis.core, old) is getattr(navis, new)


@pytest.mark.parametrize(
    "module,old,new",
    [
        (navis.core.skeleton, "TreeNeuron", "Skeleton"),
        (navis.core.mesh, "MeshNeuron", "Mesh"),
        (navis.core.voxel, "VoxelNeuron", "Voxels"),
        # pre-dates the core split, so older pickles point here
        (navis.core.neurons, "TreeNeuron", "Skeleton"),
        (navis.core.neurons, "MeshNeuron", "Mesh"),
    ],
)
def test_pickle_paths_resolve_quietly(module, old, new):
    """`pickle` resolves classes by defining module - that must not warn."""
    with no_deprecation_warning():
        assert getattr(module, old) is getattr(navis, new)


def test_roundtrip_pickle():
    n = navis.example_neurons(1, kind="skeleton")
    assert isinstance(pickle.loads(pickle.dumps(n)), navis.Skeleton)


@pytest.mark.parametrize(
    "kind,expected",
    [("skeleton", "navis.Skeleton"), ("mesh", "navis.Mesh")],
)
def test_type_property_uses_new_name(kind, expected):
    assert navis.example_neurons(1, kind=kind).type == expected
