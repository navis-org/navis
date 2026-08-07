"""Tests for the shared machinery under `navis.interfaces.base`.

Deliberately network-free: `fetch_parallel` is handed plain callables, and the
optional-import tests use package names that cannot exist.

Note this module must stay free of doctests - pytest runs with
`--doctest-modules`.
"""

import threading

import pytest

import navis

from navis.interfaces import base
from navis.interfaces.base import (
    FetchError,
    cached,
    clear_cache,
    fetch_parallel,
    optional_import,
    register_cache,
    resolve_errors,
)
from navis.utils import http


@pytest.fixture(autouse=True)
def not_strict():
    """Strict mode is global state; make sure a test never leaks it."""
    before = navis.config.strict
    navis.config.strict = False
    yield
    navis.config.strict = before


def boom(x):
    """Fail on 2, pass everything else through."""
    if x == 2:
        raise ValueError("nope")
    return x


# -----------------------------------------------------------------------------
# fetch_parallel
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("parallel", [True, False])
def test_returns_input_order(parallel):
    """The whole point: results line up with the input, not with completion.

    The old `as_completed` loops appended results as they arrived, so callers had
    to re-sort afterwards (and several didn't).
    """
    # Reversing the sleep makes the *last* item finish first
    import time

    def slow(x):
        time.sleep((10 - x) / 1000)
        return x

    items = list(range(10))
    assert fetch_parallel(slow, items, parallel=parallel, max_threads=10) == items


def test_empty_input():
    assert fetch_parallel(boom, []) == []


@pytest.mark.parametrize("parallel", [True, False])
def test_kwargs_are_forwarded(parallel):
    def add(x, *, offset):
        return x + offset

    got = fetch_parallel(add, [1, 2], parallel=parallel, offset=10)
    assert got == [11, 12]


@pytest.mark.parametrize("parallel", [True, False])
def test_errors_raise(parallel):
    with pytest.raises(FetchError) as excinfo:
        fetch_parallel(boom, [1, 2, 3], errors="raise", parallel=parallel)

    # The original error must survive - callers need to tell a 404 from a 500.
    assert isinstance(excinfo.value.__cause__, ValueError)
    # ... and the message has to say *which* item failed
    assert "2" in str(excinfo.value)


@pytest.mark.parametrize("parallel", [True, False])
def test_errors_ignore_leaves_a_hole(parallel):
    assert fetch_parallel(boom, [1, 2, 3], errors="ignore", parallel=parallel) == [
        1,
        None,
        3,
    ]


def test_errors_log(caplog):
    with caplog.at_level("ERROR"):
        got = fetch_parallel(boom, [1, 2, 3], errors="log")

    assert got == [1, None, 3]
    assert "nope" in caplog.text


def test_labels_used_in_messages():
    with pytest.raises(FetchError, match="body-two"):
        fetch_parallel(
            boom, [1, 2], labels=["body-one", "body-two"], errors="raise"
        )


def test_labels_length_is_checked():
    with pytest.raises(ValueError, match="must match"):
        fetch_parallel(boom, [1, 2, 3], labels=["a"])


def test_bad_error_policy():
    with pytest.raises(ValueError, match="errors"):
        fetch_parallel(boom, [1], errors="explode")


@pytest.mark.parametrize("parallel", [True, False])
def test_initializer_runs(parallel):
    """Serial runs must still initialise - neuprint hangs its client off this."""
    local = threading.local()

    def init(value):
        local.value = value

    def read(x):
        return getattr(local, "value", None)

    got = fetch_parallel(
        read, [1, 2], initializer=init, initargs=("set",), parallel=parallel
    )
    assert got == ["set", "set"]


def test_parallel_false_matches_max_threads_one():
    assert fetch_parallel(boom, [1, 3], parallel=False) == fetch_parallel(
        boom, [1, 3], max_threads=1
    )


# -----------------------------------------------------------------------------
# strict mode
# -----------------------------------------------------------------------------


def test_default_error_policy_follows_strict():
    assert resolve_errors() == "log"
    navis.config.strict = True
    assert resolve_errors() == "raise"


def test_explicit_errors_beats_strict():
    navis.config.strict = True
    assert resolve_errors("ignore") == "ignore"


def test_strict_makes_a_partial_fetch_fatal():
    """A server run must not quietly hand back a short result."""
    assert fetch_parallel(boom, [1, 2, 3]) == [1, None, 3]

    navis.config.strict = True
    with pytest.raises(FetchError):
        fetch_parallel(boom, [1, 2, 3])


def test_strict_is_carried_into_workers():
    assert "strict" in navis.config.WORKER_SETTINGS


# -----------------------------------------------------------------------------
# optional_import
# -----------------------------------------------------------------------------


def test_optional_import_returns_real_module():
    import numpy

    assert optional_import("numpy") is numpy


def test_missing_module_does_not_raise_on_import():
    # The point of the exercise: importing an interface whose dependency is
    # absent must not blow up.
    missing = optional_import("navis_no_such_package")
    assert not missing
    assert "navis_no_such_package" in repr(missing)


def test_missing_module_raises_on_use():
    missing = optional_import("navis_no_such_package", pip="something-else")

    with pytest.raises(ModuleNotFoundError, match="pip install something-else"):
        missing.anything

    with pytest.raises(ModuleNotFoundError):
        missing()


def test_missing_module_hint():
    missing = optional_import("navis_no_such_package", hint="Mind the gap.")
    with pytest.raises(ModuleNotFoundError, match="Mind the gap."):
        missing.anything


def test_missing_module_survives_copy_and_pickle():
    """`copy`/`pickle` probe for dunders and rebuild without `__init__`.

    Both used to recurse to a stack overflow, and would have taken down any
    `deepcopy` of an object that merely held one of these.
    """
    import copy
    import pickle

    missing = optional_import("navis_no_such_package")

    assert hasattr(missing, "__deepcopy__") is False

    for rebuilt in (copy.deepcopy(missing), pickle.loads(pickle.dumps(missing))):
        assert not rebuilt
        assert "navis_no_such_package" in repr(rebuilt)
        with pytest.raises(ModuleNotFoundError):
            rebuilt.anything


def test_interfaces_import_without_their_dependencies():
    """Every data-source interface must import on a bare install.

    `vfb` and `allen_celltypes` used to raise at import - and `vfb` additionally
    opened a connection to VFB's servers while doing so.
    """
    import importlib

    for name in (
        "brain_image_library",
        "cave_utils",
        "microns",
        "h01",
        "neuromorpho",
        "insectbrain_db",
        "allen_celltypes",
        "vfb",
    ):
        importlib.import_module(f"navis.interfaces.{name}")


# -----------------------------------------------------------------------------
# retired interfaces
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["r", "cytoscape"])
def test_retired_interface_points_at_the_replacement(name):
    """A retired interface must say what to do instead, not just fail."""
    from navis import interfaces

    with pytest.raises(ImportError) as excinfo:
        getattr(interfaces, name)

    # An `AttributeError` would be swallowed by `hasattr`, and the import
    # machinery would then report a bare "cannot import name".
    assert not isinstance(excinfo.value, AttributeError)
    assert name in str(excinfo.value)


def test_unknown_interface_still_raises_attributeerror():
    from navis import interfaces

    with pytest.raises(AttributeError):
        interfaces.not_a_real_interface


# -----------------------------------------------------------------------------
# caches & sessions
# -----------------------------------------------------------------------------


def test_cached_is_registered_and_clearable():
    calls = []

    @cached
    def count(x):
        calls.append(x)
        return x

    count(1)
    count(1)
    assert calls == [1]  # second call came from the cache

    clear_cache()

    count(1)
    assert calls == [1, 1]


def test_register_cache_returns_the_callable():
    cleared = []

    @register_cache
    def clear():
        cleared.append(True)

    clear_cache()
    assert cleared


def test_interface_caches_are_registered():
    """The point of the registry - no cache reachable only by name."""
    import navis.interfaces.microns  # noqa: F401
    import navis.interfaces.insectbrain_db  # noqa: F401
    import navis.interfaces.brain_image_library as bil

    assert bil.clear_metadata_cache in base._CACHE_CLEARERS
    assert len(base._CACHE_CLEARERS) >= 8


def test_sessions_are_shared_and_identify_navis():
    session = http.get_session()

    assert session is http.get_session()
    assert "navis" in session.headers["User-Agent"]

    # io and interfaces must not be running separate pools
    from navis.io.base import get_session as io_get_session

    assert io_get_session() is session


def test_session_config_is_part_of_the_cache_key():
    assert http.get_session(retries=0) is not http.get_session(retries=3)
    assert http.get_session({"X-Test": "1"}) is not http.get_session()


def test_clear_sessions():
    session = http.get_session()
    http.clear_sessions()
    assert http.get_session() is not session
