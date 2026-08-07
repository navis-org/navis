from pathlib import Path

# `navis.interfaces` talks to remote services, so its doctests cannot run in CI.
# Matched as a path rather than as the substring "interfaces", which would also
# swallow the tests *for* those modules (e.g. `tests/test_interfaces_base.py`).
INTERFACES = Path(__file__).resolve().parent / "navis" / "interfaces"


def pytest_ignore_collect(collection_path: Path, config):
    """Return True to prevent considering this path for collection.
    This hook is consulted for all files and directories prior to calling
    more specific hooks.
    """
    collection_path = Path(collection_path).resolve()
    if collection_path == INTERFACES or INTERFACES in collection_path.parents:
        return True

    path = str(collection_path)
    for pattern in (
        "/docs",
        "/stubs",
        "/examples",
        "/dist/",
        "/binder",
        "/site",
        "/scripts",
        # Test data, plus the scripts that generate it. `--doctest-modules`
        # would otherwise import those scripts and run them.
        "/fixtures",
        "h5reg_numba",  # this module requires numba but doesn't contain any tests
    ):
        if pattern in path:
            return True
