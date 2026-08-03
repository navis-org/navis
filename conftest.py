from pathlib import Path


def pytest_ignore_collect(collection_path: Path, config):
    """Return True to prevent considering this path for collection.
    This hook is consulted for all files and directories prior to calling
    more specific hooks.
    """
    path = str(collection_path)
    for pattern in (
        "interfaces",
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
