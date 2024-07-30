def pytest_ignore_collect(path, config):
    """Return True to prevent considering this path for collection.
    This hook is consulted for all files and directories prior to calling
    more specific hooks.
    """
    path = str(path)
    for pattern in (
        "interfaces",
        "/docs",
        "/stubs",
        "/examples",
        "/dist/",
        "/binder",
        "/site",
        "/scripts",
    ):
        if pattern in path:
            return True
