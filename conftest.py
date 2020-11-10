def pytest_ignore_collect(path, config):
    """Return True to prevent considering this path for collection.
    This hook is consulted for all files and directories prior to calling
    more specific hooks.
    """
    path = str(path)
    if 'interfaces' in path:
        return True
    if '/docs' in path:
        return True
    if '/stubs' in path:
        return True
    if '/examples' in path:
        return True
    if '/dist/' in path:
        return True
    if '/binder' in path:
        return True
