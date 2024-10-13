"""Configuration for submodule doctests.

Should be importable (but not useful)
without development dependencies.
"""
try:
    import vispy
except ModuleNotFoundError:
    collect_ignore_glob = ["*.py"]
