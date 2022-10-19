"""Configuration for submodule doctests.

Should be importable (but not useful)
without development dependencies.
"""
try:
    import k3d

except ImportError:
    collect_ignore_glob = ["*.py"]
