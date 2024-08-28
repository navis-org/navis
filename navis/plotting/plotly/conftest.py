"""Configuration for submodule doctests.

Should be importable (but not useful)
without development dependencies.
"""
try:
    import plotly.graph_objs as go
except ModuleNotFoundError:
    collect_ignore_glob = ["*.py"]
