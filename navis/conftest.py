"""Configuration for in-package doctests.

Should be importable (but not useful)
without development dependencies.

Where these files up and when they are deleted is documented
`here <https://pytest.org/en/stable/tmpdir.html#the-default-base-temporary-directory>`_.
"""
from pathlib import Path

try:
    import pytest

    decorator = pytest.fixture(autouse=True)

except ImportError:
    def decorator(fn):
        return fn


@decorator
def add_tmp_dir(doctest_namespace, tmpdir):
    """Give all doctests access to a ``tmp_dir`` variable.

    ``tmp_dir`` is a ``pathlib.Path`` to a real directory
    in pytest's tmp directory which is automatically cleaned up
    in later invocations of ``pytest``.
    """
    doctest_namespace["tmp_dir"] = Path(tmpdir)
