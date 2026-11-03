"""Tests for `navis.drop_internals` - the wrapper, not the repair.

The repair itself lives in navis-fastcore and is tested there. What navis adds
is the bookkeeping around it, and two pieces of that are easy to get wrong.

fastcore reports "I kept surface I could not vouch for" with `warnings.warn` and
a `stacklevel` meant for a direct caller, which through navis points at navis
rather than at whoever asked. navis re-says those through its logger, the way it
says everything else of the kind - without swallowing any other warning raised
on the way past.

And the per-pass log has to name the faces dropped for enclosing nothing. That
step removes whole components, so a pass that discards most of a mesh reads as a
small, tidy repair if the number is left as the gap between the other ones.
"""

import logging
import warnings

import numpy as np
import pytest

import navis
import navis_fastcore


@pytest.fixture
def mesh():
    return navis.example_neurons(1, kind="mesh")


@pytest.fixture
def noisy_fastcore(monkeypatch):
    """`drop_internals` that raises both a fastcore warning and a foreign one.

    The real `unclosed`/`spared` paths need a mesh whose boundary rings pinch,
    which is exactly what the 0.13.2 repair made hard to come by - so the two
    warnings are injected rather than provoked.
    """
    real = navis_fastcore.drop_internals

    def noisy(*args, **kwargs):
        out = real(*args, **kwargs)
        warnings.warn(
            "drop_internals pass 0: 123 faces sit in components the capping "
            "could not close",
            UserWarning,
            stacklevel=2,
        )
        warnings.warn("something else entirely", RuntimeWarning, stacklevel=2)
        return out

    monkeypatch.setattr(navis.utils.fastcore, "drop_internals", noisy)


def test_fastcore_warning_is_said_through_the_logger(mesh, noisy_fastcore, caplog):
    """Attributed to the caller's world, not to a line inside navis."""
    with caplog.at_level(logging.WARNING, logger="navis"):
        with warnings.catch_warnings(record=True) as raised:
            warnings.simplefilter("always")
            navis.drop_internals(mesh, inplace=False)

    assert "could not close" in caplog.text
    assert not [w for w in raised if "could not close" in str(w.message)], (
        "fastcore's own warning should have been re-said, not passed through"
    )


def test_other_warnings_are_left_alone(mesh, noisy_fastcore):
    """Catching fastcore's must not quietly eat anyone else's."""
    with warnings.catch_warnings(record=True) as raised:
        warnings.simplefilter("always")
        navis.drop_internals(mesh, inplace=False)

    assert [w for w in raised if w.category is RuntimeWarning], (
        "a foreign warning was swallowed"
    )


def test_pass_log_names_what_it_dropped(mesh, caplog):
    """The component filter's share of the damage has to be in the log."""
    with caplog.at_level(logging.DEBUG, logger="navis"):
        navis.drop_internals(mesh, inplace=False)

    assert "drop_internals pass 0" in caplog.text
    assert "dropped as enclosing nothing" in caplog.text


def test_repair_still_repairs(mesh):
    """A smoke test on the real thing, with the vertex map intact."""
    out = navis.drop_internals(mesh, inplace=False)

    assert out.n_faces <= mesh.n_faces
    assert out.n_vertices <= mesh.n_vertices
    # Only ever removes - every surviving vertex is one of the originals.
    kept = {tuple(v) for v in np.asarray(mesh.vertices)}
    assert all(tuple(v) in kept for v in np.asarray(out.vertices)[:200])
