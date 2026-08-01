"""Tests for `navis.config.quiet_logger` and the sites that use it.

There is only one navis logger, so a function that silences it and then fails
takes the whole library's output down with it - for the rest of the session,
and with nothing to tell the user why. These check that no such function can.
"""

import logging

import numpy as np
import pytest

import navis
from navis import config


@pytest.fixture
def logger_state():
    """Put the logger and the pbar setting back however the test leaves them."""
    level, hide = config.logger.level, config.pbar_hide
    yield
    config.logger.setLevel(level)
    config.pbar_hide = hide


def test_quiet_logger_restores_on_exception(logger_state):
    config.logger.setLevel(logging.INFO)

    with pytest.raises(ValueError):
        with config.quiet_logger():
            assert config.logger.getEffectiveLevel() == logging.ERROR
            raise ValueError('boom')

    assert config.logger.level == logging.INFO


def test_quiet_logger_restores_pbars_on_exception(logger_state):
    config.pbar_hide = False

    with pytest.raises(ValueError):
        with config.quiet_logger(pbars=True):
            assert config.pbar_hide is True
            raise ValueError('boom')

    assert config.pbar_hide is False


def test_quiet_logger_only_ever_quietens(logger_state):
    """Someone who asked for silence must not be talked at."""
    config.logger.setLevel(logging.CRITICAL)
    with config.quiet_logger(level='ERROR'):
        assert config.logger.getEffectiveLevel() == logging.CRITICAL
    assert config.logger.level == logging.CRITICAL


def test_quiet_logger_rejects_a_bad_level():
    with pytest.raises(ValueError, match='logging level'):
        config.quiet_logger('LOUD')


# The functions that silence the logger to build a throwaway downsampled copy.
# All take a skeleton with connectors and all used to leak on failure.
QUIETENERS = [navis.arbor_segregation_index, navis.bending_flow,
              navis.synapse_flow_centrality, navis.flow_centrality]


@pytest.mark.parametrize('func', QUIETENERS, ids=lambda f: f.__name__)
def test_failure_does_not_silence_navis(func, logger_state, monkeypatch):
    """Each of these downsamples first; make that step raise."""
    n = navis.example_neurons(1)
    config.logger.setLevel(logging.INFO)
    config.pbar_hide = False

    def boom(*args, **kwargs):
        raise RuntimeError('boom')

    # `synapse_flow_centrality` hands the whole computation to fastcore where it
    # can, never reaching the downsample - so take that route away
    monkeypatch.setattr(navis.utils, 'fastcore', None)
    # Both spellings - two of the four go through the function, two the method
    monkeypatch.setattr(navis.sampling, 'downsample_neuron', boom)
    monkeypatch.setattr(navis.Skeleton, 'downsample', boom)

    with pytest.raises(RuntimeError, match='boom'):
        func(n)

    assert config.logger.level == logging.INFO
    assert config.pbar_hide is False


def test_thumbnail_failure_leaves_matplotlib_alone(logger_state, monkeypatch):
    """`_gen_svg_thumbnail` also owns matplotlib's interactive mode."""
    import matplotlib.pyplot as plt

    n = navis.example_neurons(1)
    config.logger.setLevel(logging.INFO)
    # Not assumed: `NAVIS_HEADLESS` makes True the default, and CI sets it
    config.pbar_hide = False
    plt.ion()

    monkeypatch.setattr(navis.Skeleton, 'plot2d',
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('boom')))

    try:
        with pytest.raises(RuntimeError, match='boom'):
            n._gen_svg_thumbnail()

        assert config.logger.level == logging.INFO
        assert config.pbar_hide is False
        assert plt.isinteractive() is True
    finally:
        plt.ioff()
        plt.close('all')


def test_quieteners_still_work(logger_state):
    """The refactor must not have changed what they compute."""
    n = navis.example_neurons(1)
    assert np.isfinite(navis.flow_centrality(n).nodes.flow_centrality).all()
