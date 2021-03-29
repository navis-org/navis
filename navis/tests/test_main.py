""" Test suite for navis.

Examples
--------

From terminal::

  $ cd folder/to/navis
  $ pytest

"""
import warnings
import navis

import pandas as pd
import numpy as np

import doctest
import unittest
import functools

try:
    import igraph  # type: ignore
except BaseException:
    igraph = None
    warnings.warn('iGraph library not found. Will test only with NetworkX.')

# Set navis to headless -> this prevents viewer window from showing
navis.config.headless = True

globs = {'navis': navis, 'numpy': np, 'np': np, 'pandas': pd, 'pd': pd}


def try_conditions(func):
    """Runs each test under various conditions and asserts that results
    are always the same."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        navis.config.use_igraph = False
        res1 = func(*args, **kwargs)
        if igraph:
            navis.config.use_igraph = True
            res2 = func(*args, **kwargs)
            assert res1 == res2
        return res1
    return wrapper


#class TestDocstringExamples(unittest.TestCase):
#    """Test."""

#    def test_fetch_examples(self):
#        for func in navis.data.load_data.__all__:#
#            f = getattr(navis.data.load_data, func)
#            doctest.run_docstring_examples(f, globs=globs, name=f, verbose=True)


"""
@try_conditions
def test_data_load_data():
    doctest.testmod(navis.data.load_data,
                    raise_on_error=True,
                    globs=globs)


@try_conditions
def test_io_json_io():
    doctest.testmod(navis.io.json_io,
                    raise_on_error=True,
                    globs=globs)


@try_conditions
def test_vispy_viewer():
    doctest.testmod(navis.plotting.vispy.viewer,
                    raise_on_error=True,
                    globs=globs)


@try_conditions
def test_resampling():
    doctest.testmod(navis.resampling,
                    raise_on_error=True,
                    globs=globs)


@try_conditions
def test_morpho_mmetrics():
    doctest.testmod(navis.morpho.mmetrics,
                    raise_on_error=True,
                    globs=globs)


@try_conditions
def test_morpho_manipulation():
    doctest.testmod(navis.morpho.manipulation,
                    raise_on_error=True,
                    globs=globs)


@try_conditions
def test_morpho_analyze():
    doctest.testmod(navis.morpho.analyze,
                    raise_on_error=True,
                    globs=globs)


@try_conditions
def test_graph_graph_utils():
    doctest.testmod(navis.graph.graph_utils,
                    raise_on_error=True,
                    globs=globs)


@try_conditions
def test_graph_converters():
    doctest.testmod(navis.graph.converters,
                    raise_on_error=True,
                    globs=globs)


@try_conditions
def test_graph_clinic():
    doctest.testmod(navis.graph.clinic,
                    raise_on_error=True,
                    globs=globs)
"""
