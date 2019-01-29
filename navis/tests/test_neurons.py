""" Test suite for navis.

Examples
--------

From terminal:
$ cd folder/to/navis
$ pytest

From shell:
>>> import unittest
>>> from navis.test import test_neurons
>>> suite = unittest.TestLoader().loadTestsFromModule(test_neurons)
>>> unittest.TextTestRunner().run(suite)

"""

import unittest
import warnings

import navis

try:
    import igraph
except BaseException:
    igraph = None
    warnings.warn('iGraph library not found. Will test only with NetworkX.')


class TestNeurons(unittest.TestCase):
    """Test navis.core.neurons. """

    def try_conditions(func):
        """Runs each test under various conditions and asserts that results
        are always the same."""

        def wrapper(self, *args, **kwargs):
            navis.config.use_igraph = False
            res1 = func(self, *args, **kwargs)
            if igraph:
                navis.config.use_igraph = True
                res2 = func(self, *args, **kwargs)
                self.assertEqual(res1, res2)
            return res1
        return wrapper

    @try_conditions
    def test_from_swc(self):
        n = navis.example_neurons(n=1, source='swc')
        self.assertIsInstance(n, navis.TreeNeuron)

    @try_conditions
    def test_from_gml(self):
        n = navis.example_neurons(n=1, source='gml')
        self.assertIsInstance(n, navis.TreeNeuron)
