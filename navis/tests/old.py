""" Test suite for pymaid

This test suite requires a bunch of variables that need to either defined
as environment variables or in a config_test.py file

    #Catmaid server url
    server_url = ''

    #Http user
    http_user = ''

    #Http password
    http_pw = ''

    #Auth token
    token = ''

    # Test skeleton IDs
    test_skids = []

    # Test annotations
    test_annotations = []

    # Test CATMAID volume
    test_volume = ''

If you use environmental variables, give lists as comma-separated string.


Examples
--------

From terminal:
>>> python test_pymaid.py

From shell:
>>> import unittest
>>> import test_pymaid
>>> suite = unittest.TestLoader().loadTestsFromModule(test_pymaid)
>>> unittest.TextTestRunner().run(suite)

"""

import warnings
import os
import matplotlib as mpl
#if os.environ.get('DISPLAY', '') == '':
#    warnings.warn('No display found. Using template backend (nothing '
#                  'will show).')
#    mpl.use('template')
mpl.use('template')
import matplotlib.pyplot as plt

import unittest
import doctest

import datetime

import pymaid
import pandas as pd
import numpy as np
import networkx as nx

import importlib

try:
    import igraph
except BaseException:
    igraph = None
    warnings.warn('iGraph library not found. Will test only with NetworkX.')

# Silence module loggers
pymaid.set_loggers('ERROR')

# Deactivate progress bars
pymaid.set_pbars(hide=True)

# Try getting credentials from environment variable
required_variables = ['server_url', 'http_user', 'http_pw', 'token',
                      'test_skids', 'test_annotations', 'test_volume']

if False not in [v in os.environ for v in required_variables]:
    class conf:
        pass
    config_test = conf()

    for v in ['server_url', 'http_user', 'http_pw', 'token', 'test_volume']:
        setattr(config_test, v, os.environ[v])

    for v in ['test_skids', 'test_annotations']:
        setattr(config_test, v, os.environ[v].split(','))
else:
    missing = [v for v in required_variables if v not in os.environ]
    print('Missing environment variables:', ', '.join(missing))
    print('Falling back to config_test.py')
    try:
        import config_test
    except ImportError:
        raise ImportError('Unable to import configuration file.')


class TestModules(unittest.TestCase):
    """Test individual module import. """

    def test_imports(self):
        mods = ['morpho', 'core', 'plotting', 'graph', 'graph_utils', 'core',
                'connectivity', 'user_stats', 'cluster', 'resample',
                'intersect', 'fetch', 'scene3d']

        for m in mods:
            _ = importlib.import_module('pymaid.{}'.format(m))


class TestFetch(unittest.TestCase):
    """Test pymaid.fetch """

    def setUp(self):
        self.rm = pymaid.CatmaidInstance(
            config_test.server_url,
            config_test.http_user,
            config_test.http_pw,
            config_test.token)

    def try_conditions(func):
        """Runs each test under various conditions and asserts that results
        are always the same."""

        def wrapper(self, *args, **kwargs):
            pymaid.config.use_igraph = False
            res1 = func(self, *args, **kwargs)
            # if igraph:
            #    pymaid.config.use_igraph = True
            #    res2 = func(self, *args, **kwargs)
            #    self.assertEqual(res1, res2)
            return res1
        return wrapper

    @try_conditions
    def test_get_annotation_list(self):
        remote_annotation_list_url = self.rm._get_annotation_list()
        self.assertIn('annotations', self.rm.fetch(remote_annotation_list_url))

    @try_conditions
    def test_eval_skids(self):
        """ Test skeleton ID evaluation. """
        self.assertEqual(pymaid.eval_skids(
            ['12345', '67890'], remote_instance=self.rm), ['12345', '67890'])
        self.assertEqual(pymaid.eval_skids(
            [12345, 67890], remote_instance=self.rm), ['12345', '67890'])
        self.assertIsNotNone(pymaid.eval_skids(
            'annotation:{}'.format(config_test.test_annotations[0]),
            remote_instance=self.rm))

    @try_conditions
    def test_neuron_exists(self):
        self.assertIsInstance(pymaid.neuron_exists(
            config_test.test_skids[0], remote_instance=self.rm), bool)
        self.assertIsInstance(pymaid.neuron_exists(
            config_test.test_skids, remote_instance=self.rm), dict)

    @try_conditions
    def test_get_neuron_list(self):
        self.assertIsInstance(pymaid.get_neuron_list(
            node_count=20000, remote_instance=self.rm), list)

    @try_conditions
    def test_get_user_list(self):
        self.assertIsInstance(pymaid.get_user_list(
            remote_instance=self.rm), pd.DataFrame)

    @try_conditions
    def test_get_history(self):
        self.assertIsInstance(pymaid.get_history(
            remote_instance=self.rm), pd.Series)

    @try_conditions
    def test_get_annotated_skids(self):
        self.assertIsInstance(pymaid.get_skids_by_annotation(
            config_test.test_annotations[0], remote_instance=self.rm), list)

    @try_conditions
    def test_get_annotated(self):
        self.assertIsInstance(pymaid.get_annotated(
            config_test.test_annotations[0], remote_instance=self.rm),
                                                   pd.DataFrame)

    @try_conditions
    def test_get_annotations(self):
        self.assertIsInstance(pymaid.get_annotations(
            config_test.test_skids, remote_instance=self.rm), dict)
        self.assertIsInstance(pymaid.get_annotation_details(
            config_test.test_skids, remote_instance=self.rm), pd.DataFrame)

    @try_conditions
    def test_get_neuron(self):
        self.assertIsInstance(pymaid.get_neuron(config_test.test_skids,
                                                remote_instance=self.rm),
                              pymaid.CatmaidNeuronList)

    @try_conditions
    def test_get_neuron2(self):
        self.assertIsInstance(pymaid.get_arbor(
            config_test.test_skids[0], remote_instance=self.rm), pd.DataFrame)

    @try_conditions
    def test_get_treenode_table(self):
        self.assertIsInstance(pymaid.get_treenode_table(
            config_test.test_skids, remote_instance=self.rm), pd.DataFrame)

    @try_conditions
    def test_get_review(self):
        self.assertIsInstance(pymaid.get_review(
            config_test.test_skids, remote_instance=self.rm), pd.DataFrame)

    @try_conditions
    def test_get_volume(self):
        self.assertIsInstance(pymaid.get_volume(
            config_test.test_volume, remote_instance=self.rm), pymaid.Volume)

    @try_conditions
    def test_get_logs(self):
        self.assertIsInstance(pymaid.get_logs(
            remote_instance=self.rm), pd.DataFrame)

    @try_conditions
    def test_get_contributor_stats(self):
        self.assertIsInstance(pymaid.get_contributor_statistics(
            config_test.test_skids, remote_instance=self.rm), pd.Series)

    @try_conditions
    def test_get_partners(self):
        self.assertIsInstance(pymaid.get_partners(
            config_test.test_skids[0], remote_instance=self.rm), pd.DataFrame)

    @try_conditions
    def test_get_names(self):
        names = pymaid.get_names(
            config_test.test_skids, remote_instance=self.rm)
        self.assertIsInstance(names, dict)
        self.assertIsInstance(pymaid.get_skids_by_name(
            list(names.values()), remote_instance=self.rm), pd.DataFrame)

    @try_conditions
    def test_get_cn_table(self):
        self.assertIsInstance(pymaid.get_partners('annotation:%s' % config_test.test_annotations[
                              0], remote_instance=self.rm), pd.DataFrame)

    @try_conditions
    def test_get_connectors(self):
        cn = pymaid.get_connectors(
            config_test.test_skids, remote_instance=self.rm)
        self.assertIsInstance(cn, pd.DataFrame)
        self.assertIsInstance(pymaid.get_connector_details(
            cn.connector_id.tolist(), remote_instance=self.rm), pd.DataFrame)

    @try_conditions
    def test_get_connector_links(self):
        cn = pymaid.get_connector_links(config_test.test_skids,
                                        remote_instance=self.rm)
        self.assertIsInstance(cn, pd.DataFrame)

    @try_conditions
    def test_get_partners_in_volume(self):
        self.assertIsInstance(pymaid.get_partners_in_volume(config_test.test_skids[0],
                                                            config_test.test_volume),
                              pd.DataFrame)

    @try_conditions
    def test_node_details(self):
        n = pymaid.get_neuron(config_test.test_skids[0])
        self.assertIsInstance(pymaid.get_node_details(n.nodes.sample(100).node_id.values),
                              pd.DataFrame)

    @try_conditions
    def test_skid_from_treenode(self):
        n = pymaid.get_neuron(config_test.test_skids[0])
        self.assertIsInstance(pymaid.get_skid_from_treenode(n.nodes.iloc[0].node_id),
                              dict)

    @try_conditions
    def test_get_edges(self):
        self.assertIsInstance(pymaid.get_edges(config_test.test_skids),
                              pd.DataFrame)

    @try_conditions
    def test_connectors_between(self):
        self.assertIsInstance(pymaid.get_connectors_between(config_test.test_skids,
                                                            config_test.test_skids),
                              pd.DataFrame)

    @try_conditions
    def test_user_annotations(self):
        ul = pymaid.get_user_list()
        self.assertIsInstance(pymaid.get_user_annotations(ul.sample(1).iloc[0].id),
                              pd.DataFrame)

    @try_conditions
    def test_has_soma(self):
        self.assertIsInstance(pymaid.has_soma(config_test.test_skids[0]),
                              dict)

    @try_conditions
    def test_treenode_info(self):
        n = pymaid.get_neuron(config_test.test_skids[0])
        self.assertIsInstance(pymaid.get_treenode_info(n.nodes.node_id.values[0:50]),
                              pd.DataFrame)

    @try_conditions
    def test_treenode_tags(self):
        n = pymaid.get_neuron(config_test.test_skids[0])
        self.assertIsInstance(pymaid.get_node_tags(n.nodes.node_id.values[0:50],
                                                   node_type='TREENODE'),
                              dict)

    @try_conditions
    def test_review_details(self):
        self.assertIsInstance(pymaid.get_review_details(config_test.test_skids[0]),
                              pd.DataFrame)

    @try_conditions
    def test_find_neurons(self):
        self.assertIsInstance(pymaid.find_neurons(annotations=config_test.test_annotations),
                              pymaid.CatmaidNeuronList)

    @try_conditions
    def test_get_paths(self):
        paths, g = pymaid.get_paths(
            config_test.test_skids[0], config_test.test_skids[1], return_graph=True)
        self.assertIsInstance(g,
                              nx.Graph)
        self.assertIsInstance(paths,
                              list)

    @try_conditions
    def test_annotation_list(self):
        self.assertIsInstance(pymaid.get_annotation_list(),
                              pd.DataFrame)

    @try_conditions
    def test_url_to_coords(self):
        self.assertIsInstance(pymaid.url_to_coordinates((0, 0, 0), 1),
                              str)

    @try_conditions
    def test_get_segments(self):
        self.assertIsInstance(pymaid.get_segments(config_test.test_skids[0]),
                              list)

    @try_conditions
    def test_neurons_in_volume(self):
        self.assertIsInstance(pymaid.get_neurons_in_volume(config_test.test_volume),
                              list)

    @try_conditions
    def test_label_list(self):
        self.assertIsInstance(pymaid.get_label_list(),
                              pd.DataFrame)


class TestCore(unittest.TestCase):
    """Test pymaid.core """

    def try_conditions(func):
        """Runs each test under various conditions and asserts that results
        are always the same."""

        def wrapper(self, *args, **kwargs):
            pymaid.config.use_igraph = False
            res1 = func(self, *args, **kwargs)
            if igraph:
                pymaid.config.use_igraph = True
                res2 = func(self, *args, **kwargs)
                self.assertEqual(res1, res2)
            return res1
        return wrapper

    def setUp(self):
        self.rm = pymaid.CatmaidInstance(
            config_test.server_url,
            config_test.http_user,
            config_test.http_pw,
            config_test.token)

        self.nl = pymaid.get_neuron('annotation:%s' % config_test.test_annotations[
            0], remote_instance=self.rm)

    @try_conditions
    def test_init(self):
        self.assertIsInstance(pymaid.CatmaidNeuron(
            config_test.test_skids[0]), pymaid.CatmaidNeuron)
        self.assertIsInstance(pymaid.CatmaidNeuronList(
            config_test.test_skids), pymaid.CatmaidNeuronList)

    @try_conditions
    def test_loaddata(self):
        # Test explicit loading
        nl = pymaid.CatmaidNeuron(config_test.test_skids[0])
        nl.get_skeleton()
        self.assertIsInstance(nl.nodes, pd.DataFrame)
        # Test implicit loading
        nl = pymaid.CatmaidNeuron(config_test.test_skids[0])
        self.assertIsInstance(nl.nodes, pd.DataFrame)

    @try_conditions
    def test_copy(self):
        self.assertIsInstance(self.nl[0].copy(), pymaid.CatmaidNeuron)
        self.assertIsInstance(self.nl.copy(), pymaid.CatmaidNeuronList)

    @try_conditions
    def test_summary(self):
        self.assertIsInstance(self.nl[0].summary(), pd.Series)
        self.assertIsInstance(self.nl.summary(), pd.DataFrame)

        self.assertIsInstance(self.nl.sum(), pd.Series)
        self.assertIsInstance(self.nl.mean(), pd.Series)

    @try_conditions
    def test_reload(self):
        self.nl.reload()
        self.assertIsInstance(self.nl, pymaid.CatmaidNeuronList)

    @try_conditions
    def test_graph_related(self):
        self.assertIsInstance(self.nl[0].graph, nx.Graph)
        self.assertIsInstance(self.nl[0].segments, list)

    @try_conditions
    def test_indexing(self):
        skids = self.nl.skeleton_id

        self.assertIsInstance(self.nl[0], pymaid.CatmaidNeuron)
        self.assertIsInstance(self.nl[:3], pymaid.CatmaidNeuronList)
        self.assertIsInstance(
            self.nl[self.nl.n_nodes > 1000], pymaid.CatmaidNeuronList)
        self.assertIsInstance(
            self.nl[list(skids[:-1])], pymaid.CatmaidNeuronList)
        self.assertIsInstance(self.nl - self.nl[0], pymaid.CatmaidNeuronList)
        self.assertIsInstance(
            self.nl[0] + self.nl[1], pymaid.CatmaidNeuronList)
        self.assertIsInstance(self.nl + self.nl[1], pymaid.CatmaidNeuronList)

        self.assertIsInstance(self.nl.sample(2), pymaid.CatmaidNeuronList)

    @try_conditions
    def test_neuron_attributes(self):
        attr = ['graph', 'simple', 'dps', 'annotations', 'partners',
                'review_status', 'nodes', 'connectors', 'presynapses',
                'postsynapses', 'gap_junctions', 'segments', 'soma',
                'root', 'tags', 'n_open_ends', 'n_end_nodes', 'n_connectors',
                'n_presynapses', 'n_postsynapses', 'cable_length']

        for a in attr:
            _ = getattr(self.nl[0], a)

    @try_conditions
    def test_neuron_functions(self):
        n = self.nl[0]
        slab = n.nodes[n.nodes.type == 'slab'].sample(1).iloc[0].node_id

        self.assertIsInstance(n.prune_distal_to(slab, inplace=False),
                              pymaid.CatmaidNeuron)
        self.assertIsInstance(n.prune_proximal_to(slab, inplace=False),
                              pymaid.CatmaidNeuron)
        self.assertIsInstance(n.prune_by_longest_neurite(inplace=False),
                              pymaid.CatmaidNeuron)

        self.assertIsInstance(n.reload(),
                              type(None))

    @try_conditions
    def test_swc_io(self):
        n = self.nl[0]

        n.to_swc('neuron.swc')

        self.assertIsInstance(pymaid.CatmaidNeuron.from_swc('neuron.swc'),
                              pymaid.CatmaidNeuron)

    @try_conditions
    def test_json_io(self):
        n_string = pymaid.neuron2json(self.nl[:2])

        self.assertIsInstance(n_string, str)

        n = pymaid.json2neuron(n_string)

        self.assertIsInstance(n, pymaid.CatmaidNeuronList)

    @try_conditions
    def test_selection_io(self):
        self.nl.to_selection('selection.json')

        n = pymaid.CatmaidNeuronList.from_selection('selection.json')

        self.assertIsInstance(n, pymaid.CatmaidNeuronList)

    @try_conditions
    def test_has_annotation(self):
        self.assertIsInstance(self.nl.has_annotation(self.nl.annotations[0][0]),
                              pymaid.CatmaidNeuronList)

    @try_conditions
    def test_remove_duplicates(self):
        t = pymaid.CatmaidNeuronList([self.nl[0], self.nl[0], self.nl[0]])
        t2 = t.remove_duplicates(inplace=False)
        self.assertLess(len(t2), len(t))


class TestMorpho(unittest.TestCase):
    """ Test morphological operations """

    def try_conditions(func):
        """Runs each test under various conditions and asserts that results
        are always the same."""

        def wrapper(self, *args, **kwargs):
            pymaid.config.use_igraph = False
            res1 = func(self, *args, **kwargs)
            if igraph:
                pymaid.config.use_igraph = True
                res2 = func(self, *args, **kwargs)
                self.assertEqual(res1, res2)
            return res1
        return wrapper

    def setUp(self):
        self.rm = pymaid.CatmaidInstance(
            config_test.server_url,
            config_test.http_user,
            config_test.http_pw,
            config_test.token)

        self.nl = pymaid.get_neuron(config_test.test_skids,
                                    remote_instance=self.rm)

    @try_conditions
    def test_downsampling(self):
        nl2 = self.nl.copy()
        nl2.downsample(4)
        self.assertLess(nl2.n_nodes.sum(), self.nl.n_nodes.sum())

    @try_conditions
    def test_resampling(self):
        nl2 = self.nl.resample(10000, inplace=False)
        self.assertNotEqual(nl2.n_nodes.sum(), self.nl.n_nodes.sum())

    @try_conditions
    def test_prune_by_strahler(self):
        nl2 = self.nl.prune_by_strahler(inplace=False, to_prune=1)
        self.assertLess(nl2.n_nodes.sum(), self.nl.n_nodes.sum())

    @try_conditions
    def test_axon_dendrite_split(self):
        self.assertIsInstance(pymaid.split_axon_dendrite(self.nl[0]),
                              pymaid.CatmaidNeuronList)

    @try_conditions
    def test_segregation_index(self):
        self.assertIsInstance(pymaid.segregation_index(self.nl[0]),
                              float)

    @try_conditions
    def test_bending_flow(self):
        self.assertIsInstance(pymaid.bending_flow(self.nl[0]),
                              type(None))

    @try_conditions
    def test_flow_centrality(self):
        self.assertIsInstance(pymaid.flow_centrality(self.nl[0]),
                              type(None))

    @try_conditions
    def test_stitching(self):
        self.assertIsInstance(pymaid.stitch_neurons(self.nl[:2],
                                                    method='NONE'),
                              pymaid.CatmaidNeuron)
        self.assertIsInstance(pymaid.stitch_neurons(self.nl[:2],
                                                    method='LEAFS'),
                              pymaid.CatmaidNeuron)

    @try_conditions
    def test_averaging(self):
        self.assertIsInstance(pymaid.average_neurons(self.nl[:2]),
                              pymaid.CatmaidNeuron)

    @try_conditions
    def test_tortuosity(self):
        self.assertIsInstance(pymaid.tortuosity(self.nl[0]),
                              float)
        self.assertIsInstance(pymaid.tortuosity(self.nl),
                              pd.DataFrame)

    @try_conditions
    def test_arbor_confidence(self):
        self.assertIsInstance(pymaid.arbor_confidence(self.nl[0],
                                                      inplace=False),
                              pymaid.CatmaidNeuron)

    @try_conditions
    def test_calc_cable(self):
        self.assertIsNotNone(pymaid.calc_cable(self.nl[0]))

    @try_conditions
    def test_remove_branches(self):
        self.assertIsInstance(pymaid.remove_tagged_branches(self.nl[0],
                                                            'not a branch',
                                                            how='segment',
                                                            preserve_connectors=True,
                                                            inplace=False),
                              pymaid.CatmaidNeuron)

        self.assertIsInstance(pymaid.remove_tagged_branches(self.nl[0],
                                                            'not a branch',
                                                            how='distal',
                                                            preserve_connectors=True,
                                                            inplace=False),
                              pymaid.CatmaidNeuron)

    @try_conditions
    def test_despike_neuron(self):
        self.assertIsInstance(pymaid.despike_neuron(self.nl[0],
                                                    inplace=False),
                              pymaid.CatmaidNeuron)

    @try_conditions
    def test_guess_radius(self):
        self.assertIsInstance(pymaid.guess_radius(self.nl[0],
                                                  inplace=False),
                              pymaid.CatmaidNeuron)


class TestGraphs(unittest.TestCase):
    """Test pymaid.graph and pymaid.graph_utils """

    def try_conditions(func):
        """Runs each test under various conditions and asserts that results
        are always the same."""

        def wrapper(self, *args, **kwargs):
            pymaid.config.use_igraph = False
            res1 = func(self, *args, **kwargs)
            if igraph:
                pymaid.config.use_igraph = True
                res2 = func(self, *args, **kwargs)
                self.assertEqual(res1, res2)
            return res1
        return wrapper

    def setUp(self):
        self.rm = pymaid.CatmaidInstance(
            config_test.server_url,
            config_test.http_user,
            config_test.http_pw,
            config_test.token)

        self.nl = pymaid.get_neuron(config_test.test_skids[0:2],
                                    remote_instance=self.rm)

        self.n = self.nl[0]
        self.n.reroot(self.n.soma)

        # Get some random leaf node
        self.leaf_id = self.n.nodes[self.n.nodes.type == 'end'].sample(
            1).iloc[0].node_id
        self.slab_id = self.n.nodes[self.n.nodes.type == 'slab'].sample(
            1).iloc[0].node_id

    @try_conditions
    def test_reroot(self):
        self.assertIsNotNone(self.n.reroot(self.leaf_id, inplace=False))
        self.assertIsNotNone(self.nl.reroot(self.nl.soma, inplace=False))

    @try_conditions
    def test_distal_to(self):
        self.assertTrue(pymaid.distal_to(self.n, self.leaf_id, self.n.root))
        self.assertFalse(pymaid.distal_to(self.n, self.n.root, self.leaf_id))

    @try_conditions
    def test_distance(self):
        leaf_id = self.n.nodes[self.n.nodes.type == 'end'].iloc[0].node_id

        self.assertIsNotNone(pymaid.dist_between(self.n,
                                                 leaf_id,
                                                 self.n.root))
        self.assertIsNotNone(pymaid.dist_between(self.n,
                                                 self.n.root,
                                                 leaf_id))

    @try_conditions
    def test_find_bp(self):
        self.assertIsNotNone(pymaid.find_main_branchpoint(self.n,
                                                          reroot_to_soma=False))

    @try_conditions
    def test_split_fragments(self):
        self.assertIsNotNone(pymaid.split_into_fragments(self.n,
                                                         n=2,
                                                         reroot_to_soma=False))

    @try_conditions
    def test_longest_neurite(self):
        self.assertIsNotNone(pymaid.longest_neurite(self.n,
                                                    n=2,
                                                    reroot_to_soma=False))

    @try_conditions
    def test_cut_neuron(self):
        dist, prox = pymaid.cut_neuron(
            self.n,
            self.slab_id,
        )
        self.assertNotEqual(dist.nodes.shape, prox.nodes.shape)

        # Make sure dist and prox check out
        self.assertTrue(pymaid.distal_to(self.n, dist.root, prox.root))

    @try_conditions
    def test_subset(self):
        self.assertIsInstance(pymaid.subset_neuron(self.n,
                                                   self.n.segments[0]),
                              pymaid.CatmaidNeuron)

    @try_conditions
    def test_node_sorting(self):
        self.assertIsInstance(pymaid.node_label_sorting(self.n),
                              list)


class TestConnectivity(unittest.TestCase):
    """Test pymaid.plotting """

    def try_conditions(func):
        """Runs each test under various conditions and asserts that results
        are always the same."""

        def wrapper(self, *args, **kwargs):
            pymaid.config.use_igraph = False
            res1 = func(self, *args, **kwargs)
            if igraph:
                pymaid.config.use_igraph = True
                res2 = func(self, *args, **kwargs)
                self.assertEqual(res1, res2)
            return res1
        return wrapper

    def setUp(self):
        self.rm = pymaid.CatmaidInstance(
            config_test.server_url,
            config_test.http_user,
            config_test.http_pw,
            config_test.token)

        self.n = pymaid.get_neuron(config_test.test_skids[0],
                                   remote_instance=self.rm)

        self.cn_table = pymaid.get_partners(config_test.test_skids[0],
                                            remote_instance=self.rm)

        self.nB = pymaid.get_neuron(self.cn_table.iloc[0].skeleton_id,
                                    remote_instance=self.rm)

        self.adj = pymaid.adjacency_matrix(
            self.cn_table[self.cn_table.relation == 'upstream'].iloc[:10].skeleton_id.values)

    @try_conditions
    def test_adjacency_matrix(self):
        self.assertIsInstance(self.adj, pd.DataFrame)

    @try_conditions
    def test_adjacency_matrix2(self):
        nl = pymaid.get_neurons(
            self.cn_table[self.cn_table.relation == 'upstream'].iloc[:10].skeleton_id.values)
        self.assertIsInstance(pymaid.adjacency_matrix(nl, use_connectors=True),
                              pd.DataFrame)

    @try_conditions
    def test_group_matrix(self):
        gr_adj = pymaid.group_matrix(self.adj,
                                     row_groups={n: 'group1' for n in self.adj.index.values})
        self.assertIsInstance(gr_adj, pd.DataFrame)

    @try_conditions
    def test_connectivity_filter(self):
        dist, prox = pymaid.cut_neuron(
            self.n, pymaid.find_main_branchpoint(self.n))

        vol = pymaid.get_volume(config_test.test_volume)

        # Connectivity table by neuron
        self.assertIsInstance(pymaid.filter_connectivity(self.cn_table, prox),
                              pd.DataFrame)
        # Adjacency matrix by neuron
        self.assertIsInstance(pymaid.filter_connectivity(self.adj, prox),
                              pd.DataFrame)

        # Connectivity table by volume
        self.assertIsInstance(pymaid.filter_connectivity(self.cn_table, vol),
                              pd.DataFrame)
        # Adjacency matrix by volume
        self.assertIsInstance(pymaid.filter_connectivity(self.adj, vol),
                              pd.DataFrame)

    @try_conditions
    def test_calc_overlap(self):
        self.assertIsInstance(pymaid.cable_overlap(self.n, self.nB),
                              pd.DataFrame)

    @try_conditions
    def test_pred_connectivity(self):
        self.assertIsInstance(pymaid.predict_connectivity(self.n,
                                                          self.nB,
                                                          remote_instance=self.rm),
                              pd.DataFrame)

    @try_conditions
    def test_cn_table_from_connectors(self):
        self.assertIsInstance(pymaid.cn_table_from_connectors(self.n,
                                                              remote_instance=self.rm),
                              pd.DataFrame)

    @try_conditions
    def test_adjacency_from_connectors(self):
        nl = pymaid.get_neurons(
            self.cn_table[self.cn_table.relation == 'upstream'].iloc[:10].skeleton_id.values)
        self.assertIsInstance(pymaid.adjacency_from_connectors(nl,
                                                               remote_instance=self.rm),
                              pd.DataFrame)


class TestCluster(unittest.TestCase):
    """Test pymaid.cluster """

    def try_conditions(func):
        """Runs each test under various conditions and asserts that results
        are always the same."""

        def wrapper(self, *args, **kwargs):
            pymaid.config.use_igraph = False
            res1 = func(self, *args, **kwargs)
            if igraph:
                pymaid.config.use_igraph = True
                res2 = func(self, *args, **kwargs)
                self.assertEqual(res1, res2)
            return res1
        return wrapper

    def setUp(self):
        self.rm = pymaid.CatmaidInstance(
            config_test.server_url,
            config_test.http_user,
            config_test.http_pw,
            config_test.token)

    @try_conditions
    def test_connectivity_cluster(self):
        self.assertIsInstance(pymaid.cluster_by_connectivity(config_test.test_skids),
                              pymaid.ClustResults)

    @try_conditions
    def test_synapse_cluster(self):
        self.assertIsInstance(pymaid.cluster_by_synapse_placement(config_test.test_skids),
                              pymaid.ClustResults)

    @try_conditions
    def test_clustresults(self):
        res = pymaid.cluster_by_connectivity(config_test.test_skids)

        self.assertIsNotNone(res.linkage, np.ndarray)
        self.assertIsNotNone(res.leafs)
        self.assertIsNotNone(res.cophenet)
        self.assertIsNotNone(res.agg_coeff)
        res.to_selection()
        self.assertIsInstance(res.get_colormap(k=2), dict)
        self.assertIsInstance(res.get_clusters(k=2), list)


class TestPlot(unittest.TestCase):
    """Test pymaid.plotting """

    def try_conditions(func):
        """Runs each test under various conditions and asserts that results
        are always the same."""

        def wrapper(self, *args, **kwargs):
            pymaid.config.use_igraph = False
            res1 = func(self, *args, **kwargs)
            if igraph:
                pymaid.config.use_igraph = True
                res2 = func(self, *args, **kwargs)
                self.assertEqual(res1, res2)
            return res1
        return wrapper

    def setUp(self):
        self.rm = pymaid.CatmaidInstance(
            config_test.server_url,
            config_test.http_user,
            config_test.http_pw,
            config_test.token)

        self.nl = pymaid.get_neuron(config_test.test_skids,
                                    remote_instance=self.rm)

        self.vol = pymaid.get_volume(config_test.test_volume)

    """
    @try_conditions
    def test_plot3d_plotly(self):
        self.assertIsNotNone(self.nl.plot3d(backend='plotly'))
        self.assertIsNotNone(pymaid.plot3d(self.nl, backend='plotly'))
        self.assertIsNotNone(pymaid.plot3d(
            [self.nl, self.vol], backend='plotly'))

    @try_conditions
    def test_plot3d_vispy(self):
        # self.assertIsNotNone(self.nl.plot3d(backend='vispy'))
        # pymaid.close3d()
        # self.assertIsNotNone(pymaid.plot3d(self.nl, backend='vispy'))
        # pymaid.close3d()
        # self.assertIsNotNone(pymaid.plot3d([self.nl, self.vol],
        #                                   backend='vispy'))
        # pymaid.close3d()
        self.assertIsNotNone(pymaid.plotting._neuron2vispy(self.nl))
        self.assertIsNotNone(pymaid.plotting._volume2vispy(self.vol))
    """

    @try_conditions
    def test_plot2d(self):
        plt.close()
        self.assertIsNotNone(self.nl.plot2d(method='2d'))
        plt.close()
        self.assertIsNotNone(self.nl.plot2d(method='3d_complex'))
        plt.close()
        self.assertIsNotNone(self.nl.plot2d(method='3d'))
        plt.close()

    def tearDown(self):
        plt.close()


class TestTiles(unittest.TestCase):
    """Test pymaid.tiles """

    def setUp(self):
        self.rm = pymaid.CatmaidInstance(config_test.server_url,
                                         config_test.http_user,
                                         config_test.http_pw,
                                         config_test.token)

    def test_tiles(self):
        from pymaid import tiles
        # Generate the job
        job = tiles.LoadTiles([119000, 119500, 36000, 36500, 4050],
                              stack_id=5,
                              coords='PIXEL')
        # Load, stich and crop the required EM image tiles
        job.load_in_memory()
        # Render image
        ax = job.render_im(slider=False, figsize=(12, 12))
        # Add treenodes
        job.render_nodes(ax, treenodes=True, connectors=False)
        # Add scalebar
        job.scalebar(size=1000, ax=ax, label=False)
        # Show
        plt.close()

    def tearDown(self):
        plt.close()


class TestUserStats(unittest.TestCase):
    """Test pymaid.user_stats """

    def setUp(self):
        self.rm = pymaid.CatmaidInstance(
            config_test.server_url,
            config_test.http_user,
            config_test.http_pw,
            config_test.token)

        self.n = pymaid.get_neuron(config_test.test_skids[0],
                                   remote_instance=self.rm)

    def test_time_invested(self):
        ds = self.n.downsample(20, inplace=False)
        self.assertIsInstance(pymaid.get_time_invested(ds,
                                                       mode='SUM',
                                                       remote_instance=self.rm),
                              pd.DataFrame)
        self.assertIsInstance(pymaid.get_time_invested(ds,
                                                       mode='ACTIONS',
                                                       remote_instance=self.rm),
                              pd.DataFrame)
        self.assertIsInstance(pymaid.get_time_invested(ds,
                                                       mode='OVER_TIME',
                                                       remote_instance=self.rm),
                              pd.DataFrame)

    def test_team_contributions(self):
        ds = self.n.downsample(20, inplace=False)
        ul = pymaid.get_user_list().set_index('id')
        teams = {'test_team' : [ul.loc[u, 'login'] for u in ds.nodes.creator_id.unique()]}
        self.assertIsInstance(pymaid.get_team_contributions(teams,
                                                            neurons=ds,
                                                            remote_instance=self.rm),
                              pd.DataFrame)

    def test_user_contributions(self):
        self.assertIsInstance(pymaid.get_user_contributions(
            config_test.test_skids, remote_instance=self.rm), pd.DataFrame)

    def test_user_actions(self):
        # Get a day on which a random user has done some work
        last_week = datetime.date.today() - datetime.timedelta(days=7)
        h = pymaid.get_history(start_date=last_week,
                               end_date=datetime.date.today())
        u = h.treenodes.loc[h.treenodes.sum(axis=1) > 10].sample(1).index[0]

        user_actions = pymaid.get_user_actions(users=u, start_date=last_week,
                                               end_date=datetime.date.today())

        self.assertIsInstance(user_actions, pd.DataFrame)


class TestExamples(unittest.TestCase):
    """Test pymaid.tiles """

    def setUp(self):
        self.rm = pymaid.CatmaidInstance(config_test.server_url,
                                         config_test.http_user,
                                         config_test.http_pw,
                                         config_test.token)

    def test_fetch_examples(self):
        for func in pymaid.fetch.__all__:
            f = getattr(pymaid.fetch, func)
            doctest.run_docstring_examples(f, globals(), name=f)

    def test_user_stats_examples(self):
        for func in pymaid.user_stats.__all__:
            f = getattr(pymaid.user_stats, func)
            doctest.run_docstring_examples(f, globals(), name=f)


if __name__ == '__main__':
    unittest.main()
