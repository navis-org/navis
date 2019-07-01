.. _overview_link:

Neuron objects
==============

Single neurons and lists of neurons are represented in ``navis`` by:

.. autosummary::
    :toctree: generated/

 	~navis.TreeNeuron
 	~navis.NeuronList

Navis comes with a couple example neurons from the FAFB project published
in `Zheng et al <http://www.cell.com/cell/retrieve/pii/S0092867418307876?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS0092867418307876%3Fshowall%3Dtrue>`_
, Cell (2018)::

	>>> n = navis.example_neurons(1)
	>>> n
	type              TreeNeuron
	name            neuron_38885
	n_nodes                 6365
	n_connectors               0
	n_branches               235
	n_leafs                  243
	cable_length     1.21335e+06
	soma                  [3490]
	dtype: object

:class:`~navis.TreeNeuron` stores nodes and other data as attached DataFrames:

	>>> n.nodes.head()
	   node_id label         x         y         z  radius  parent_id  type
	0        1     0  489192.0  186290.0  138040.0     0.0         -1  root
	1        2     0  489233.0  186198.0  138080.0     0.0          1  slab
	2        3     0  489235.0  186174.0  138120.0     0.0          2  slab
	3        4     0  489229.0  186200.0  138160.0     0.0          3  slab
	4        5     0  489251.0  186141.0  138200.0     0.0          4  slab



List of neurons are represented as :class:`~navis.NeuronList`::

	>>> nl = navis.example_neurons(2)
	>>> nl
	<class 'navis.core.neuronlist.NeuronList'> of 2 neurons
	          type  n_nodes  n_connectors  n_branches  n_leafs  cable_length    soma
	0  TreeNeuron     6365             0         235      243  1.213347e+06  [3490]
	1  TreeNeuron     8187             0         305      322  1.447682e+06  [3604]

A :class:`~navis.NeuronList` works similar to normal lists with a few
additional perks::

	>>> nl[0]
	type              TreeNeuron
	name            neuron_38885
	n_nodes                 6365
	n_connectors               0
	n_branches               235
	n_leafs                  243
	cable_length     1.21335e+06
	soma                  [3490]
	dtype: object

They allow easy and fast access to data across all neurons::

	>>> nl.n_nodes
	array([6365, 8187])

	>>> nl.cable_length
	array([1213347.43506801, 1447681.63642537])


In addition to these **attributes**, both :class:`~navis.TreeNeuron` and
:class:`~navis.NeuronList` have shortcuts (called **methods**) to
other navis functions. These lines of code are equivalent::

	>>> n.reroot(n.soma, inplace=True)
	>>> navis.reroot_neuron(n, n.soma, inplace=True)

	>>> n.plot3d(color='red')
	>>> navis.plot3d(n, color='red')

	>>> n.prune_by_volume('LH_R', inplace=True)
	>>> navis.in_volume(n, 'LH_R', inplace=True)

The ``inplace`` parameter is part of many navis functions and works like that
in the excellent pandas library. If ``inplace=True`` operations are performed
on the original. If ``inplace=False`` operations are performed on a copy of the
original which is then returned::

	>>> n = navis.example_neurons(1)
	>>> lh = navis.example_volume('LH')
	>>> n_lh = n.prune_by_volume(lh, inplace=False)
	>>> n.n_nodes, n_lh.n_nodes
	(6365, 1299)

Please see other sections and the docstrings of
:class:`~navis.TreeNeuron` and :class:`~navis.NeuronList` for
more examples.

Neuron attributes
-----------------

This is a *selection* of :class:`~navis.TreeNeuron` and
:class:`~navis.NeuronList` class attributes:

- ``uuid``: a unique identified
- ``nodes``: node table
- ``connectors``: connector table (optional)
- ``cable_length``: cable length(s)
- ``soma``: node ID(s) of soma (if applicable)
- ``root``: root node ID(s)
- ``segments``: list of linear segments
- ``graph``: NetworkX graph representation of the neuron
- ``igraph``: iGraph representation of the neuron (if library available)
- ``dps``: Dotproduct representation of the neuron

All attributes are accessible through auto-completion.

Reference
---------

See :class:`~navis.TreeNeuron` or :ref:`API <api_neurons>`.
