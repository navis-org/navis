.. _overview_link:

Neuron objects
==============

Single neurons and lists of neurons are represented in ``pymaid`` by:

.. autosummary::
    :toctree: generated/

 	~pymaid.CatmaidNeuron
 	~pymaid.CatmaidNeuronList

They can be minimally initialized with just skeleton IDs. So you can do
something like this::

	>>> n = pymaid.CatmaidNeuron(16)
	>>> n
	type              <class 'pymaid.core.CatmaidNeuron'>
	neuron_name                                        NA
	skeleton_id                                        16
	n_nodes                                            NA
	n_connectors                                       NA
	n_branch_nodes                                     NA
	n_end_nodes                                        NA
	n_open_ends                                        NA
	cable_length                                       NA
	review_status                                      NA
	soma                                               NA
	dtype: object


This gives you an **empty** neuron (note ``NA`` entries) which you can
use to retrieve more data. Ordinarily, however, you would get neurons from
functions like :func:`~pymaid.get_neuron` that already contain some data::

	>>> n = pymaid.get_neuron(16)
	>>> n
	type              <class 'pymaid.core.CatmaidNeuron'>
	neuron_name                  PN glomerulus VA6 017 DB
	skeleton_id                                        16
	n_nodes                                         12743
	n_connectors                                     2028
	n_branch_nodes                                    774
	n_end_nodes                                       823
	n_open_ends                                       280
	cable_length                                  2866.11
	review_status                                      NA
	soma                                          2941309

:func:`pymaid.get_neuron` returned a :class:`~pymaid.CatmaidNeuron` with name,
nodes, connectors and tags::

	>>> n.nodes
	   treenode_id parent_id  creator_id       x       y       z  radius  \
	0     17909304  26337791         123  355710  153742  152440      -1
	1     26337791  26337787         117  355738  153802  152400      -1
	2     26337787   3148134         117  355790  153922  152320      -1
	3      6532788  25728462          94  349025  160905  154160      -1
	4     25728462   6532787         117  349059  160783  154360      -1

	   confidence  type
	0           5   end
	1           5  slab
	2           5  slab
	3           5  slab
	4           5  slab


Missing data, e.g. the review status or annotations, are either fetched and
stored automatically upon first accessing the respective attributes (see also
list below)::

	>>> n.annotations
	['test_cremi_c',
	 'WTPN2017_AL_PN',
	 ...
	 'right',
 	 'right excitatory']

Or by calling the respective function explicitly. This serves also to update
data from the server::

	>>> n.get_annotations()
	['test_cremi_c',
	 'WTPN2017_AL_PN',
	 ...
	 'right',
 	 'glomerulus DA1 right excitatory']


Functions such as :func:`~pymaid.get_neuron` return multiple neurons as
:class:`~pymaid.CatmaidNeuronList`::

	>>> nl = pymaid.get_neuron([16, 27295])
	>>> nl
	<class 'pymaid.core.CatmaidNeuronList'> of 2 neurons
                 	  neuron_name skeleton_id  n_nodes  n_connectors  \
	0    PN glomerulus VA6 017 DB          16    12743          2028
	1  PN glomerulus DA1 27296 BH       27295     9973           469

	   n_branch_nodes  n_end_nodes  open_ends  cable_length review_status  soma
	0             774          823        280   2866.105439            NA  True
	1             212          219         58   1591.519821            NA  True

A :class:`~pymaid.CatmaidNeuronList` works similar to normal lists with a few
additional perks::

	>>> nl[0]
	type              <class 'pymaid.core.CatmaidNeuron'>
	neuron_name                  PN glomerulus VA6 017 DB
	skeleton_id                                        16
	n_nodes                                         12743
	n_connectors                                     2028
	n_branch_nodes                                    774
	n_end_nodes                                       823
	n_open_ends                                       280
	cable_length                                  2866.11
	review_status                                      NA
	soma                                          2941309

	>>> nl.skid[27295]
	type              <class 'pymaid.core.CatmaidNeuron'>
	neuron_name                  PN glomerulus VA6 017 DB
	skeleton_id                                        16
	n_nodes                                         12743
	n_connectors                                     2028
	n_branch_nodes                                    774
	n_end_nodes                                       823
	n_open_ends                                       280
	cable_length                                  2866.11
	review_status                                      NA
	soma                                          2941309

	>>> nl.has_annotations('glomerulus VA6')
	<class 'pymaid.core.CatmaidNeuronList'> of 1 neurons
                 	  neuron_name skeleton_id  n_nodes  n_connectors  \
	0    PN glomerulus VA6 017 DB          16    12743          2028

	   n_branch_nodes  n_end_nodes  open_ends  cable_length review_status  soma
	0             774          823        280   2866.105439            NA  True


They allow easy and fast access to data across all neurons::

	>>> nl.skeleton_id
	array(['16', '27295'], dtype='<U5')

	>>> nl.cable_length
	array([2866.10543944, 1591.51982146])


In addition to these **attributes**, both :class:`~pymaid.CatmaidNeuron` and
:class:`~pymaid.CatmaidNeuronList` have shortcuts (called **methods**) to
other pymaid functions. These lines of code are equivalent::

	>>> n.reroot(n.soma, inplace=True)
	>>> pymaid.reroot_neuron(n, n.soma, inplace=True)

	>>> n.plot3d(color='red')
	>>> pymaid.plot3d(n, color='red')

	>>> n.prune_by_volume('LH_R', inplace=True)
	>>> pymaid.in_volume(n, 'LH_R', inplace=True)

The ``inplace`` parameter is part of many pymaid functions and works like that
in the excellent pandas library. If ``inplace=True`` operations are performed
on the original. Ff ``inplace=False`` operations are performed on a copy of the
original which is then returned::

	>>> n = pymaid.get_neuron(16)
	>>> n_lh = n.prune_by_volume('LH_R', inplace=False)
	>>> n.n_nodes, n_lh.n_nodes
	(12743, 3564)

Please see other sections and the docstrings of
:class:`~pymaid.CatmaidNeuron` and :class:`~pymaid.CatmaidNeuronList` for
more examples.

Neuron attributes
-----------------

This is a *selection* of :class:`~pymaid.CatmaidNeuron` and
:class:`~pymaid.CatmaidNeuronList` class attributes:

- ``skeleton_id``: neurons' skeleton ID(s)
- ``neuron_name``: neurons' name(s)
- ``nodes``: treenode table
- ``connectors``: connector table
- ``presynapses``: connector table for presynapses only
- ``postsynapses``: connector table for postsynapses only
- ``gap_junctions``: connector table for gap junctions only
- ``partners``: connectivity table
- ``tags``: node tags (dict)
- ``annotations``: list of neurons' annotations
- ``cable_length``: cable length(s) in nm
- ``review_status``: review status of neuron(s)
- ``soma``: treenode ID of soma (if applicable)
- ``root``: root treenode ID
- ``segments``: list of linear segments
- ``graph``: NetworkX graph representation of the neuron
- ``igraph``: iGraph representation of the neuron (if library available)
- ``dps``: Dotproduct representation of the neuron

All attributes are accessible through auto-completion.

Reference
---------

See :class:`~pymaid.CatmaidNeuron` or :ref:`API <api_neurons>`.
