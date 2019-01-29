Connectivity analyses
+++++++++++++++++++++

Getting connectivity information for a set of neurons is pretty straight
forward. You can get connectivity tables, edges between neurons, connectors
between neurons. Check out functions below and in "Fetching data from the
server"!

Here, we will focus on more complex examples: restricting connectivity
to specific parts of a neuron (e.g. dendrite vs axon) or to a given volume
(e.g. the lateral horn).

.. important::
   Pymaid (like CATMAID) uses the skeleton ID to *uniquely* identify a neuron.
   If you break a neuron into fragments (e.g. axon + dendrites) and you pass
   both as neuronlist to a basic function (e.g. :func:`~pymaid.get_partners`),
   pymaid ignores the fact that the same skeleton ID exists twice. This is
   also true for functions that respect a neuron's morphology (e.g.
   :func:`~pymaid.filter_connectivity`): *they work fine with a single fragment
   per neuron* but if you pass multiple fragments of the same neuron, they will
   collapse these duplicate skeleton IDs back into a single neuron. There are
   two ways to deal with this.

   **Option A**: run your analysis on each fragment separately and merge results
   at the end.

   **Option B**: use the "fragment-safe" functions
   :func:`~pymaid.adjacency_from_connectors` or
   :func:`~pymaid.cn_table_from_connectors`. Both functions are able to deal
   with non-unique skeleton IDs. Because they bypass higher level CATMAID API
   and regenerate connectivity from scratch they are somewhat slow though.

Let's start with an easy example: Subset connectivity to the axons of a set of
neurons (no duplicate skeleton IDs)

>>> # Get a set of projection neurons
>>> nl = pymaid.get_neurons('annotation:glomerulus DA1 right excitatory')
>>> # Split get only the axonal branches in the laterl horn using a tag
>>> nl.reroot(nl.soma)
>>> nl_axon = nl.prune_proximal_to('SCHLEGEL_LH', inplace=False)
>>> # Get a list of the all partners (axon + dendrites)
>>> # get_partners() does not care about whether neurons are pruned
>>> cn_table = pymaid.get_partners(nl_axon)
>>> # Subset connectivity to just the LH
>>> cn_lh = pymaid.filter_connectivity(cn_table, nl_axon)
>>> cn_lh.head()

The second example is more complex: we will split neurons into axon/dendrites
and compare their connectivity (duplicate skeleton IDs!)

>>> import pandas as pd
>>> import seaborn as sns
>>> import matplotlib.pyplot as plt
>>> import numpy as np
>>> # Get a set of neurons
>>> nl = pymaid.get_neurons('annotation:PD2a1/b1')
>>> # Split into axon dendrite by using synapse flow
>>> nl.reroot(nl.soma)
>>> nl_split = pymaid.split_axon_dendrite(nl)
>>> # Get a list of partners
>>> cn_table = pymaid.get_partners(nl)
>>> ds_partners = cn_table[ cn_table.relation == 'downstream' ]
>>> us_partners = cn_table[ cn_table.relation == 'upstream' ]
>>> # Take the top 10 up- and downstream partners
>>> top10 = np.append(ds_partners.iloc[:10].skeleton_id.values,
...                   us_partners.iloc[:10].skeleton_id.values)
>>> # We have to retrieve full skeletons for these partners
>>> top10 = pymaid.get_neurons(top10)
>>> # Now generate an adjaceny matrix
>>> adj = pymaid.adjacency_from_connectors(nl_split, top10)
>>> # Matrix is labeled by skeleton ID but the order of both
>>> # neuronlists is preserved. Let's rename columns and rows
>>> adj.columns = top10.neuron_name
>>> adj.index = nl_split.neuron_name
>>> # Plot heatmap using seaborn
>>> ax = sns.heatmap(adj)
>>> plt.show()

Above example illustrate how to subset connectivity to given part(s) of a
neuron. Subsetting to a volume is even easier:
Following up on above example, we will next subset the connectivity table to
connections in a given CATMAID volume:

>>> # Get a CATMAID volume
>>> vol = pymaid.get_volume('LH_R')
>>> cn_table_lh = pymaid.filter_connectivity(cn_table, vol)

Reference
=========

Connectivity table
------------------
.. autosummary::
    :toctree: generated/

    ~pymaid.get_partners
    ~pymaid.cn_table_from_connectors

Graphs
------
.. autosummary::
    :toctree: generated/

    ~pymaid.neuron2nx
    ~pymaid.neuron2igraph
    ~pymaid.neuron2KDTree
    ~pymaid.network2nx
    ~pymaid.network2igraph

Predict connectivity
--------------------
.. autosummary::
    :toctree: generated/

	~pymaid.predict_connectivity

Matrices
--------
.. autosummary::
    :toctree: generated/

    ~pymaid.adjacency_matrix
    ~pymaid.group_matrix
    ~pymaid.adjacency_from_connectors

Clustering
----------
.. autosummary::
    :toctree: generated/

    ~pymaid.cluster_by_connectivity
    ~pymaid.cluster_by_synapse_placement
    ~pymaid.ClustResults

Plotting
--------
.. autosummary::
    :toctree: generated/

    ~pymaid.plot_network

Filtering/Subsetting
--------------------
.. autosummary::
    :toctree: generated/

	~pymaid.filter_connectivity
    ~pymaid.cn_table_from_connectors
    ~pymaid.adjacency_from_connectors
