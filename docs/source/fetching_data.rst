Fetching data from the server
*****************************
With minor exceptions, pymaid covers every API endpoint of your CATMAID server - i.e. you should be able to get all the data that your webbrowser has access to.

Parsing skeleton IDs
====================
All functions that require neurons/skeleton IDs as inputs (e.g. :func:`~pymaid.get_neuron`) accept either:

1. skeleton ID(s) (int or str)::

    n = pymaid.get_neuron(16)
    nl = pymaid.get_neuron([16, 57241])

2. neuron name(s) (str)::

    n = pymaid.get_neuron('PN glomerulus DC3 57242 ML')
    nl = pymaid.get_neuron(['PN glomerulus DC3 57242 ML', 'PN glomerulus VA6 017 DB'])

3. annotation(s) (str)::

    n = pymaid.get_neuron('annotation:glomerulus VA6')
    nl = pymaid.get_neuron(['annotation:glomerulus DC3', 'annotation:glomerulus VA6'])

4. CatmaidNeuron or CatmaidNeuronList object::

    n = pymaid.get_neuron(16)
    cn_table = pymaid.get_partners(n)

By default, annotations and neuron names must match exactly. You can use regex on names and annotations, using a leading ``/``::

    # Get all neurons that have 'VA6' in their name
    nl = pymaid.get_neuron('/VA6')

    # Get all neurons annotated with a 'VA6'-containing annotation
    nl = pymaid.get_neuron('annotation:/VA6')

    # Get all skeleton IDs that have an annotation starting with 'AV1.R':
    skids = pymaid.get_skids_by_annotation('/^AV1.R')


See `here <https://medium.com/factory-mind/regex-tutorial-a-simple-cheatsheet-by-examples-649dc1c3f285>`_ for detailed explanation of regex patterns.


Functions for fetching data
===========================

Neurons
-------
.. autosummary::
    :toctree: generated/

    ~pymaid.get_neuron
    ~pymaid.delete_neuron
    ~pymaid.find_neurons
    ~pymaid.get_arbor
    ~pymaid.get_neurons_in_volume
    ~pymaid.get_neuron_list
    ~pymaid.get_skids_by_annotation
    ~pymaid.get_skids_by_name
    ~pymaid.rename_neurons
    ~pymaid.get_names

Annotations
-----------
.. autosummary::
    :toctree: generated/

    ~pymaid.add_annotations
    ~pymaid.get_annotations
    ~pymaid.get_annotation_details
    ~pymaid.get_user_annotations
    ~pymaid.remove_annotations

Treenodes
----------
.. autosummary::
    :toctree: generated/

    ~pymaid.get_treenode_table
    ~pymaid.get_treenode_info
    ~pymaid.get_skid_from_treenode
    ~pymaid.get_node_details

Tags
----
.. autosummary::
    :toctree: generated/

    ~pymaid.get_label_list
    ~pymaid.add_tags
    ~pymaid.delete_tags
    ~pymaid.get_node_tags

Connectivity
------------
.. autosummary::
    :toctree: generated/

    ~pymaid.get_connectors
    ~pymaid.get_connector_details
    ~pymaid.get_connectors_between
    ~pymaid.get_edges
    ~pymaid.get_partners
    ~pymaid.get_partners_in_volume
    ~pymaid.get_paths
    ~pymaid.get_connector_links

User stats
----------
.. autosummary::
    :toctree: generated/

    ~pymaid.get_user_list
    ~pymaid.get_history
    ~pymaid.get_time_invested
    ~pymaid.get_user_contributions
    ~pymaid.get_contributor_statistics
    ~pymaid.get_logs
    ~pymaid.get_transactions

Volumes
-------
.. autosummary::
    :toctree: generated/

    ~pymaid.get_volume

Misc
----
.. autosummary::
    :toctree: generated/

    ~pymaid.CatmaidInstance
    ~pymaid.url_to_coordinates
    ~pymaid.get_review
    ~pymaid.get_review_details
