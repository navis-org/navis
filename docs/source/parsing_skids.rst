Parsing skeleton IDs
********************
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

Regex
-----

By default, annotations and neuron names must match exactly. You can use regex on names and annotations, using a leading ``/``::

    # Get all neurons that have 'VA6' in their name
    nl = pymaid.get_neuron('/VA6')

    # Get all neurons annotated with a 'VA6'-containing annotation
    nl = pymaid.get_neuron('annotation:/VA6')

    # Get all skeleton IDs that have an annotation starting with 'AV1.R':
    skids = pymaid.get_skids_by_annotation('/^AV1.R')


See `here <https://medium.com/factory-mind/regex-tutorial-a-simple-cheatsheet-by-examples-649dc1c3f285>`_ for detailed explanation of regex patterns.