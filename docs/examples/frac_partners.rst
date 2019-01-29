Fraction of inputs
------------------

In this code snippet we will take a look at the fraction of inputs to a set of neurons.

First, load all packages and initialise a CATMAID remote instance.

>>> import pymaid
>>> from matplotlib.pyplot import plt
>>> rm = pymaid.CatmaidInstances( 'server_url',
...                               'http_user',
...                               'http_password',
...                               'auth_token')


Next, load some lateral horn neurons and get their connectivity table.

>>> lhns = pymaid.find_neurons(annotations='example_neurons')
>>> cn_table = pymaid.get_partners(lhns, min_size=1)
>>> # Have a look at the table
>>> cn_table.head()

Let's check fraction of inputs from projection neurons

>>> # Get PNs as (mostly) empty CatmaidNeuronList
>>> pns = pymaid.find_neurons(annotations='uPN right')
>>> # Subset connectivity table by inputs
>>> inputs = cn_table[ cn_table.relation == 'upstream' ]

``cn_table`` and its subset ``inputs`` are Pandas DataFrames which come with some neat ways to group data.
We will write a small function that sorts neurons into "PNs" and "other" and apply this to the inputs table:

>>> # Generate function that lets us sort the table
>>> def is_PN(x):
...     """ Returns True if PN """
...     if x.skeleton_id in pns.skeleton_id:
...         return 'PN'
...     else:
...         return 'other'
>>> # Sorting is based on the index -> we have to change the index to skeleton IDs
>>> inputs.set_index('skeleton_id', inplace=True, drop=True)
>>> # Now apply function explicitly to the columns that hold connectivity information
>>> grouped = inputs[lhns].groupby(by=is_PN).sum()

In above last line, we grouped by PNs/non-PNs and summed up the number of synapses. Instead of ``.sum()``
we can also use e.g. ``.mean()``, ``.std()`` or ``.min()``.

For now, let's stick with the sum and do some plotting. Conveniently, Pandas DataFrames also have some very
nice wrappers for visualisations:

>>> # Plot bar plots per neuron (notice that we transpose the table first with ".T")
>>> ax = grouped.T.plot(kind='bar')
>>> plt.show()
>>> # Plot a pie chart for fraction across all neurons
>>> ax = grouped.sum(axis=1).plot(kind='pie', autopct='%.2f', figsize=(6,6))
>>> plt.show()
