"""
Lists of Neurons
================

This tutorial will show you how to use NeuronLists to efficiently work with many neurons at a time.

!!! note
    If you haven't please check out the [neuron types tutorial](../tutorial_basic_01_neurons) first.

{{ navis }} will typically collect multiple neurons into a [`navis.NeuronList`][] as container.
This container behaves like a mix of lists, numpy arrays and pandas dataframes, and allows you
to quickly sort, filter and manipulate neurons.
"""

# %%
# ## Overview

import navis

# Grab three example skeletons (TreeNeurons) as a NeuronList
nl = navis.example_neurons(n=3)
nl

# %%
# !!! note
#     Note how just printing `nl` at the end of the cell will produce a nice summary table?
#     If you want to get this table as pandas DataFrame, use the `summary()` method:
#     ```python
#     df = nl.summary()
#     ```

import matplotlib.pyplot as plt

navis.plot2d(nl, view=('x', '-z'), method='2d')
plt.tight_layout()

# %%
# ## Creating NeuronLists
#
# To create a [`NeuronList`][navis.NeuronList] from scratch simply pass a list of neurons to the constructor:

n = navis.example_neurons(n=1)
nl = navis.NeuronList([n, n, n])  # a list with 3x the same neuron
nl

# %%
# ## Accessing Neuron Attributes
#
# [`NeuronLists`][navis.NeuronList] give you quick and easy access to data and across all neurons:

# Get the number of nodes in the first skeleton
nl = navis.example_neurons(n=3)
nl[0].n_nodes

# %%
# Use the [`NeuronList`][navis.NeuronList] to collect number of nodes across all neurons:

nl.n_nodes

# %%
# This works on any neuron attribute:

nl.cable_length

# %%
# !!! note
#     The `n_{attribute}` pattern works with any "countable" neuron attributes like nodes, connectors, etc.
#
# If the neuron attribute is a dataframe, the [`NeuronList`](navis.NeuronList) will concatenate them and
# add a new column with the neuron ID:

nl.nodes  # note the `neuron` column

# %%
# [`NeuronLists`][navis.NeuronList] can also contain a mix of different neuron types:

nl_mix = navis.example_neurons(n=2, kind='mix')
nl_mix

# %%
# Note how `nl_mix` contains a [`TreeNeuron`][navis.TreeNeuron] and a [`MeshNeuron`][navis.MeshNeuron]?
#
# In such cases you have to be a bit more careful about asking for attributes that are not shared across all neurons:
#
# ```python
# # This will throw an error because MeshNeurons
# # don't have a `cable_length` attribute
# nl_mix.cable_length
# ```

# Instead use the `get_neuron_attributes()` method with a default value:
nl_mix.get_neuron_attributes('cable_length', None)


# %%
# ## Indexing NeuronLists
#
# A [`NeuronList`][navis.NeuronList] works similar to normal lists with a bunch of additional perks:

# Get the first neuron in the list
nl = navis.example_neurons(n=3)
nl[0]

# %%
# ### Index by position
#
# [`NeuronLists`][navis.NeuronList] are designed to behave similar to numpy arrays in that they allow some fancing
# indexing.
#
# You've already seen how to extract a single neuron from a [`NeuronList`][navis.NeuronList] using a single integer
# index. Like with numpy arrays, this also works for lists of indices...

# %%
nl = navis.example_neurons(n=3)
nl[[0, 2]]

# %%
# ... or slices

# %%
nl[:2]

# %%
# ### Index by attributes
#
# You can index [`NeuronLists`][navis.NeuronList] by boolean `numpy.arrays` - that includes
# neuron attributes, e.g. `n_nodes`, `cable_length`, `soma`, etc.
#
# Index using node count:

# %%
subset = nl[nl.n_branches > 700]
subset

# %%
# Here is an example where we subset to neurons that have a soma:

# %%
subset = nl[nl.soma != None]  # Index by boolean array
subset

# %%
# ### Index by name
#
# [`navis.TreeNeuron`][] can (but don't have to) have names (`.name`). If you, for example,
# import neurons from SWC files they will inherit their name from the file by default.
#
# Our example neurons all have the same name, so to demo this feature we will need to make
# those names unique:

# %%
for i, n in enumerate(nl):
    n.name = n.name + str(i + 1)
nl

# %%
# You can index by single...

# %%
nl["DA1_lPN_R1"]

# %%
# ... or multiple names:

# %%
nl[["DA1_lPN_R1", "DA1_lPN_R2"]]

# %%
# #### Using regex
#
# Under the hood {{ navis }} uses `re.fullmatch` to match neuron names - so you can use regex!

# %%
nl[".*DA1.*"]

# %%
# ### Index by ID
#
# All neurons have an ID - even if you don't explicitly assign one, a UUID will assigned under the hood.

# %%
nl[0].id

# %%
# Neuron lists can be indexed by their ID (similar to `.loc[]` in pandas DataFrames) by using the `.idx` indexer:

# %%
nl.idx[1734350908]

# %%
# ## Neuron Math
#
# {{ navis }} implements a very simple and intuitive syntax to add and remove items from a [`navis.NeuronList`][]:
#
# ### Addition
#
# To merge two lists in Python, you can simply add them:

# %%
[1] + [3]

# %%
# [`navis.NeuronList`][] works exactly the same:

# %%
nl[:2] + nl[2:]

# %%
# This also works on with two single [`navis.TreeNeurons`][navis.TreeNeuron]! You can use that to combine them into a list:

# %%
nl[0] + nl[1]

# %%
# ### Substraction
#
# To remove an item from a Python list, you would call the `.pop()` method:

# %%
l = [1, 2, 3]
l.pop(2)
l

# %%
# For [`navis.NeuronList`][] you can use substraction:

# %%
nl - nl[2]

# %%
# ### Bitwise AND
#
# To find the intersection between two lists, you would use `sets` and the `&` operator:

# %%
set([0, 1, 2]) & set([2, 3, 4])

# %%
# [`navis.NeuronList`][] work similarly:

# %%
nl[[0, 1]] & nl[[1, 2]]

# %%
# ### Bitwise OR
#
# To generate the union between two lists, you would use `sets` and the `|` operator:

# %%
set([0, 1, 2]) | set([2, 3, 4])

# %%
# [`navis.NeuronLists`][navis.NeuronList] work similarly:

# %%
nl[[0, 1]] | nl[[1, 2]]

# %%
# !!! important
#     Be aware that bitwise AND and OR will likely change the order of the neurons in the list.

# %%
# ### Multiplication and Division
#
# So far, all operations have led to changes in the structure of the [`navis.NeuronList`][].
# **Multiplication and division are different**! Just like multiplying/dividing individual neurons
# by a number, multiplying/dividing a [`navis.NeuronList`][] will change the *coordinates* of nodes, vertices, etc.
# (including associated data such as radii or connector positions) of the neurons in the list:

# %%
nl.units  # our neurons are originally in 8x8x8 nm voxels

# %%

nl_um = nl * 8 / 1000  # convert neurons: voxels -> nm -> um
nl_um.units

# %%
# The above will have changed the coordinates for all neurons in the list.
#
# ## Comparing NeuronLists
#
# [`navis.NeuronList`][] implements some of the basic arithmetic and comparison operators that you might
# know from standard `lists` or `numpy.arrays`. Most this should be fairly intuitive (I hope) but there
# are a few things you should be aware of. The following examples will illustrate that.
#
#
# In Python the `==` operator compares two elements:

# %%
1 == 1

# %%
2 == 1

# %%
# For [`navis.TreeNeuron`][] this is comparison done by looking at the neurons' attribues:
# morphologies (soma & root nodes, cable length, etc) and meta data (name).

# %%
nl[0] == nl[0]

# %%
nl[0] == nl[1]

# %%
# To find out which attributes are compared, check out:

# %%
navis.TreeNeuron.EQ_ATTRIBUTES

# %%
# Edit this list to establish your own criteria for equality.
#
# For [`NeuronList`][navis.NeuronList], we do the same comparison pairwise between the neurons in both
# lists:

# %%
nl == nl

# %%
nl == nl[:2]

# %%
# Because the comparison is done pairwise and **in order**, shuffling a [`NeuronList`][navis.NeuronList]
# will result in a failed comparison:

# %%
nl == nl[[2, 1, 0]]

# %%
# Comparisons are safe against copying but making any changes to the neurons will cause inequality:

# %%
nl[0] == nl[0].copy()

# %%
nl[0] == nl[0].downsample(2, inplace=False)

# %%
# You can also ask if a neuron is in a given [`NeuronList`][navis.NeuronList]:

# %%
nl[0] in nl

# %%
nl[0] in nl[1:]

# %%
# ## Operating on NeuronLists
#
# With very few exceptions, all {{ navis }} functions that work on individual neurons also work on [`navis.NeuronList`][].
#

# %%
# !!! note
#
#     In general, {{ navis }} functions expect multiple neurons to be passed as a `NeuronList` - not as a list of neurons:
#     ```python
#     n1, n2 = navis.example_neurons(2)  # grab two individual neurons
#
#     # This will raise an error
#     navis.downsample_neuron([n1, n2], 2)
#
#     # This will work
#     navis.downsample_neuron(navis.NeuronList([n1, n2]), 2)
#     ```

# %%
# ### NeuronList methods
#
# Similar to individual neurons, [`navis.NeuronLists`][navis.NeuronList] have a number of methods that
# allow you to manipulate the neurons in the list. In fact, (almost) all shorthand methods on individual
# neurons also work on neuron lists:
#
# === "Operating on individual neurons"
#     ```python
#     nl = navis.example_neurons(2)
#     for n in nl:
#        n.reroot(n.soma, inplace=True)  # reroot the neuron to its soma
#     ```
#
# === "Using the neuronlist"
#     ```python
#     nl = navis.example_neurons(2)
#     nl.reroot(nl.soma, inplace=True)  # reroot the neuron to its soma
#     ```
#
# In addition [`navis.NeuronLists`][navis.NeuronList] have a number of specialised methods:

# %%

nl = navis.example_neurons(3)  # load a neuron list
df = nl.summary()  # get a summary table with all neurons
df.head()

# %%

# Quickly map new attributes onto the neurons
nl.set_neuron_attributes(['Huey', 'Dewey', 'Louie'], name='name')
nl.set_neuron_attributes(['Nephew1', 'Nephew2', 'Nephew3'], name='id')
nl

# %%

# Sort the neurons by their name
nl.sort_values('name')  # this is always done inplace
nl


# %%
# Of course there are also a number of `NeuronList`-specific properties:
#
# - `is_mixed`: returns `True` if list contains more than one neuron type
# - `is_degenerated`: returns `True` if list contains neurons with non-unique IDs
# - `types`: tuple with all types of neurons in the list
# - `shape`: size of neuronlist `(N, )`
#
# All attributes and methods are accessible through auto-completion.

# %%
# ## What next?
#
# <div class="grid cards" markdown>
#
# -   :octicons-file-directory-symlink-16:{ .lg .middle } __Neuron I/O__
#     ---
#
#     Learn about how to load your own neurons into {{ navis }}.
#
#     [:octicons-arrow-right-24: I/O Tutorials](../../gallery#import-export)
#
# -   :material-toothbrush-paste:{ .lg .middle } __Visualizations__
#     ---
#
#     Check out the guides on visualizations.
#
#     [:octicons-arrow-right-24: I/O Tutorials](../../gallery#plotting)
#
# </div>
