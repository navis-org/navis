"""
Lists of Neurons
================

Work with many neurons at once using NeuronLists: indexing, filtering and batch operations.

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
# If the neuron attribute is a dataframe, the [`NeuronList`][navis.NeuronList] will concatenate them and
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
# !!! warning "Missing attributes"
#     ```python
#     # MeshNeurons have no `cable_length` - so this raises an error:
#     nl_mix.cable_length
#     ```

# Instead use the `get_neuron_attributes()` method with a default value:
nl_mix.get_neuron_attributes('cable_length', None)


# %%
# ## Indexing NeuronLists
#
# A [`NeuronList`][navis.NeuronList] indexes like a cross between a Python `list`, a numpy array and a
# pandas DataFrame. The tabs below cover the main styles - each returns a new [`NeuronList`][navis.NeuronList]:
#
# === "By position"
#     Integers, lists of integers and slices - just like numpy:
#     ```python
#     nl[0]         # a single neuron
#     nl[[0, 2]]    # first and third neuron
#     nl[:2]        # first two neurons
#     ```
#
# === "By attribute"
#     Index with a boolean array - which includes any neuron attribute (`n_nodes`, `cable_length`, `soma`, ...):
#     ```python
#     nl[nl.n_branches > 700]   # neurons with many branches
#     nl[nl.soma != None]       # neurons that have a soma
#     ```
#
# === "By name"
#     Match against the neurons' `.name`. Pass a single name, several names or - since {{ navis }}
#     matches with `re.fullmatch` - a regex pattern:
#     ```python
#     nl["DA1_lPN_R1"]                   # single name
#     nl[["DA1_lPN_R1", "DA1_lPN_R2"]]   # multiple names
#     nl[".*DA1.*"]                       # regex
#     ```
#
# === "By ID"
#     Every neuron has an `.id` (a random UUID if you didn't set one). Use the `.idx` indexer to select
#     by ID, much like pandas' `.loc[]`:
#     ```python
#     nl.idx[1734350908]
#     ```
#
# Let's see one in action. First, give our three neurons unique names:

# %%
nl = navis.example_neurons(n=3)
for i, n in enumerate(nl):
    n.name = n.name + str(i + 1)
nl

# %%
# Now subset to the neurons whose name matches the "DA1" pattern:
nl[".*DA1.*"]

# %%
# ## Neuron Math
#
# {{ navis }} implements an intuitive syntax for combining and subsetting [`NeuronLists`][navis.NeuronList].
# If you know how Python's `list` and `set` operators behave, these will feel right at home:
#
# | Operator          | On a `NeuronList`                        | Familiar from |
# |-------------------|------------------------------------------|---------------|
# | `A + B`           | concatenate (also combines two neurons)  | `list + list` |
# | `A - B`           | remove the neurons in `B` from `A`       | `list.pop()`  |
# | `A & B`           | keep only neurons present in both        | `set & set`   |
# | <code>A &#124; B</code> | union of both lists                | <code>set &#124; set</code> |
# | `A * x`, `A / x`  | **scale coordinates** by `x`             | —             |
#
# The first four operators change *which* neurons are in the list:
#
# === "Add `+`"
#     Concatenate two lists - or combine two single neurons into a list:
#     ```python
#     nl[:2] + nl[2:]     # -> a list of 3 neurons
#     nl[0] + nl[1]       # two single neurons -> a NeuronList
#     ```
#
# === "Subtract `-`"
#     Drop neurons from the list:
#     ```python
#     nl - nl[2]          # remove the third neuron
#     ```
#
# === "AND `&`"
#     Intersection - keep only neurons present in *both* lists:
#     ```python
#     nl[[0, 1]] & nl[[1, 2]]     # -> just neuron 1
#     ```
#
# === "OR `|`"
#     Union of both lists:
#     ```python
#     nl[[0, 1]] | nl[[1, 2]]     # -> neurons 0, 1 and 2
#     ```
#
# !!! warning "Order is not preserved"
#     Bitwise `&` and `|` will likely reorder the neurons in the resulting list.
#
# ### Multiplication & division: scaling coordinates
#
# **Multiplication and division are the odd ones out.** Rather than changing *which* neurons are in the
# list, they scale the *coordinates* of every neuron in it - nodes, vertices, connectors and radii alike:

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
# know from standard `lists` or `numpy.arrays`. Most of this should be fairly intuitive (I hope) but there
# are a few things you should be aware of. The following examples will illustrate that.
#
#
# In Python the `==` operator compares two elements:

# %%
1 == 1

# %%
2 == 1

# %%
# For [`navis.TreeNeuron`][] this comparison is done by looking at the neurons' attributes:
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
#     [:octicons-arrow-right-24: I/O Tutorials](index.md#import-export)
#
# -   :material-toothbrush-paste:{ .lg .middle } __Visualizations__
#     ---
#
#     Check out the guides on visualizations.
#
#     [:octicons-arrow-right-24: I/O Tutorials](index.md#plotting)
#
# </div>
