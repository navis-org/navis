"""
The Basics
==========

This tutorial will introduce some of the basics concepts in NAVis.

This is not supposed to be comprehensive but rather to give you a flavor of how things work. For inspiriation,
explore the [example gallery](../index.md) and for detailed explanations have a look at the [API documentation](../../api/).

## Single Neurons

{{ navis }} lets you import neurons from a variety of local and remote sources. For demonstration purposes {{ navis }} comes with a
bunch of fruit fly neurons from the [Janelia hemibrain](https://neuprint.janelia.org) project:


!!! note

    We will cover loading neurons from different sources in a different tutorial.
"""

# %%
import navis
# mkdocs_gallery_thumbnail_path = '_static/favicon.png'

# Load a single neuron
n = navis.example_neurons(n=1, kind='skeleton')
n

# %%
# In above code we loaded one of the example neurons. {{ navis }} represents neurons as
# [`navis.TreeNeuron`][], [`navis.MeshNeuron`][], [`navis.VoxelNeuron`][] or [`navis.Dotprops`][].
#
# In the above example we asked for a skeleton, so the neuron returned is a [`TreeNeuron`][navis.TreeNeuron].
# This class is essentially a wrapper around the actual neuron data (the SWC table in this case) and has
# some convenient features.
#
# Node data is stored as `pandas.DataFrame`:

# %%
n.nodes.head()

# %%
# !!! note "Pandas"
#
#     [pandas](https://pandas.pydata.org) is **the** data science library for Python and will help you analyze and visualize your data.
#     We highly recommend familiarizing yourself with pandas! There are plenty of good tutorials out there but pandas' own
#     [10 Minutes to pandas](https://pandas.pydata.org/pandas-docs/stable/10min.html) is a good place to start.
#
# Try typing in "`n.`" and hitting tab: most attributes and functions are accessible via autocompletion.
# If you don't know what a function does, check out the documentation using `help()` or via the [API documentation](../../api):
# ```python
# help(navis.TreeNeuron.root)
# ```
#
# ```python
# help(navis.TreeNeuron.downsample)
# ```

# %%
# You will notice that many {{ navis }} functions that accept neurons have an `inplace` parameter. This is analogous to pandas:

# %%
# Downsample a copy, leaving the original unchanged
# (this is the default for almost all functions)
n_ds = n.downsample(10, inplace=False)

# Downsample original neuron
n.downsample(10, inplace=True)

n

# %%
# [`navis.TreeNeuron`][] functions such as [`.downsample()`][navis.TreeNeuron.downsample] are shorthands for calling the actual
# {{ navis }} functions. So above code is equivalent to:

# %%
n = navis.example_neurons(n=1, kind='skeleton')
n_ds = navis.downsample_neuron(n, downsampling_factor=10, inplace=False)
n_ds

# %%
# ## Lists of Neurons
#
# For multiple neurons, {{ navis }} uses [`navis.NeuronList`][]:

# %%
nl = navis.example_neurons(n=3, kind='skeleton')
nl

# %%
# [`navis.NeuronList`][] is a container and behaves like a glorified `list`:

# %%
nl[0]

# %%
# [`navis.NeuronList`][] lets you run/access all functions (methods) and properties of the neurons it contrains:

# %%
nl.cable_length

# %%
nl_ds = nl.downsample(10, inplace=False)

nl_ds

# %%
# Let's finish this primer with some eye candy

# %%
nl.plot3d(backend='plotly')


# %%
# ## What next?
#
# <div class="grid cards" markdown>
#
# -   :material-tree-outline:{ .lg .middle } __Neuron types__
#     ---
#
#     Find out more about the different neuron types in {{ navis }}.
#
#     [:octicons-arrow-right-24: Neuron types tutorial](../tutorial_basic_01_neurons)
#
# -   :fontawesome-solid-list-ul:{ .lg .middle } __Lists of Neurons__
#     ---
#
#     Check out the guide on lists of neurons.
#
#     [:octicons-arrow-right-24: NeuronLists tutorial](../tutorial_basic_02_neuronlists)
#
# -   :octicons-file-directory-symlink-16:{ .lg .middle } __Neuron I/O__
#     ---
#
#     Learn about how to load your own neurons into {{ navis }}.
#
#     [:octicons-arrow-right-24: I/O Tutorials](../../gallery#import-export)
#
# </div>
