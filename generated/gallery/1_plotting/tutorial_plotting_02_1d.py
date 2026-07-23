"""
Neuron "Barcodes"
=================

In this tutorial we'll explore a unique way to visualize the branching pattern of neurons using "barcodes".

This visualization technique is based on [Cuntz et al (2010)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000877)
and turns a neuron's branching pattern into a unique "barcode" using toplogical sorting.
"""

import navis
import matplotlib.pyplot as plt

n = navis.example_neurons(n=1, kind="skeleton")  # we need skeletons for this

n.reroot(n.soma, inplace=True)  # set the somas as the root

ax = navis.plot1d(n)
plt.tight_layout()

# %%
# !!! important
#     [`navis.plot1d`][] only works with [`navis.TreeNeurons`][navis.TreeNeuron]. These neurons must be have
#     only a single root (i.e. a single connected component) which is used as the point of reference for
#     the sorting.

# %%
# So what's actually happening here? The barcode is a way to represent the branching pattern of a neuron
# in a unique way. The barcode is created by breaking the neuron into linear segments between branch points and
# sorting them by walking from the root to the leafs in a depth-first manner. At each branch point
# we prioritize the branch that maximizes the distance to the root.
#
# In the plot each segment is represented by a rectangle where the width correspond to the segment's length.
# Segments that terminate in a leaf node are plotted with a darker color.
#
# Here's a simple example to illustrate this:
#
# ![barcode](../../../_static/barcode.png)

# %%
#
# The example skeleton we used above is pretty complex with lots of little side branches which makes the
# barcode quite complicated. Let's simplify the neurons a bit to make the barcode easier to read:

# Prune each neuron to its 20 longest branches
n_pruned = navis.longest_neurite(n, n=20)

ax = navis.plot1d(n_pruned, lw=1)
plt.tight_layout()

# %%
# !!! note
#     Instead of [`navis.longest_neurite`][] you could also use e.g. [`navis.prune_twigs`][] to
#     remove small branches below a certain size.
#
# That's easier to interpet! Let's pull up the neuron to compare:

# Draw the full neuron in black
fig, ax = navis.plot2d(n, view=('x', '-z'), method='2d', color='k')
# Add the pruned neuron in red on top
fig, ax = navis.plot2d(n_pruned, view=('x', '-z'), method='2d', color='r', ax=ax, lw=1.1)

# %%
# With that side-by-side comparison you can hopefully see how the barcode captures the branching pattern of the neuron.
#
# ## Multiple Neurons
#
# We can also plot barcodes for multiple neurons at a time:

nl = navis.example_neurons(n=5, kind="skeleton")
nl = nl[nl.soma != None]  # remove neurons without soma
navis.heal_skeleton(nl, inplace=True)  # heal any potential breaks in the skeletons
nl.reroot(nl.soma)  # reroot all neurons to their soma
navis.longest_neurite(nl, 10, inplace=True)

fig, ax = plt.subplots(figsize=(18, 6))

# Plot the barcodes
ax = navis.plot1d(nl, ax=ax)
plt.tight_layout()

# %%
# ## Customizing the Plot
#
# Similar to other plotting functions in {{ navis }}, you can customize the appearance of the barcode plot:

ax = navis.plot1d(n_pruned, color="red")
plt.tight_layout()

# %%
# We can also color segments based on some property of the neuron:

# Add a "root_dist" column to the neuron's node table
n_pruned.nodes['root_dist'] = n_pruned.nodes.node_id.map(
    navis.dist_to_root(n_pruned)
)

# Plot the barcode with segments colored by their distance to the root
ax = navis.plot1d(n_pruned, color_by="root_dist", palette="jet")
