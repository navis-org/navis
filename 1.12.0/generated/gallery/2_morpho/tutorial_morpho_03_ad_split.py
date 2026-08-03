"""
Axon-Dendrite Splits
====================

This tutorial shows you how to split a neuron into its axonal and dendritic compartment.

## Background

Neurons, generally, have two main compartments: axon and dendrites. In textbooks, the axon is the
part of the neuron that sends signals to other neurons, while the dendrites are the parts that
receive signals from other neurons.

![Created in https://BioRender.com](../../../_static/axon_v_dendrite.png "Created in https://BioRender.com")

In reality, the distinction between axon and dendrite is not always so clear-cut: axons can also receive
signals and dendrites can also send signals. In particular invertebrate neurons can be very mixed
with the postsynapses (inputs) and presynapses (outputs) almost evenly distributed across the whole neuron.
Whether a connection between two neurons is axo-dendritic or axo-axonic likely makes a big difference.

_So how do we determine what's axon and what's dendrite?_

## Two Methods

{{ navis }} currently implements two methods for axon-dendrite splits:

1. Flow-based splits via [`navis.split_axon_dendrite`][navis.split_axon_dendrite]
2. Label propagation via [`navis.split_axon_dendrite_prop`][navis.split_axon_dendrite_prop]

Both methods work on neuron skeletons (`TreeNeurons`) as well meshes (`MeshNeuron`), and require
the neuron to have pre- and postsynapses as `.connectors`. Please see also the table at the bottom
for a direct comparison.

### Synapse Flow Centrality

Splitting a neuron into axon and dendrite using synapse flow centrality (SFC) was first proposed by
[Schneider-Mizell _et al._ (2016)](). In a nutshell: we draw paths from all inputs (postsynapses)
to all outputs (presynapses). For each segment in the neuron, we counts the paths that go across
it. If we split the neuron at the segment(s) with the highest SFC, we separate it into axon
and dendrite.

![Synapse Flow Centrality Split](../../../_static/sfc_split.png "Synapse flow centrality split")

!!! note
    [`navis.split_axon_dendrite`][navis.split_axon_dendrite] also implements a number of other
    flow-based methods. See the `metric` parameter for details!

This method is very fast and works well for "normal" neurons. But see for yourself:

"""

# %%
import navis
import numpy as np

# Load an example neuron. This is a mesh but the same process works for skeletons as well.
n = navis.example_neurons(1, kind="mesh")

# %%
# Like all example neurons, this one also comes with a connector table containing both pre- and postsynapses:

n.connectors.head()

# %%
#

fig, ax = navis.plot2d(n, connectors=True, view=("x", "-z"), color="k")

# %%
# In above plot, the cyan dots are postsynapses and red dots are presynapses.
# This is a unipolar invertebrate neuron: a single process extends from the
# soma and branches into a proximal dendrite and a distal axon. Note how both
# compartments contain both pre- and postsynapses?
#
# Let's try to split it!

splits = navis.split_axon_dendrite(n)
splits

# %%
# By default, `split_axon_dendrite` will take one or more neurons and try to break them into
# axon and dendrite which it returns as a `NeuronList`. But wait: why are there **three** pieces
# in that neuronlist?
#
# That's because the more-or-less synapse-free bit that connects the axon and the dendrites is
# returned as a separate "linker":

splits.compartment

# %%
# So far so good but what do those splits look like? For your convenience, the split function
# has added not just a `.compartment` but also a `.color` property to the splits - makes
# visualisation easy:

fig, ax = navis.plot2d(splits, color=splits.color, view=("x", "-z"))

# %%
# The axon is red, dendrites are blue and the linker is grey. Looks reasonable!
#
# If you like, you can use this to calculate the "segregation index" - also from Schneider-Mizell
# _et al._ - which tells you how well pre- and postsynapses separate between axon and dendrite:

navis.segregation_index(splits[splits.compartment != "linker"])

# %%
# If you're working with skeletons, you can also just label the neuron instead of splitting it:

s = navis.example_neurons(1, kind="skeleton")
_ = navis.split_axon_dendrite(s, label_only=True)
s.nodes[["compartment"]]

# %%
# ### Label Propagation
#
# The second method is based on label propagation. In a nutshell: we're using the locations of
# pre- and postsynapses as sparse initial labels for "axon" and "dendrite" and propagate those
# labels across the neurons.
#
# ![Label Propagation Split](../../../_static/label_split.png "Label propagation split")
#
# By doing that, we fill in both missing labels but also allow the initial labels to be refined
# - for example, if there is a single postsynapse surrounded by many presynapses
# , we probably want to ignore that postsynapse and consider this part of the axon.
# For a more detailed explanation, check out the
# [tutorial on label propagation](../../2_morpho/tutorial_morpho_02_label_prop).
#
# Let's give it a shot!

splits = navis.split_axon_dendrite_prop(n)
fig, ax = navis.plot2d(splits, color=splits.color, view=("x", "-z"))

# %%
# Alread looks good! However, we're not actually making use of the biggest advantage of this method:
# it gives us probabilities!
#
# To get probabilities, we need to use label-only mode:

n = navis.split_axon_dendrite_prop(n, label_only=True)

# The compartment is given on a per-vertex or per-node basis
n.compartment

# %%
# Note how there is at least one `nan` among the compartments? Nodes/vertices that weren't
# reached because they are either disconnected from the initial set of labels or because we didn't
# run enough iterations (see the `max_iter` parameter) will not be given a compartment.

# Same for probabilities
n.compartment_prob

# %%
# Let's try visualising those probabilities!

# The probabilities
axon_prob = n.compartment_prob.copy()

# Unvisited (i.e. likely disconnected) nodes will have NaN as probability
axon_prob[np.isnan(axon_prob)] = 0

# Where the predicted compartment is "dendrite", the probablity for the axon is 1 - probability
axon_prob[n.compartment == "dendrite"] = 1 - axon_prob[n.compartment == "dendrite"]

# Plot
fig, ax = navis.plot2d(n, color_by=axon_prob, palette="Reds", view=("x", "-z"))
_ = ax.set_title("axon probability")

# %%

# Same for the dendrites
dend_prob = n.compartment_prob.copy()
dend_prob[np.isnan(dend_prob)] = 0
dend_prob[n.compartment == "axon"] = 1 - dend_prob[n.compartment == "axon"]

fig, ax = navis.plot2d(n, color_by=dend_prob, palette="Blues", view=("x", "-z"))
_ = ax.set_title("dendrite probability")

# %%
# As you can see there is a smooth transition between axon and dendrite.
# You could use that to e.g. refine the axon/dendrite assignment into high
# and low-confidence, or try to define a linker.
#
#
# ## Comparison
#
# Below table compares flow- and propagation-based functions:
#
# |                             | [`split_axon_dendrite`][navis.split_axon_dendrite] (flow)    | [`split_axon_dendrite_prop`](navis.split_axon_dendrite_prop) (propagation) |
# |-----------------------------|---------------------------|----------------------------|
# | Works on `TreeNeurons`?     | Yes                       | Yes                        |
# | Works on `MeshNeurons`?     | Yes but still operates on the skeleton         | Yes   |
# | Returns probabilities?      | No                        | Yes                        |
# | Defines a "linker"?         | Yes                       | Not directly               |
# | Finds cell body fibers?     | Yes                       | No                         |
# | Which method is faster?     | Faster                    | Slower (but depends on parameters)   |

# mkdocs_gallery_thumbnail_path = '_static/axon_v_dendrite.png'
