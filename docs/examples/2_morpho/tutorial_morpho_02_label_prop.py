"""
Label Propagation
=================

This tutorial will show you how to propagate an initial set of labels across a neuron. The basic idea is that
you start with a sparse set of labels (e.g. from a manual annotation) and then use the neuron's topology to propagate
those labels across the whole neuron.

One of the most intuitive applications for this is to try to label the axon and dendrite of a neuron based on
the distribution of pre- and postsynapses alone.

## Background

The function that does the heavy lifting is [`navis.propagate_labels()`][navis.propagate_labels]. It takes a neuron
(skeleton or mesh) and a set of initial labels and then propagates those labels across the neuron based on its topology.

![label_propagation](../../../_static/label_propagation.png)

The way this is implemented in {{ navis }} is effectively as a series of matrix multiplications. The neuron's topology
- node-to-node for skeletons and vertex-to-vertex for meshes - is represented as an adjacency matrix, and the labels
are represented as a vector. By multiplying the adjacency matrix by the label vector, we can propagate the labels
across the neuron. This process is repeated iteratively until convergence: we either stop when the labels stop changing
(`tol` parameter) or after a certain number of iterations (`max_iter` parameter).

There are a number of ways to customize the propagation process. First, by default all input labels are treated the same
but we can specify `weight` to give different importance to different labels, making some more likely to propagate than others.
Second, we can decide whether we want to allow the initial set of labels to be changed during the propagation process by
changing the `clamping` parameter.

## Axon-Dendrite Labeling

To demonstrate label propagation, we will create labels from its pre- and postsynapses, and propagate those labels across the
neuron to try to label the axon and dendrite.

"""

# %%
import navis

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


sns.set_color_codes("muted")

# Load the example neuron. This is a mesh but the same process works for skeletons as well.
n = navis.example_neurons(1, kind='mesh')

# %%
# First, we need to collect our initial sets of labels. There are various options but here we will create a vector that, for each vertex
# in the mesh, says whether it contains pre- or postsynapses.

# Snap each pre/postsynapses to the closest vertex
pre_vertices, _ = n.snap(n.presynapses[['x', 'y', 'z']].values)
post_vertices, _ = n.snap(n.postsynapses[['x', 'y', 'z']].values)

# Create an empty vector of labels
labels = np.full(n.n_vertices, fill_value=np.nan, dtype=object)

# Fill in the labels for the pre- and postsynapses
labels[post_vertices] = 'post'
labels[pre_vertices] = 'pre'
# Note that if a vertex contains both pre- and postsynapses, it will be labeled as 'pre' because we overwrite the 'post' label.
# There are more clever ways to handle this (e.g. using weights) but here we'll keep it simple.

# Check the distribution of our initial labels
np.unique(labels.astype(str), return_counts=True)

# %%
# Now we have our initial labels, we can propagate them across the neuron. By default, the initial labels are "clamped", meaning
# they cannot change during the propagation process. In insect neurons, pre- and postsynapses are typically mixed - i.e.
# the axon will have some amount of postsynapses and vice versa for the dendrites. If we want a smooth labeling across the neuron,
# we need to allow the initial labels to change during the propagation process. We can do this by setting `clamping=False`.
# Alternatively, we can also provide an alpha value via e.g. `clamping="soft:0.9"` to determine how resilient the initial labels are to change.
#
# Let's propagate those labels across the neuron and see what we get!

pred = navis.propagate_labels(n, labels, clamping=False, max_iter=10000, tol=1e-6, verbose=True)

# %%
# Note how the prediction didn't converge? In the worst case scenario, that means we haven't reached the entirety of the neuron and some vertices
# remain unlabeled but in our case it just means that the probabilities haven't fully stabilised yet - which may not even affect the final labels.
# You can increase the max number of iterations (or decrease the tolerance) to get a fully converged solution. If you run your own experiments,
# it's worth playing around with these parameters to find the optimum balance between convergence and runtime.
#
# Let's check the distribution of our predicted labels:
np.unique(pred.astype(str), return_counts=True)

# %%
# So we seem to be having some `None` values in our predictions. On closer inspection, these on little internal "flakes" that are disconnected from the rest
# of the neuron mesh and thus never receive any signal during the propagation. We will ignore those for now!
#
# Next, we will visualise the predicted labels across the neuron.

# Fill the Nones with a new category for plotting
pred[pred == None] = "unlabeled"

navis.plot3d(
    n,
    color_by=pred,
    palette={"pre": "red", "post": "cyan", "unlabeled": "lightgray"},
    backend="plotly",
)

# %%
# That doesn't look too bad! But we can do even better: our `pred` vector is effectively a hard assignment of each vertex to a single label, but the underlying
# propagation process actually gives us probabilities for each label at each vertex. We can use those probabilities to get a more nuanced view of the predicted labels across the neuron.

pred, prob, order = navis.propagate_labels(n, labels, clamping=False, max_iter=10000, tol=1e-6, verbose=True, return_probs=True)

# `order` gives us the order of labels in the `prob` array
print("Label order:", order)

# Inspect probability vector
prob

# %%
# Let's visualise the probabilities for the "post" label across the neuron:

navis.plot3d(
    n,
    color_by=prob[:, 0],
    palette="coolwarm",
    backend="plotly",
)


# %%
# As you can see, the probability for "post" decreases (red -> blue) as we move from the proximal dendrites to the distal axon - which is exactly what we would expect!
# If you wanted, you could use those probabilities to make finer categories such low vs high confidence axon/dendrite labels, or even just use the probabilities directly for
# downstream analyses instead of hard labels.
#
# ## Final Notes
#
# 1. If your neuron is not contiguous (i.e. has multiple disconnected components), your labels can't propagate across those components.
# 2. For large mesh neurons it can be computationally expensive to propagate labels across the entire neuron. In that case, you can try to speed things up by downsampling the neuron first.
#
# That's it for now! Please see the [axon/dendrite split tutorial](../../2_morpho/tutorial_morpho_03_ad_split) for a deep dive into determining axon vs dendrite, including
# alternative approaches using flow metrics.

# %%

# mkdocs_gallery_thumbnail_path = '_static/label_propagation.png'