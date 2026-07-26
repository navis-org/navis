"""
Masking
=======

Restrict analyses, plots and processing to part of a neuron with masks.

!!! example "Experimental"
    Masking is a powerful tool that allows you to restrict analyses, visualization or other operations to specific
    parts of a neuron. It's a new feature in {{ navis }} version `1.10` and still somewhat experimental.
    Feel free to take it for a spin - we are keen to hear your feedback on it! Please see also the [caveats](#caveats) section.

Prior to {{ navis }} version `1.10` you could already use [`navis.subset_neuron`][] to subset e.g. a skeleton to a specific
set of nodes or a region of interest. This approach is fine if your workflow is unidirectional but it becomes cumbersome if you
want to switch back and forth between the full version and subsets of a neuron. With version `1.10` we introduced a masking
feature that allows for more flexible and efficient masking of neurons.

Masking has two main advantages over subsetting:

1. Reversible
2. Memory efficient as it avoids making unnecessary copies (see also the [caveats](#caveats) section)

There are two ways to mask neurons:

- direct masking
- the `NeuronMask` context manager (recommended)

We will start with direct masking to illustrate the principle:

## Direct masking

You can directly mask a neuron by calling the `.mask()` method on the neuron object:

"""

# mkdocs_gallery_thumbnail_path = '_static/masking_thumbnail.png'

# %%
import navis

# Load an example skeleton
n = navis.example_neurons(kind="skeleton", n=1)

# Inspect the neuron prior to masking
n

# %%
# First, we need to create a mask consisting of boolean (`True`/`False`) values of the same
# length as the number of nodes in our skeleton.
#
# !!! important
#     Masks are inclusive, meaning nodes/vertices/points where the mask is `True` will be kept and
#     data where mask is `False` will be removed.

# With this mask we only keep nodes above a certain y-coordinate
mask = n.nodes.y > 35_000
mask

# %%
# We apply the mask by feeding it into the neuron's `.mask()` method:

n_masked = n.mask(mask)

# Let's inspect the masked neuron
n_masked


# %%
# The masked neuron contains only the nodes that satisfy the mask, and hence the node count
# is lower for the masked neuron:

print(f"Nodes in mask {mask.sum():,}")
print(f"Nodes in masked neuron {len(n_masked.nodes):,}")

# %%
# Meanwhile the original neuron is still intact:
n

# %%
# !!! important
#     `n` and `n_masked` may look like separate neurons but `n_masked` is actually just a "view" into the original neuron.
#     Edits made to the underlying data (the `.nodes` table in case of this skeleton) will propagate back to the original neuron!
#
# You can also release the mask to get the original neuron "back":
n_masked.unmask()

# %%
# Masks can be `numpy` arrays, `pandas.Series`, a boolean neuron property or a function that
# takes a neuron as its input and returns a boolean array.
#
# We can achieve the same result as above using a function:
n_masked = n.mask(lambda x: x.nodes.y > 35_000)
n_masked

# %%
# The expected size of the mask depends on the neuron type:
#
#  - [`TreeNeurons`][navis.TreeNeuron]: number of nodes
#  - [`MeshNeurons`][navis.MeshNeuron]: number of faces or vertices
#  - [`Dotprops`][navis.Dotprops] number of points
#
# You can always check by using `len()`:
len(n)

# %%
# ### A practical example
#
# Let's showcase a practical example for masking! Here, we will label the axon and dendrites of a neuron and then
# calculate neuron properties for each compartment separately.
#
# First, we need to label axon and dendrites in our neuron

# This function will label the axon and dendrites of a neuron
navis.split_axon_dendrite(n, label_only=True)
n.nodes.head()  # note the new `compartment` column

# %%
# Now we can mask the neuron to get the length of the axon and dendrites:

comp = n.nodes.compartment
print("Axon length:", n.mask(comp == "axon").cable_length * n.units)
print("Dendrite length:", n.mask(comp == "dendrite").cable_length * n.units)

# %%
# This also works for visualization

# Visualize the compartments separately
fig, ax = navis.plot2d(
    [
        n.mask(~n.nodes.compartment.isin(("axon", "dendrite"))),
        n.mask(n.nodes.compartment == "axon"),
        n.mask(n.nodes.compartment == "dendrite"),
    ],
    color=["gray", "red", "cyan"],  # color the compartments differently
    view=("x", "-y"),
    radius=True,
)

# %%
# !!! tip
#     The above visualization could have also been achieved using the `color_by` argument of the
#     [`plot2d`][navis.plot2d] function. See the tutorial on
#     [coloring neurons](../1_plotting/tutorial_plotting_01_colors#coloring-neurites) for an example.
#
# ## Masking context manager
#
# In the previous examples we directly masked a single neuron using the `.mask()` method.
# The alternative and *the recommended way to use masks* is the [`navis.NeuronMask`][] context manager!
# It lets you mask multiple neurons within a specific context and automatically removes the mask afterwards:

# Load multiple example neurons
nl = navis.example_neurons(3, kind="skeleton")

# Label their axon and dendrites
navis.split_axon_dendrite(nl, label_only=True)

# Mask all neurons to their axon
with navis.NeuronMask(nl, lambda x: x.nodes.compartment == "axon"):
    print("Axon cable lengths:", nl.cable_length * nl.units)

# Mask all neurons to their dendrites
with navis.NeuronMask(nl, lambda x: x.nodes.compartment == "dendrite"):
    print("Dendrite cable lengths:", nl.cable_length * nl.units)

# %%
# ## Operations using masks
#
# So far we have run read-only analyses on masked neurons. You can, however, also modify masked neurons!
# Restricting operations to specific parts of a neuron can be a very powerful tool:
#
#  - prune twigs around the soma but leave the rest of the neuron intact
#  - remove all but the longest neurite of only the axon
#  - smooth the backbone of the neuron but leave the tips untouched
#
# Let's demonstrate with a simple example:

# %%
# Load a single neuron:
n = navis.example_neurons(kind="skeleton", n=1)

# %%
# Label axon and dendrites:
navis.split_axon_dendrite(n, label_only=True)

# %%
# Keep a copy of the neuron for comparison:
n_original = n.copy()

# %%
# Downsample the axon only:
with navis.NeuronMask(n, lambda x: x.nodes.compartment == "axon"):
    n.downsample(5)

# %%
# Plot side-by-side for comparison (original on the left, modified on the right)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot neuron
navis.plot2d([n_original, n_original.nodes[["x", "y", "z"]].values], ax=axes[0])
navis.plot2d([n, n.nodes[["x", "y", "z"]].values], ax=axes[1])

axes[0].set_title("original")
axes[1].set_title("downsampled axon")


# %%
# ## Caveats
#
# Masking can be incredibly powerful but it comes with a few caveats and gotchas that you should be aware of:
#
# 1. By default we don't copy any data when masking. This makes the process fast and memory efficient but it also means
#    that if you modify the masked neuron you will also modify the original neuron's data. If that's what you want, great!
#    But if you want to keep the original neuron intact you should consider making a copy of the neuron before or during masking.
