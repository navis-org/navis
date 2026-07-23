"""
Neuron Topology
===============

This example demonstrates how to plot neurons' topology using various different layouts.

Skeletons in {{ navis }} are hierarchical trees (hence the name [`TreeNeuron`][navis.TreeNeuron]).
As such they can be visualized as dendrograms or flat, graph-like plots using [`navis.plot_flat`][].
This is useful to unravel otherwise complicated, overlapping branching patterns - e.g. when
you want to show compartments or synapse positions within the neuron. Let's demo some layouts!

## Subway Maps

First up: the `subway` layout. This one is nice for side-by-side comparisons of neurons.
"""

# %%

import navis
import matplotlib.pyplot as plt

# Load example neurons
nl = navis.example_neurons(n=4, kind="skeleton")

# Convert example neurons from voxels to nanometers
nl_nm = nl.convert_units("nm")

# Reroot to soma
nl_nm[nl_nm.soma != None].reroot(nl_nm[nl_nm.soma != None].soma, inplace=True)

# Generate one axis for each neuron
fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

navis.plot_flat(nl_nm[0], layout="subway", connectors=True, ax=axes[0])
navis.plot_flat(nl_nm[1], layout="subway", connectors=True, ax=axes[1])
navis.plot_flat(nl_nm[3], layout="subway", connectors=True, ax=axes[2])

# Clean up the axes
for ax in axes[:-1]:
    ax.set_axis_off()

for sp in ["left", "right", "top"]:
    axes[-1].spines[sp].set_visible(False)
axes[-1].set_yticks([])

_ = axes[-1].set_xlabel("distance [nm]")
plt.tight_layout()

# %%
# !!! important
#     For the other layouts you will need to have [pygraphviz](https://pygraphviz.github.io/) and the underlying `graphviz` library
#     installed. For details on the layout, check out the [graphviz docs](https://graphviz.org/about/).
#
# ## Dendrograms
#
# `dot` and `twopi` are dendrogram layouts. They (should) preserve branch lengths similar to `subway`:

# %%
ax = navis.plot_flat(nl_nm[0], layout="dot", connectors=True, color="k")
plt.tight_layout()

# %%
n = nl_nm[0]
ax = navis.plot_flat(n, layout="twopi", connectors=True, color="k")
plt.tight_layout()

# %%
# You can also highlight specific connectors by their ID (here we just use the first 100):

# %%
highlight = n.connectors.connector_id[:100]

ax = navis.plot_flat(
    nl_nm[0],
    layout="dot",
    highlight_connectors=highlight,
    color="k",
    syn_marker_size=2,
)
plt.tight_layout()

# %%
#
# ## Force-Directed Layouts
#
# `neato`, `fdp` and `sfdp` are force-directed layouts.
#
# Some layouts (like `neato` & `fdp`) can be quite expensive to calculate in which case it's worth
# downsampling your neuron before plotting
ds = navis.downsample_neuron(nl_nm[0], 10, preserve_nodes="connectors")

ax = navis.plot_flat(ds, layout="neato", connectors=True, color="k")
plt.tight_layout()

# %%
ax = navis.plot_flat(nl[0], layout="sfdp", connectors=True, color="k")
plt.tight_layout()
