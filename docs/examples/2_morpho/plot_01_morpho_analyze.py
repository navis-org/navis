"""
Analyzing Neuron Morphology
===========================

This tutorial will give you an overview of how to analyze neuron morphology.

Disclaimer: As you might imagine some properties can be gathered for all/most neuron types while others will only work
on specific types. For example, topological properties such as cable length, branch points, etc. are easy to get for
a skeleton (i.e. a [`TreeNeuron`][navis.TreeNeuron]) but not for an image (i.e. a [`VoxelNeuron`][navis.VoxelNeuron]).
Make sure to check the respective function's docstring for details!


## Basic Properties

{{ navis }} provides some basic morphometrics as neuron properties. Others need to be computed using a specific function
(e.g. [`navis.tortuosity`][navis.tortuosity]) - mostly because they require/allow some parameters to be set.

With that in mind, let's dive right in:
"""

# %%
import navis
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_color_codes("muted")

# %%
# Accessing attributes for a single neuron:

# Load a single skeleton (i.e. a TreeNeuron)
n = navis.example_neurons(n=1, kind="skeleton")
print(f"This neuron has {n.cable_length} of cable")

# %%
# If your neuron has its `units` properties set, you can also combine those:

print(f"This neuron has {(n.cable_length * n.units).to('microns')} of cable")

# %%
# Accessing the same properties via a [`navis.NeuronList`][] will return an array:

nl = navis.example_neurons(kind="skeleton")
print(f"Cable lengths for neurons in this list:\n{nl.cable_length}")

# %%
# [`navis.NeuronList.summary`][] is a useful way to collect some of the basic parameters:

# %%
df = nl.summary()
df.head()

# %%
# For [`MeshNeurons`][navis.MeshNeuron] the available properties look different. For example,
# you can get its volume:

# Load a single MeshNeuron
m = navis.example_neurons(n=1, kind="mesh")
print(f"Neuron volume: {m.volume}")

# Again, we can use the `units` property to convert:
print(f"Neuron volume: {(m.volume * n.units **3).to('microns ** 3')}")

# %%
# For topological properties, we need to convert to skeleton. The fastest way is to simply access
# the [`MeshNeuron`][navis.MeshNeuron]'s `.skeleton` property:

print(f"Mesh cable length: {m.skeleton.cable_length * m.units}")

# %%
# !!! note
#     The above `.skeleton` property is simply an automatically generated [`TreeNeuron`][navis.TreeNeuron]
#     representation of the mesh. It uses sensible defaults but as said initially: it's good practice to
#     create and check the skeleton yourself via [`navis.skeletonize`][].
#
# Importantly, some {{ navis }} functions (e.g. [`navis.segment_analysis`][], see below) that accept
# [`MeshNeurons`][navis.MeshNeuron] as input, really use this skeleton representation under-the-hood.
#
# The skeleton representation of the mesh lets us access many toplogical properties:

m.skeleton.n_leafs

# %%
m.skeleton.n_branches

# %%
# You may have already noticed here and in other examples the use of `n_{property}` (e.g. `n.n_leafs`).
# These are in fact generic: you can use any `n_...` and - assuming that property exists - {{ navis }} will
# return a count:

m.n_vertices

# %%
m.skeleton.n_nodes

# %%

# Illustrate with a random property
m.my_counts = [1, 2, 3, 4, 5]
m.n_my_counts

# %%
# ## Segment Analysis
#
# [`navis.segment_analysis`][] is a great entry point for collecting a bunch of morphometrics for your neuron(s) of
# interest. It returns Strahler index, cable length, distance to root, radius and tortuosity for each linear segment:

sa = navis.segment_analysis(m)
sa.head()

# %%

# See if segment length correlates with radius
ax = sns.scatterplot(
    data=sa, x="length", y="radius_mean", size="strahler_index", alpha=0.7
)
ax.set_xscale("log")
ax.set_yscale("log")

# %%
# ## Sholl Analysis
#
# For an example of a Sholl analyses, check out the [MICrONS tutorial](../4_remote/plot_02_remote_microns).
#
# ## Geodesic Distances
#
# Working with Euclidean distances is straight forward and we won't cover this extensively but here is an example where
# we are measuring the average distances between a node and its parent (= the sampling rate):

import numpy as np

# Get nodes but remove the root (has no parent)
nodes = nl[0].nodes[nl[0].nodes.parent_id > 0]

# Get the x/y/z coordinates of all nodes (except root)
node_locs = nodes[["x", "y", "z"]].values

# For each node, get its parent's location
parent_locs = (
    nl[0].nodes.set_index("node_id").loc[nodes.parent_id.values, ["x", "y", "z"]].values
)

# Calculate Euclidean distances
distances = np.sqrt(np.sum((node_locs - parent_locs) ** 2, axis=1))

# Use the neuron's units to convert into nm
distances = distances * nl[0].units

print(
    f"Mean distance between nodes: {np.mean(distances):.2f} (+/- {np.std(distances):.2f})"
)

# %%
# What if you wanted to know the distance between the soma and all terminal nodes? In that case Euclidean distance
# would be insufficient as the neuron is not a straight line. Instead, you need the geodesic, the "along-the-arbor" distance.
#
# {{ navis }} comes with a couple functions that help you get geodesic distances. For single node-to-node queries,
# [`navis.dist_between`][] should be sufficient:

n = nl[0]

end = n.nodes[n.nodes.type == "end"].node_id.values[0]

d_geo = navis.dist_between(n, n.soma, end) * n.units

print(f"Euclidean distance between soma and terminal node {end}: {d_geo:.2f}")

# %%
# Let's visualize this:

import networkx as nx

# First we need to find the path between the soma and the terminal node
path = nx.shortest_path(n.graph.to_undirected(), n.soma, end)

# Get coordinates for the path
path_co = n.nodes.set_index("node_id").loc[path, ["x", "y", "z"]].copy()

# Add a small offset
path_co.x += 500
path_co.y -= 500

# Plot neuron
fig, ax = navis.plot2d(n, c="blue", method="2d", view=("x", "-z"))

# Add geodesic path
ax.plot(path_co.x, path_co.z, c="r", ls="--")

# Add Euclidean path
end_loc = n.nodes.set_index("node_id").loc[end, ["x", "y", "z"]]
soma_loc = n.nodes.set_index("node_id").loc[n.soma, ["x", "y", "z"]]
ax.plot([soma_loc.x, end_loc.x], [soma_loc.z, end_loc.z], c="g", ls="--")

d_eucl = np.sqrt(np.sum((end_loc - soma_loc) ** 2)) * n.units

# Annotate distances
_ = ax.text(
    x=0.1,
    y=0.3,
    s=f"Euclidean distance:\n{d_eucl.to_compact():.0f}",
    transform=ax.transAxes,
    c="g",
)
_ = ax.text(
    x=0.85,
    y=0.5,
    s=f"Geodesic distance:\n{d_geo.to_compact():.0f}",
    transform=ax.transAxes,
    c="r",
    ha="right",
)


# %%
# If you want to look at geodesic distances across many nodes, you should consider using [`navis.geodesic_matrix`][].
# To demonstrate, let's generate a geodesic distance matrix between all terminal nodes:

# Calculate distances from all end nodes to all other nodes
ends = n.nodes[n.nodes.type == "end"].node_id.values
m = navis.geodesic_matrix(n, from_=ends)

# Subset to only end-nodes-to-end_nodes
m = m.loc[ends, ends]

m.head()

# %%
# Let's see if we can visualize any clusters using a `seaborn` clustermap:

import seaborn as sns

from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

# Generate a linkage from the distances
Z = linkage(squareform(m, checks=False), method="ward")

# Plot
cm = sns.clustermap(m, cmap="Greys", col_linkage=Z, row_linkage=Z)

cm.ax_heatmap.set_xticks([])
cm.ax_heatmap.set_yticks([])

# %%
# As you can see in the heatmap, the dendrites and the axon nicely separate.
#
# That's it for now! Please see the [NBLAST tutorial](../../5_nblast/plot_00_nblast_intro) for morphological comparisons using NBLAST and
# the :[API reference](../../../api.md) for a full list of morphology-related functions.
