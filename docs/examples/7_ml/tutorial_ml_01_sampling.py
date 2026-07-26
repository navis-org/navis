"""
From Neurons to Model Inputs
============================

Turn neurons into fixed-size model inputs: feature point clouds and batchable fragments.

Neurons are variable-sized graphs and meshes with uneven node/vertex density.
Most models want a fixed number of points, sampled evenly. The `navis.ml` module
has two complementary primitives for that:

- **sampling** draws a point cloud from a whole neuron (optionally carrying
  per-node/vertex features), and
- **chunking** tiles a neuron into fixed-size, spatially-coherent fragments.

!!! note
    This tutorial follows on from [Normalizing Neurons](tutorial_ml_00_normalize.md).
    In practice you would usually normalize *before* sampling so every cloud lives
    in the same canonical frame.
"""

# %%
import navis
import numpy as np
import matplotlib.pyplot as plt

navis.config.pbar_hide = True

n = navis.example_neurons(1, kind="skeleton")
m = navis.example_neurons(1, kind="mesh")

# %%
# ## Sampling a skeleton: `sample_cable`
#
# Skeleton nodes are rarely spaced evenly along the cable, so drawing nodes at
# random over-weights densely-noded regions. [`navis.ml.sample_cable`][] instead
# samples points at uniform arclength along the *edges* and interpolates node
# attributes onto each sample. Draws are **stratified** (one jittered sample per
# equal-measure bin) for exactly even coverage.

pts = navis.ml.sample_cable(n, n_points=1000, interpolate="radius", random_state=0)
pts.head()

# %%
# You get back a tidy DataFrame with the sampled coordinates, the requested
# interpolated columns, and a `source_id` recording which node each sample came
# from (join it to pull in any other per-node attribute):

print(list(pts.columns))

# %%
# Let's compare the raw nodes (density tracks the reconstruction) against the
# arclength-uniform sample (density tracks the *cable*). Point size encodes the
# interpolated radius.

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

navis.plot2d(
    n.nodes[["x", "y", "z"]].values, method="2d", view=("x", "-z"),
    ax=axes[0], scatter_kws=dict(c="k", s=2),
)
axes[0].set_title(f"Raw nodes (n={n.n_nodes})")

navis.plot2d(
    pts[["x", "y", "z"]].values, method="2d", view=("x", "-z"),
    ax=axes[1], scatter_kws=dict(c=pts["radius"].values, s=4, cmap="viridis"),
)
axes[1].set_title("sample_cable (n=1000, colored by radius)")
plt.tight_layout()

# %%
# !!! tip
#     Pass `weights="radius"` to bias sampling towards thicker cable (a
#     lateral-surface-area-like weighting), or `interpolate=True` to attach every
#     non-structural node column.
#
# ## Sampling a mesh: `sample_surface`
#
# Mesh vertex density tracks surface detail, not the arbor, so meshes are best
# sampled by **surface area**. [`navis.ml.sample_surface`][] does area-weighted
# (blue-noise) sampling and, like `sample_cable`, records each sample's source
# vertex so per-vertex labels transfer onto the cloud.

surf = navis.ml.sample_surface(m, n_points=2000, random_state=0)

fig, ax = plt.subplots(figsize=(6, 6))
navis.plot2d(
    surf[["x", "y", "z"]].values, method="2d", view=("x", "-z"),
    ax=ax, scatter_kws=dict(c="C3", s=2),
)
ax.set_title("sample_surface (n=2000)")
plt.tight_layout()

# %%
# Per-vertex attributes ride along via the `attributes` argument - here we fake a
# per-vertex label and transfer it onto the sampled points:

label = (m.vertices[:, 2] > np.median(m.vertices[:, 2])).astype(int)
surf = navis.ml.sample_surface(m, 2000, attributes={"label": label}, random_state=0)
print(surf["label"].value_counts().to_dict())

# %%
# ## Subsampling a cloud: `sample_points_uniform`
#
# If you already have a point cloud (e.g. dotprops points, or the output of the
# samplers above) and just want *fewer, evenly-spread* points,
# [`navis.ml.sample_points_uniform`][] subsamples it. It uses farthest-point
# sampling for the most uniform coverage, falling back to a cheaper decimation on
# large clouds (`method="auto"`).

cloud = pts[["x", "y", "z"]].values                 # the 1000-point cable sample
sub = navis.ml.sample_points_uniform(cloud, size=250)

fig, ax = plt.subplots(figsize=(6, 6))
navis.plot2d(cloud, method="2d", view=("x", "-z"), ax=ax, scatter_kws=dict(c="lightgrey", s=6))
navis.plot2d(sub, method="2d", view=("x", "-z"), ax=ax, scatter_kws=dict(c="C0", s=12))
ax.set_title("1000 points (grey) subsampled to 250 (blue)")
plt.tight_layout()

# %%
# !!! tip
#     Use `output="indices"` or `output="mask"` to get the selection rather than
#     the points - handy for subsetting a matching feature array in lock-step.
#
# ## Chunking into fixed-size fragments: `chunk_neuron`
#
# Instead of one cloud per neuron, you sometimes want to tile a neuron into many
# fixed-size fragments - e.g. to feed a fixed input size, or to make per-fragment
# predictions. [`navis.ml.chunk_neuron`][] returns a list of **positional index**
# arrays, each addressing rows of `x.nodes` (skeleton) or `x.vertices` (mesh).
#
# By default fragments grow *along the arbor* (geodesic distance), so each one is
# a connected piece:

chunks = navis.ml.chunk_neuron(n, size=300, mode="partition", undersized="keep")
print(f"{len(chunks)} fragments")

co = n.nodes[["x", "y", "z"]].values
fig, ax = plt.subplots(figsize=(6, 6))
cmap = plt.get_cmap("tab20")
for i, ch in enumerate(chunks):
    real = ch[ch >= 0]                              # drop any pad slots (see below)
    navis.plot2d(
        co[real], method="2d", view=("x", "-z"),
        ax=ax, scatter_kws=dict(c=[cmap(i % 20)], s=4),
    )
ax.set_title(f"chunk_neuron: {len(chunks)} connected fragments")
plt.tight_layout()

# %%
# For a stackable batch, keep the default `undersized="pad"`: every fragment is
# padded up to `size` with a pad token (default `-1`) so they stack into one
# array.

chunks = navis.ml.chunk_neuron(n, size=256, mode="partition")   # undersized="pad"
batch = np.stack(chunks)
print("batch shape:", batch.shape)                              # (n_fragments, 256)

# %%
# !!! warning
#     Don't index coordinates with a padded fragment directly - a negative pad
#     value is a *valid* (wrong) index and silently returns real rows. Mask first:
#
#     ```python
#     real = chunk[chunk >= 0]          # drop pad slots
#     patch = co[real]                  # (<= size, 3)
#     node_ids = n.nodes.node_id.values[real]
#     ```
#
# `chunk_neuron` has a few `mode`s worth knowing:
#
# - `"partition"` - non-overlapping tiles (the default).
# - `"cover"` - overlapping, covering every node at least once.
# - `"random"` / `"spaced"` - `k` fragments from random / evenly-spread seeds,
#   great for oversampling a training set.
#
# And `connected=False` grows fragments by *Euclidean* distance instead (the
# `size` nearest points in space), which packs tightly but ignores the arbor.

spaced = navis.ml.chunk_neuron(n, size=200, mode="spaced", k=6, random_state=0)
print(f"{len(spaced)} evenly-spaced fragments of {[len(c) for c in spaced]} nodes")

# %%
# ## A typical pipeline
#
# Putting it together, a common recipe for feeding a batch of skeletons to a
# point-cloud model is: normalize -> sample -> stack.

nl = navis.example_neurons(3, kind="skeleton")

batch = np.stack([
    navis.ml.sample_cable(
        navis.ml.normalize_neuron(nrn),     # canonical frame (tutorial 00)
        n_points=1024, random_state=0,
    )[["x", "y", "z"]].values
    for nrn in nl
])
print("model input batch:", batch.shape)    # (n_neurons, 1024, 3)

# %%
# ## What's next
#
# - [Normalizing Neurons](tutorial_ml_00_normalize.md) - the canonical-pose step
#   that usually comes first.
# - [Augmenting Neurons](tutorial_ml_02_augment.md) - expand a training set with
#   realistic perturbations.
#
# See the [Machine Learning API reference](../../../api.md#machine-learning) for
# the full list of functions.
