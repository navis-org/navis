"""
From Neurons to Model Inputs
============================

Turn neurons into fixed-size model inputs: feature point clouds and batchable patches.

Neurons are variable-sized graphs and meshes with uneven node/vertex density.
Most models want a fixed number of points, sampled evenly and at a controlled
resolution. The `navis.ml` toolkit answers three separate questions:

1. **Where do the points come from?** The cable of a skeleton, the surface of a
   mesh, or an existing point cloud.
2. **How many / how dense?** A fixed count (`n_points`), a fixed resolution
   (`density`), or a fixed inter-point distance (`spacing`) - see the next section.
3. **One cloud per neuron, or many patches?** Sample the whole neuron, or tile it
   into fixed-size fragments/patches.

| Function | Input | What you get |
|----------|-------|--------------|
| [`navis.sample_skeleton`][] | skeleton | equal-spacing points, deterministic, coordinates only |
| [`navis.ml.sample_cable`][] | skeleton | stratified arclength points **+ interpolated features** |
| [`navis.ml.sample_surface`][] | mesh | area-weighted surface points **+ per-vertex features** |
| [`navis.ml.sample_points_uniform`][] | point cloud | an evenly-spread *subsample* |
| [`navis.ml.estimate_spacing`][] | point cloud | the measured spacing/density (the inverse of the knobs) |
| [`navis.ml.chunk_neuron`][] | skeleton or mesh | fixed-size fragments of *existing* points (indices) |
| [`navis.ml.sample_patches`][] | skeleton or mesh | *resample* at a density, then tile into fixed-size patches |

!!! note
    This tutorial follows on from [Normalizing Neurons](tutorial_ml_00_normalize.md).
    In practice you would usually normalize *before* sampling so every cloud lives
    in the same canonical frame - and so `density`/`spacing` are expressed in that
    frame's units.
"""

# %%
import navis
import numpy as np
import matplotlib.pyplot as plt

navis.config.pbar_hide = True

n = navis.example_neurons(1, kind="skeleton")
m = navis.example_neurons(1, kind="mesh")

# %%
# ## One dial, three faces: `n_points`, `density`, `spacing`
#
# A neuron's size ties together *how many* points you draw and *how far apart* they
# are, so you can pin **exactly one** of these three (the samplers reject more than
# one). They are shared by [`navis.sample_skeleton`][], [`navis.ml.sample_cable`][]
# and [`navis.ml.sample_surface`][]:
#
# === "`n_points` — fixed count"
#     Draw the same number of points from every neuron. The resolution then varies
#     with size: a large arbor is sampled sparsely, a small one densely.
#
#     **Use when the model needs fixed-size inputs** (e.g. a 1024-point tensor).
#
#     ```python
#     navis.ml.sample_cable(n, n_points=1000)
#     navis.ml.sample_surface(m, n_points=2000)
#     ```
#
# === "`density` — fixed resolution"
#     Draw a fixed number of points *per unit* cable length (skeleton) or surface
#     area (mesh); the count scales with neuron size.
#
#     **Use when local neighbourhoods must mean the same physical scale across
#     neurons** - exactly what k-NN / ball-query models (PointNet++, DGCNN, point
#     transformers) assume.
#
#     ```python
#     navis.ml.sample_cable(n, density=5e-3)     # points per unit length
#     navis.ml.sample_surface(m, density=3e-5)   # points per unit area
#     ```
#
# === "`spacing` — fixed distance"
#     Fix the distance between neighbouring points. For a skeleton this is the
#     *exact* step along the cable; for a mesh it is a Poisson-disk *minimum*
#     distance, so the count comes out of rejection sampling and may vary.
#
#     Related to density by `density ~ 1 / spacing` (cable) or `~ 1 / spacing**2`
#     (surface).
#
#     ```python
#     navis.ml.sample_cable(n, spacing=250)      # distance between samples
#     navis.ml.sample_surface(m, spacing=200)    # mode="even" only
#     ```
#
# Here are all three on the mesh surface. `n_points` fixes the count; `density` and
# `spacing` fix the resolution and let the count fall out:

by_count = navis.ml.sample_surface(m, n_points=2000, random_state=0)
by_density = navis.ml.sample_surface(m, density=3e-5, random_state=0)
by_spacing = navis.ml.sample_surface(m, spacing=250, random_state=0)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, s, title in zip(
    axes,
    (by_count, by_density, by_spacing),
    (f"n_points=2000  (n={len(by_count)})",
     f"density=3e-5  (n={len(by_density)})",
     f"spacing=250  (n={len(by_spacing)})"),
):
    navis.plot2d(s[["x", "y", "z"]].values, method="2d", view=("x", "-z"),
                 ax=ax, scatter_kws=dict(c="C3", s=2))
    ax.set_title(title)
plt.tight_layout()

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
# The same neuron sampled by `density` or `spacing` gives a count that tracks its
# cable length instead:

for kw in (dict(n_points=1000), dict(density=5e-3), dict(spacing=250)):
    out = navis.ml.sample_cable(n, **kw, random_state=0)
    print(f"{kw} -> {len(out)} points")

# %%
# !!! tip
#     Pass `weights="radius"` to bias sampling towards thicker cable (a
#     lateral-surface-area-like weighting), or `interpolate=True` to attach every
#     non-structural node column.
#
# ### `sample_cable` vs `sample_skeleton`
#
# There are two ways to turn a skeleton into points; pick by whether you need
# features and randomness:
#
# === "`sample_cable`  (navis.ml)"
#     **Stratified** (randomised) arclength sampling. Interpolates `radius`/other
#     columns onto each point and records `source_id`. Returns a DataFrame. The
#     richer, ML-facing sampler - use it to build feature clouds.
#
# === "`sample_skeleton`  (top-level)"
#     **Deterministic** equal-spacing sampling: the same points every call,
#     coordinates only (no attributes). Returns a plain `(n, 3)` array. Use it when
#     you want a reproducible geometric skeletonisation without features.
#
#     ```python
#     navis.sample_skeleton(n, n_points=1000)   # or density= / spacing=
#     ```
#
# ## Sampling a mesh: `sample_surface`
#
# Mesh vertex density tracks surface detail, not the arbor, so meshes are best
# sampled by **surface area**. [`navis.ml.sample_surface`][] does area-weighted
# (blue-noise) sampling and, like `sample_cable`, records each sample's source
# vertex so per-vertex labels transfer onto the cloud.
#
# Per-vertex attributes ride along via the `attributes` argument - here we fake a
# per-vertex label and transfer it onto the sampled points:

label = (m.vertices[:, 2] > np.median(m.vertices[:, 2])).astype(int)
surf = navis.ml.sample_surface(m, n_points=2000, attributes={"label": label}, random_state=0)
print(surf["label"].value_counts().to_dict())

# %%
# !!! note
#     `mode="even"` (the default) is blue-noise sampling; `mode="surface"` is plain
#     area-weighted random sampling; `mode="vertex"` draws existing vertices. Only
#     `"even"` supports `spacing` (it is the Poisson-disk minimum distance).
#
# ## Measuring resolution: `estimate_spacing`
#
# [`navis.ml.estimate_spacing`][] is the inverse of the `spacing`/`density` knobs:
# give it any cloud and it reports the characteristic nearest-neighbour distance.
# Use it to audit a cloud, or to pick a `density`/`spacing` for a whole dataset.

print("cable sample  :", round(navis.ml.estimate_spacing(pts[["x", "y", "z"]].values), 1))
print("surface sample:", round(navis.ml.estimate_spacing(surf[["x", "y", "z"]].values), 1))

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
# fixed-size fragments - e.g. to make per-fragment predictions. [`navis.ml.chunk_neuron`][]
# returns a list of **positional index** arrays, each addressing rows of `x.nodes`
# (skeleton) or `x.vertices` (mesh). It tiles the neuron's *existing* points; it
# does **not** resample, so it has no density control (see `sample_patches` below
# for that).
#
# By default fragments grow *along the arbor* (geodesic distance), so each one is a
# connected piece:

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
# padded up to `size` with a pad token (default `-1`) so they stack into one array.

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
# `chunk_neuron` has four `mode`s; the first two emerge a fragment count, the last
# two let you set it via `k` / `sampling_factor`:
#
# === "`partition`"
#     Non-overlapping tiles covering every point once (the default). The remainder
#     that can't fill a whole tile is handled per `undersized`.
#
# === "`cover`"
#     Overlapping tiles that cover every point *at least* once - tiles may reuse
#     points to stay full-size.
#
# === "`random`"
#     `k` (or `sampling_factor x`) fragments from random seeds. Good for
#     oversampling a training set.
#
# === "`spaced`"
#     Like `random`, but seeds are evenly spread by farthest-point sampling
#     (deterministic unless you pass `random_state`). Good for even oversampling.
#
# `connected=False` grows fragments by *Euclidean* distance instead (the `size`
# nearest points in space), which packs tightly but ignores the arbor.

spaced = navis.ml.chunk_neuron(n, size=200, mode="spaced", k=6, random_state=0)
print(f"{len(spaced)} evenly-spaced fragments of {[len(c) for c in spaced]} nodes")

# %%
# ## Patches at a controlled resolution: `sample_patches`
#
# `chunk_neuron` tiles *existing* points, so a fragment's density is whatever the
# reconstruction had. When you want fixed-size patches **and** a fixed resolution -
# the standard "group `K` points within radius `r`" input for point-cloud models -
# use [`navis.ml.sample_patches`][]. It **resamples** the neuron to a uniform cloud
# at your `density`/`spacing` (via `sample_surface`/`sample_cable`) and then tiles
# that cloud into `n_points`-sized spatial patches.
#
# Because the cloud is uniform, `n_points` + `density` pin each patch's physical
# extent - and that extent is the same across patches and across neurons.
#
# The result is one tidy DataFrame with a `chunk_id` column (overlapping patches
# simply repeat a point's row under each patch it belongs to):

patches = navis.ml.sample_patches(
    m, n_points=128, density=3e-5, mode="spaced", random_state=0
)
print(f"{patches.chunk_id.nunique()} patches, {len(patches)} rows total")
print(list(patches.columns))

fig, ax = plt.subplots(figsize=(6, 6))
for cid, g in patches.groupby("chunk_id"):
    navis.plot2d(
        g[["x", "y", "z"]].values, method="2d", view=("x", "-z"),
        ax=ax, scatter_kws=dict(c=[cmap(cid % 20)], s=4),
    )
ax.set_title(f"sample_patches: {patches.chunk_id.nunique()} patches of 128 points")
plt.tight_layout()

# %%
# `groupby("chunk_id")` gives you the patches; stack them into a batch tensor:

batch = np.stack([g[["x", "y", "z"]].values for _, g in patches.groupby("chunk_id")])
print("patch batch:", batch.shape)                  # (n_patches, 128, 3)

# %%
# ## A typical pipeline
#
# Putting it together, a common recipe for feeding a batch of skeletons to a
# point-cloud model is: normalize -> sample -> stack. Sampling by `density` here
# would instead give every neuron the same *resolution* (and a variable count).

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
