"""
Coloring
========
<!-- difficulty: beginner -->

Control neuron colors, palettes, opacity and depth cues - in every backend.

Color is the one part of {{ navis }}' plotting API that works the same everywhere: the parameters
below behave identically whether you are drawing a static figure with [`plot2d`][navis.plot2d] or
spinning a scene around in [`plot3d`][navis.plot3d]. Everything that is specific to one renderer -
shading, halos, depth sorting - lives in the [2D](../1b_plotting_2d/tutorial_plotting_2d_00_skeletons)
and [3D](../1c_plotting_3d/tutorial_plotting_3d_00_skeletons) tutorials instead.

New to plotting in {{ navis }}? Start with the [overview](../tutorial_plotting_00_intro).

| Parameter    | What it does                                              | Applies to               |
|--------------|-----------------------------------------------------------|--------------------------|
| `color`      | One color, one per neuron, or an `ID -> color` dict        | Neurons and volumes      |
| `palette`    | Where colors come from when you did not name them          | Neurons and volumes      |
| `color_by`   | Color *by a property* - per neuron, or per node/vertex     | Skeletons and meshes     |
| `shade_by`   | Same, but onto the alpha channel                           | Skeletons and meshes     |
| `vmin`/`vmax`| Clamp the normalization of numerical `color_by`            | Numerical `color_by`     |
| `alpha`      | Flat opacity, one value or one per neuron                  | Neurons and volumes      |
| `depth_coloring` | Color by distance from the camera                      | [`plot2d`][navis.plot2d] |

"""

# mkdocs_gallery_thumbnail_path = '_static/coloring_thumbnail.png'

# %%
# ## Naming colors
#
# In {{ navis }} you can control the color of individual neurons, their compartments, their synapses
# and so on. There are four ways to say which colors you want:
#
# === "Single color"
#     One color for all neurons:
#     ```python
#     navis.plot2d(nl, color="r", view=("x", "-z"), method="2d")
#     ```
#
# === "List of colors"
#     One color per neuron:
#     ```python
#     navis.plot2d(nl, color=["r", "g", "b"], view=("x", "-z"), method="2d")
#     ```
#
# === "Palette"
#     Draw colors from a named palette:
#     ```python
#     navis.plot2d(nl, palette="Greens", view=("x", "-z"), method="2d")
#     ```
#
# === "`ID -> color`"
#     A dictionary mapping neuron ID to color:
#     ```python
#     colors = dict(zip(nl.id, ["r", "g", "b"]))
#     navis.plot2d(nl, color=colors, view=("x", "-z"), method="2d")
#     ```
#
# All four side by side:

import navis
import matplotlib.pyplot as plt

nl = navis.example_neurons(3, kind="mesh")

fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
for ax, (title, kwargs) in zip(
    axes,
    [
        ("color='r'", dict(color="r")),
        ("color=[...]", dict(color=["r", "g", "b"])),
        ("palette='Greens'", dict(palette="Greens")),
        ("color={id: ...}", dict(color=dict(zip(nl.id, ["r", "g", "b"])))),
    ],
):
    navis.plot2d(nl, ax=ax, method="2d", view=("x", "-z"), **kwargs)
    ax.set_title(title, fontsize=11, y=0.98)
    ax.set_axis_off()
plt.tight_layout()

# %%
# Individual colors can be given in any format `matplotlib` understands, and you can mix them:
#
# | Format            | Example                                |
# |-------------------|----------------------------------------|
# | Name              | `"red"`, `"green"`, `"blue"`           |
# | Hex code          | `"#FF0000"`, `"#00FF00"`, `"#0000FF"`  |
# | RGB / RGBA tuple  | `(1, 0, 0)` for red                    |

navis.plot2d(nl, color=["red", "#FF0000", (0, 0, 0)], view=("x", "-z"), method="2d")
plt.tight_layout()

# %%
# ## Coloring by a property
#
# `color_by` is the workhorse. What it does depends on what you hand it - one value per *neuron*
# gives each neuron a single color, one value per *node or vertex* colors within a neuron:
#
# ```mermaid
# graph LR
#   A["color_by=..."] --> B{"one value per<br>neuron, or per<br>node/vertex?"}
#   B -->|per neuron| C["one color<br>per neuron"]
#   B -->|per node/vertex| D["colors vary<br>along the neuron"]
#   C --> E{"categorical or<br>numerical?"}
#   D --> E
#   E -->|categorical| F["palette assigns<br>one color per label"]
#   E -->|numerical| G["palette used as a<br>colormap, normalized<br>by vmin/vmax"]
# ```
#
# ### One color per neuron
#
# Pass a list of labels - one per neuron - to color by type, brain region, hemisphere, whatever:

types = ["typeA", "typeB", "typeA"]

navis.plot2d(nl, color_by=types, palette="tab10", view=("x", "-z"), method="2d")
plt.tight_layout()

# %%
# {{ navis }} assigns a color to each unique label from the palette. Pass a dict to pick them yourself:

palette = {"typeA": "red", "typeB": "blue"}

navis.plot2d(nl, color_by=types, palette=palette, view=("x", "-z"), method="2d")
plt.tight_layout()

# %%
# ### Colors that vary along a neuron
#
# The same parameter colors *within* a neuron - say, its axon red and its dendrites blue. Give it
# the name of a column in the node table (for [`Skeletons`][navis.Skeleton]) or of a property (for
# [`Meshes`][navis.Mesh]):

n = navis.example_neurons(1, kind="skeleton")

# This adds a "compartment" label to every node
navis.split_axon_dendrite(n, label_only=True)

n.nodes.head()

# %%
# Categorical labels work exactly as they did per neuron - a palette name, or a dict:

fig, axes = plt.subplots(1, 2, figsize=(9, 5))
navis.plot2d(n, color_by="compartment", palette="tab10", ax=axes[0], method="2d", view=("x", "-z"))
axes[0].set_title("palette='tab10'", fontsize=11, y=0.98)
navis.plot2d(
    n,
    color_by="compartment",
    palette={"axon": "coral", "dendrite": "cyan", "linker": "limegreen"},
    ax=axes[1],
    method="2d",
    view=("x", "-z"),
)
axes[1].set_title("palette={'axon': 'coral', ...}", fontsize=11, y=0.98)
for ax in axes:
    ax.set_axis_off()
plt.tight_layout()

# %%
# Numerical values are normalized and run through the palette as a colormap instead. This is what
# you want for Strahler index, branch order, distance from the soma and so on:

n = navis.example_neurons(1, kind="skeleton")

# This adds a `strahler_index` column to the node table
navis.strahler_index(n)

navis.plot2d(
    n, color_by="strahler_index", palette="viridis", view=("x", "-z"), method="2d"
)
plt.tight_layout()

# %%
# !!! tip "Controlling the normalization"
#     `vmin`/`vmax` clamp the range that gets mapped onto the colormap, and `norm_global=False`
#     normalizes each neuron on its own instead of across the whole `NeuronList`.
#
# ??? tip "Passing values directly instead of a column name"
#     Everywhere above you can hand `color_by` an array of values instead of a name:
#     ```python
#     navis.plot2d(n, color_by=n.nodes.strahler_index, palette="viridis")
#     ```
#
# It all works on [`Meshes`][navis.Mesh] too - there you need one value per vertex:

m = navis.example_neurons(1, kind="mesh")
navis.strahler_index(m)
m.strahler_index  # an array with one value per vertex

# %%

# Let's use plot3d this time - the parameters are the same
navis.plot3d(m, color_by="strahler_index", palette="viridis", backend="plotly", legend=False)

# %%
# !!! warning "Not every neuron type can be colored per element"
#     Per-node/vertex coloring needs a node or vertex table, so it covers
#     [`Skeletons`][navis.Skeleton] and [`Meshes`][navis.Mesh].
#     [`Dotprops`][navis.Dotprops] and [`Voxels`][navis.Voxels] raise a `TypeError` - color those
#     per neuron instead.
#
# ## Opacity
#
# `alpha` takes a single value for everything, or one per neuron - handy for pushing context
# neurons back and leaving one in front:

fig, axes = plt.subplots(1, 2, figsize=(9, 5))
navis.plot2d(nl, alpha=0.25, ax=axes[0], method="2d", view=("x", "-z"))
axes[0].set_title("alpha=0.25", fontsize=11, y=0.98)
navis.plot2d(nl, color="k", alpha=[1, 0.2, 0.2], ax=axes[1], method="2d", view=("x", "-z"))
axes[1].set_title("alpha=[1, 0.2, 0.2]", fontsize=11, y=0.98)
for ax in axes:
    ax.set_axis_off()
plt.tight_layout()

# %%
# ## Shading
#
# `shade_by` is `color_by`'s sibling: same inputs, but it drives the *alpha* channel rather than the
# hue. Here we fade a neuron out with distance from its soma:

n = navis.example_neurons(1, kind="skeleton")
n.reroot(n.soma, inplace=True)
n.nodes["root_dist"] = n.nodes.node_id.map(navis.dist_to_root(n, weight="weight")) * -1

navis.plot2d(n, shade_by="root_dist", view=("x", "-z"), radius=True, method="2d")
plt.tight_layout()

# %%
# The two compose, so you can color *and* fade by the same property at once:

navis.plot2d(
    n,
    color_by="root_dist",
    shade_by="root_dist",
    palette="viridis",
    view=("x", "-z"),
    method="2d",
    radius=True,
)
plt.tight_layout()

# %%
# ## Depth coloring
#
# The obvious problem with a 2D plot is that it is... well, 2D. `depth_coloring=True` recovers some
# of the missing dimension by coloring the neuron along the viewing axis - near the camera at one
# end of the colormap, far from it at the other:

n = navis.example_neurons(1, kind="skeleton")

navis.plot2d(n, depth_coloring=True, method="2d", view=("x", "-z"))
plt.tight_layout()

# %%
# For this neuron the ventral dendrites are closest to the camera and the dorsal axon furthest away.
# The default colormap is `jet`; `palette` swaps it for any
# [matplotlib colormap](https://matplotlib.org/stable/users/explain/colors/colormaps.html):

fig, axes = plt.subplots(1, 3, figsize=(13, 5))
for ax, palette in zip(axes, [None, "hsv", "cividis"]):
    navis.plot2d(
        n,
        depth_coloring=True,
        palette=palette,
        depth_scale=False,
        ax=ax,
        method="2d",
        view=("x", "-z"),
    )
    ax.set_title(f"palette={palette!r}" if palette else "jet (default)", fontsize=11, y=0.98)
    ax.set_axis_off()
plt.tight_layout()

# %%
# !!! note "Depth coloring is a `plot2d` feature"
#     It works for [`Skeletons`][navis.Skeleton], [`Dotprops`][navis.Dotprops] and
#     [`Meshes`][navis.Mesh], with `method="2d"` and `method="3d"` (but not `"3d_complex"`, which
#     raises). Volumes and voxels keep their own colors. `depth_scale=False` drops the colorbar -
#     which is what we did above to keep the three panels the same size.
#
#     In 3D you do not need it: the renderer gives you real depth, and you can just rotate the scene.
#
# ## Where to next
#
# <div class="grid cards" markdown>
#
# -   :material-image-outline: **2D plotting**
#
#     Shading, halos, depth sorting and radius for static figures.
#
#     [:octicons-arrow-right-24: Skeletons](../1b_plotting_2d/tutorial_plotting_2d_00_skeletons) ·
#     [Meshes](../1b_plotting_2d/tutorial_plotting_2d_01_meshes) ·
#     [Volumes](../1b_plotting_2d/tutorial_plotting_2d_02_volumes)
#
# -   :material-rotate-3d: **3D plotting**
#
#     Interactive scenes, lighting and snapshots.
#
#     [:octicons-arrow-right-24: Skeletons](../1c_plotting_3d/tutorial_plotting_3d_00_skeletons) ·
#     [Meshes](../1c_plotting_3d/tutorial_plotting_3d_01_meshes) ·
#     [Volumes](../1c_plotting_3d/tutorial_plotting_3d_02_volumes)
#
# </div>
