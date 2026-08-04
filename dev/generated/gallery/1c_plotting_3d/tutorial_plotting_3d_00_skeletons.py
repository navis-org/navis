"""
Skeletons
=========
<!-- difficulty: intermediate -->

Explore skeletons in an interactive 3D viewer - then render what you set up into a figure.

Where [`plot2d`][navis.plot2d] fakes depth, [`plot3d`][navis.plot3d] has a real renderer: proper
occlusion, proper perspective, and a camera you can throw around. The trade-off is control - there
is no `taper`, no `halo`, no `linestyle`, because those are `matplotlib` tricks for faking what a 3D
renderer just *does*.

If you have not read the [plotting overview](../1a_plotting_general/tutorial_plotting_00_intro) yet,
start there; colors and palettes work identically in every backend and are covered in
[coloring](../1a_plotting_general/tutorial_plotting_01_colors).

| Parameter    | Effect                                                        | octarine | plotly | k3d |
|--------------|---------------------------------------------------------------|----------|--------|-----|
| `radius`     | Draw skeletons as tubes rather than lines                     | :material-check: | :material-check: | :material-check: |
| `linewidth`  | Line thickness, and the radius scaling factor                 | :material-check: | :material-check: | :material-check: |
| `soma`       | Draw the soma as a sphere                                     | :material-check: | :material-check: | :material-check: |
| `connectors` | Add synapses as a scatter                                     | :material-check: | :material-check: | :material-check: |
| `snapshot`   | Render offscreen into a `matplotlib` figure                   | :material-check: | :material-close: | :material-close: |

"""

# %%
# ## The interactive viewer
#
# This is what [`plot3d`][navis.plot3d] is for. Hand it neurons and you get a live scene to spin,
# zoom and pick apart - no parameters required:

import navis
import matplotlib.pyplot as plt

n = navis.example_neurons(1, kind="skeleton")
nl = navis.example_neurons(5, kind="skeleton")

# %%
# All three backends take the same neuron parameters - what differs is what you get back:
#
# === "Octarine"
#     A `Viewer` object you keep adding to, removing from and recolouring. Fastest, and the only one
#     that works from a plain terminal as well as from Jupyter.
#     ```python
#     viewer = navis.plot3d(nl, backend="octarine", radius=True)
#     viewer.set_view("XZ")
#     ```
#
# === "Plotly"
#     An inline figure that persists in the notebook and exports to standalone HTML.
#     ```python
#     navis.plot3d(nl, backend="plotly", radius=True, legend_orientation="h")
#     ```
#
# === "k3d"
#     Jupyter-only, in-memory, fast.
#     ```python
#     navis.plot3d(nl, backend="k3d", radius=True)
#     ```
#
# Here is the plotly version live - drag to rotate, scroll to zoom, click the legend to hide neurons:

navis.plot3d(nl, backend="plotly", radius=True, legend_orientation="h", height=600)

# %%
# !!! tip "Plotly styling"
#     `background` (`"light"`, `"white"`, `"dark"`, or a colour), `projection`
#     (`"perspective"`/`"orthographic"`) and `dragmode` (`"orbit"`/`"turntable"`) are plotly-only and
#     change the whole scene rather than the neurons. `projection="orthographic"` is usually what you
#     want for a figure - no perspective distortion.
#
# ## Snapshots
#
# Interactive is how you *look* at neurons; a figure is how you publish them. `snapshot=True` bridges
# the two: octarine renders the scene offscreen and hands back a `matplotlib` `(fig, ax)` with the
# image on it. Real occlusion and lighting, in a figure you can annotate, compose and `savefig`:

navis.plot3d(n, snapshot=True, view=("x", "-z"), color="#4C6EF5")

# %%
# It needs the octarine backend, and it works headlessly - so it is also how you render 3D figures
# from a script or on a cluster node. Pass `ax=` to draw into an axis you already have, which is all a
# comparison sheet is - and is how every side-by-side below is built:

fig, axes = plt.subplots(1, 4, figsize=(17, 5))
for ax, (title, kwargs) in zip(
    axes,
    [
        ("radius=False (default)", dict(radius=False)),
        ("radius=True", dict(radius=True)),
        ("radius=True, linewidth=2", dict(radius=True, linewidth=2)),
        ("soma=False", dict(soma=False)),
    ],
):
    navis.plot3d(n, snapshot=True, ax=ax, view=("x", "-z"), color="#4C6EF5", **kwargs)
    ax.set_title(title, fontsize=11, y=0.98)
    ax.set_axis_off()
plt.tight_layout()

# %%
# !!! note "`linewidth` means two things"
#     With `radius=False` it is the line thickness. With `radius=True` it *multiplies* the node radii,
#     the same as in [`plot2d`][navis.plot2d]. Note that unlike `plot2d`, `radius=True` here builds
#     real tube geometry - so it is genuinely lit and occluded, not outlined.
#
# ## Pointing the camera
#
# `view` takes the same axis pairs as [`plot2d`][navis.plot2d]: the first entry is the axis pointing
# right, the second the axis pointing up. Prefix with `-` to flip:

fig, axes = plt.subplots(1, 4, figsize=(17, 5))
for ax, view in zip(axes, [("x", "y"), ("x", "-y"), ("x", "-z"), ("z", "-y")]):
    navis.plot3d(n, snapshot=True, ax=ax, view=view, color="#4C6EF5")
    ax.set_title(f"view={view}", fontsize=11, y=0.98)
    ax.set_axis_off()
plt.tight_layout()

# %%
# You can also hand it a string (`"xy"`, `"x-z"`), `None` to leave the camera alone, or a camera
# state dict from `viewer.get_view()` - which is what ties the two halves of this page together:
#
# ```python
# viewer = navis.plot3d(n)          # spin it around until it looks right
# state = viewer.get_view()         # grab the camera
# navis.plot3d(n, snapshot=True, view=state)   # render exactly that
# ```
#
# ## Snapshots keep their coordinates
#
# The rendered image is placed in *data* coordinates, not pixels. That means the `matplotlib` axes
# around it are real, and `hide_axes=False` shows them - so a snapshot can carry a scale bar, a
# labelled arrow or anything else you would add to an ordinary plot:

fig, ax = navis.plot3d(
    n, snapshot=True, view=("x", "-z"), hide_axes=False, color="#4C6EF5", figsize=(6.5, 6.5)
)

# these neurons are in 8 x 8 x 8 nm voxels, so one micron is 125 units
per_um = 1000 / n.units.to("nm").magnitude

ax.plot([6000, 6000 + 20 * per_um], [27800, 27800], lw=3, c="k", solid_capstyle="butt")
ax.text(6000 + 10 * per_um, 27400, "20 µm", ha="center", fontsize=10)

# %%
# !!! tip "Where the units come from"
#     `n.units` knows that these coordinates are 8 nm voxels, so the bar can be defined in microns and
#     converted - rather than hard-coding a number that silently breaks when you switch datasets.
#
# !!! warning "Only for axis-aligned views"
#     Data coordinates survive because an axis-aligned camera maps one data axis to each screen axis.
#     For an arbitrary camera that is no longer possible; the axes then carry world units in the view
#     plane instead - still to scale, so a scale bar still works, but no longer tied to a data axis.
#
# ## Framing and background
#
# `margin` pads the scene, `bgcolor` fills the canvas (by default the render is transparent and picks
# up whatever is underneath), and `figsize`/`dpi` control the `matplotlib` figure as usual:

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, (title, kwargs) in zip(
    axes,
    [
        ("margin=0.05 (default)", dict(margin=0.05)),
        ("margin=0.3", dict(margin=0.3)),
        ('bgcolor="#111"', dict(bgcolor="#111111")),
    ],
):
    navis.plot3d(n, snapshot=True, ax=ax, view=("x", "-z"), color="#4C6EF5", **kwargs)
    ax.set_title(title, fontsize=11, y=0.98)
    ax.set_axis_off()
plt.tight_layout()

# %%
# ## Several neurons
#
# Everything from the [coloring tutorial](../1a_plotting_general/tutorial_plotting_01_colors) applies
# unchanged - and here you get real occlusion for free, which is the whole reason 2D needs
# `depth_sort` and `halo` in the first place:

fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
navis.plot3d(nl, snapshot=True, ax=axes[0], view=("x", "-z"), radius=True)
axes[0].set_title("5 neurons, radius=True", fontsize=11, y=0.98)
navis.plot3d(
    nl, snapshot=True, ax=axes[1], view=("x", "-z"),
    color_by="strahler_index", palette="viridis", radius=True,
)
axes[1].set_title('color_by="strahler_index"', fontsize=11, y=0.98)
for ax in axes:
    ax.set_axis_off()
plt.tight_layout()

# %%
# !!! info "Connectors"
#     `connectors=True` adds the neuron's synapses as a scatter, here as in every other backend. It
#     has a page of its own: [connectors](../1d_plotting_misc/tutorial_plotting_misc_00_connectors).
#
# !!! info "Meshes and volumes"
#     Skeletons are lines, so there is not much surface to light. For neurons and neuropils that
#     *do* have a surface, see [3D meshes](../tutorial_plotting_3d_01_meshes) and
#     [3D volumes](../tutorial_plotting_3d_02_volumes).
