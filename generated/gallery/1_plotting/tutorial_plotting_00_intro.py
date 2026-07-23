"""
Plotting Overview
=================

This tutorial gives an overview of the plotting capabilities of NAVis. We will cover 2D
and 3D plotting with various backends and their pro's and con's.

{{ navis }} contains functions for (static) 2D and (interactive) 3D plotting. These functions
can use various different backends for plotting. For 2D plots we use [`matplotlib`](http://www.matplotlib.org)
and for 3D plots we use either [`octarine`](https://schlegelp.github.io/octarine/),
[`vispy`](http://www.vispy.org), [`plotly`](http://plot.ly) or [`k3d`](https://k3d-jupyter.org).

Which plotting method (2D/3D) and which backend (octarine, plotly, etc.) to use depends on
what you are after (e.g. static, publication quality figures vs interactive data exploration)
and your environment (e.g. Jupyter/VS code or terminal). Here's a quick summary:

| Backend    | Used in              | Pros                                                                 | Cons                                                                                      |
|------------|----------------------|----------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| matplotlib | [`navis.plot2d`][]<br>[`navis.plot1d`][]<br>[`navis.plot_flat`][]   | - high quality (vector graphics!)<br>- works in Jupyter and terminal<br>- exports to vector graphics<br>- myriads of ways to adjust plots | - struggles with correct depth layering in complex scenes<br>- not very interactive (although you can adjust perspective)<br> - slow for large scenes<br>- not good for voxel data (e.g. image volumes) |
| octarine   | [`navis.plot3d`][]   | - blazingly fast thanks to WGPU backend<br>- works in terminal and Jupyter<br>- very interactive | - may not work on older systems (use `vispy` instead)<br>- not persistent (i.e. dies with notebook kernel)<br>- can't share interactive plot (screenshots only) |
| vispy      | [`navis.plot3d`][]   | - very interactive<br>- good render quality and performance    | - can't share interactive plot (screenshots only)<br>- not persistent (i.e. dies with notebook kernel)<br>- deprecated in favor of octarine |
| plotly     | [`navis.plot3d`][]   | - works "inline" for Jupyter environments<br>- persistent (i.e. plots get saved alongside notebook)<br>- can produce offline HTML plots for sharing  | - not very fast for large scenes<br>- large file sizes (i.e. makes for large `.ipynb` notebook files)<br>- horrendous for voxel data (i.e. images) |
| k3d        | [`navis.plot3d`][]   | - works "inline" for Jupyter environments<br>- super fast and performant<br>- in memory (i.e. does not increase notebook file size) | - does not work in terminal sessions<br>- not persistent (i.e. dies with notebook kernel)<br>- can't share interactive plot (screenshots only) |

In theory there is feature parity across backends but due to built-in limitations there are minor differences.

If you installed {{ navis }} via `pip install navis[all]` all of the above backends should be available to you.
If you ran a minimal install via `pip install navis` you may need to install the backends separately
- {{ navis }} will complain if you try to use a backend that is not installed!

!!! note

    The plots in this tutorial are optimized for light-mode. If you are using dark-mode, you may have trouble seeing e.g. axis or labels.

## 2D plots

This uses `matplotlib` to generate static 2D plots. The big advantage is that you can save these plots as vector graphics.
Unfortunately, matplotlib's capabilities regarding 3D data are limited. The main problem is that depth (z) is only
simulated by trying to layer objects (lines, vertices, etc.) according to their z-order rather than doing proper rendering which
is why you might see some neurons being plotted in front of others even though they are actually behind them. It's still great
for plotting individual neurons or small groups thereof!

Let's demonstrate with a simple example using the default settings:
"""

# %%
import navis
import matplotlib.pyplot as plt

nl = navis.example_neurons(kind="skeleton")

# Plot using default settings
fig, ax = navis.plot2d(nl, view=("x", "-z"), method="2d")
plt.tight_layout()

# %%
# !!! note
#     We set `view("x", "-z")` above to get a frontal view of the example neurons. You may need to adjust this depending on
#     the orientation of your neurons.
#
# Above plot used the default `matplotlib` 2D plot. You might notice that the plot looks rather "flat" - i.e. neurons seem
# to be layered on top of each other without intertwining. That is one of the limitations of `matplotlib`'s 3d backend.
# We can try to ameliorate this by adjust the `method` parameter:

# Plot settings for more complex scenes - comes at a small performance cut though
fig, ax = navis.plot2d(nl, method="3d_complex", view=("x", "-z"))
plt.tight_layout()

# %%
# Looks better now, doesn't it? Now what if we wanted to adjust the perspective? For 3d axes, `matplotlib` lets
# us adjust the viewing angle by setting the `elev`, `azim` and `roll` attributes.
# See also [this official explanation](https://matplotlib.org/stable/api/toolkits/mplot3d/view_angles.html).
#
# Let's give that a shot:

# %%
# Plot again
fig, ax = nl.plot2d(
    method="3d_complex", view=("x", "-z"), non_view_axes3d="show", radius=True
)

# Change view to see the neurons from a different angle
ax.elev = -20
ax.azim = 45
ax.roll = 180

plt.tight_layout()

# %%
# !!! note
#     Did you note that we set `non_view_axes3d='show'` in above example? By default, {{ navis }} hides the axis that is parallel to
#     the viewing direction is hidden to not clutter the image. Because we were going to change the perspective,
#     we set it to `show`. FYI: if the plot is rendered in a separate window (e.g. if you run Python from
#     terminal), you can change the perspective by dragging the image.
#
# We can use this to generate small animations:
#
# ```python
# # Render 3D rotation
# for i in range(0, 360, 10):
#    # Change rotation
#    ax.azim = i
#    # Save each incremental rotation as frame
#    plt.savefig('frame_{0}.png'.format(i), dpi=200)
# ```
#
# ![rotation](../../../_static/rotation.gif)

# %%
# ## 3D plots
#
# By "3D plots" we typically mean interactive 3D plots as opposed to the (mostly) static 2D or semi-3D plots above.
# 3D plots are great for great for exploring your data interactively but you can also use them to generate high-quality
# static images.
#
# As laid out at the top of this page: for 3D plots, we are using either [octarine](https://schlegelp.github.io/octarine/),
# [vispy](https://github.com/vispy/vispy), [plotly](https://plotly.com/) or [k3d](https://k3d-jupyter.org). In brief:
#
# | backend             | Jupyter | Terminal |
# |---------------------|---------|----------|
# | octarine            | yes     | yes      |
# | plotly              | yes     | yes but only via export to html |
# | vispy (depcrecated) | yes     | yes      |
# | k3d                 | yes     | no       |
#
# By default, the choice is automatic and depends on (1) what backends are installed and (2) the context:
#
#   - from IPython/Terminal/scripts: octarine :material-code-greater-than: vispy :material-code-greater-than: plotly
#   - from Jupyter Lab/Notebook: plotly :material-code-greater-than: octarine :material-code-greater-than: k3d
#
# You can always force a specific backend using the `backend` parameter in [`navis.plot3d`][]:
#
# === "Automatic (default)"
#     ```python
#     n = navis.example_neurons()
#     navis.plot3d(n)
#     ```
# === "Octarine"
#     ```python
#     n = navis.example_neurons()
#     navis.plot3d(n, backend='octarine')
#     ```
# === "Plotly"
#     ```python
#     n = navis.example_neurons()
#     navis.plot3d(n, backend='plotly')
#     ```
# === "Vispy"
#     ```python
#     n = navis.example_neurons()
#     navis.plot3d(n, backend='vispy')
#     ```
# === "k3d"
#     ```python
#     n = navis.example_neurons()
#     navis.plot3d(n, backend='k3d')
#     ```
#
# Alternatively, you can also set the default backend using an environment variable. For example:
# ```shell
# export NAVIS_PLOT3D_BACKEND="octarine"
# ```
#
# !!! important "Google Collaboratory"
#     The `jupyter_rfb` used by Octarine and Vispy to render 3D plots in Jupyter does not work in Google Collaboratory.
#     If you are using Google Collaboratory, we recommend you use the plotly backend.
#
# With that out of the way, let's have a look at some 3D plots! You will notice that for the `octarine`, `vispy` and `k3d`
# backends we're just showing screenshots - that's because their interactive plots can't be embedded into this documentation.
#
# ### Octarine/Vispy
#
# Octarine and Vispy are pretty similar in that they both work via a `Viewer` object which allows you to interactively
# add/remove objects, change colors, etc. The main difference is that Octarine uses modern WGPU instead of OpenGL which makes
# it much faster than Vispy:
#
# === "Octarine"
#     ```python
#     nl = navis.example_neurons()
#     viewer = navis.plot3d(nl, backend='octarine')
#     ```
#     ![octarine](../../../_static/octarine_viewer.png)
# === "Vispy"
#     ```python
#     nl = navis.example_neurons()
#     viewer = navis.plot3d(nl, backend='vispy')
#     ```
#     ![vispy](../../../_static/vispy_viewer.png)
#
# !!! important
#     If you are using Octarine/Vispy from Jupyter, we may have to explicitly call the `viewer.show()` method *in the last line of the cell*
#     for the widget to show up.
#
#
# A few important notes regarding the Octarine/Vispy backends:
#
# - The `viewer` is dynamic: you can keep adding/removing items in other cells. However, it will die with the kernel (unlike `plotly`)!
# - By default, {{ navis }} will track the "primary" viewer and subsequent calls of [`navis.plot3d`][] will add object to that primary viewer
#
#     * if you want to force a new viewer: `navis.plot3d(nl, viewer='new')`
#     * if you want to add to a specific viewer: `navis.plot3d(nl, viewer=viewer)`
#
# - You can dynamically resize the canvas (in Jupyter by dragging the lower right corner)
# - For Jupyter: the rendering runs in your Jupyter Kernel and the frames are sent to Jupyter via a remote frame buffer (`jupyter_rbf`). If your
#   Jupyter kernel runs on a remote machine you might experiences some lag depending on the connection speed and quality.
#
# Some important methods for the `viewer` object:
#
# ```python
# # Close the viewer
# viewer.close()
#
# # Close the current primary viewer
# navis.close3d()
#
# # Add neurons to the primary viewer
# navis.plot3d(nl)
#
# # Add neurons to a specific
# navis.plot3d(nl, viewer=viewer)
#
# # Clear viewer
# viewer.clear3d()
#
# # Clear the primary viewer
# navis.clear3d()
# ```
#
# The Octarine viewer itsel has a bunch of neat features - check out the [documentation](https://schlegelp.github.io/octarine/) to learn more.
#
# !!! important
#     The Vispy backend is deprecated and will be removed in future versions of {{ navis }}. If you can please switch to Octarine.
#     If you have any issues with Octarine and want us to keep the Vispy backend, please let us know!

# %%
# ### K3d
#
# `k3d` plots work in Jupyter (and only there) but unlike `plotly` don't persist across sessions. Hence we will only briefly demo
# them using static screenshots and then move on to plotly. Almost everything you can do with the `plotly` backend can also be done
# with `k3d` (or `octarine/vispy` for that matter)!

# %%
p = navis.plot3d(nl, backend="k3d")

# %%
# ![k3d](../../../_static/k3d_viewer.png)
#
# ### Plotly
#
# Last but not least, we have the `plotly` backend. This is the only backend which allows us to embed interactive 3D plots into
# this documentation. The main advantage of `plotly` is that it works "inline" in Jupyter notebooks and that you can export the
# plots as standalone HTML files. The main disadvantage is that it can be quite slow for large scenes and that the resulting
# `.ipynb` notebook files can get quite large.
#
# Using plotly as backend generates "inline" plots by default (i.e. they are rendered right away):

navis.plot3d(
    nl,
    backend="plotly",
    connectors=False,
    radius=True,  # use node radii for skeletons
    legend_orientation="h",  # horizontal legend (more space for plot)
)

# %%
# Instead of inline plotting, you can also export your plotly figure as 3D html file that can be opened in any browser:
#
# ```python
# import plotly
#
# # Prevent inline plotting
# fig = nl.plot3d(backend='plotly', connectors=False, width=1400, height=1000, inline=False)
#
# # Save figure to html file
# plotly.offline.plot(fig, filename='~/Documents/3d_plot.html')
# ```
#
# ### Navigating the 3D viewers
#
# | Action          | Octarine/Vispy | Plotly | K3d |
# |-----------------|----------------|--------|-----|
# | Rotate          | Left Mouse + Drag          | Left Mouse + Drag  | Left Mouse + Drag  |
# | Zoom            | Mousewheel                 | Mousewheel         | Mousewheel         |
# | Pan             | Right Mouse + Drag         | Right Mouse + Drag | Right Mouse + Drag |
# | Hide/Unhide<br>objects  | `viewer.hide()`<br>`viewer.show()` | Click on legend | Click on legend |
#
# !!! note "Camera rotation"
#     If the camera rotation using plotly causes problems, try clicking on the "Orbital rotation" in the upper right tool bar.
#     ![plotly orbital](../../../_static/plotly_orbital.png)
#
# %%
# ## High-quality renderings
#
# Above we demo'ed making a little GIF using matplotlib. While that's certainly fun, it's not
# very high production value. For high quality videos and renderings I recommend you check out
# the tutorial on navis' [Blender interface](../3_interfaces/tutorial_interfaces_02_blender). Here's a little taster:
#
#  <iframe width="560" height="315" src="https://www.youtube.com/embed/wl3sFG7WQJc" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
#
