"""
Transformations
===============

Move neurons between brain templates and mirror them across the midline.

As of version `0.5.0`, {{ navis }} can transform and mirror spatial data such as neurons. The functionality
splits into high-level functions (which most users want) and the low-level building blocks. {{ navis }}
supports several transform types, each backed by a class you can also construct yourself:

| Transform type | navis class | Built from |
|----------------|-------------|------------|
| [CMTK](https://www.nitrc.org/projects/cmtk/) warp | [`navis.transforms.CMTKtransform`][] | a CMTK registration |
| [Hdf5](https://github.com/saalfeldlab/template-building/wiki/Hdf5-Deformation-fields) deformation field | [`navis.transforms.H5transform`][] | an HDF5 deformation field |
| [Elastix](https://elastix.lumc.nl/) | [`navis.transforms.ElastixTransform`][] | an elastix transformation file |
| landmark thin-plate spline | [`navis.transforms.TPStransform`][] | landmark correspondences |
| affine | [`navis.transforms.AffineTransform`][] | a 4x4 matrix |

## flybrains

{{ navis }} provides the machinery but ships no transforms of its own. Here we use
[flybrains](https://github.com/navis-org/navis-flybrains), which registers a suite of *Drosophila*
transforms directly with {{ navis }}. Registering your own is covered [further down](#low-level-functions).

!!! note "Requires flybrains"
    This tutorial needs [flybrains](https://github.com/navis-org/navis-flybrains) installed **and** its
    bridging registrations downloaded. Follow the flybrains instructions - including the data-download
    step - before running the code below.
"""

import flybrains

# %%
# Importing `flybrains` automatically registers the transforms with {{ navis }}. This in turn allows {{ navis }} to plot a
# sequence of bridging transformations to map between any connected template spaces.
#
# ![Flybrain Bridging Graph](https://raw.githubusercontent.com/navis-org/navis-flybrains/main/_static/bridging_graph.png)
#
# In addition to those bridging transforms, `flybrains` also contains mirror registrations (we will cover those later), meta data
# and meshes for the template brains:

# This is the Janelia "hemibrain" template brain
print(flybrains.JRCFIB2018F)

# %%
import navis
import matplotlib.pyplot as plt

# This is the hemibrain neuropil surface mesh
fig, ax = navis.plot2d(flybrains.JRCFIB2018F, view=("x", "-z"))
plt.tight_layout()

# %%
# You can check the registered transforms like so:

navis.transforms.registry.summary()

# !!! note
#     The documentation is built in an environment with a minimal number of transforms registered. If you have installed
#     and imported `flybrains`, you should see a lot more than what is shown above.

# %%
# ## Using ``xform_brain``
#
# For high-level transforming, you will want to use [`navis.xform_brain`][]. This function takes a `source` and `target` argument
# and tries to find a bridging sequence that gets you to where you want. Let's try it out:
#
# !!! info
#     Incidentally, the example neurons that {{ navis }} ships with are from the Janelia hemibrain project and are therefore in
#     `JRCFIB2018raw` space ("raw" means uncalibrated voxel space which is 8x8x8nm for this dataset). We will be using those
#     but there is nothing stopping you from using the {{ navis }} interface with neuPrint (the tutorials on
#     [interfaces](../#interfaces)) to fetch your favourite hemibrain neurons and transform those.

# Load the example hemibrain neurons (JRCFIB2018raw space)
nl = navis.example_neurons()
nl

# %%
fig, ax = navis.plot2d([nl, flybrains.JRCFIB2018Fraw], view=("x", "-z"))
plt.tight_layout()

# %%
# Let's say we want these neurons in `JRC2018F` template space. Before running the transform, it's worth
# tracing the path [`navis.xform_brain`][] will take through the bridging graph:
#
# ??? info "What is JRC2018F?"
#     `JRC2018F` is a standard brain made from averaging over multiple fly brains. See
#     [Bogovic et al., 2020](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0236495) for details.
#
# ```mermaid
# graph LR
#     A["JRCFIB2018Fraw"] -->|"affine: voxels to nm"| B["nanometers"];
#     B -->|"affine: nm to um"| C["JRCFIB2018Fum"];
#     C -->|"Hdf5 (Saalfeld lab)"| D["JRC2018F"];
# ```
#
# Two affine steps take us from raw voxels to micrometers, then a Saalfeld-lab Hdf5 transform maps
# `JRCFIB2018Fum` onto `JRC2018F`. Arrows show each transform's forward direction, but {{ navis }} can invert
# any of them to traverse the graph in reverse.

xf = navis.xform_brain(nl, source="JRCFIB2018Fraw", target="JRC2018F")

# %%
# Let's see if it worked:

# %%
# Plot the transformed neurons and the JRC2018F template brain
fig, ax = navis.plot2d([xf, flybrains.JRC2018F], color="r", view=("x", "-y"))
plt.tight_layout()

# %%
# Take a look at [`navis.xform_brain`][]'s parameters to fine-tune the transform.
#
# ## Using ``mirror_brain``
#
# [`navis.mirror_brain`][] mirrors neurons across the midline - e.g. from the left to the right side of a brain.
#
# ??? info "How mirroring works"
#     Mirroring happens in two steps:
#
#     1. Reflect coordinates about the midpoint of the mirror axis (an affine transformation).
#     2. Optionally apply a warping transform to compensate for left/right asymmetries.
#
#     Step 1 needs the length of the mirror axis, so - as with registered transforms - {{ navis }} must have
#     meta data about the template space (its bounding box) available.
#
#     Step 2 is optional: `JRC2018F` and `JRC2018U` are averaged from many brains and are already mirror
#     symmetrical, so they need no warping. `flybrains` does ship mirror transforms for others, e.g. `FCWB`,
#     `VNCIS1` or `JFRC2`.
#
# Since our neurons are already in `JRC2018F` space, let's try mirroring them:

mirrored = navis.mirror_brain(xf, template="JRC2018F")

# %%
fig, ax = navis.plot2d(
    [xf, mirrored, flybrains.JRC2018F], color=["r"] * 5 + ["g"] * 5, view=("x", "-y")
)
plt.tight_layout()

# %%
# As noted above, this only works if the `template` is registered with {{ navis }} and contains its bounding box.
# If you only have the bounding box but no template brain, check out the lower-level function [`navis.transforms.mirror`][].
#
# ## Low-level functions
#
# ### Adding your own transforms
#
# To add your own transform, pick the matching class from the table at the top of this page and construct it
# directly. Here we build a thin-plate spline transform with [`navis.transforms.TPStransform`][].
# If you look at the bridging graph again, you might note the `"FAFB14"` template brain: it stands for
# `"Full Adult Fly Brain"` (the `14` is a version number for the alignment). We will use landmarks to generate a
# mapping between this 14th and the previous 13th iteration.
#
# First we will grab the landmarks from the Saalfeld's lab [elm](https://github.com/saalfeldlab/elm) repository:

import pandas as pd

# These landmarks map between FAFB (v14 and v13) and a light level template
# We will use only the v13 and v14 landmarks
landmarks_v14 = pd.read_csv(
    "https://github.com/saalfeldlab/elm/raw/master/lm-em-landmarks_v14.csv", header=None
)
landmarks_v13 = pd.read_csv(
    "https://github.com/saalfeldlab/elm/raw/master/lm-em-landmarks_v13.csv", header=None
)

# Name the columns
landmarks_v14.columns = landmarks_v13.columns = [
    "label",
    "use",
    "lm_x",
    "lm_y",
    "lm_z",
    "fafb_x",
    "fafb_y",
    "fafb_z",
]

landmarks_v13.head()

# %%
# Now we can use those landmarks to generate a thin plate spline transform:

# %%
from navis.transforms.thinplate import TPStransform

tr = TPStransform(
    landmarks_source=landmarks_v14[["fafb_x", "fafb_y", "fafb_z"]].values,
    landmarks_target=landmarks_v13[["fafb_x", "fafb_y", "fafb_z"]].values,
)
# note: navis.transforms.MovingLeastSquaresTransform has similar properties

# %%
# The transform has a method that we can use to transform points but first we need some data in `FAFB14` space:

# Transform our neurons into FAFB 14 space
xf_fafb14 = navis.xform_brain(nl, source="JRCFIB2018Fraw", target="FAFB14")

# %%
# Now let's see if we can use the v14:octicons-arrow-right-24:v13 transform:

# Transform the nodes of the first two neurons
pts_v14 = xf_fafb14[:2].nodes[["x", "y", "z"]].values
pts_v13 = tr.xform(pts_v14)

# %%
# Quick check how the v14 and v13 coordinates compare:

# Original in black, transformed in red
fig, ax = navis.plot2d(pts_v14, scatter_kws=dict(c="k"), view=("x", "-y"))
_ = navis.plot2d(pts_v13, scatter_kws=dict(c="r"), ax=ax, view=("x", "-y"))

# %%
# Next, we will register this new transform with {{ navis }} so we can use it with the higher-level functions:

# Register the transform
navis.transforms.registry.register_transform(
    tr, source="FAFB14", target="FAFB13", transform_type="bridging"
)

# %%
# Now that's done we can use `FAFB13` with [`navis.xform_brain`][]:

# Transform our neurons into FAFB 14 space
xf_fafb13 = navis.xform_brain(xf_fafb14, source="FAFB14", target="FAFB13")

# %%
fig, ax = navis.plot2d(xf_fafb14, c='k', view=("x", "-y"))
_ = navis.plot2d(xf_fafb13, c='r', ax=ax)

# %%
# ### Registering Template Brains
#
# For completeness, let's also have a quick look at registering additional template brains.
#
# Template brains are represented in navis as [`navis.transforms.templates.TemplateBrain`][] and there is currently no canonical way of
# constructing them: you can associate as much or as little data with them as you like. However, for them to be useful they should have
# a `name`, a `label` and a `boundingbox` property.
#
# Minimally, you could do something like this:

# Construct template brain from base class
my_brain = navis.transforms.templates.TemplateBrain(
    name="My template brain",
    label="my_brain",
    boundingbox=[[0, 100], [0, 100], [0, 100]],
)

# Register with navis
navis.transforms.registry.register_templatebrain(my_brain)

# Now you can use it with mirror_brain:
import numpy as np

pts = np.array([[10, 10, 10]])
pts_mirrored = navis.mirror_brain(pts, template="my_brain")

# Plot the points
fig, ax = plt.subplots()
ax.scatter(pts[:, 0], pts[:, 1], c="k", alpha=1, s=50, label="Original")
ax.scatter(
    pts_mirrored[:, 0], pts_mirrored[:, 1], c="r", alpha=1, s=50, label="Mirrored"
)
ax.legend()

# %%
# While this is a working solution, it's not very pretty: for example, `my_brain` does have the default docstring and no fancy string
# representation (e.g. for `print(my_brain)`). I highly recommend you take a look at how [flybrains](https://github.com/navis-org/navis-flybrains)
# constructs and packages the templates.
#
# ## Acknowledgments
#
# Much of the transform module is modelled after functions written by Greg Jefferis for the [natverse](http://natverse.org). Likewise,
# [flybrains](https://github.com/navis-org/navis-flybrains) is a port of data collected by Greg Jefferis for `nat.flybrains` and `nat.jrcbrains`.
