"""
Skeletons from light-level data
===============================

This tutorial will show you how to extract skeletons from confocal microscopy stacks.

!!! important "This example is not executed"
    In contrast to almost all other tutorials, this one is not executed when the documentation is built.
    Consequently, it also does not display any actual code output or plots - images shown are statically
    embedded. The main reason for this is that the example requires downloading a large-ish file which
    is a pain in the neck to get to work in the CI enviroment.

Extracting neuron skeletons from microscopy data is a common but non-trivial task. There are about
as many ways to do this as there are people doing it - from fully manual to fully automated tracing.

In this tutorial, we will show you a fully automated way using a number of easy-to-install Python
packages. If this isn't for you, check out the [Alternatives](#alternatives) section at the end of this tutorial.

## Requirements:

Please make sure you have the following packages installed:

- [`pynrrd`](https://github.com/mhe/pynrrd) to load image stacks
  ```shell
  pip install pynrrd -U
  ```
- [`connected-components-3d`](https://github.com/seung-lab/connected-components-3d) (cc3d) to label connected components
  ``` shell
  pip install connected-components-3d -U
  ```
- [`kimimaro`](https://github.com/seung-lab/kimimaro) to extract the skeletons
  ```shell
  pip install kimimaro -U
  ```

## Preparing the data

The pipeline we're using here was designed for pre-segmented data, so there is little in the way
of dealing with noisy data. Fortunately, the image stack we will use is exceptionally clean which
makes the skeletonization process very straightforward.

In practice, you may have to do some pre-processing to clean up your data before running the skeletonization.
If your run-of-the-mill thresholding, denoising, etc. doesn't cut it, you can also try more advanced
segmentation techniques.

There are various fairly easy-to-use tools available for this, e.g. [Ilastik](https://www.ilastik.org) (see the
[pixel classification](https://www.ilastik.org/documentation/pixelclassification/pixelclassification) and
[voxel segmentation](https://www.ilastik.org/documentation/voxelsegmentation/voxelsegmentation) tutorials) or
[DeepImageJ](https://deepimagej.github.io/).

### Download Image Stack

As example data, we will use a confocal stack from the [Janelia Split-Gal4 collection](https://splitgal4.janelia.org/cgi-bin/splitgal4.cgi).
We picked the [SS00731](https://flweb.janelia.org/cgi-bin/view_splitgal4_imagery.cgi?line=SS00731)
line because it's already fairly clean as is and there are high-resolution stacks
with stochastic multi-color labeling of individual neurons available for download.

Scroll all the way to the bottom of the page and in the dropdown for the left-most image,
select "Download H5J stack: Unaligned".

![download](../../../_static/lm_tut/download.png)

### Convert to NRRD

Next, we need to open this file in [Fiji/ImageJ](https://imagej.net/software/fiji/) to convert it to
a format we can work with in Python:

1. Fire up Fiji/ImageJ
2. Drag & drop the `SS00731-20140620_20_C5-f-63x-ventral-Split_GAL4-unaligned_stack.h5j` file into Fiji
3. Go to "Image" -> "Colors" -> "Split Channels" to split the image into the channels
4. Discard all but the red "C1" channel with our neurons
5. Go to "Image" -> "Type" -> "8-bit" to convert the image to 8-bit (optional but recommended)
6. Save via "File" -> "Save As" -> "NRRD" and save the file as `neuron.nrrd`

![Z stack](../../../_static/lm_tut/C1.gif)

## Extracting the Skeleton

Now that we have that file in a format we can load it into Python, we can get started:
"""

# %%
import kimimaro
import nrrd
import navis
import cc3d
import numpy as np

# %%
# First load the image stack:

# `im` is numpy array, `header` is a dictionary
im, header = nrrd.read(
    "neuron.nrrd"
)

# %%
# Next, we need to find some sensible threshold to binarize the image. This is not strictly
# necessary (see the further note down) but at least for starters this more intuitive.

# Threshold the image
mask = (im >= 20).astype(np.uint8)

# %%
# You can inspect the mask to see if the thresholding worked as expected:
# ```python
# import matplotlib.pyplot as plt
# plt.imshow(mask.max(axis=2))
# ```
#
# With the `octarine` backend, you can also visualize the volume in 3D:
# ```python
# # spacing can be found in the `header` dictionary
# import octarine as oc
# v = oc.Viewer()
# v.add_volume(mask, spacing=(.19, .19, .38))
# ```
#
# ![mask](../../../_static/lm_tut/mask.png)
#
# A couple notes on the thresholding:
#
# - feel free to test the thresholding in e.g. ImageJ/Fiji
# - remove as much background as possible without disconnecting neurites
# - perfection is the enemy of progress: we can denoise/reconnect during postprocessing
#
# Next, we we need to label the connected components in the image:

# %%
# Extract the labels
labels, N = cc3d.connected_components(mask, return_N=True)

# %%
# Visualize the labels:
# ```python
# import cmap
# import octarine as oc
# v = oc.Viewer()
# v.add_volume(labels, spacing=(.19, .19, .38), color=cmap.Colormap('prism'))
# ```
#
# ![labels](../../../_static/lm_tut/labels.png)
#
# !!! experiment
#     `cc3d.connected_component` also works with non-thresholded image - see the `delta` parameter.

# Collect some statistics
stats = cc3d.statistics(labels)

print("Total no. of labeled componenents:", N)
print("Per-label voxel counts:", np.sort(stats["voxel_counts"])[::-1])
print("Label IDs:", np.argsort(stats["voxel_counts"])[::-1])

# %%
# ```
# Total no. of labeled componenents: 37836
# Per-label voxel counts: [491996140    527374    207632 ...         1         1         1]
# Label IDs: [    0  6423  6091 ... 22350 22351 18918]
# ```
#
# Note how label `0` has suspiciously many voxels? That's because this is the background label.
# We need to make sure to exlude it from the skeletonization process:
to_skeletonize = np.arange(1, N)


# %%
# Now we can run the actual skeletonization!
#
# !!! note "Skeletonization paramters"
#     There are a number of parameters that are worth explaining
#     first because you might want to tweak them for your data:
#
#     - `scale` & `const`: control how detailed your skeleton will be: lower = more detailed but more noise
#     - `anisotropy`: controls the voxel size - see the `header` dictionary for the voxel size of our image
#     - `dust_threshold`: controls how small connected components are skipped
#     - `object_ids`:  a list of labels to process (remember that we skipped the background label)
#     - `max_path`: if this is set, the algorithm will only process N paths in each skeleton - you can use
#       this to finish early (e.g. for testing)
#
#     See the [`kimimaro` repository](https://github.com/seung-lab/kimimaro) for a detailed explanation
#     of the parameters!

skels = kimimaro.skeletonize(
    labels,
    teasar_params={
        "scale": 1.5,
        "const": 1,  # physical units (1 micron in our case)
        "pdrf_scale": 100000,
        "pdrf_exponent": 4,
        "soma_acceptance_threshold": 3.5,  # physical units
        "soma_detection_threshold": 1,  # physical units
        "soma_invalidation_const": 0.5,  # physical units
        "soma_invalidation_scale": 2,
        "max_paths": None,  # default None
    },
    object_ids=list(to_skeletonize), # process only the specified labels
    dust_threshold=500,  # skip connected components with fewer than this many voxels
    anisotropy=(0.19, .19, 0.38),  # voxel size in physical units
    progress=True,  # show progress bar
    parallel=6,  # <= 0 all cpu, 1 single process, 2+ multiprocess
    parallel_chunk_size=1,  # how many skeletons to process before updating progress bar
)

# %%
# `skels` is a dictionary of `{label: cloudvolume.Skeleton}`. Let's convert these to {{ navis }} neurons:

# Convert skeletons to NAVis neurons
nl = navis.NeuronList([navis.read_swc(s.to_swc(), id=i) for i, s in skels.items()])

# %%
# Based on the voxel sizes in `stats`, we can make an educated guess that label `6423` is one of our neurons.
# Let's visualize it in 3D:
#
# ```python
# import octarine as oc
# v = oc.Viewer()
# v.add_neurons(nl.idx[6423], color='r', linewidth=2, radius=False))
# v.add_volume(im, spacing=(.19, .19, .38), opacity=.5)
# ```
#
# ![stack animation](../../../_static/lm_tut/stack.gif)
#
# This looks pretty good off the bat! Now obviously we will have the other large neuron (label `6091`)
# plus bunch of smaller skeletons in our NeuronList. Let's have a look at those as well:
#
# ![all skeletons](../../../_static/lm_tut/all_skeletons.png)
#
# Zooming in on `6091` you will see that it wasn't fully skeletonized: some of the branches are missing
# and others are disconnected. That's either because our threshold for the mask was too high (this neuron
# had a weaker signal than the other) and/or we dropped too many fragments during the skeletonization process
# (see the `dust_threshold` parameter).
#
# ![zoom in](../../../_static/lm_tut/zoom_in.png)
#
# ## Alternatives
#
# If the pipeline described in this tutorial does not work for you, there are a number of alternatives:
#
# 1. [Simple Neurite Tracer](https://imagej.net/plugins/snt/index) is a popular ImageJ plugin for semi-automated tracing
# 2. Folks at the Allen Institute for Brain Science have published a [protocol for reconstructing neurons](https://portal.brain-map.org/explore/toolkit/morpho-reconstruction/vaa3d-mozak)
# 3. [NeuTube](https://neutracing.com/tutorial/) is an open-source software for reconstructing neurongs from fluorescence microscopy images
#
# ## Acknowledgements
#
# The packages we used here were written by the excellent Will Silversmith from the Seung lab in Princeton.
# The image stack we processed is from the Janelia Split-Gal4 collection and was published as part of the
# [Cheong, Eichler, Stuerner, _et al._ (2024)](https://elifesciences.org/reviewed-preprints/96084v1) paper.

# %%

# mkdocs_gallery_thumbnail_path = '_static/lm_tut/z_stack.png'