"""
Neuroglancer & CloudVolume
==========================

This tutorial will show you how to access data from `Neuroglancer` using `CloudVolume`.

[Neuroglancer](https://github.com/google/neuroglancer) is a WebGL-based viewer for volumetric data. You may have used it to browse
some of the recent large EM datasets. If you want to programmatically access/download these data, you need
[CloudVolume](https://github.com/seung-lab/cloud-volume). `CloudVolume` is an excellent Python library developed by
William Silversmith (Seung lab, Princeton) and others. While `CloudVolume` is not directly related to `Neuroglancer`,
it shares much of its functionality. As a rule of thumb: if you can view a dataset in `Neuroglancer`, you can download
that data using `CloudVolume`. For example:

1. [FlyWire](https://flywire.ai/) is a segmentation of an entire *Drosophila* brain. This dataset is very much work in progress and you
   will to register and apply for access. Check out [FAFBseg](https://fafbseg-py.readthedocs.io) for a fairly mature interface built on
   top of {{ navis }}.
2. [Google's flood-filling segmentation](http://fafb-ffn1.storage.googleapis.com/landing.html) of an entire *Drosophila* brain.
3. The Allen Institute's [MICrONs datasets](https://www.microns-explorer.org/). We have a separate [tutorial](../tutorial_remote_02_microns) on this!
4. The Janelia [hemibrain connectome](https://neuprint.janelia.org).

`CloudVolume` supports the backends/data formats of these and many up-and-coming datasets. You can use it to query the segmentation directly,
and to fetch meshes and skeletons (if available). {{ navis }} & friends provide simple interfaces for some of the datasets (see e.g. the
neuPrint and the MICrONs tutorials) but there is also some lower-level option to pull neurons into {{ navis }} via `CloudVolume`.

First of all, you will want to make sure to `cloud-volume` is installed and up-to-date:

```shell
pip install cloud-volume -U
```

Once that's done we can start pulling data using `cloud-volume`. In this example here, we will use the Google segmentation of the FAFB dataset:
"""

# %%
import navis
import cloudvolume as cv

# %%
# *Before* we connect to the database we have to "monkey patch" `cloudvolume` using [`navis.patch_cloudvolume`][]. That will
# teach `cloudvolume` to return {{ navis }} neurons:

# This needs to be run only once at the beginning of each session
navis.patch_cloudvolume()

# %%
# Now we can connect to our data source. Here we connect to the Google segmentation of the FAFB dataset:

# Don't forget to set `use_https=True` to avoid having to setup Google credentials
vol = cv.CloudVolume(
    "precomputed://gs://fafb-ffn1-20200412/segmentation", use_https=True, progress=False
)

# %%
# Fetch some (mesh) neurons:

# Setting `as_navis=True` we will get us MeshNeurons
m = vol.mesh.get([4335355146, 2913913713, 2137190164, 2268989790], as_navis=True, lod=3)
m

# %%
navis.plot3d(
    m,
    legend_orientation="h"  # few neurons, so we can afford a horizontal legend
    )

# %%

# And one 2d plot for the tutorial thumbnail
import matplotlib.pyplot as plt
fig, ax = navis.plot2d(m[1], method='2d', view=("x", "-y"))
ax.set_axis_off()
ax.grid(False)
plt.tight_layout()

# %%
# This also works for skeletons. Note though that not all datasets contain precomputed skeletons! For those cases you
# might want to check out [`navis.skeletonize`][] to skeletonize neuron meshes.

sk = vol.skeleton.get([4335355146, 2913913713, 2137190164, 2268989790], as_navis=True)
sk

# %%
# !!! experiment "Try it out!"
#     If you are working a lot with NeuroGlancer and need to e.g. generated or parse URLs, you might want to check out the
#     [`nglscenes`](https://github.com/schlegelp/nglscenes) package.
