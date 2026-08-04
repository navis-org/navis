"""
Neuroglancer & CloudVolume
==========================
<!-- difficulty: intermediate -->

Pull neurons and meshes from Neuroglancer sources via CloudVolume.

[Neuroglancer](https://github.com/google/neuroglancer) is a WebGL-based viewer for volumetric data. You may have used it to browse
some of the recent large EM datasets. If you want to programmatically access/download these data, you need
[CloudVolume](https://github.com/seung-lab/cloud-volume). `CloudVolume` is an excellent Python library developed by
William Silversmith (Seung lab, Princeton) and others. While `CloudVolume` is not directly related to `Neuroglancer`,
it shares much of its functionality. As a rule of thumb: if you can view a dataset in `Neuroglancer`, you can download
that data using `CloudVolume`. For example:

| Dataset | Description |
|---------|-------------|
| [FlyWire](https://flywire.ai/) | Segmentation of an entire *Drosophila* brain. Very much work in progress; you'll need to register and apply for access. See [FAFBseg](https://fafbseg-py.readthedocs.io) for a mature {{ navis }}-based interface. |
| [Google FFN](http://fafb-ffn1.storage.googleapis.com/landing.html) | Google's flood-filling segmentation of an entire *Drosophila* brain. |
| [MICrONS](https://www.microns-explorer.org/) | The Allen Institute's datasets - see the dedicated [MICrONS tutorial](../tutorial_remote_02_microns). |
| [hemibrain](https://neuprint.janelia.org) | The Janelia hemibrain connectome. |

You can find the source for the data you want to access by right-clicking on the layer in question and selecting the "Source" tab on the right:

![Neuroglancer source](../../../_static/neuroglancer_source.png)

`CloudVolume` supports pretty much all the backends/data formats that neuroglancer does. You can use it to programmatically query the segmentation itself,
and to fetch meshes and skeletons (if available). {{ navis }} & friends provide simple interfaces for some of the datasets (see e.g. the
neuPrint and the MICrONs tutorials) but there is also some lower-level option to pull neurons into {{ navis }} via `CloudVolume`.

First of all, you will want to make sure `cloud-volume` is installed and up-to-date:

```shell
pip install cloud-volume -U
```

!!! note "Network access"
    This tutorial downloads data from remote Neuroglancer sources, so it needs an internet connection.
    Some datasets (e.g. FlyWire) additionally require you to register and authenticate.

Once that's done we can start pulling data using `cloud-volume`. In this example here, we will use the Google segmentation of the FAFB dataset:
"""

# %%
import navis
import cloudvolume as cv

# This tutorial pulls from a remote source that may be unreachable when the docs
# are built, so we pin a static thumbnail rather than rely on a scraped figure.
# mkdocs_gallery_thumbnail_path = '_static/neuroglancer_source.png'

# %%
# !!! important "Patch before you connect"
#     [`navis.patch_cloudvolume`][] monkey-patches `cloudvolume` so that its `get()` methods return
#     {{ navis }} neurons. Run it *before* you create the `CloudVolume` object, and only once per session.

# This needs to be run only once at the beginning of each session
navis.patch_cloudvolume()

# %%
# Now we can connect to our data source. Here we connect to the Google segmentation of the FAFB dataset:

# Don't forget to set `use_https=True` to avoid having to setup Google credentials!
vol = cv.CloudVolume(
    "precomputed://gs://fafb-ffn1-20200412/segmentation", use_https=True, progress=False
)

# %%
# Fetch neuron meshes:

# Setting `as_navis=True` will get us Meshes
m = vol.mesh.get([4335355146, 2913913713, 2137190164, 2268989790], as_navis=True, lod=3)
m

# %%
# !!! note "Shortcut"
#     Instead of `vol.mesh.get(..., as_navis=True)` you can also use the shortcut
#     `vol.mesh.get_navis(...)` which is equivalent.

# %%
# Plot!
navis.plot3d(
    m,
    legend_orientation="h",  # few neurons, so we can afford a horizontal legend
)

# %%

# And one 2D plot (for the tutorial thumbnail)
import matplotlib.pyplot as plt

fig, ax = navis.plot2d(m[1], method="2d", view=("x", "-y"))
ax.set_axis_off()
ax.grid(False)
plt.tight_layout()

# %%
# This also works for skeletons:

sk = vol.skeleton.get([4335355146, 2913913713, 2137190164, 2268989790], as_navis=True)
sk

# %%
# Note that not all datasets contain precomputed skeletons! In that case you
# could download the meshes and use [`navis.skeletonize`][] to skeletonize them.
#
# !!! experiment "Try it out!"
#     If you are working a lot with NeuroGlancer and need to e.g. generate or parse URLs, you might want to check out the
#     [`nglscenes`](https://github.com/schlegelp/nglscenes) package.
#
# *[EM]: Electron Microscopy.
