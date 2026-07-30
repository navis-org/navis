"""
NBLAST using light-level data
=============================

Match light-level neurons against EM skeletons with NBLAST.

!!! important "This example is not executed"
    In contrast to almost all other tutorials, this one is not executed when the documentation is built.
    Consequently, it also does not show any code output or figures. That's because this example requires
    downloading a large dataset (~1.8GB) and running an NBLAST against it, it is simply not feasible to
    run this as part of the documentation build process.

One of the applications of NBLAST is to match neurons across data sets. Here we will illustrate this by taking a light-level,
confocal image stack and finding the same neuron in an EM connectome.

Specifically, we will use an image from Janelia's collection of split-Gal4 driver lines and match it against the neurons in the
[hemibrain connectome](https://neuprint.janelia.org).

!!! note "Requirements"
    Before you start, make sure:

    - {{ navis }} is installed and up-to-date
    - [`flybrains`](https://github.com/navis-org/navis-flybrains) is installed and you've downloaded the Saalfeld lab's and VFB
      bridging transforms (see the `flybrains.download_...` functions)
    - you've downloaded and extracted [hemibrain-v1.2-skeletons.tar](https://storage.googleapis.com/hemibrain/v1.2/hemibrain-v1.2-skeletons.tar.gz)
      (kindly provided by Stuart Berg, Janelia Research Campus)

Next we need to pick an image stack to use as query. You can browse the expression patterns of the Janelia split-Gal4 lines
[here](https://splitgal4.janelia.org/cgi-bin/splitgal4.cgi). I ended up picking `LH1112` which is a very clean line containing a couple
of WED projection neurons. Among other data, you can download these stacks as "gendered" (i.e. female or male) or "unisex" space.
Unfortunately, all image stacks are in Janelia's `.h5j` format which I haven't figured out how to read straight into Python.

Two ways to get a `.nrrd`:

=== "Fiji"
    Load the stack into Fiji and save the GFP signal channel as a `.nrrd` file.

=== "VirtualFlyBrain"
    Go to [VirtualFlyBrain](http://www.virtualflybrain.org/), search for your line of interest (LH1112; not all lines are
    available on VFB) and download the "Signal(NRRD)" at the bottom of the Term Info panel on the right-hand side.

I went with VirtualFlyBrain and downloaded `VFB_001013cg.nrrd`. This is the neuron we'll be searching for:

![LH1112 z-stack](https://s3.amazonaws.com/janelia-flylight-imagery/Lateral+Horn+2019/LH1112/LH1112-20150313_46_A2-f-20x-brain-Split_GAL4-multichannel_mip.png)

Let's get started!
"""

# mkdocs_gallery_thumbnail_path = '_static/nblast_hemibrain_thumbnail.png'

# %%
import navis

# %%
# First we need to load the image stack and turn it into dotprops:

query = navis.read_nrrd("VFB_001013cg.nrrd", output="dotprops", k=20, threshold=100)
query.id = "LH1112"  # manually set the ID to the Janelia identifier
query

# %%

# Inspect the results
navis.plot3d(query)

# %%
# !!! note
#     Depending on your image you will have to play around with the `threshold` parameter to get a decent dotprop representation.
#
# Next we need to load the hemibrain skeletons and convert them to dotprops:

# Make sure to adjust the path to where you extracted the skeletons
sk = navis.read_swc(
    "hemibrain-v1.2-skeletons/", include_subdirs=True, fmt="{name,id:int}.swc"
)

# %%
# These 97k skeletons include lots of small fragments - there are only ~25k proper neurons or substantial fragments thereof in the hemibrain dataset.
# So to make our life a little easier, we will keep only the largest 30k skeletons:

# %%
sk.sort_values("n_nodes")
sk = sk[:30_000]
sk

# %%
# Next up: turning those skeletons into dotprops:

# %%
# Note that we are resampling to 1 point for every micron of cable
# Because these skeletons are in 8nm voxels we have to use 1000/8
targets = navis.make_dotprops(sk, k=5, parallel=True, resample=1000 / 8)

# %%
# !!! note
#     Making the dotprops may take a while (mostly because of the resampling). You can dedicate more
#     cores via the `n_cores` parameter. It may also make sense to save the dotprops for future
#     use e.g. by pickling them.
#
# Last but not least we need to convert the image's dotprops from their current brain space (`JRC2018U`, `U` for "unisex")
# to hemibrain (`JRCFIB2018Fraw`, `raw` for voxels) space.

# %%
import flybrains

query_xf = navis.xform_brain(query, source="JRC2018U", target="JRCFIB2018Fraw")

# %%
# Now we can run the actual NBLAST:

# %%

# Note that we convert from the JRCFIB2018F voxel (8x8x8nm) space to microns
scores = navis.nblast(query_xf / 125, targets / 125, scores="mean")
scores.head()

# %%
# ??? example "Speeding up NBLAST"
#     You can greatly speed up NBLAST by installing the optional [pykdtree](https://github.com/storpipfugl/pykdtree) library:
#     ```shell
#     pip3 install pykdtree
#     ```
#
# Now we can sort those scores to find the top matches:

# %%
scores.loc["LH1112"].sort_values(ascending=False)

# %%
# So we did get a couple of hits here. Let's visualize the top 3:

# %%
fig = navis.plot3d([query_xf, targets.idx[[2030342003, 2214504597, 1069223047]]])

# %%
# A final note: the scores for these matches are rather low (1 = a perfect match).
#
# The main reason is that our image stack contains two neurons (left and right), so half of the `query` has no match in any
# single hemibrain neuron. Subsetting the query to the approximate hemibrain bounding box fixes this — also a good idea for
# bilateral neurons whose arbour extends outside the hemibrain volume:

# %%
# Remove the left-hand-side neuron based on the position
# along the x-axis (this is just one of the possible approaches)
# This is the approximate LHS boundary of the volume
flybrains.JRCFIB2018Fraw.mesh.vertices[:, 0].max()

# %%
query_ss = navis.subset_neuron(query_xf, query_xf.points[:, 0] <= 35_000)
query_ss

# %%
# !!! tip "Improving your scores"
#     Two levers for better matches:
#
#     - **Subset the query.** Re-running with `query_ss` drops the unmatched half of the image and should yield much
#       higher scores.
#     - **Pre-process the image dotprops.** Image-derived dotprops are denser than the skeleton-derived ones because the image
#       contains several individuals of the same cell type. Downsampling or thinning the image beforehand helps.
