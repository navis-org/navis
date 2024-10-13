"""
Skeletons
=========

This tutorial will show you how to load and save skeletons.

Skeletons are probably the most common representation of neurons and are stored as a series
of connected nodes (the "skeleton"). In {{ navis }}, skeletons are represented by the
[`navis.TreeNeuron`][] class.

You can either construct these manually (see bottom of this page) or use one of the built-in
functions to them from one of the various file formats:

!!! note
    {{ navis }} has dedicated interfaces for loading skeletons from remote data sources
    (e.g. the MICrONS, neuromorpho, Virtual Fly Brain or Janelia hemibrain datasets).
    These are covered in separate [tutorials](../../gallery).

    If you have light-level microscopy data, you might also be interested in the
    tutorial on [skeletons from light-level data](../zzz_tutorial_io_05_skeletonize).

## From SWC files

SWC is a common format for storing neuron skeletons. Thus {{ navis }} provides functions to both
read and write SWC files. To demo these, we will be using supplemental data from
Bates, Schlegel et al. (Current Biology, 2020). If you want to follow along, please download
Supplemental Data S1 ([link](https://doi.org/10.1016/j.cub.2020.06.042)). If you do, make sure
to adjust the filepaths in the examples according to where you saved it to.

"""

# %%
# I extracted the archive with the supplemental data inside my downloads folder.
#
# It contains a bunch of CSV files with meta data but the important file for us is the
# `"skeletons_swc.zip"`. Now you could extract that zip archive too but {{ navis }} can
# actually read directly from (and write to) zip files!

# %%
import navis
skeletons = navis.read_swc(
    'mmc2/skeletons_swc.zip',
    include_subdirs=True
)
skeletons

# %%
# Let's say you are looking at a huge collection of SWC files and you only want to sample a few of them:

# %%
# Load only the first 10 skeletons
sample = navis.read_swc(
    './mmc2/skeletons_swc.zip',
    include_subdirs=True,
    limit=10
)
sample

# %%
# We can also point [`navis.read_swc()`][navis.read_swc] at single files instead of folders or zip archives:

# %%
# For this I extraced the skeletons_swc.zip archive
s = navis.read_swc('./mmc2/swc/CENT/11519759.swc')
s

# %%
# You can even use URLs or FTP servers directly:

# %%

# From URL:
s = navis.read_swc('https://v2.virtualflybrain.org/data/VFB/i/jrch/jup2/VFB_00101567/volume.swc')

# %%

# From an FTP folder:
nl = navis.read_swc('ftp://download.brainlib.org:8811/biccn/zeng/pseq/morph/200526/', limit=3)


# !!! tip
#     [`read_swc`][navis.read_swc] is super flexible and can handle a variety of inputs (file names, folders, archives, URLs, etc.).
#     Importantly, it also let you customize which/how neurons are loaded. For example:
#      - the `limit` parameter can also be used to load only files matching a given pattern
#      - the `fmt` parameter lets you specify how to parse filenames into neuron names and ids
#     Many of the other `navis.read_*` functions share these features!

# %%
# ## To SWC files
#
# Now let's say you have skeletons and you want to save them to disk. Easy!

# %%

# Write a single neuron:
navis.write_swc(s, './mmc2/my_neuron.swc')

# %%

# Write a whole list of skeletons to a folder and use the neurons' `name` property as filename:
navis.write_swc(sample, './mmc2/{neuron.name}.swc')

# %%

# Write directly to a zip file:
navis.write_swc(sample, './mmc2/skeletons.zip')

# %%

# Write directly to a zip file and use the neuron name as filename:
navis.write_swc(sample, './mmc2/{neuron.name}.swc@skeletons.zip')

# %%
# See [`navis.write_swc`][] for further details!
#
# ## From NMX files
#
# NMX is a xml-based format used e.g. by [pyKNOSSOS](https://github.com/adwanner/PyKNOSSOS) to store skeletons plus meta data.
# {{ navis }} supports reading (but not writing) this format. If you want to follow
# along download [this dataset](https://doi.org/10.5281/zenodo.58985) by Wanner et al. (2016).
# Just like the SWCs, I extracted the archive to my downloads folder:

# %%
# Read a single file
s = navis.read_nmx('./WannerAA201605_SkeletonsGlomeruli/Neuron_id0001.nmx')
s

# %%
# Read all files in folder
nl = navis.read_nmx('./WannerAA201605_SkeletonsGlomeruli/')
nl

# %%
navis.plot2d(nl[:10], method='2d', radius=False)

# %%
# !!! note
#     If you encounter an error message while reading: NMX files don't always contain skeletons.
#     If {{ navis }} comes across one that can't be turned into a [`navis.TreeNeuron`][],
#     it will skip the file and produce a warning.
#
# ## From Neuroglancer
#
# Among other formats, neuroglancer supports a "precomputed" format for skeletons
# (see specs [here](https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/skeletons.md).
# This binary format is more compact than uncompressed SWC files but is not used outside of neuroglancer as far as I know.
# That said: {{ navis }} lets you read and write skeletons from/to precomputed format using [`navis.read_precomputed`][] and
# [`navis.write_precomputed`][]. Note that these functions work on both precomputed skeletons and meshes.
#
# ## Manual construction
#
# What if you have some obscure data format for which {{ navis }} does not have a read function? The data underlying
# a [`navis.TreeNeuron`][] is a simple SWC table - so as long as you can produce that from your data, you can create
# your own skeletons.
#
# Here's a quick & dirty example:

# %%
import pandas as pd

# Create a mock SWC table for a 2-node skeleton
swc = pd.DataFrame()
swc['node_id'] = [0, 1]
swc['parent_id'] = [-1, 0]   # negative indices indicate roots
swc['x'] = [0, 1]
swc['y'] = [0, 1]
swc['z'] = [0, 1]
swc['radius'] = 0

swc

# %%
# This SWC can now be used to construct a [`TreeNeuron`][navis.TreeNeuron]:

# %%
s = navis.TreeNeuron(swc, name='my_neuron', units='microns')
s

# %%
# There are a few other ways to construct a [`navis.TreeNeuron`][] (e.g. using a graph) - see the docstring for details.
#
# Also note that all {{ navis }} neurons can be stored to disk using `pickle` - see the [pickling tutorial](../tutorial_io_04_pickle).
#
# Hopefully the above has given you some entry points on how to load your data. See also the [I/O API reference](../../../api.md#importexport).
#
# Please also keep in mind that you can also convert one neuron type into another - for example by skeletonizing [`MeshNeurons`][navis.MeshNeuron]
# (see also the API reference on [neuron conversion](../../../api.md#converting-between-types)).

