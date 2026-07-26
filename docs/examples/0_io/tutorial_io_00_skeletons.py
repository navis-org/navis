"""
Skeletons
=========

Load and save skeletons from SWC and other formats, or build them from scratch.

Skeletons are probably the most common representation of neurons and are stored as a series
of connected nodes (the "skeleton"). In {{ navis }}, skeletons are represented by the
[`navis.TreeNeuron`][] class.

Build them manually (see the [bottom of this page](#manual-construction)) or use one of the built-in
readers for the various skeleton file formats:

<div class="grid cards" markdown>

-   :material-file-document-outline:{ .lg .middle } __SWC__

    ---

    The most common skeleton format - read & write, including from zip archives, URLs and FTP.

    [:octicons-arrow-right-24: From SWC files](#from-swc-files)

-   :material-xml:{ .lg .middle } __NMX__

    ---

    XML-based format used by pyKNOSSOS (read-only).

    [:octicons-arrow-right-24: From NMX files](#from-nmx-files)

-   :material-cube-outline:{ .lg .middle } __Precomputed__

    ---

    Neuroglancer's compact binary format - read & write.

    [:octicons-arrow-right-24: From Neuroglancer Precomputed](#from-neuroglancer-precomputed)

-   :material-hammer-wrench:{ .lg .middle } __Manual__

    ---

    Roll your own [`TreeNeuron`][navis.TreeNeuron] from an SWC-like table.

    [:octicons-arrow-right-24: Manual construction](#manual-construction)

</div>

!!! note
    {{ navis }} has dedicated interfaces for loading skeletons from remote data sources
    (e.g. the MICrONS, neuromorpho, Virtual Fly Brain or Janelia hemibrain datasets).
    These are covered in separate [tutorials](../index.md).

    If you have light-level microscopy data, see the tutorial on
    [skeletons from light-level data](../zzz_tutorial_io_05_skeletonize).

## From SWC files

SWC is the most common format for storing neuron skeletons, and {{ navis }} reads and writes it.
The examples below use Supplemental Data S1 from
[Bates, Schlegel et al. (2020)](https://doi.org/10.1016/j.cub.2020.06.042) - download it to follow
along and adjust the filepaths to match where you saved it.

"""

# %%
# The archive holds a bunch of metadata CSVs, but the file we want is `skeletons_swc.zip`.
# No need to unzip it - {{ navis }} reads directly from (and writes to) zip archives.

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
# For this I extracted the skeletons_swc.zip archive
s = navis.read_swc('./mmc2/swc/CENT/11519759.swc')
s

# %%
# You can even use URLs or FTP servers directly:

# %%

# From URL:
s = navis.read_swc('https://v2.virtualflybrain.org/data/VFB/i/jrch/jup2/VFB_00101567/volume.swc')

# %%
# You can even read straight from an FTP folder (this particular server is currently offline, shown for reference):
#
# ```python
# nl = navis.read_swc('ftp://download.brainimagelibrary.org/biccn/zeng/pseq/morph/200526/', limit=3)
# ```
#
# !!! tip "`read_swc` is flexible"
#     [`read_swc`][navis.read_swc] handles a whole range of inputs - file names, folders, archives, URLs and more -
#     and lets you customise *which* and *how* neurons are loaded:
#
#     - `limit` can also take a pattern to load only matching files
#     - `fmt` controls how filenames are parsed into neuron names and IDs
#
#     Many of the other `navis.read_*` functions share these features!

# %%
# ## To SWC files
#
# Saving skeletons back to disk works the same way. Write a single neuron:

# %%

navis.write_swc(s, './mmc2/my_neuron.swc')

# %%
# The magic is all in the filepath: use a `{neuron.name}` placeholder to name files by a neuron
# property, and append `@archive.zip` to write straight into a zip. These compose freely:
#
# | Filepath pattern | Result |
# |------------------|--------|
# | `my_neuron.swc` | A single neuron to one file. |
# | `{neuron.name}.swc` | One file per neuron in a `NeuronList`, named by each neuron's `.name`. |
# | `skeletons.zip` | The whole `NeuronList` bundled into a single zip archive. |
# | `{neuron.name}.swc@skeletons.zip` | One file per neuron, named by `.name`, *inside* a zip archive. |
#
# The placeholder can reference any neuron property, e.g. `{neuron.id}.swc`. See
# [`navis.write_swc`][] for further details.
#
# ## From NMX files
#
# NMX is an xml-based format used e.g. by [pyKNOSSOS](https://github.com/adwanner/PyKNOSSOS) to store skeletons plus meta data.
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
# ## From Neuroglancer Precomputed
#
# Among other formats, neuroglancer supports a "precomputed" format for skeletons
# (see specs [here](https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/skeletons.md).
# This binary format is more compact than uncompressed SWC files but is not used outside of neuroglancer as far as I know.
# That said: {{ navis }} lets you read and write skeletons from/to precomputed format using [`navis.read_precomputed`][] and
# [`navis.write_precomputed`][]. Note that these functions work on both precomputed skeletons and meshes.
#
# Also check out the [tutorial](../4_remote/tutorial_remote_01_cloudvolume.md) on reading skeletons straight from
# a neuroglancer source using `cloud-volume`.
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
#
# *[SWC]: A plain-text format storing a neuron skeleton as a table of connected nodes.
# *[FTP]: File Transfer Protocol - a standard way of serving files over a network.

# mkdocs_gallery_thumbnail_path = '_static/skeleton_thumbnail.png'