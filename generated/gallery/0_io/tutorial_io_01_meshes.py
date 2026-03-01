"""
Meshes
======

This tutorial will teach you how to load and save meshes.

{{ navis }} knows two types of meshes:

1. [`navis.MeshNeuron`][] for neurons
2. [`navis.Volume`][] for meshes that are not neurons, e.g. neuropil or brain meshes

Both of these are subclasses of `trimesh.Trimesh` and can be used as such.

!!! note
    {{ navis }} has dedicated interfaces for loading meshes from remote data sources
    (e.g. the MICrONS, neuromorpho, Virtual Fly Brain or Janelia hemibrain datasets).
    These are covered in separate [tutorials](../../gallery).

## From files

For reading run-of-the-mill files containing meshes, {{ navis }} provides a single function: [`navis.read_mesh`][].
Under the hood, that function uses `trimesh.load_mesh` which supports most of the common formats (`.obj`, `.ply`, `.stl`, etc.).

"""
# %%
import navis

# %%
# ```python
# # Load an example file (here a FlyWire neuron I downloaded and saved locally)
# mesh = navis.read_mesh('test_neuron.stl')
# ```

# %%
# The interface is similar to [`navis.read_swc`][] in that you can point
# [`navis.read_mesh`][] at single file or at folders with multiple files:

# %%
# ```python
#  # When reading all files in folder you have to specificy the file extension (e.g. *.stl)
#  meshes = navis.read_mesh('neurons/*.stl')
# ```

# %%
# By default, [`navis.read_mesh`][] will return neurons. Use the `output` parameter to get
# a [`navis.Volume`][] or a `trimesh.Trimesh` instead:

# %%
# ```python
# # Load a mesh file into a Volume
# vol = navis.read_mesh('test_mesh.stl', output='volume')
# ```

# %%
# ## Manual construction
#
# It's super easy to construct [`navis.MeshNeuron`][] or [`navis.Volume`][] from scratch -
# they are just vertices and faces after all.
#
# So if e.g. your mesh file format is not covered by [`navis.read_mesh`][] or you created
# the mesh yourself (e.g. using a marching cube algorithm), just create the objects yourself:

# %%
import numpy as np

# Create some mock vertices
vertices = np.array([[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]])
# Make a single triangular face using the vertex indices
faces = np.array([[0, 1, 2]])

# %%
# Turn into MeshNeuron
m = navis.MeshNeuron((vertices, faces), name='my_mesh', id=1, units='microns')
m

# %%

navis.plot3d(m)

# %%
# Turn into Volume
vol = navis.Volume(vertices, faces, name='my_volume')
vol

# %%
# ## To files
#
# For saving [`navis.MeshNeurons`][navis.MeshNeuron] or [`navis.Volumes`][navis.Volume] to disk, use [`navis.write_mesh`][].

# %%
# Save single neuron to file:
# ```python
# m = navis.example_neurons(1, kind='mesh')
# navis.write_mesh(m, '~/Downloads/neuron.obj')
# ```


# %%
# Save a bunch of neurons to mesh:
# ```python
# nl = navis.example_neurons(3, kind='mesh')
# navis.write_mesh(nl, '~/Downloads/', filetype='obj')
# ```

# %%
# By default, [`navis.write_mesh`][] will write multiple neurons to files named `{neuron.id}.obj`.
# You can change this behavior by specifying the format in the filename:
# ```python
# # Use the neuron name instead of the id
# navis.write_mesh(nl, '~/Downloads/{neuron.name}.obj')
# ```

# %%
# !!! important
#     One thing to keep in mind here is that {{ navis }} only works with triangular faces,
#     i.e. no quads or polygons! Please see the documentation of [`navis.MeshNeuron`][] and
#     [`navis.Volume`][] for details.
#
# This tutorial has hopefully given you some entry points on how to load your data. See also the [I/O API reference](../../../api.md#importexport).
# Also note that all {{ navis }} neurons can be stored to disk using ``pickle`` - see the [pickling tutorial](../tutorial_io_04_pickle).
#
# Please also keep in mind that you can also convert one neuron type into another - for example by skeletonizing [`MeshNeurons`][navis.MeshNeuron]
# (see also the API reference on [neuron conversion](../../../api.md#converting-between-types)).