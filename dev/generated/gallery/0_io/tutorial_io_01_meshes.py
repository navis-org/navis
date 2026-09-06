"""
Meshes
======
<!-- difficulty: beginner -->

Load and save mesh neurons in OBJ, PLY, STL and other formats.

{{ navis }} knows two types of meshes, both subclasses of `trimesh.Trimesh` (and usable as such):

| Class | Use for |
|-------|---------|
| [`navis.Mesh`][] | Neurons stored as meshes, e.g. from EM segmentation. |
| [`navis.Volume`][]     | Meshes that are *not* neurons, e.g. neuropils or brain outlines. |

!!! note
    {{ navis }} has dedicated interfaces for loading meshes from remote data sources
    (e.g. the MICrONS, neuromorpho, Virtual Fly Brain or Janelia hemibrain datasets).
    These are covered in separate [tutorials](../index.md).

## From files

For reading run-of-the-mill files containing meshes, {{ navis }} provides a single function: [`navis.read_mesh`][].
Under the hood, that function uses `trimesh.load_mesh` which supports most of the common formats (`.obj`, `.ply`, `.stl`, etc.).

"""
# %%
import navis

# %%
# Like [`navis.read_swc`][], you can point [`navis.read_mesh`][] at a single file or a folder:
#
# ```python
# mesh = navis.read_mesh('test_neuron.stl')        # (1)!
#
# meshes = navis.read_mesh('neurons/*.stl')        # (2)!
# ```
#
# 1.  A single mesh file - returns a [`navis.Mesh`][].
# 2.  All matching files in a folder. You must give the extension (e.g. `*.stl`) so {{ navis }} knows what to read.

# %%
# By default [`navis.read_mesh`][] returns neurons. Use `output` to get a [`navis.Volume`][] or a
# raw `trimesh.Trimesh` instead:
#
# ```python
# vol = navis.read_mesh('test_mesh.stl', output='volume')   # (1)!
# ```
#
# 1.  `output` also accepts `'neuron'` (the default) and `'trimesh'`.

# %%
# ## Manual construction
#
# It's super easy to construct [`navis.Mesh`][] or [`navis.Volume`][] from scratch -
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
# Turn into Mesh
m = navis.Mesh((vertices, faces), name='my_mesh', id=1, units='microns')
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
# Save [`navis.Meshes`][navis.Mesh] or [`navis.Volumes`][navis.Volume] with [`navis.write_mesh`][]:
#
# ```python
# m = navis.example_neurons(1, kind='mesh')
# navis.write_mesh(m, '~/Downloads/neuron.obj')          # (1)!
#
# nl = navis.example_neurons(3, kind='mesh')
# navis.write_mesh(nl, '~/Downloads/', filetype='obj')   # (2)!
# ```
#
# 1.  A single neuron to a named file. The extension (`.obj`, `.ply`, `.stl`, ...) sets the format.
# 2.  A whole `NeuronList` to a folder - `filetype` is required here because the path has no extension.

# %%
# Just like [`write_swc`][navis.write_swc], the filepath controls how batches of neurons are named:
#
# | Filepath pattern | Result |
# |------------------|--------|
# | `~/Downloads/` (+ `filetype`) | One file per neuron, named `{neuron.id}.obj` (the default). |
# | `~/Downloads/{neuron.name}.obj` | One file per neuron, named by each neuron's `.name`. |
# | `~/Downloads/{neuron.id}.obj` | One file per neuron, named by each neuron's `.id`. |

# %%
# !!! warning "Triangular faces only"
#     {{ navis }} works exclusively with triangular faces - no quads or polygons. See the
#     [`navis.Mesh`][] and [`navis.Volume`][] docs for details.
#
# This tutorial has hopefully given you some entry points on how to load your data. See also the [I/O API reference](../../../api.md#importexport).
# Also note that all {{ navis }} neurons can be stored to disk using ``pickle`` - see the [pickling tutorial](../tutorial_io_04_pickle).
#
# Please also keep in mind that you can also convert one neuron type into another - for example by skeletonizing [`Meshes`][navis.Mesh]
# (see also the API reference on [neuron conversion](../../../api.md#converting-between-types)).

# mkdocs_gallery_thumbnail_path = '_static/mesh_thumbnail.png'