## Storing neuron data in HDF5 files

### Preamble
There are a few file formats that can store neuron morphology. To name but a few:
- [SWC](http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html)
  for simple skeletons
- [NeuroMl](https://en.wikipedia.org/wiki/NeuroML) is an XML-based format
  primarily used for modelling but can store compartment models (i.e. skeletons)
  of neurons and meta data
- [NWB](https://www.nwb.org) (neurodata without borders) is an HDF5-based format
  focused on physiology
- [NRRD](http://teem.sourceforge.net/nrrd/format.html) files can be used to
  store dotprops

_Why then start a new format?_

Because none of the existing formats tick all the boxes:

I need a file format that can hold:

1. thousands of neurons,
2. multiple representations (mesh, skeleton, dotprops) of a given neuron,
3. annotations (e.g. synapses) associated with each neuron and
4.meta data such as names, soma positions, etc.

Enter HDF5: basically a filesystem-in-a-file. The important thing for me
is that I don't have to worry about how data is en-/decoded because
other libraries (like `h5py` for Python or `hdf5r` for R) take care of
that. All I have to do is come up with a schema.

### Schema
HDF5 knows groups (`grp`), datasets (`ds`) and attributes (`attr`). The
basic idea for our schema is this:

- the `root` contains info about the format as attributes
- each group in `root` represents a neuron and the group's name is the neuron's ID
- a neuron group holds the neuron's representations (mesh, skeleton
  and/oo dotprops), annotations and meta data in separate sub-groups

To illustrate the basic principle:

```
.
├── attrs: format-related meta data
├── group: neuron1
│   ├── attrs: neuron-related meta data
│   ├── group: skeleton
│   |    ├── attrs: skeleton-related meta data
|   |    └── datasets: node table, etc
│   ├── group: dotprops
│   |    ├── attrs: dotprops-related meta data
|   |    └── datasets: points, tangents, alpha, etc
│   ├── group: mesh
│   |    ├── attrs: mesh-related meta data
|   |    └── datasets: vertices, faces, etc
|   └── group: annotations
|       └── group: e.g. connectors
|           ├── attrs: connector-related meta data
|           └── datasets: connector data
├── group: neuron2
|   ├── ...  
...
```

#### Root attributes

The root meta data must contain two attributes:
- `format_spec` specifies format and version
- `format_url` points to a library or format specifications

```
.
├── attr['format_spec']: str = 'navis_hdf5_v1'   
├── attr['format_url']: str = 'https://github.com/schlegelp/navis'
...
```

#### Neuron base groups

Each neuron group contains properties that apply to all
the neuron's potential representations - for example a `neuron_name`.
If an attribute is defined in the neuron group and at a deeper level
(i.e. the skeleton, mesh or dotprops), the deeper attribute takes
precedence.

```
.
└── group['123456']  # note that numeric IDs will be "stringified"
    ├── attr["neuron_name"]: str = "some name"
...
```

#### Skeletons  

Attributes:
- `units_nm` (float | int | tuple, optional): specifies the units in
  nanometer space - can be a tuple of `(x, y, z)` if units are
  non-isotropic  
- `soma` (int, optional): the node ID of the soma  

Datasets:
- `node_id` (int): IDs for the nodes
- `parent_id` (int): for each node, the ID of it's parent, nodes with
  out parents (i.e. roots) have `parent_id` of `-1`
- `x`, `y`, `z` (float | int): node coordinates
- `radius` (float | int, optional): radius for each node

```
└── group['123456']
    ├── attr['neuron_name'] = "example neuron with a skeleton"
    ├── attr['units_nm'] = (4, 4, 40)
    └── grp['skeleton']
         ├── attr['soma']: 1
         ├── ds['node_id']: (N, ) array
         ├── ds['parent_id']: (N, ) array
         ├── ds['x']: (N, ) array
         ├── ds['y']: (N, ) array
         ├── ds['z']: (N, ) array
         └── ds['radius']: (N, ) array, optional
```

#### Meshes  

Attributes:
- `units_nm` (float | int | tuple, optional): specifies the units in
  nanometer space - can be a tuple of `(x, y, z)` if units are
  non-isotropic  
- `soma` (tuple, optional): tuple of `(x, y, z)` coordinates of the soma

Datasets:
- `vertices` (int | float): (N, 3) array of vertex positions
- `faces` (int): (M, 3) array of vertex IDs forming the faces
- `skeleton_map` (int, optional): (N, ) array mapping each vertex to a
  node ID in the skeleton

```
└── group['4353421']
    ├── attr['neuron_name'] = "example neuron with a mesh"
    ├── attr['units_nm'] = (4, 4, 40)
    └── grp['mesh']
         ├── attr['soma']: (1242, 6533, 400)
         ├── ds['vertices']: (N, 3) array
         ├── ds['faces']: (M, 3) array
         └── ds['skeleton_map']: (N, ) array, optional

```

#### Dotprops  

Attributes:
- `k` (int): number of k-nearest neighbours used to calculate the tangent
  vectors from the point cloud
- `units_nm` (float | int | tuple, optional): specifies the units in
  nanometer space - can be a tuple of `(x, y, z)` if units are
  non-isotropic  
- `soma` (tuple, optional): tuple of `(x, y, z)` coordinates of the soma

Datasets:
- `points` (int | float): (N, 3) array of x/y/z positions
- `vect` (int | float, optional): (N, 3) array of tangent vectors -    
  generated if not provided
- `alpha` (int | float, optional): (N, ) array of alpha values for each
  point in ``points`` generated if not provided

```
└── group['65432']    
    ├── attr['neuron_name'] = "example neuron with dotprops"    
    └── grp['mesh']
        ├── attr['k'] = 5
        ├── attr['units_nm'] = (4, 4, 40)
        ├── attr['soma']: (1242, 6533, 400)
        ├── ds['points']: (N, 3) array
        ├── ds['vect']: (N, 3) array
        └── ds['alpha']: (N, ) array
```

#### Annotations
Annotations are meant to be flexible and are principally parsed into
pandas DataFrames. Because they won't follow a common format, it is
good practice to leave some (optional) meta data pointing to columns
containing data relevant for e.g. plotting:

Attributes:
- `point_col` (str | list thereof): pointer to the column(s) containing
   x/y/z positions
- `type_col` (str): pointer to a column specifying types
- `skeleton_map` (str): pointer to a column associating the row with
   a node ID in the skeleton

Let's illustrate this with a mock synapse table:

```
└── group['32434566']
    ├── attr['neuron_name'] = "example neuron with synapse annotations"
    ├── attr['units_nm'] = (4, 4, 40)
    └── grp['annotations']
         └── grp['synapses']
             ├── attr['points']: ['x', 'y', 'z']
             ├── attr['types']: 'prepost'
             ├── attr['skeleton_map']: 'node_id'
             ├── ds['x']: (N, ) array
             ├── ds['x']: (N, ) array
             ├── ds['z']: (N, ) array
             ├── ds['prepost']: (N, ) array of [0, 1, 2, 3, 4]
             └── ds['node_id']: (N, )
```

The current version of the format is 1.0.

Changes:
- 2021/02/01: Version 1.0

### A final remark
The above schema describes a "minimal" layout - i.e. we expect no less
data than that. However, the `navis` implementations for reading/writing
the schema are flexible: you can add more attributes or datasets
and `navis` will by default try to read and attach them to the neuron.

### Is this stable?
Ish? The format is versioned and I will maintain readers/writers for
past versions in ``navis``. In other good news: the HDF5 backend is
stable - so even if `navis` acts up when parsing your file, you can
always read it manually using `h5py`.
