---
icon: simple/keepachangelog
hide:
  - navigation
---

# :simple-keepachangelog: Changelog

This is a selection of features added, changes made and bugs fixed with each version.
For a full list of changes please see the [commits history](https://github.com/navis-org/navis/commits/master) on
{{ navis }}' Github repository.

## dev
_Date: ongoing_

To install the current `dev` version of {{ navis }}:

```shell
pip uninstall navis -y
pip install git+https://github.com/navis-org/navis@master
```

## Version `1.9.1` { data-toc-label="1.9.1" }
_Date: 24/10/24_

##### Improvements
- MICrONS & H01 interfaces:
    - `fetch_neurons` now accepts a `materialization` parameter that determines which materialization version is used for synapse and nucleus queries; defaults to "auto" which means `navis` will try to find a materialization version matching the queried root IDs
    - `fetch_neurons` will now also assign soma positions for H01 neurons (see the `.soma_pos` neuron property)
- `CloudVolume.mesh.get_navis` (see [`navis.patch_cloudvolume`][]) now accepts a `process` (default is `False`) parameter that determines whether the NeuroGlancer mesh is processed (deduplication of vertices, etc.); contribute by Forrest Collman

##### Fixes
- fixed a bug in [`navis.subset_neuron`][] that caused branch points to not be re-calculated

**Full Changelog**: [v1.9.0...v1.9.1](https://github.com/navis-org/navis/compare/v1.9.0...v1.9.1)

## Version `1.9.0` { data-toc-label="1.9.0" }
_Date: 17/10/24_

This version brings a shiny new interface to the [H01 human cortex](https://h01-release.storage.googleapis.com/landing.html) dataset
and various other quality of life improvements.

##### Breaking
- The default for `radius` ([`navis.plot2d`][] and [`navis.plot3d`][]) was changed to `False` (from `"auto"`); this is to make sure that the defaults allow visualizing large numbers of skeletons, i.e. prioritizing performance over beauty

##### Additions
- New interface to the [H01](https://h01-release.storage.googleapis.com/landing.html) dataset (by Jinhan Choi and Jakob Troidl; see the new tutorial for details)

##### Improvements
- I/O:
    - [`read_nrrd`][navis.read_nrrd], [`read_tiff`][navis.read_tiff] and [`read_mesh`][navis.read_mesh] now use the same backend as e.g. [`read_swc`][navis.read_swc] which enables some niceties such as reading directly from URLs and archives, parallel processing, etc
    - all `read_*` functions now have an `error` parameter that can be used to skip errors
- Image data:
    - new function: [`navis.thin_voxels`][] can be used to thin images and `VoxelNeurons` to single-pixel width (see also below)
    - new `thin` parameter for [`read_nrrd`][navis.read_nrrd], [`read_tiff`][navis.read_tiff]
- [`TreeNeurons`][navis.TreeNeuron]:
    - skeletons can now be initialized from a `(vertices, edges)` tuple - see also [`navis.edges2neuron`][]
    - new property: `TreeNeuron.vertices` gives read-only to node (vertex) coordinates
- [`VoxelNeurons`][navis.VoxelNeuron]:
    - new properties: `VoxelNeuron.nnz` and `VoxelNeuron.density`
- [`navis.drop_fluff`][] and [`navis.neuron2nx`][] now also works with [`Dotprops`][navis.Dotprops]

##### Experimental
- setting `navis.config.add_units=True` (default is `False` for the time being) will add units to certain neuron properties such as `.cable_length` to make them human-readable

**Full Changelog**: [v1.8.0...v1.9.0](https://github.com/navis-org/navis/compare/v1.8.0...v1.9.0)

## Version `1.8.0` { data-toc-label="1.8.0" }
_Date: 22/09/24_

This version contains a major internal rework of both [`navis.plot2d`][] and [`navis.plot3d`][] to make them
more consistent and easier to use.

##### Breaking
- Plotting: the `synapse_layout` parameter was renamed to `cn_layout` (matching e.g. other parameters such as `cn_colors`)
- Negative views in [`navis.plot2d`][] (e.g. `view=("x", "-z")`) will now invert axes rather than changing the underlying data
- Minimum version of `matplotlib` is now `3.9` (was `3.6`)
- The `plotly` backend is not part of a minimal install anymore (still installed using `navis[all]`)
- The Vispy backend is now deprecated and will be removed in a future release
- Removed `navis.screenshot` - please use the Octarine/Vispy viewer's `.screenshot()` method instead
- [`navis.tortuosity`][] now calculates tortuosity as-is (i.e. without resampling) by default

##### Additions
- Added [Octarine](https://github.com/schlegelp/octarine) as the default backend for plotting from terminal
- New Function: [`navis.ivscc_features`][] computes some basic IVSCC features
- New function: [`navis.graph.skeleton_adjacency_matrix`][] computes the node adjacency for skeletons
- New function: [`navis.graph.simplify_graph`][] simplifies skeleton graphs to only root, branch and leaf nodes while preserving branch length (i.e. weights)
- New [`NeuronList`][navis.NeuronList] method: [`get_neuron_attributes`][navis.NeuronList.get_neuron_attributes] is analagous to `dict.get`
- [`NeuronLists`][navis.NeuronList] now implement the `|` (`__or__`) operator which can be used to get the union of two [`NeuronLists`][navis.NeuronList]
- [`navis.Volume`][] now have an (optional) `.units` property similar to neurons
- `Tree/MeshNeurons` and `Dotprops` now support addition/subtraction (similar to the already existing multiplication and division) to allow offsetting neurons

##### Improvements
- Plotting:
    - [`navis.plot3d`][]:
      - `legendgroup` parameter (plotly backend) now also sets the legend group's title
      - new parameters for the plotly backend:
          - `legend` (default `True`): determines whether legends is shown
          - `legend_orientation` (default `v`): determines whether legend is aranged vertically (`v`) or horizontally (`h`)
          - `linestyle` (default `-`): determines line style for skeletons
      - default for `radius` is now `"auto"`
    - [`navis.plot2d`][]:
      - the `view` parameter now also works with `methods` `3d` and `3d_complex`
      - the `color_by` and `shade_by` parameters now also work when plotting skeletons with `radius=True`
      - new defaults: `radius="auto"`, `alpha=1`, `figsize=None` (use matplotlib defaults)
      - new parameters for methods `3d` and `3d_complex`: `mesh_shade=False` and `non_view_axes3d`
      - the `scalebar` and `soma` parameters can now also be dictionaries to style (color, width, etc) the scalebar/soma
    - the `connectors` parameter can now be used to show specific connector types (e.g. `connectors="pre"`)
- I/O:
    - `read_*` functions are now able to read from FTP servers (`ftp://...`)
    - the `limit` parameter used in many `read_*` functions can now also be a regex pattern or a `slice`
- New parameter in [`navis.resample_skeleton`][]: use `map_column` to include arbitrary columns in the resampling
- [`navis.prune_twigs`][] and [`navis.morpho.cable_length`][] now accept a `mask` parameter
- General improvements to docs and tutorials

##### Fixes
- Memory usage of `Neuron/Lists` is now correctly re-calculated when the neuron is modified
- Various fixes and improvements for the MICrONS interface (`navis.interfaces.microns`)
- [`navis.graph.node_label_sorting`][] now correctly prioritizes total branch length
- [`navis.TreeNeuron.simple`][] now correctly drops soma nodes if they aren't root, branch or leaf points themselves

**Full Changelog**: [v1.7.0...v1.8.0](https://github.com/navis-org/navis/compare/v1.7.0...v1.8.0)

## Version `1.7.0` { data-toc-label="1.7.0" }
_Date: 25/07/24_

##### Breaking
- Plotting: dropped the `cluster` parameter in favor of an improved `color_by` logic (see below)

##### Additions
- {{ navis }} now uses `navis-fastcore` if present to dramatically speed up core functions (see updated install instructions)
- New method `navis.NeuronList.add_metadata` to quickly add metadata to neurons

##### Improvements
- `navis.find_soma` and `navis.graph.neuron2nx` (used under the hood) are now much faster
- All I/O functions such as `navis.read_swc` now show which file caused an error (if any); original filenames are tracked as `file` property
- `navis.NeuronList` will only search the first 100 neurons for autocompletion to avoid freezing with large lists
- Plotting functions: `color_by` now accepts either a list of labels (one per neuron) or the name of a neuron property
- `navis.subset_neuron` is now faster and more memory efficient when subsetting meshes
- `navis.TreeNeuron.cable_length` is now faster
- Fixed a bug in plotting when using vertex colors
- Fixed the progress bar in `navis.interfaces.neuprint.fetch_mesh_neuron`
- Fixed a bug in `navis.synblast` that caused multiprocessing to fail (pickling issue with `pykdtree`)
- `navis.interfaces.neuprint.fetch_mesh_neuron` will now ignore the `lod` parameter if the data source does not support it instead of breaking
- Fixed a number of deprecation warnings in the codebase

**Full Changelog**: [v1.6.0...v1.7.0](https://github.com/navis-org/navis/compare/v1.6.0...v1.7.0)

## Version `1.6.0` { data-toc-label="1.6.0" }
_Date: 07/04/24_

##### Breaking
- Dropped support for Python 3.8, per NEP 29
- `navis.write_swc` no longer supports writing Dotprops to SWC files

##### Additions
- New property `TreeNeuron.surface_area`
- New (experimental) functions `navis.read_parquet` and `navis.write_parquet` store skeletons and dotprops in parquet files (see [here](https://github.com/clbarnes/neurarrow) for format specs)
- New `navis.read_nml` function to read single NML file
- New `navis.NeuronConnector` class for creating connectivity graphs from groups neurons with consistent connector IDs
- New method for CMTKtransforms: `navis.transforms.CMTKTransform.xform_image`

##### Improvements
- Improved performance for adding recordings to `CompartmentModel`
- `navis.heal_skeleton` and `navis.resample_skeleton` are now faster
- Improved logic for splitting NBLASTs across multiple cores
- `navis.xform_brain` now allows specifying multiple intermediate template spaces through the `via` parameter and to ignore spaces through the `avoid` parameter
- I/O functions can now read directly from `.tar` or `.tar.gz` files
- `navis.read_precomputed` now accepts a `limit` parameter similar to `navis.read_swc`

##### Fixes
- Fixed interface to InsectBrainDB
- `navis.read_precomputed`:
    - now correctly parses the `info` file depending on the source
    - reading large files (i.e. meshes) directly from a URL should not break anymore
- Fixed writing vertex properties in `navis.write_precomputed`
- Fixed a bug in `navis.resample_skeleton`
- Fixed an occasional issue when plotting skeletons with radii
- Fixed bug in `navis.subset_neuron` that caused connectors to be dropped when using mask
- Fixed a bug in `navis.despike_skeleton` that caused the `reverse` argument to be ignored
- Fixed two small bugs in `navis.interfaces.neuprint.fetch_mesh_neuron`

**Full Changelog**: [v1.5.0...v1.6.0](https://github.com/navis-org/navis/compare/v1.5.0...v1.6.0)

## Version `1.5.0` { data-toc-label="1.5.0" }
_Date: 27/07/23_

##### Breaking
- Dropped support for Python 3.7

##### Additions
- New function: `navis.pop3d` removes the most recently added object from the vispy 3d viewer
- New experimental functions for (pairwise) alignment of neurons using the `pycpd` package: `navis.nblast_align`, `navis.align.align_deform`, `navis.align.align_rigid`, `navis.align.align_pca`, `navis.align.align_pairwise`
- New `NeuronList` method: `navis.NeuronList.set_neuron_attributes`
- New utility functions: `navis.nbl.compress_scores`, `navis.nbl.nblast_prime`

##### Improvements
- `navis.xform_brain` now recognizes the target template's units if available
- Improved persistence functions: `navis.persistence_distances`, `navis.persistence_vector`, `navis.persistence_diagram`
- `navis.longest_neurite` and `navis.cell_body_fiber` now also allow removing the longest neurite and CBF, respectively
- `navis.heal_skeleton` now accepts a `mask` parameter that allows restricting where fragments are stitched

##### Fixes
- Various other bugfixes

**Full Changelog**: [v1.4.0...v1.5.0](https://github.com/navis-org/navis/compare/v1.4.0...v1.5.0)

## Version `1.4.0` { data-toc-label="1.4.0" }
_Date: 21/12/22_

##### Breaking
- `navis.flow_centrality` was renamed to `navis.synapse_flow_centrality` and a new non-synaptic `navis.flow_centrality` function was added. This also impacts the `method` parameter in `navis.split_axon_dendrite`!
- `vispy` is now a soft dependency

##### Additions
- New function: `navis.read_tiff` to read image stacks from TIFF files
- New utility function: `navis.nbl.extract_matches`

##### Improvements
- NBLASTs: single progress bar instead of one for each process
- New `via` parameter for `navis.xform_brain`
- `navis.write_swc` can now save Dotprops to SWC files
- `navis.make_dotprops` can now downsample point cloud inputs
- Various improvements to `navis.split_axon_dendrite`, `navis.nblast_allbyall`, `navis.interfaces.neuprint.fetch_mesh_neuron`, `navis.interfaces.neuprint.fetch_skeletons`

##### Fixes
- Tons of bug fixes

**Full Changelog**: [v1.3.1...v1.4.0](https://github.com/navis-org/navis/compare/v1.3.1...v1.4.0)

## Version `1.3.1` { data-toc-label="1.3.1" }
_Date: 10/06/22_

##### Fixes
- Various bugs fixed

**Full Changelog**: [v1.3.0...v1.3.1](https://github.com/navis-org/navis/compare/v1.3.0...v1.3.1)

## Version `1.3.0` { data-toc-label="1.3.0" }
_Date: 10/05/22_

##### Breaking
- As of this version `pip install navis` won't install a vispy backend

##### Additions
- New interface to fetch data from Virtual Fly Brain: `navis.interfaces.vfb`
- Tools to build custom NBLAST score matrices
- Bayesian implementation of the network traversal model: `navis.models.network_models.BayesianTraversalModel`
- New morphometrics functions: `navis.segment_analysis` & `navis.form_factor`
- New function to write meshes: `navis.write_mesh`

##### Improvements
- NBLASTs: new `approx_nn` parameter
- Example neurons now come with some meta data

##### Fixes
- Lots of fixes and improvements in particular for I/O-related functions

**Full Changelog**: [v1.2.1...v1.3.0](https://github.com/navis-org/navis/compare/v1.2.1...v1.3.0)

## Version `1.2.1` { data-toc-label="1.2.1" }
_Date: 25/02/22_

##### Fixes
- Hot fix for `navis.split_axon_dendrite`

**Full Changelog**: [v1.2.0...v1.2.1](https://github.com/navis-org/navis/compare/v1.2.0...v1.2.1)

## Version `1.2.0` { data-toc-label="1.2.0" }
_Date: 24/02/22_

##### Additions
- New function: `navis.betweeness_centrality`
- New function: `navis.combine_neurons` to simply concatenate neurons
- New set of persistence functions: `navis.persistence_vectors`, `navis.persistence_points` and `navis.persistence_distances`
- Added a new interface with the Allen Cell Types Atlas

##### Improvements
- Improvements to various functions: e.g. `navis.bending_flow`, `navis.synapse_flow_centrality`, `navis.split_axon_dendrite`, `navis.longest_neurite`
- `navis.write_nrrd` and `navis.read_nrrd` can now be used to write/read Dotprops to/from NRRD files
- `navis.read_swc` now accepts a `limit` parameter that enables reading on the first N neurons
- `navis.nblast` (and variants) now accept a `precision` parameter
- `navis.simplify_mesh` (and therefore `navis.downsample_neuron` with skeletons) now uses the `pyfqmr` if present
- Improved the interface to Neuromorpho

##### Fixes
- Myriads of small and big bugfixes

**Full Changelog**: [v1.1.0...v1.2.0](https://github.com/navis-org/navis/compare/v1.1.0...v1.2.0)

## Version `1.1.0` { data-toc-label="1.1.0" }
_Date: 18/11/21_

##### Additions
- New function: `navis.sholl_analysis`
- Plotly is now correctly chosen as default backend in Google colab

##### Fixes
- Fixed a critical bug with plotting skeletons with plotly `5.4.0`

**Full Changelog**: [v1.0.0...v1.1.0](https://github.com/navis-org/navis/compare/v1.0.0...v1.1.0)

## Version `1.0.0` { data-toc-label="1.0.0" }
_Date: 11/11/21_

##### Breaking
- `navis.MeshNeuron`: `__getattr__` does not search `trimesh` representation anymore
- NBLASTs: queries/targets now MUST be `navis.Dotprops` (no more automatic conversion, use `navis.make_dotprops`)
- Renamed functions to make it clear they work only on `TreeNeurons`:
- `smooth_neuron` :octicons-arrow-right-24: `navis.smooth_skeleton`
- `reroot_neuron` :octicons-arrow-right-24: `navis.reroot_skeleton`
- `rewire_neuron` :octicons-arrow-right-24: `navis.rewire_skeleton`
- `despike_neuron` :octicons-arrow-right-24: `navis.despike_skeleton`
- `average_neurons` :octicons-arrow-right-24: `navis.average_skeletons`
- `heal_fragmented_neuron` :octicons-arrow-right-24: `navis.heal_skeleton`
- `stitch_neurons` :octicons-arrow-right-24: `navis.stitch_skeletons`
- `cut_neuron` :octicons-arrow-right-24: `navis.cut_skeleton`
- Removals and other renamings:
    - `navis.clustering` module was removed and with it `navis.cluster_xyz` and `ClustResult` class
    - renamed `cluster_by_synapse_placement` :octicons-arrow-right-24: `navis.synapse_similarity`
    - renamed `cluster_by_connectivity` :octicons-arrow-right-24: `navis.connectivity_similarity`
    - renamed `sparseness` :octicons-arrow-right-24: `navis.connectivity_sparseness`
    - renamed `navis.write_google_binary` :octicons-arrow-right-24: `navis.write_precomputed`
- `navis.geodesic_matrix` renamed parameter `tn_ids` :octicons-arrow-right-24: `from_`

##### Additions & Improvements
- `navis.NeuronList.apply()` now allows omitting failures
- `navis.VoxelNeuron`:
    - new class representing neurons as voxels
    - new (experimental) class representing neurons as voxels
    - `navis.read_nrrd` now returns `VoxelNeuron` instead of `Dotprops` by default
    - currently works with only a selection of functions
- `navis.TreeNeuron`:
    - can now be initialized directly with `skeletor.Skeleton`
    - new method: `navis.TreeNeuron.snap`
- `navis.MeshNeuron`:
    - `navis.in_volume`, `navis.subset_neuron` and `navis.break_fragments` now work on `MeshNeurons`
    - new properties: `.skeleton`, `.graph` and `.igraph`
    - new methods: `navis.MeshNeuron.skeletonize` and `navis.MeshNeuron.snap`
    - can now be initialized with `skeletor.Skeleton` and `(vertices, faces)` tuple
    - plotting: `color_by` parameter now works with `MeshNeurons`
- `navis.Dotprops`:
    - new property: `.sampling_resolution` (used e.g. for scaling vectors for plotting)
    - new method: `navis.Dotprops.snap`
- Experimental support for non-isometric `.units` for neurons
- NBLASTs:
    - new parameter `limit_dist` allows speeding up NBLASTs with minor precision loss
    - new experimental parameter `batch_size` to NBLAST neurons in batches
    - overall faster initialization with large lists of neurons
- SWC I/O (`navis.read_swc` & `navis.write_swc`):
    - by default we will now deposit neuron meta data (name, id, units) in the SWC header (see `write_meta` parameter)
    - meta data in SWC header can also be read back (see `read_meta` parameter)
    - filenames can now be parsed into specific neuron properties (see `fmt` parameter)
    - node IDs now start with 0 instead of 1 when writing SWC files
- I/O to/from Google neuroglancer's precomputed format:
    - total rework of this module
    - renamed `navis.write_google_binary` :octicons-arrow-right-24: `navis.write_precomputed`
    - new function: `navis.read_precomputed`
- Plotting:
    - new function `navis.plot_flat` plots neurons as dendrograms
    - `navis.plot3d` with plotly backend now returns a plotly `Figure` instead of a figure dictionary
    - new [k3d](https://k3d-jupyter.org) backend for plotting in Jupyter environments: try `navis.plot3d(x, backend='k3d')`
    - new parameter for `navis.plot2d` and `navis.plot3d`: use `clusters=[0, 0, 0, 1, 1, ...]` to assigns clusters and have them automatically coloured accordingly
    - `navis.plot2d` now allows `radius=True` parameter
- Transforms:
    - support for elastix (`navis.transforms.ElastixTransform`)
    - whether transforms are invertible is now determined by existence of `__neg__` method
- Most functions that work with `TreeNeurons` now also work with `MeshNeurons`
- New high-level wrappers to convert neurons: `navis.voxelize`, `navis.mesh` and `navis.skeletonize`
- `navis.make_dotprops` now accepts `parallel=True` parameter for parallel processing
- `navis.smooth_skeleton` can now be used to smooth arbitrary numeric columns in the node table
- New function `navis.drop_fluff` removes small disconnected bits and pieces from neurons
- New function `navis.patch_cloudvolume` monkey-patches `cloudvolume` (see the new tutorial)
- New function `navis.write_nrrd` writes `VoxelNeurons` to NRRD files
- New functions to read/write `MeshNeurons`: `navis.read_mesh` and `navis.write_mesh`
- New function `navis.read_nmx` reads pyKNOSSOS files
- New function `navis.smooth_mesh` smoothes meshes and `MeshNeurons`
- Improved/updated the InsectBrain DB interface (see the tutorial)
- Under-the-hood fixes and improvements to: `navis.plot2d`, `navis.split_axon_dendrite`, `navis.tortuosity`, `navis.resample_skeleton`, `navis.mirror_brain`
- First pass at a `NEURON` interface (see the new tutorial)
- First pass at interface with the Allen's MICrONS datasets (see the new tutorial)
- `NAVIS_SKIP_LOG_SETUP` environment variable prevents default log setup for library use
- Improved `navis.cable_overlap`

##### Fixes
- Under-the-hood fixes and improvements

**Full Changelog**: [v0.6.0...v1.0.0](https://github.com/navis-org/navis/compare/v0.6.0...v1.0.0)

## Version `0.6.0` { data-toc-label="0.6.0" }
_Date: 12/05/21_

##### Additions
- new functions: `navis.prune_at_depth`, `navis.read_rda`, `navis.cell_body_fiber`
- new functions to map units into neuron space: `BaseNeuron.map_units` and `navis.to_neuron_space`

##### Improvements
- many spatial parameters (e.g. in `navis.resample_skeleton`) can now be passed as unit string, e.g. `"5 microns"`
- many functions now accept a `parallel=True` parameter to use multiple cores (depends on `pathos`)
- `navis.read_swc` and `navis.write_swc` can now read/write directly from/to zip files
- reworked `navis.read_json`, and `navis.write_json`
- `nblast` functions now let you use your own scoring function (thanks to Ben Pedigo!)
- added `threshold` parameter to `navis.read_nrrd`
- `navis.nblast_smart`: drop `quantile` and add `score` criterion
- functions that manipulate neurons will now always return something (even if `inplace=True`)
- `navis.cut_skeleton` now always returns a single `NeuronList`
- `navis.mirror_brain` now works with `k=0/None` Dotprops
- all `reroot_to_soma` parameters have been renamed to `reroot_soma`
- `navis.TreeNeuron` now has a `soma_pos` property that can also be used to set the soma by position
- made transforms more robust against points outside deformation fields
- better deal if node ID of soma is `0` (e.g. during plotting)
- `navis.neuron2tangents` now drops zero-length vectors

##### Fixes
- fixed `navis.guess_radius`
- fixed NBLAST progress bars in notebook environments
- fixed a couple bugs with `CMTK` transforms

**Full Changelog**: [v0.5.3...v0.6.0](https://github.com/navis-org/navis/compare/v0.5.3...v0.6.0)

## Version `0.5.3` { data-toc-label="0.5.3" }
_Date: 10/04/21_

##### Additions
- new functions: `navis.nblast_smart`, `navis.synblast`, `navis.symmetrize_brain`
- `navis.plot2d`: `rasterize=True` will rasterize neurons (but not axes or labels) to help keep file sizes low
- `navis.plot3d` (plotly): `hover_name=True` will show neuron names on hover

##### Improvements
- `navis.simplify_mesh` now supports 3 backends: Blender3D, `open3d` or `pymeshlab`
- `navis.make_dotprops` can now produce `Dotprops` purely from skeleton edges (set `k=None`)
- reworked `navis.write_swc` (faster, easier to work with)
- a new type of landmark-based transform: moving least square transforms (thanks to Chris Barnes)
- vispy `navis.Viewer`: press B to show a bounding box
- moved tests from Travis to Github Actions (this now also includes testing tutorial notebooks)

##### Fixes
- a great many small and big bug fixes

**Full Changelog**: [v0.5.2...v0.5.3](https://github.com/navis-org/navis/compare/v0.5.2...v0.5.3)

## Version `0.5.2` { data-toc-label="0.5.2" }
_Date: 02/02/21_

##### Additions
- new functions: `navis.xform`, `navis.write_precomputed`

##### Improvements
- `navis.downsample_neuron` now also works on `Dotprops`
- Neurons: connectors are now included in bounding box calculations
- NeuronLists: added progress bar for division / multiplication

**Full Changelog**: [v0.5.1...v0.5.2](https://github.com/navis-org/navis/compare/v0.5.1...v0.5.2)

## Version `0.5.1` { data-toc-label="0.5.1" }
_Date: 10/01/21_

##### Fixes
- Various under-the-hood improvements and bugfixes

**Full Changelog**: [v0.5.0...v0.5.1](https://github.com/navis-org/navis/compare/v0.5.0...v0.5.1)

## Version `0.5.0` { data-toc-label="0.5.0" }
_Date: 05/01/21_

##### Additions
- new functions for transforming spatial data (locations, neurons, etc) between brain spaces:
    - `navis.xform_brain` transforms data from one space to another
    - `navis.mirror_brain` mirrors data about given axis
    - see the new tutorials for explanations
- low-level interfaces to work with affine, H5-, CMTK- and thin plate spline transforms

##### Improvements
- de-cluttered top level namespace: some more obscure functions are now only available through modules

**Full Changelog**: [v0.4.3...v0.5.0](https://github.com/navis-org/navis/compare/v0.4.3...v0.5.0)

## Version `0.4.3` { data-toc-label="0.4.3" }
_Date: 22/12/20_

##### Fixes
- Small bugfixes

**Full Changelog**: [v0.4.2...v0.4.3](https://github.com/navis-org/navis/compare/v0.4.2...v0.4.3)

## Version `0.4.2` { data-toc-label="0.4.2" }
_Date: 22/12/20_

##### Fixes
- Small bugfixes

**Full Changelog**: [v0.4.1...v0.4.2](https://github.com/navis-org/navis/compare/v0.4.1...v0.4.2)

## Version `0.4.1` { data-toc-label="0.4.1" }
_Date: 06/12/20_

##### Fixes
- Critical bugfix in NBLAST

**Full Changelog**: [v0.4.0...v0.4.1](https://github.com/navis-org/navis/compare/v0.4.0...v0.4.1)

## Version `0.4.0` { data-toc-label="0.4.0" }
_Date: 06/12/20_

##### Additions
- native implementation of NBLAST: `navis.nblast` and `navis.nblast_allbyall`!
- new parameter `navis.plot3d` (plotly backend) with `hover_id=True` will show node IDs on hover
- `navis.Volume.resize` has now `inplace=False` as default

**Full Changelog**: [v0.3.4...v0.4.0](https://github.com/navis-org/navis/compare/v0.3.4...v0.4.0)

## Version `0.3.4` { data-toc-label="0.3.4" }
_Date: 24/11/20_

##### Improvements
- improved `navis.Dotprops`:
- more control over generation in `navis.make_dotprops`
- `navis.Dotprops` now play nicely with R interface

**Full Changelog**: [v0.3.3...v0.3.4](https://github.com/navis-org/navis/compare/v0.3.3...v0.3.4)

## Version `0.3.3` { data-toc-label="0.3.3" }
_Date: 23/11/20_

##### Additions
- new module: `models` for modelling networks and neurons
- new functions `navis.resample_along_axis`, `navis.insert_nodes`, `navis.remove_nodes`
- full rework of `navis.Dotprops`:
- make them a subclass of BaseNeuron
- implement `nat:dotprops` in `navis.make_dotprops`
- added `navis.read_nrrd` and `navis.write_nrrd`
- side-effect: renamed `navis.from_swc` :octicons-arrow-right-24: `read_swc` and `navis.to_swc` :octicons-arrow-right-24: `write_swc`
- improved conversion between nat and {{ navis }} `Dotprops`
- full rework of topology-related functions:
- `navis.strahler_index`, `navis.segregation_index`, `navis.bending_flow`, `navis.synapse_flow_centrality` and `navis.split_axon_dendrite` now work better, faster and more accurately. See their docs for details.
- new function: `navis.arbor_segregation_index`
- new `color_by` and `shade_by` parameters for `plot3d` and `plot2d` that lets you color/shade a
neuron by custom properties (e.g. by Strahler index or compartment)

##### Improvements
- neurons are now more memory efficient:
    - pandas "categoricals" are used for connector and node "type" and "label" columns
    - add a `.memory_usage` method analogous to that of `pandas.DataFrames`
- `navis.NeuronList` can now be pickled!
- made `navis.Viewer` faster
- `navis.prune_twigs` can now (optionally) prune by `exactly` the desired length
- improved `navis.NeuronList.apply`

##### Fixes
- small bugfixes and improvements

**Full Changelog**: [v0.3.2...v0.3.3](https://github.com/navis-org/navis/compare/v0.3.2...v0.3.3)

## Version `0.3.2` { data-toc-label="0.3.2" }
_Date: 18/10/20_

##### Improvements
- `navis.plot2d` and `navis.plot3d` now accept `trimesh.Trimesh` directly
- `navis.in_volume` now works with any mesh-like object, not just `navis.Volumes`

##### Fixes
- lots of small bugfixes and improvements

**Full Changelog**: [v0.3.1...v0.3.2](https://github.com/navis-org/navis/compare/v0.3.1...v0.3.2)

## Version `0.3.1` { data-toc-label="0.3.1" }
_Date: 07/10/20_

##### Additions
- new function `navis.rewire_skeleton`

##### Improvements
- `navis.heal_skeleton` and `navis.stitch_skeletons` are now much much faster
- `navis.reroot_skeleton` can now reroot to multiple roots in one go
- `navis.plot3d` now accepts a `soma` argument
- improved caching for neurons
- improved multiplication/division of neurons
- faster `r.nblast` and `r.nblast_allbyall`
- `r.xform_brain` now also adjusts the soma radius
- `neuprint.fetch_skeletons` now returns correct soma radius

##### Fixes
- lots of small bugfixes

**Full Changelog**: [v0.3.0...v0.3.1](https://github.com/navis-org/navis/compare/v0.3.0...v0.3.1)

## Version `0.3.0` { data-toc-label="0.3.0" }
_Date: 06/10/20_

##### Additions
- Started module to manipulate mesh data (see e.g. `navis.simplify_mesh`)

##### Improvements
- Improved interfaces with R NBLAST and `xform_brain`
- Improved attribute caching for neurons

**Full Changelog**: [v0.2.3...v0.3.0](https://github.com/navis-org/navis/compare/v0.2.3...v0.3.0)

## Version `0.2.3` { data-toc-label="0.2.3" }
_Date: 06/09/20_

##### Additions
- New Neuron property `.label` that if present will be used for plot legends
- New function for R interface: `navis.interfaces.r.load_rda`

##### Improvements
- Blender interface: improved scatter plot generation

## Version `0.2.2` { data-toc-label="0.2.2" }
_Date: 15/08/20_

##### Additions
- New `plot3d` parameter: with plotly backend, use `fig` to add data to existing plotly figure
- New `plot3d` parameter: with vispy backend, use `center=False` to not re-center camera on adding new data
- New `r.mirror_brain` parameter: use e.g. `via='FCWB'` if source space does not have mirror transform
- New `NeuronList` method: `append()` works like `list.append()`
- First implementation of smarter (re-)calculation of temporary Neuron properties using `.is_stale` property
- Neurons can now be multiplied/divided by array/list of x/y/z coordinates for non-isometric transforms

##### Fixes
- Fix issues with newer rpy2 versions
- Various improvements and bug fixes

## Version `0.2.1` { data-toc-label="0.2.1" }
_Date: 20/04/20_

##### Additions
- New `plot3d` parameter: with plotly backend, use `radius=True` plots TreeNeurons with radius
- New `plot2d` parameter: `orthogonal=False` sets view to perspective

##### Improvements
- Various improvements to e.g. `nx2neuron`

## Version `0.2.0` { data-toc-label="0.2.0" }
_Date: 29/06/20_

##### Breaking
- `navis.nx2neuron` now returns a `navis.TreeNeuron` instead of a `DataFrame`

##### Additions
- New neuron class `navis.MeshNeuron`
- New `navis.TreeNeuron` property `.volume`
- New example data from the Janelia hemibrain data set

##### Improvements
- Clean-up in neuromorpho interface
- We now use [ncollpyde](https://pypi.org/project/ncollpyde) for ray casting (intersections)

##### Fixes
- Fix bugs in `navis.Volume` pickling

## Version `0.1.16` { data-toc-label="0.1.16" }
_Date: 26/05/20_

##### Fixes
- Many small bugfixes

## Version `0.1.15` { data-toc-label="0.1.15" }
_Date: 15/05/20_

##### Improvements
- Improvements to R and Blender interface
- Improved loading from SWCs (up to 2x faster)
- `TreeNeurons`: allow rerooting by setting the `.root` attribute

## Version `0.1.14` { data-toc-label="0.1.14" }
_Date: 05/05/20_

##### Fixes
- Emergency fixes for critical bugs

## Version `0.1.13` { data-toc-label="0.1.13" }
_Date: 05/05/20_

##### Additions
- new function: `navis.vary_color`

##### Improvements
- improvements to Blender interface and various other functions

## Version `0.1.12` { data-toc-label="0.1.12" }
_Date: 02/04/20_

##### Imnprovements
- `navis.Volume` is now sublcass of `trimesh.Trimesh`

## Version `0.1.11` { data-toc-label="0.1.11" }
_Date: 28/02/20_

##### Improvements
- improved `navis.stitch_neurons`: much faster now if you have iGraph

##### Fixes
- fixed errors when using multiprocessing (e.g. in `NeuronList.apply`)
- fixed bugs in `navis.downsample_neuron`

## Version `0.1.10` { data-toc-label="0.1.10" }
_Date: 24/02/20_

##### Fixes
- Fixed bugs in Blender interface introduced in 0.1.9

## Version `0.1.9` { data-toc-label="0.1.9" }
_Date: 24/02/20_

##### Fixes
- Removed hard-coded swapping and translation of axes in the Blender interface
- Fixed bugs in `navis.stitch_neurons`

## Version `0.1.8` { data-toc-label="0.1.8" }
_Date: 21/02/20_

##### Fixes
- Again lots of fixed bugs

## Version `0.1.0` { data-toc-label="0.1.0" }
_Date: 23/05/19_

##### Fixes
- Many small bugfixes

## Version `0.0.1` { data-toc-label="0.0.1" }
_Date: 29/01/19_

##### Fixes
- First commit, lots to fix.

