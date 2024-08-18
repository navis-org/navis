
.. _whats_new:

What's new?
===========
This is a selection of features added, changes made and bugs fixed with each version.
For a full list of changes please see the
`commits <https://github.com/navis-org/navis/commits/master>`_ on navis' Github
repository.

.. list-table::
   :widths: 7 7 86
   :header-rows: 1

   * - Version
     - Date
     -
   * - 1.7.0
     - 25/07/24
     - - BREAKING:
          - plotting functions: dropped the ``cluster`` parameter in favour of an improved ``color_by`` logic (see below)
       - Additions:
          - ``navis`` now uses `navis-fastcore <https://github.com/schlegelp/fastcore-rs/tree/main/py>`
            if present to dramatically speed up core functions (see updated install instructions)
          - new method :meth:`navis.NeuronList.add_metadata` to quickly add metadata to neurons
       - Improvements:
          - :func:`navis.find_soma` and :func:`navis.graph.neuron2nx` (used under the hood) are now much faster
          - all I/O functions such as :func:`navis.read_swc` now show which file caused an error (if any); original
            filenames are tracked as ``file`` property
          - :class:`navis.NeuronList` will only search the first 100 neurons for autocompletion to avoid freezing with
            large lists
          - plotting functions: `color_by` now accepts either a list of labels (one per neuron) or the name of a neuron property
          - :func:`navis.subset_neuron` is now faster and more memory efficient when subsetting meshes
          - :meth:`navis.TreeNeuron.cable_length` is now faster
       - Fixes:
          - all I/O functions such as :func:`navis.read_swc` now ignore hidden files (filename starts with `._`)
          - :func:`navis.read_swc` now actually uses the soma label (if present) to set the soma node
          - fixed a bug in plotting when using vertex colors
          - fixed the progress bar in :func:`navis.interfaces.neuprint.fetch_mesh_neuron`
          - fixed a bug in :func:`navis.synblast` that caused multiprocessing to fail (pickling issue with `pykdtree`)
          - :func:`navis.interfaces.neuprint.fetch_mesh_neuron` will now ignore the `lod` parameter if the data source does not
            support it instead of breaking
          - fixed  a number of depcrecation warnings in the codebase
   * - 1.6.0
     - 07/04/24
     - - BREAKING:
          - dropped support for Python 3.8, per `NEP 29 <https://numpy.org/neps/nep-0029-deprecation_policy.html>`_
          - :func:`navis.write_swc` no longer supports writing Dotprops to SWC files
       - Additions:
          - new property ``Treenode.surface_area``
          - new (experimental) functions :func:`navis.read_parquet` and :func:`navis.write_parquet`
            store skeletons and dotprops in parquet files (see `here <https://github.com/clbarnes/neurarrow>`_
            for format specs)
          - new :func:`navis.read_nml` function to read single NML file (complements
            existing :func:`navis.read_nmx` files which are collections of NMLs)
          - :class:`navis.NodeConnectorRelation` is an :class:`enum.IntEnum`
            encoding relationships between (tree)nodes and connector nodes,
            used in neurons' connector tables.
          - new :class:`navis.NeuronConnector` class for creating connectivity graphs
            from groups neurons with consistent connector IDs.
          - new method for CMTKtransforms: :meth:`navis.transforms.CMTKTransform.xform_image`
       - Improvements:
          - improved performance for:
            - adding recordings to ``CompartmentModel``
            - :func:`navis.heal_skeleton` and :func:`navis.resample_skeleton`
          - improved logic for splitting NBLASTs across multiple cores
          - :func:`navis.xform_brain`: now allows to specify multiple intermediate
            template spaces through the ``via`` parameter and to ignore spaces
            through the ``avoid`` parameter
          - i/o functions can now read directly from `.tar` or `.tar.gz` files (`.zip`
            was already supported)
          - :func:`navis.read_precomputed` now accepts a ``limit`` parameter similar
            to :func:`navis.read_swc`
       - Fixes:
          - fixed interface to InsectBrainDB
          - :func:`navis.read_precomputed`:
            - now correctly parses the `info` file depending on the source
            - reading large files (i.e. meshes) directly from a URL should not break anymore
          - fixed writing vertex properties in :func:`navis.write_precomputed`
          - fixed a bug in :func:`navis.resample_skeleton`
          - fixed an occasional issue when plotting skeletons with radii
          - fix bug in :func:`navis.subset_neuron` that caused connectors to be dropped when using mask
          - fixed a bug in :func:`navis.despike_skeleton` that cause the `reverse` argument to be ignored
          - fixed two small bugs in :func:`navis.interfaces.neuprint.fetch_mesh_neuron`
   * - 1.5.0
     - 27/07/23
     - - BREAKING: dropped support for Python 3.7
       - new function: :func:`navis.pop3d` removes the most recently added object from the vispy 3d viewer
       - new experimental functions for (pairwise) alignment of neurons using the ``pycpd`` package:
         :func:`navis.nblast_align`, :func:`navis.align.align_deform`, :func:`navis.align.align_rigid`,
         :func:`navis.align.align_pca`, :func:`navis.align.align_pairwise`
       - :func:`navis.xform_brain` now recognizes the target template's units if available
       - new ``NeuronList`` method: :func:`navis.NeuronList.set_neuron_attributes`
       - new utility functions: :func:`navis.nbl.compress_scores`, :func:`navis.nbl.nblast_prime`
       - improved persistence functions: :func:`navis.persistence_distances`, :func:`navis.persistence_vector`, :func:`navis.persistence_diagram`
       - :func:`navis.longest_neurite` and :func:`navis.cell_body_fiber` now also allow
         removing the longest neurite and CBF, respectively
       - :func:`navis.heal_skeleton` now accepts a `mask` parameter that allows restricting where fragments are stitched
       - various other bugfixes
   * - 1.4.0
     - 21/12/22
     - - BREAKING: ``navis.flow_centrality`` was renamed to :func:`navis.synapse_flow_centrality`
         and a new non-synaptic :func:`navis.flow_centrality` function was added. This also
         impacts the ``method`` parameter in :func:`navis.split_axon_dendrite`!
       - `vispy` is now a soft dependency
       - new function: :func:`navis.read_tiff` to read image stacks from TIFF files
       - NBLASTs: single progress bar instead of one for each process
       - new ``via`` parameter for :func:`navis.xform_brain`
       - new utility function: :func:`navis.nbl.extract_matches`
       - :func:`navis.write_swc` can now save Dotprops to SWC files
       - :func:`navis.make_dotprops` can now downsample point cloud inputs
       - various improvements to :func:`navis.split_axon_dendrite`, :func:`navis.nblast_allbyall`,
         :func:`navis.interfaces.neuprint.fetch_mesh_neuron`, :func:`navis.interfaces.neuprint.fetch_skeletons`
       - tons of bug fixes
   * - 1.3.1
     - 10/06/22
     - - fixed various bugs
   * - 1.3.0
     - 10/05/22
     - - as of this version ``pip install navis`` won't install a vispy backend (see :ref:`install instructions <installing>` for details)
       - new interface to fetch data from Virtual Fly Brain: ``navis.interfaces.vfb``
       - tools to build custom NBLAST score matrices (big thanks to Chris Barnes!), see the new :ref:`tutorial <smat_intro>`
       - Bayesian implementation of the network traversal model: :class:`~navis.models.network_models.BayesianTraversalModel` (big thanks to Andrew Champion!)
       - NBLASTs: new ``approx_nn`` parameter (sacrifices precision for speed)
       - example neurons now come with some meta data
       - new morphometrics functions: :func:`navis.segment_analysis` & :func:`navis.form_factor`
       - new function to write meshes: :func:`navis.write_mesh`
       - lots of fixes and improvements in particular for i/o-related functions
   * - 1.2.1
     - 25/02/22
     - - hot fix for :func:`navis.split_axon_dendrite`
   * - 1.2.0
     - 24/02/22
     - - new function: :func:`navis.betweeness_centrality`
       - new function: :func:`navis.combine_neurons` to simply concatenate neurons
       - new set of persistence functions: :func:`navis.persistence_vectors`,
         :func:`navis.persistence_points` and :func:`navis.persistence_distances`
       - improvements to various functions: e.g. :func:`navis.bending_flow`,
         :func:`navis.synapse_flow_centrality`, :func:`navis.split_axon_dendrite`,
         :func:`navis.longest_neurite`
       - :func:`navis.read_swc` now accepts a ``limit`` parameter that enables
         reading on the the first N neurons (useful to sample large collections)
       - :func:`navis.write_nrrd` and :func:`navis.read_nrrd` can now be used to
         write/read Dotprops to/from NRRD files
       - :func:`navis.nblast` (and variants) now accept a ``precision`` parameter
         that allows setting the datatype for the matrix (useful to keep memory
         usage low for large NBLASTs)
       - :func:`navis.simplify_mesh` (and therefore :func:`navis.downsample_neuron`
         with skeletons) now uses the ``pyfqmr`` if present (much faster!)
       - improved the interface to Neuromorpho
       - added a new interface with the Allen Cell Types Atlas (see
         :mod:`navis.interfaces.allen_celltypes`)
       - myriads of small and big bugfixes
   * - 1.1.0
     - 18/11/21
     - - new function :func:`navis.sholl_analysis`
       - plotly is now correctly chosen as default backend in Google colab
       - fixes a critical bug with plotting skeletons with plotly `5.4.0`
   * - 1.0.0
     - 11/11/21
     - Breaking changes:

       - :class:`~navis.MeshNeuron`:
           - ``__getattr__`` does not search ``trimesh`` representation anymore
       - NBLASTs:
           - queries/targets now MUST be :class:`~navis.Dotprops` (no more automatic conversion, use :func:`~navis.make_dotprops`)
       - renamed functions to make it clear they work only on ``TreeNeurons`` (i.e. skeletons):
           - ``smooth_neuron`` -> :func:`~navis.smooth_skeleton`
           - ``reroot_neuron`` -> :func:`~navis.reroot_skeleton`
           - ``rewire_neuron`` -> :func:`~navis.rewire_skeleton`
           - ``despike_neuron`` -> :func:`~navis.despike_skeleton`
           - ``average_neurons`` -> :func:`~navis.average_skeletons`
           - ``heal_fragmented_neuron`` -> :func:`~navis.heal_skeleton`
           - ``stitch_neurons`` -> :func:`~navis.stitch_skeletons`
           - ``cut_neuron`` -> :func:`~navis.cut_skeleton`
       - removals and other renamings:
           - ``navis.clustering`` module was removed and with it ``navis.cluster_xyz`` and ``ClustResult`` class
           - renamed ``cluster_by_synapse_placement`` -> :func:`~navis.synapse_similarity`
           - renamed ``cluster_by_connectivity`` -> :func:`~navis.connectivity_similarity`
           - renamed ``sparseness`` -> :func:`~navis.connectivity_sparseness`
           - renamed ``navis.write_google_binary`` -> :func:`~navis.write_precomputed`
       - :func:`~navis.geodesic_matrix` renamed parameter ``tn_ids`` -> ``from_``

       New things & Bugfixes:

       - :class:`~navis.NeuronList`:
           - :meth:`~navis.NeuronList.apply` now allows omitting failures (see ``omit_failures`` parameter)
       - :class:`~navis.VoxelNeuron`:
           - new (experimental) class representing neurons as voxels
           - :func:`~navis.read_nrrd` now returns ``VoxelNeuron`` instead of ``Dotprops`` by default
           - currently works with only a selection of functions
       - :class:`~navis.TreeNeuron`:
           - can now be initialized directly with ``skeletor.Skeleton``
           - new method: :meth:`~navis.TreeNeuron.snap`
       - :class:`~navis.MeshNeuron`:
           - :func:`~navis.in_volume`, :func:`~navis.subset_neuron` and :func:`~navis.break_fragments` now work on ``MeshNeurons``
           - new properties: ``.skeleton``, ``.graph`` and ``.igraph``
           - new methods: :meth:`~navis.MeshNeuron.skeletonize` and :meth:`~navis.MeshNeuron.snap`
           - can now be initialized with ``skeletor.Skeleton`` and ``(vertices, faces)`` tuple
           - plotting: ``color_by`` parameter now works with ``MeshNeurons``
       - :class:`~navis.Dotprops`:
           - new property: ``.sampling_resolution`` (used e.g. for scaling vectors for plotting)
           - new method :meth:`~navis.Dotprops.snap`
       - experimental support for non-isometric ``.units`` for neurons
       - NBLASTs:
           - new parameter ``limit_dist`` allows speeding up NBLASTs with minor precision loss
           - new experimental parameter ``batch_size`` to NBLAST neurons in batches
           - overall faster initialization with large lists of neurons
       - SWC I/O (:func:`~navis.read_swc` & :func:`~navis.write_swc`):
           - by default we will now deposit neuron meta data (name, id, units) in the SWC header (see ``write_meta`` parameter)
           - meta data in SWC header can also be read back (see ``read_meta`` parameter)
           - filenames can now be parsed into specific neuron properties (see ``fmt`` parameter)
           - node IDs now start with 0 instead of 1 when writing SWC files
       - I/O to/from Google neuroglancer's precomputed format:
           - total rework of this module
           - renamed ``navis.write_google_binary`` -> :func:`~navis.write_precomputed`
           - new function: :func:`~navis.read_precomputed`
       - plotting:
           - new function :func:`navis.plot_flat` plots neurons as dendrograms
           - :func:`~navis.plot3d` with plotly backend now returns a plotly ``Figure`` instead of a figure dictionary
           - new `k3d <https://k3d-jupyter.org>`_ backend for plotting in Jupyter environments: try ``navis.plot3d(x, backend='k3d')``
           - new parameter for :func:`~navis.plot2d` and :func:`~navis.plot3d`: use ``clusters=[0, 0, 0, 1, 1, ...]`` to assigns
             clusters and have them automatically coloured accordingly
           - :func:`~navis.plot2d` now allows ``radius=True`` parameter
       - transforms:
           - support for elastix (:class:`navis.transforms.ElastixTransform`)
           - whether transforms are invertible is now determined by existence of ``__neg__`` method
       - most functions that work with ``TreeNeurons`` now also work with ``MeshNeurons``
       - new high-level wrappers to convert neurons: :func:`navis.voxelize`, :func:`navis.mesh` and :func:`navis.skeletonize`
       - :func:`~navis.make_dotprops` now accepts ``parallel=True`` parameter for parallel processing
       - :func:`~navis.smooth_skeleton` can now be used to smooth arbitrary numeric columns in the node table
       - new function :func:`navis.drop_fluff` removes small disconnected bits and pieces from neurons
       - new function :func:`navis.patch_cloudvolume` monkey-patches `cloudvolume` (see the new :ref:`tutorial <cloudvolume_tut>`)
       - new function :func:`navis.write_nrrd` writes ``VoxelNeurons`` to NRRD files
       - new functions to read/write ``MeshNeurons``: :func:`~navis.read_mesh` and :func:`navis.write_mesh`
       - new function :func:`navis.read_nmx` reads pyKNOSSOS files
       - new function :func:`~navis.smooth_mesh` smoothes meshes and ``MeshNeurons``
       - improved/updated the InsectBrain DB interface (see the :ref:`tutorial <insectbraindb_tut>`)
       - under-the-hood fixes and improvements to: :func:`~navis.plot2d`, :func:`~navis.split_axon_dendrite`, :func:`~navis.tortuosity`, :func:`~navis.resample_skeleton`, :func:`~navis.mirror_brain`
       - first pass at a ``NEURON`` interface (see the new :ref:`tutorial <neuron_tut>`)
       - first pass at interface with the Allen's MICrONS datasets (see the new :ref:`tutorial <microns_tut>`)
       - ``NAVIS_SKIP_LOG_SETUP`` environment variable prevents default log setup for library use
       - improved :func:`~navis.cable_overlap`
   * - 0.6.0
     - 12/05/21
     - - new functions: :func:`navis.prune_at_depth`, :func:`navis.read_rda`, :func:`navis.cell_body_fiber`
       - many spatial parameters (e.g. in :func:`navis.resample_skeleton`) can now be passed as unit string, e.g. ``"5 microns"``
       - many functions now accept a ``parallel=True`` parameter to use multiple cores (depends on ``pathos``)
       - :func:`navis.read_swc` and :func:`navis.write_swc` can now read/write directly from/to zip files
       - reworked :func:`navis.read_json`, and :func:`navis.write_json`
       - ``nblast`` functions now let you use your own scoring function (thanks to Ben Pedigo!)
       - added ``threshold`` parameter to :func:`navis.read_nrrd`
       - fixed NBLAST progress bars in notebook environments
       - :func:`navis.nblast_smart`: drop ``quantile`` and add ``score`` criterion
       - new functions to map units into neuron space: :func:`~BaseNeuron.map_units` and :func:`navis.to_neuron_space`
       - functions that manipulate neurons will now always return something (even if ``inplace=True``)
       - :func:`navis.cut_skeleton` now always returns a single ``NeuronList``
       - :func:`navis.mirror_brain` now works with ``k=0/None`` Dotprops
       - all ``reroot_to_soma`` parameters have been renamed to ``reroot_soma``
       - :class:`navis.TreeNeuron` now has a ``soma_pos`` property that can also be used to set the soma by position
       - fixed a couple bugs with `CMTK` transforms
       - made transforms more robust against points outside deformation fields
       - better deal if node ID of soma is ``0`` (e.g. during plotting)
       - :func:`navis.neuron2tangents` now drops zero-length vectors
       - fixed :func:`navis.guess_radius`
   * - 0.5.3
     - 10/04/21
     - - new functions: :func:`navis.nblast_smart`, :func:`navis.synblast`, :func:`navis.symmetrize_brain`
       - :func:`navis.plot3d` (plotly): ``hover_name=True`` will show neuron names on hover
       - :func:`navis.plot2d`: ``rasterize=True`` will rasterize neurons (but not axes or labels) to help keep file sizes low
       - :func:`navis.simplify_mesh` now supports 3 backends: Blender3D, ``open3d`` or ``pymeshlab``
       - :func:`navis.make_dotprops` can now produce ``Dotprops`` purely from skeleton edges (set ``k=None``)
       - reworked :func:`navis.write_swc` (faster, easier to work with)
       - a new type of landmark-based transform: moving least square transforms (thanks to Chris Barnes)
       - vispy :class:`navis.Viewer`: press B to show a bounding box
       - moved tests from Travis to Github Actions (this now also includes testing tutorial notebooks)
       - a great many small and big bug fixes
   * - 0.5.2
     - 02/02/21
     - - new functions: :func:`navis.xform`, :func:`navis.write_precomputed`
       - :func:`navis.downsample_neuron` now also works on ``Dotprops``
       - Neurons:
         - connectors are now included in bounding boxes
       - NeuronLists:
         - added progress bar for division / multiplication
   * - 0.5.1
     - 10/01/21
     - - a couple under-the-hood improvements and bugfixes
   * - 0.5.0
     - 05/01/21
     - - new functions for transforming spatial data (locations, neurons, etc) between brain spaces:
           - :func:`navis.xform_brain` transforms data from one space to another
           - :func:`navis.mirror_brain` mirrors data about given axis
           - see the new :ref:`tutorials<example_gallery>` for explanations
           - low-level interfaces to work with affine, H5-, CMTK- and thin plate spline transforms
       - de-cluttered top level namespace: some more obscure functions are now only available through modules
   * - 0.4.3
     - 22/12/20
     - - more small bug fixes
   * - 0.4.2
     - 22/12/20
     - - some small bug fixes
   * - 0.4.1
     - 06/12/20
     - - hotfix for critical bug in NBLAST
   * - 0.4.0
     - 06/12/20
     - - native implementation of NBLAST: :func:`navis.nblast` and :func:`navis.nblast_allbyall`!
       - new parameter :func:`navis.plot3d` (plotly backend) with ``hover_id=True`` will show node IDs on hover
       - :func:`navis.Volume.resize` has now ``inplace=False`` as default
   * - 0.3.4
     - 24/11/20
     - - improved :class:`navis.Dotprops`:
           - more control over generation in :func:`navis.make_dotprops`
           - :class:`navis.Dotprops` now play nicely with R interface
   * - 0.3.3
     - 23/11/20
     - - new module: ``models`` for modelling networks and neurons
       - new functions :func:`navis.resample_along_axis`, :func:`navis.insert_nodes`, :func:`navis.remove_nodes`
       - full rework of :class:`navis.Dotprops`:
           - make them a subclass of BaseNeuron
           - implement ``nat:dotprops`` in :func:`navis.make_dotprops`
           - added :func:`navis.read_nrrd` and :func:`navis.write_nrrd`
           - side-effect: renamed ``navis.from_swc`` -> ``read_swc`` and ``navis.to_swc`` -> ``write_swc``
           - improved conversion between nat and navis ``Dotprops``
       - full rework of topology-related functions:
           - :func:`navis.strahler_index`, :func:`navis.segregation_index`, :func:`navis.bending_flow`, :func:`navis.synapse_flow_centrality` and :func:`navis.split_axon_dendrite` now work better, faster and more accurately. See their docs for details.
           - new function: :func:`navis.arbor_segregation_index`
       - new ``color_by`` and ``shade_by`` parameters for ``plot3d`` and ``plot2d`` that lets you color/shade a
         neuron by custom properties (e.g. by Strahler index or compartment)
       - neurons are now more memory efficient:
           - pandas "categoricals" are used for connector and node "type" and "label" columns
           - add a ``.memory_usage`` method analogous to that of ``pandas.DataFrames``
       - :class:`navis.NeuronList` can now be pickled!
       - made :class:`navis.Viewer` faster
       - :func:`navis.prune_twigs` can now (optionally) prune by `exactly` the desired length
       - improved ``navis.NeuronList.apply``
       - small bugfixes and improvements
   * - 0.3.2
     - 18/10/20
     - - :func:`navis.plot2d` and :func:`navis.plot3d` now accept ``trimesh.Trimesh`` directly
       - :func:`navis.in_volume` now works with any mesh-like object, not just ``navis.Volumes``
       - lots of small bugfixes and improvements
   * - 0.3.1
     - 07/10/20
     - - new function :func:`navis.rewire_skeleton`
       - improve :func:`navis.heal_skeleton` and :func:`navis.stitch_skeletons`: now much much faster
       - :func:`navis.reroot_skeleton` can now reroot to multiple roots in one go
       - :func:`navis.plot3d` now accepts a ``soma`` argument
       - improved caching for neurons
       - improved multiplication/division of neurons
       - faster ``r.nblast`` and ``r.nblast_allbyall``
       - ``r.xform_brain`` now also adjusts the soma radius
       - ``neuprint.fetch_skeletons`` now returns correct soma radius
       - lots of small bugfixes
   * - 0.3.0
     - 06/10/20
     - - started module to manipulate mesh data: see :func:`navis.simplify_mesh`
       - improved interfaces with R NBLAST and ``xform_brain``
       - improved attribute caching for neurons
   * - 0.2.3
     - 06/09/20
     - - new Neuron property ``.label`` that if present will be used for plot legends
       - new function for R interface: :func:`navis.interfaces.r.load_rda`
       - Blender interface: improved scatter plot generation
   * - 0.2.2
     - 15/08/20
     - - new ``plot3d`` parameter: with plotly backend, use ``fig`` to add data to existing plotly figure
       - new ``plot3d`` parameter: with vispy backend, use ``center=False`` to not re-center camera on adding new data
       - new ``r.mirror_brain`` parameter: use e.g. ``via='FCWB'`` if source space does not have mirror transform
       - new ``NeuronList`` method: ``append()`` works like ``list.append()``
       - first implementation of smarter (re-)calculation of temporary Neuron properties using ``.is_stale`` property
       - Neurons can now be multiplied/divided by array/list of x/y/z coordinates for non-isometric transforms
       - fix issues with newer rpy2 versions
       - various improvements and bug fixes
   * - 0.2.1
     - 20/07/20
     - - new ``plot3d`` parameter: with plotly backend, use ``radius=True`` plots TreeNeurons with radius
       - new ``plot2d`` parameter: ``orthogonal=False`` sets view to perspective
       - various improvements to e.g. ```nx2neuron``
   * - 0.2.0
     - 29/06/20
     - - new neuron class :class:`~navis.MeshNeuron` that consists of vertices and faces
       - new :class:`~navis.TreeNeuron` property ``.volume``
       - we now use `ncollpyde <https://pypi.org/project/ncollpyde>`_ for ray casting (intersections)
       - clean-up in neuromorpho interface
       - fix bugs in :class:`~navis.Volume` pickling
       - new example data from the Janelia hemibrain data set
       - breaking changes: :func:``~navis.nx2neuron`` now returns a :class:`~navis.TreeNeuron` instead of a ``DataFrame``
   * - 0.1.16
     - 26/05/20
     - - many small bugfixes
   * - 0.1.15
     - 15/05/20
     - - improvements to R and Blender interface
       - improved loading from SWCs (up to 2x faster)
       - TreeNeurons: allow rerooting by setting the ``.root`` attribute
   * - 0.1.14
     - 05/05/20
     - - emergency fixes for 0.1.13
   * - 0.1.13
     - 05/05/20
     - - new function :func:`navis.vary_color`
       - improvements to Blender interface and various other functions
   * - 0.1.12
     - 02/04/20
     - - :class:`~navis.Volume` is now sublcass of ``trimesh.Trimesh``
   * - 0.1.11
     - 28/02/20
     - - removed hard-coded swapping and translation of axes in the Blender interface
       - improved ``navis.stitch_neurons``: much faster now if you have iGraph
       - fixed errors when using multiprocessing (e.g. in ``NeuronList.apply``)
       - fixed bugs in :func:`navis.downsample_neuron`
   * - 0.1.10
     - 24/02/20
     - - fixed bugs in Blender interface introduced in 0.1.9
   * - 0.1.9
     - 24/02/20
     - - removed hard-coded swapping and translation of axes in the Blender interface
       - fixed bugs in ``stitch_neurons``
   * - 0.1.8
     - 21/02/20
     - - Again lots of fixed bugs
       - Blame myself for not keeping track of changes
   * - 0.1.0
     - 23/05/19
     - - Made lots of fixes
       - Promised myself to be better at tracking changes
   * - 0.0.1
     - 29/01/19
     - - First commit, lots to fix.
