.. _api:

API Reference
=============

``navis`` has grown a lot! Last I looked, there were ~110 functions exposed
at top level (e.g. ``navis.plot3d``) and easily another 100 secondary functions
available via submodules (e.g. ``navis.morpho.find_soma``). This can be a bit
daunting at first - especially if you don't exactly know what you are looking
for.

I recommend you either just have a browse, use the search field
(upper right) or simply search in page (CONTROL/CMD-F). Failing that, please
feel free to open an `issue <https://github.com/navis-org/navis/issues>`_ on
the Github repo with your question.

This API reference is a more or less complete account of the primary functions:

1. :ref:`Neuron- and NeuronList functions and methods <api_neurons>`
2. :ref:`Functions for visualization<api_plot>`
3. :ref:`Manipulate or analyze neuron morphology<api_morph>`
4. :ref:`Transforming and mirroring data<transfm>`
5. :ref:`Analyze connectivity<api_con>`
6. :ref:`Import/Export<api_io>`
7. :ref:`Utility functions<api_utility>`
8. :ref:`Which functions work with which neuron types?<api_func_matrix>`

In addition ``navis`` has interfaces to various external datasets and tools:

- :ref:`NEURON simulator<api_interfaces.neuron>`
- :ref:`Neuromorpho<api_interfaces.neuromorpho>`
- :ref:`neuPrint<api_interfaces.neuprint>`
- :ref:`InsectBrain DB<api_interfaces.insectdb>`
- :ref:`Blender 3D<api_interfaces.blender>`
- :ref:`Cytoscape<api_interfaces.cytoscape>`
- :ref:`Allen MICrONS datasets<api_interfaces.microns>`
- :ref:`Allen Cell Type Atlas<api_interfaces.celltypes>`
- :ref:`R and the natverse libraries<api_interfaces.r>`

Most of these functions include examples of how to use them. Click on them to
learn more!

.. _api_neurons:

Neuron/List
+++++++++++
``TreeNeurons``, ``MeshNeurons``, ``VoxelNeurons`` and ``Dotprops`` are neuron
classes. ``NeuronLists`` are containers thereof.

.. autosummary::
    :toctree: generated/

    navis.BaseNeuron
    navis.TreeNeuron
    navis.MeshNeuron
    navis.VoxelNeuron
    navis.Dotprops
    navis.NeuronList

General Neuron methods
----------------------
Despite being fundamentally different data types, all neurons share some common
methods (i.e. functions) which they inherit from their (abstract) parent
class ``BaseNeurons``.

.. autosummary::
    :toctree: generated/

    ~navis.BaseNeuron.copy
    ~navis.BaseNeuron.plot3d
    ~navis.BaseNeuron.plot2d
    ~navis.BaseNeuron.summary
    ~navis.BaseNeuron.convert_units
    ~navis.BaseNeuron.map_units
    ~navis.BaseNeuron.memory_usage

In addition to methods, neurons also have properties. These properties common
to all neurons:

.. autosummary::
    :toctree: generated/

    ~navis.BaseNeuron.bbox
    ~navis.BaseNeuron.connectors
    ~navis.BaseNeuron.postsynapses
    ~navis.BaseNeuron.presynapses
    ~navis.BaseNeuron.datatables
    ~navis.BaseNeuron.id
    ~navis.BaseNeuron.name
    ~navis.BaseNeuron.units
    ~navis.BaseNeuron.soma
    ~navis.BaseNeuron.type


TreeNeurons
-----------
These are class methods available specific for ``TreeNeurons``. Most of them are
simply short-hands for the other navis functions:

.. autosummary::
    :toctree: generated/

    ~navis.TreeNeuron.convert_units
    ~navis.TreeNeuron.cell_body_fiber
    ~navis.TreeNeuron.downsample
    ~navis.TreeNeuron.get_graph_nx
    ~navis.TreeNeuron.get_igraph
    ~navis.TreeNeuron.prune_by_longest_neurite
    ~navis.TreeNeuron.prune_by_strahler
    ~navis.TreeNeuron.prune_by_volume
    ~navis.TreeNeuron.prune_distal_to
    ~navis.TreeNeuron.prune_proximal_to
    ~navis.TreeNeuron.prune_twigs
    ~navis.TreeNeuron.reload
    ~navis.TreeNeuron.reroot
    ~navis.TreeNeuron.resample
    ~navis.TreeNeuron.snap

In addition ``TreeNeurons`` have a range of different properties:

.. autosummary::
    :toctree: generated/

    ~navis.TreeNeuron.cable_length
    ~navis.TreeNeuron.created_at
    ~navis.TreeNeuron.cycles
    ~navis.TreeNeuron.downsample
    ~navis.TreeNeuron.igraph
    ~navis.TreeNeuron.is_tree
    ~navis.TreeNeuron.n_branches
    ~navis.TreeNeuron.n_leafs
    ~navis.TreeNeuron.n_skeletons
    ~navis.TreeNeuron.n_trees
    ~navis.TreeNeuron.nodes
    ~navis.TreeNeuron.root
    ~navis.TreeNeuron.sampling_resolution
    ~navis.TreeNeuron.segments
    ~navis.TreeNeuron.simple
    ~navis.TreeNeuron.soma_pos
    ~navis.TreeNeuron.subtrees
    ~navis.TreeNeuron.volume


MeshNeurons
-----------
These are methods and properties specific to ``MeshNeurons``.

.. autosummary::
    :toctree: generated/

    ~navis.MeshNeuron.faces
    ~navis.MeshNeuron.vertices
    ~navis.MeshNeuron.skeletonize
    ~navis.MeshNeuron.snap
    ~navis.MeshNeuron.trimesh
    ~navis.MeshNeuron.volume
    ~navis.MeshNeuron.validate


VoxelNeurons
------------
VoxelNeurons (e.g. from confocal stacks) are a relatively new addition to
navis and the interface might still change.
These are methods and properties specific to ``VoxelNeurons``.

.. autosummary::
    :toctree: generated/

    ~navis.VoxelNeuron.grid
    ~navis.VoxelNeuron.voxels
    ~navis.VoxelNeuron.shape
    ~navis.VoxelNeuron.strip


Dotprops
--------
These are methods and properties specific to ``Dotprops``.

.. autosummary::
    :toctree: generated/

    ~navis.Dotprops.points
    ~navis.Dotprops.vect
    ~navis.Dotprops.alpha
    ~navis.Dotprops.to_skeleton
    ~navis.Dotprops.snap

Dotprops are typically indirectly generated from e.g. skeletons or
point clouds using :func:`navis.make_dotprops`.


Conversion
----------
There are a couple functions to convert from one neuron type to another:

.. autosummary::
    :toctree: generated/

    navis.make_dotprops
    navis.skeletonize
    navis.mesh
    navis.voxelize


NeuronList methods
------------------
``NeuronLists`` let you access all the properties and methods of the neurons
they contain. In addition there are a few ``NeuronList``-specific methods and
properties.

Methods:

.. autosummary::
    :toctree: generated/

    ~navis.NeuronList.apply
    ~navis.NeuronList.head
    ~navis.NeuronList.itertuples
    ~navis.NeuronList.mean
    ~navis.NeuronList.remove_duplicates
    ~navis.NeuronList.sum
    ~navis.NeuronList.summary
    ~navis.NeuronList.tail
    ~navis.NeuronList.unmix

Properties:

.. autosummary::
    :toctree: generated/

    ~navis.NeuronList.bbox
    ~navis.NeuronList.empty
    ~navis.NeuronList.id
    ~navis.NeuronList.idx
    ~navis.NeuronList.is_degenerated
    ~navis.NeuronList.is_mixed
    ~navis.NeuronList.shape
    ~navis.NeuronList.types

.. _api_plot:

Visualization
+++++++++++++
Various functions for plotting neurons and volumes.

.. autosummary::
    :toctree: generated/

    navis.plot3d
    navis.plot2d
    navis.plot1d
    navis.plot_flat
    navis.clear3d
    navis.close3d
    navis.pop3d
    navis.get_viewer
    navis.screenshot

Plotting Volumes/Meshes
-----------------------
To plot meshes, you can pass ``trimesh.Trimesh`` objects directly to ``plot3d``
or ``plot2d``. However, ``navis`` has a custom class to represent meshes that
has some useful perks: :class:`navis.Volume`.

.. autosummary::
    :toctree: generated/

    navis.Volume
    navis.Volume.combine
    navis.Volume.plot3d
    navis.Volume.validate
    navis.Volume.resize

Vispy 3D viewer
---------------
Using :func:`navis.plot3d` from a terminal will spawn a Vispy 3D viewer object
which has a bunch of useful methods.
Note that this requires one of navis' ``vispy-*`` extras to be installed,
so that vispy has a backend.

.. autosummary::
    :toctree: generated/

    navis.Viewer
    navis.Viewer.add
    navis.Viewer.clear
    navis.Viewer.close
    navis.Viewer.colorize
    navis.Viewer.set_colors
    navis.Viewer.hide_neurons
    navis.Viewer.unhide_neurons
    navis.Viewer.screenshot
    navis.Viewer.show
    navis.Viewer.toggle_bounds


.. _api_morph:

Neuron Morphology
+++++++++++++++++
Collection of functions to analyze and manipulate neuronal morphology.

Manipulation
------------
Functions to edit morphology:

.. autosummary::
    :toctree: generated/

    navis.average_skeletons
    navis.break_fragments
    navis.despike_skeleton
    navis.drop_fluff
    navis.cell_body_fiber
    navis.combine_neurons
    navis.cut_skeleton
    navis.guess_radius
    navis.heal_skeleton
    navis.longest_neurite
    navis.prune_by_strahler
    navis.prune_twigs
    navis.prune_at_depth
    navis.reroot_skeleton
    navis.split_axon_dendrite
    navis.split_into_fragments
    navis.stitch_skeletons
    navis.subset_neuron
    navis.smooth_skeleton
    navis.smooth_mesh
    navis.smooth_voxels


Resampling
----------
Functions to down- or resample neurons.

.. autosummary::
    :toctree: generated/

    navis.resample_skeleton
    navis.resample_along_axis
    navis.downsample_neuron
    navis.simplify_mesh

Morphometrics
-------------
Functions to analyze morphology.

.. autosummary::
    :toctree: generated/

    navis.find_main_branchpoint
    navis.form_factor
    navis.persistence_points
    navis.persistence_vectors
    navis.strahler_index
    navis.segment_analysis
    navis.sholl_analysis
    navis.tortuosity
    navis.betweeness_centrality

Functions to compare morphology.

.. autosummary::
    :toctree: generated/

    navis.nblast
    navis.nblast_smart
    navis.nblast_allbyall
    navis.nblast_align
    navis.vxnblast
    navis.synblast
    navis.persistence_distances

Utilities for creating your own score matrices for NBLAST can be found in

.. autosummary::
    :toctree: generated/

    navis.nbl.smat.Lookup2d
    navis.nbl.smat.Digitizer
    navis.nbl.smat.LookupDistDotBuilder

Utilities for NBLAST

.. autosummary::
    :toctree: generated/

    navis.nbl.make_clusters
    navis.nbl.update_scores
    navis.nbl.compress_scores
    navis.nbl.extract_matches
    navis.nbl.nblast_prime


Polarity metrics
----------------
.. autosummary::
    :toctree: generated/

    navis.bending_flow
    navis.flow_centrality
    navis.synapse_flow_centrality
    navis.arbor_segregation_index
    navis.segregation_index

Distances
---------
Functions to calculate Euclidian and geodesic ("along-the-arbor") distances.

.. autosummary::
    :toctree: generated/

    navis.cable_overlap
    navis.distal_to
    navis.dist_between
    navis.dist_to_root
    navis.geodesic_matrix
    navis.segment_length

Intersection
------------
Functions to intersect points and neurons with volumes.

.. autosummary::
    :toctree: generated/

    navis.in_volume
    navis.intersection_matrix

.. _transfm:

Transforming and Mirroring
++++++++++++++++++++++++++
Functions to transform spatial data e.g. between template brains.
Check out the :ref:`tutorials<example_gallery>` for examples on how to use them.

High-level functions:

.. autosummary::
    :toctree: generated/

    navis.xform
    navis.xform_brain
    navis.symmetrize_brain
    navis.mirror_brain
    navis.transforms.mirror
    navis.align.align_rigid
    navis.align.align_deform
    navis.align.align_pca
    navis.align.align_pairwise

``navis`` supports several types of transforms:

.. autosummary::
    :toctree: generated/

    ~navis.transforms.AffineTransform
    ~navis.transforms.ElastixTransform
    ~navis.transforms.CMTKtransform
    ~navis.transforms.H5transform
    ~navis.transforms.TPStransform
    ~navis.transforms.AliasTransform
    ~navis.transforms.MovingLeastSquaresTransform

The ``TemplateRegistry`` keeps track of template brains, transforms and such:

.. autosummary::
    :toctree: generated/

    ~navis.transforms.templates.TemplateRegistry

This relevant instance of this class is ``navis.transforms.registry``. So to
register a new transform you would for example do this::

  >>> navis.transforms.registry.register_transform(transform, ...)

These are the methods and properties of ``registry``:

.. autosummary::
    :toctree: generated/

    ~navis.transforms.templates.TemplateRegistry.register_transform
    ~navis.transforms.templates.TemplateRegistry.register_transformfile
    ~navis.transforms.templates.TemplateRegistry.register_templatebrain
    ~navis.transforms.templates.TemplateRegistry.register_path
    ~navis.transforms.templates.TemplateRegistry.scan_paths
    ~navis.transforms.templates.TemplateRegistry.plot_bridging_graph
    ~navis.transforms.templates.TemplateRegistry.find_mirror_reg
    ~navis.transforms.templates.TemplateRegistry.find_bridging_path
    ~navis.transforms.templates.TemplateRegistry.shortest_bridging_seq
    ~navis.transforms.templates.TemplateRegistry.clear_caches
    ~navis.transforms.templates.TemplateRegistry.summary
    ~navis.transforms.templates.TemplateRegistry.transforms
    ~navis.transforms.templates.TemplateRegistry.mirrors
    ~navis.transforms.templates.TemplateRegistry.bridges

.. _api_con:

Connectivity
++++++++++++
Collection of functions to work with graphs and adjacency matrices.

Graphs
------
Functions to convert neurons and networkx to iGraph or networkX graphs.

.. autosummary::
    :toctree: generated/

    navis.neuron2nx
    navis.neuron2igraph
    navis.neuron2KDTree
    navis.network2nx
    navis.network2igraph
    navis.rewire_skeleton
    navis.insert_nodes
    navis.remove_nodes

Connectivity metrics
--------------------
Functions to analyse/cluster neurons based on connectivity.

.. autosummary::
    :toctree: generated/

    navis.connectivity_similarity
    navis.connectivity_sparseness
    navis.cable_overlap
    navis.synapse_similarity

.. _api_io:

Import/Export
+++++++++++++
Functions to import/export neurons.

.. autosummary::
    :toctree: generated/

    navis.read_swc
    navis.write_swc
    navis.read_nrrd
    navis.write_nrrd
    navis.read_mesh
    navis.write_mesh
    navis.read_tiff
    navis.read_nmx
    navis.read_nml
    navis.read_rda
    navis.read_json
    navis.write_json
    navis.write_precomputed
    navis.read_precomputed
    navis.read_parquet
    navis.write_parquet
    navis.scan_parquet


.. _api_utility:

Utility
+++++++
Various utility functions.

.. autosummary::
    :toctree: generated/

    navis.health_check
    navis.set_pbars
    navis.set_loggers
    navis.set_default_connector_colors
    navis.config.remove_log_handlers
    navis.patch_cloudvolume
    navis.example_neurons
    navis.example_volume


.. _api_interfaces:

Interfaces
++++++++++
Interfaces with various external tools/websites. These modules have to be
imported explicitly as they are not imported at top level.

.. _api_interfaces.neuron:


Network Models
++++++++++++++
Navis comes with a simple network traversal model
(see `Schlegel, Bates et al., 2021 <https://elifesciences.org/articles/66018>`_).

.. autosummary::
   :toctree: generated/

   navis.models.network_models.TraversalModel
   navis.models.network_models.BayesianTraversalModel

NEURON simulator
++++++++++++++++
Functions to facilitate creating models of neurons/networks. Please see
the :ref:`tutorials<example_gallery>` for details.

Not imported at top level! Must be imported explicitly::

    import navis.interfaces.neuron as nrn


Compartment models
------------------
A single-neuron compartment model is represented by
:class:`~navis.interfaces.neuron.comp.CompartmentModel`:

.. autosummary::
   :toctree: generated/

   navis.interfaces.neuron.comp.CompartmentModel
   navis.interfaces.neuron.comp.DrosophilaPN

The :class:`~navis.interfaces.neuron.comp.DrosophilaPN` class is a subclass
of :class:`~navis.interfaces.neuron.comp.CompartmentModel` with
properties used from Tobin et al.

.. currentmodule:: navis.interfaces.neuron.comp

.. rubric:: Class methods

.. autosummary::
   :toctree: generated/

   ~CompartmentModel.add_current_record
   ~CompartmentModel.add_spike_detector
   ~CompartmentModel.add_synaptic_current
   ~CompartmentModel.add_synaptic_input
   ~CompartmentModel.add_voltage_record
   ~CompartmentModel.clear_records
   ~CompartmentModel.clear_stimuli
   ~CompartmentModel.connect
   ~CompartmentModel.get_node_section
   ~CompartmentModel.get_node_segment
   ~CompartmentModel.inject_current_pulse
   ~CompartmentModel.plot_results
   ~CompartmentModel.insert
   ~CompartmentModel.uninsert

.. rubric:: Class attributes

.. autosummary::

   ~CompartmentModel.Ra
   ~CompartmentModel.cm
   ~CompartmentModel.label
   ~CompartmentModel.n_records
   ~CompartmentModel.n_sections
   ~CompartmentModel.n_stimuli
   ~CompartmentModel.nodes
   ~CompartmentModel.records
   ~CompartmentModel.sections
   ~CompartmentModel.stimuli
   ~CompartmentModel.synapses
   ~CompartmentModel.ts

Network models
--------------
A network of point-processes is represented by
:class:`~navis.interfaces.neuron.network.PointNetwork`:

.. currentmodule:: navis.interfaces.neuron.network

.. autosummary::
   :toctree: generated/

   navis.interfaces.neuron.network.PointNetwork

.. rubric:: Class methods

.. autosummary::
   :toctree: generated/

   ~PointNetwork.__init__
   ~PointNetwork.add_background_noise
   ~PointNetwork.add_neurons
   ~PointNetwork.add_stimulus
   ~PointNetwork.connect
   ~PointNetwork.from_edge_list
   ~PointNetwork.get_spike_counts
   ~PointNetwork.plot_raster
   ~PointNetwork.plot_traces
   ~PointNetwork.run_simulation
   ~PointNetwork.set_labels

.. rubric:: Class attributes

.. currentmodule:: navis.interfaces.neuron.network

.. autosummary::
   :toctree: generated/

   ~PointNetwork.edges
   ~PointNetwork.ids
   ~PointNetwork.labels
   ~PointNetwork.neurons

.. currentmodule:: navis.interfaces
.. _api_interfaces.neuromorpho:

NeuroMorpho API
+++++++++++++++
Set of functions to grab data from `NeuroMorpho <http://neuromorpho.org>`_
which hosts thousands of neurons (see :ref:`tutorials<example_gallery>`).

Not imported at top level! Must be imported explicitly::

    from navis.interfaces import neuromorpho

.. autosummary::
    :toctree: generated/

    neuromorpho.get_neuron_info
    neuromorpho.get_neuron
    neuromorpho.get_neuron_fields
    neuromorpho.get_available_field_values


.. currentmodule:: navis.interfaces
.. _api_interfaces.neuprint:

neuPrint API
++++++++++++
NAVis wraps `neuprint-python <https://github.com/connectome-neuprint/neuprint-python>`_
and adds a few navis-specific functions. You must have `neuprint-python`
installed for this to work::

    pip install neuprint-python

You can then import neuprint from navis like so::

    from navis.interfaces import neuprint


These are the additional functions added by navis:

.. autosummary::
    :toctree: generated/

    neuprint.fetch_roi
    neuprint.fetch_skeletons
    neuprint.fetch_mesh_neuron

Please also check out the :ref:`tutorials<example_gallery>` for examples of how
to fetch and work with data from neuPrint.


.. currentmodule:: navis.interfaces
.. _api_interfaces.insectdb:

InsectBrain DB API
++++++++++++++++++
Set of functions to grab data from `InsectBrain <https://www.insectbraindb.org>`_
which hosts some neurons and standard brains (see :ref:`tutorials<example_gallery>`).

Not imported at top level! Must be imported explicitly::

    from navis.interfaces import insectbrain_db

.. autosummary::
    :toctree: generated/

    insectbrain_db.authenticate
    insectbrain_db.get_brain_meshes
    insectbrain_db.get_species_info
    insectbrain_db.get_available_species
    insectbrain_db.get_skeletons
    insectbrain_db.get_skeletons_species
    insectbrain_db.search_neurons


.. currentmodule:: navis.interfaces
.. _api_interfaces.blender:

Blender API
+++++++++++
Functions to be run inside `Blender 3D <https://www.blender.org/>`_ and import
CATMAID data (see Examples). Please note that this requires Blender >2.8 as
earlier versions are shipped with Python <3.6. See the
:ref:`tutorials<example_gallery>` for an introduction of how to use ``navis`` in
Blender.

Not imported at top level! Must be imported explicitly::

    from navis.interfaces import blender

The interface is realised through a :class:`~navis.interfaces.blender.Handler`
object. It is used to import objects and facilitate working with them
programmatically once they are imported.

.. autosummary::
    :toctree: generated/

    blender.Handler

Objects
-------
.. autosummary::
    :toctree: generated/

    blender.Handler.add
    blender.Handler.clear
    blender.Handler.select
    blender.Handler.hide
    blender.Handler.unhide

Materials
---------
.. autosummary::
    :toctree: generated/

    blender.Handler.color
    blender.Handler.colorize
    blender.Handler.emit
    blender.Handler.use_transparency
    blender.Handler.alpha
    blender.Handler.bevel

Selections
----------
.. autosummary::
    :toctree: generated/

    blender.Handler.select

    blender.ObjectList.select
    blender.ObjectList.color
    blender.ObjectList.colorize
    blender.ObjectList.emit
    blender.ObjectList.use_transparency
    blender.ObjectList.alpha
    blender.ObjectList.bevel
    blender.ObjectList.hide
    blender.ObjectList.unhide
    blender.ObjectList.hide_others
    blender.ObjectList.delete
    blender.ObjectList.to_json


.. currentmodule:: navis.interfaces
.. _api_interfaces.cytoscape:

Cytoscape API
+++++++++++++
Functions to use `Cytoscape <https://cytoscape.org/>`_ via the cyREST API.

Not imported at top level! Must be imported explicitly::

    from navis.interfaces import cytoscape

.. autosummary::
    :toctree: generated/

    cytoscape.generate_network
    cytoscape.get_client


.. currentmodule:: navis.interfaces
.. _api_interfaces.microns:

Allen MICrONS datasets
+++++++++++++++++++++++
Functions to fetch neurons (including synapses) from the Allen Institute's
`MICrONS <https://www.microns-explorer.org/>`_ EM datasets.

Requires ``caveclient`` and ``cloud-volume`` as additional dependencies::

    pip3 install caveclient cloud-volume -U

Please see ``caveclient's`` `docs <https://caveclient.readthedocs.io>`_ for
details on how to retrieve and set credentials.

Not imported at top level! Must be imported explicitly::

    from navis.interfaces import microns

.. autosummary::
    :toctree: generated/

    microns.fetch_neurons
    microns.get_somas

Please also see the :ref:`MICrONS tutorial<microns_tut>`.


.. currentmodule:: navis.interfaces
.. _api_interfaces.r:

R interface
+++++++++++
Bundle of functions to use R natverse libraries.

Not imported at top level! Must be imported explicitly::

    from navis.interfaces import r

.. autosummary::
    :toctree: generated/

    r.data2py
    r.get_neuropil
    r.init_rcatmaid
    r.load_rda
    r.nblast
    r.nblast_allbyall
    r.NBLASTresults
    r.neuron2py
    r.neuron2r
    r.xform_brain
    r.mirror_brain



.. _api_func_matrix:


Neuron types and functions
++++++++++++++++++++++++++

As you can imagine not all functions will work on all neuron types. For example
it is currently not possible to find the longest neurite
(:func:`navis.longest_neurite`) in a ``VoxelNeuron``. Conversely, some
functionality like "smoothing" makes sense for multiple neuron types but the
application is so vastly different between e.g. meshes and skeletons that
there is no single function but one for each neuron type.

Below table has an overview for which functions work with which neuron types.


.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 1

   * - Description
     - TreeNeuron
     - MeshNeuron
     - VoxelNeuron
     - Dotprops
   * - :func:`navis.plot2d`
     - yes
     - yes
     - limited
     - yes
   * - :func:`navis.plot3d`
     - yes
     - yes
     - limited
     - yes
   * - :func:`navis.plot1d`
     - yes
     - no
     - no
     - no
   * - :func:`navis.plot_flat`
     - yes
     - no
     - no
     - no
   * - :func:`navis.subset_neuron`
     - yes
     - yes
     - yes
     - yes
   * - :func:`navis.in_volume`
     - yes
     - yes
     - yes
     - yes
   * - smoothing
     - :func:`navis.smooth_skeleton`
     - :func:`navis.smooth_mesh`
     - :func:`navis.smooth_voxels`
     - no
   * - :func:`navis.downsample_neuron`
     - yes
     - yes
     - yes
     - yes
   * - resampling (e.g. :func:`navis.resample_skeleton`)
     - yes
     - no
     - no
     - no
   * - :func:`navis.make_dotprops`
     - yes
     - yes
     - yes
     - -
   * - NBLAST (e.g. :func:`navis.nblast`)
     - no
     - no
     - no
     - yes
   * - :func:`navis.xform_brain`
     - yes
     - yes
     - yes (slow!)
     - yes
   * - :func:`navis.mirror_brain`
     - yes
     - yes
     - no
     - yes
   * - :func:`navis.skeletonize`
     - no
     - yes
     - no
     - no
   * - :func:`navis.mesh`
     - yes
     - no
     - yes
     - no
   * - :func:`navis.voxelize`
     - yes
     - yes
     - no
     - yes
   * - :func:`navis.drop_fluff`
     - yes
     - yes
     - no
     - no
   * - :func:`navis.break_fragments`
     - yes
     - yes
     - no
     - no
