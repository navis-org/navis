.. _api:

API Reference
=============

``navis`` has grown a lot! Last I looked, there were ~110 functions exposed
at top level (e.g. ``navis.plot3d``) and easily another 100 secondary functions
available via submodules (e.g. ``navis.morpho.find_soma``). This can be a bit
daunting at first - especially if you don't exactly know what you are looking
for. I recommend you either just have a browse, use the search field
(upper right) or simply search in page (CONTROL/CMD-F). Failing that, please
feel free to open an `issue <https://github.com/schlegelp/navis/issues>`_ on
the Github repo with your question.

This API reference is a more or less complete account of the primary functions:

1. :ref:`Neuron- and NeuronList functions and methods <api_neurons>`
2. :ref:`Functions for visualization<api_plot>`
3. :ref:`Manipulate or analyze neuron morphology<api_morph>`
4. :ref:`Transforming and mirroring data<transfm>`
5. :ref:`Analyze connectivity<api_con>`
6. :ref:`Import/Export<io>`
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
which has a bunch of useful methods:

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
    navis.cut_neuron
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
    navis.tortuosity

Resampling
----------
Functions to down- or resample neurons.

.. autosummary::
    :toctree: generated/

    navis.resample_skeleton
    navis.resample_along_axis
    navis.downsample_neuron
    navis.simplify_mesh

Analysis
--------
Functions to analyze morphology.

.. autosummary::
    :toctree: generated/

    navis.find_main_branchpoint
    navis.strahler_index
    navis.nblast
    navis.nblast_smart
    navis.nblast_allbyall
    navis.synblast

Polarity metrics
----------------
.. autosummary::
    :toctree: generated/

    navis.bending_flow
    navis.flow_centrality
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
Functions to transform spatial data between (e.g. neurons) template brains.
Check out the :ref:`tutorials<example_gallery>` for example on how to use them.

High-level functions:

.. autosummary::
    :toctree: generated/

    navis.xform
    navis.xform_brain
    navis.symmetrize_brain
    navis.mirror_brain
    navis.transforms.mirror

``navis`` supports several types of transforms:

.. autosummary::
    :toctree: generated/

    ~navis.transforms.AffineTransform
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
    navis.rewire_neuron
    navis.insert_nodes
    navis.remove_nodes

Adjacency matrices
------------------
Functions to work with adjacency matrices.

.. autosummary::
    :toctree: generated/

    navis.group_matrix

Connectivity clustering
-----------------------
Functions to cluster neurons based on connectivity.

.. autosummary::
    :toctree: generated/

    navis.cluster_by_connectivity
    navis.cluster_by_synapse_placement

.. _io:

Import/Export
+++++++++++++
Functions to import/export neurons.

.. autosummary::
    :toctree: generated/

    navis.read_swc
    navis.write_swc
    navis.read_nrrd
    navis.write_nrrd
    navis.read_nmx
    navis.read_rda
    navis.read_json
    navis.write_json
    navis.write_precomputed
    navis.read_precomputed

.. _api_interfaces:

Interfaces
++++++++++
Interfaces with various external tools/websites. These modules have to be
imported explicitly as they are not imported at top level.

.. _api_interfaces.neuron:

NEURON simulator
++++++++++++++++
Functions to facilitate creating compartment models of neurons. Please see
the :ref:`tutorials<example_gallery>` for details.

Not imported at top level! Must be imported explicitly::

    import navis.interfaces.neuron as nrn

.. autosummary::
    :toctree: generated/

    navis.interfaces.neuron.cmp.CompartmentModel
    navis.interfaces.neuron.cmp.DrosophilaPN
    navis.interfaces.neuron.network.PointNetwork


.. _api_interfaces.neuromorpho:

NeuroMorpho API
+++++++++++++++
Set of functions to grab data from `NeuroMorpho <http://neuromorpho.org>`_
which hosts thousands of neurons (see :ref:`tutorials<example_gallery>`).

Not imported at top level! Must be imported explicitly::

    from navis.interfaces import neuromorpho

.. autosummary::
    :toctree: generated/

    navis.interfaces.neuromorpho.get_neuron_info
    navis.interfaces.neuromorpho.get_neuron
    navis.interfaces.neuromorpho.get_neuron_fields
    navis.interfaces.neuromorpho.get_available_field_values


.. _api_interfaces.neuprint:

neuPrint API
++++++++++++
NAVis wraps `neuprint-python <https://github.com/connectome-neuprint/neuprint-python>`_
and adds a few navis-specific functions. You must have `neuprint-python`
installed for this to work::

    pip install neuprint-python

You can then import neuprint from navis like so::

    import navis.interfaces.neuprint as neu

.. autosummary::
    :toctree: generated/

    navis.interfaces.neuprint.fetch_roi
    navis.interfaces.neuprint.fetch_skeletons
    navis.interfaces.neuprint.fetch_mesh_neuron

Please also check out the :ref:`tutorials<example_gallery>` for examples of how
to fetch and work with data from neuPrint.

.. _api_interfaces.insectdb:

InsectBrain DB API
++++++++++++++++++
Set of functions to grab data from `InsectBrain <https://www.insectbraindb.org>`_
which hosts some neurons and standard brains (see :ref:`tutorials<example_gallery>`).

Not imported at top level! Must be imported explicitly::

    from navis.interfaces import insectbrain_db

.. autosummary::
    :toctree: generated/

    navis.interfaces.insectbrain_db.authenticate
    navis.interfaces.insectbrain_db.get_brain_meshes
    navis.interfaces.insectbrain_db.get_species_info
    navis.interfaces.insectbrain_db.get_available_species
    navis.interfaces.insectbrain_db.get_skeletons
    navis.interfaces.insectbrain_db.get_skeletons_species
    navis.interfaces.insectbrain_db.search_neurons


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

    navis.interfaces.blender.Handler

Objects
-------
.. autosummary::
    :toctree: generated/

    navis.interfaces.blender.Handler.add
    navis.interfaces.blender.Handler.clear
    navis.interfaces.blender.Handler.select
    navis.interfaces.blender.Handler.hide
    navis.interfaces.blender.Handler.unhide

Materials
---------
.. autosummary::
    :toctree: generated/

    navis.interfaces.blender.Handler.color
    navis.interfaces.blender.Handler.colorize
    navis.interfaces.blender.Handler.emit
    navis.interfaces.blender.Handler.use_transparency
    navis.interfaces.blender.Handler.alpha
    navis.interfaces.blender.Handler.bevel

Selections
----------
.. autosummary::
    :toctree: generated/

    navis.interfaces.blender.Handler.select

    navis.interfaces.blender.ObjectList.select
    navis.interfaces.blender.ObjectList.color
    navis.interfaces.blender.ObjectList.colorize
    navis.interfaces.blender.ObjectList.emit
    navis.interfaces.blender.ObjectList.use_transparency
    navis.interfaces.blender.ObjectList.alpha
    navis.interfaces.blender.ObjectList.bevel
    navis.interfaces.blender.ObjectList.hide
    navis.interfaces.blender.ObjectList.unhide
    navis.interfaces.blender.ObjectList.hide_others
    navis.interfaces.blender.ObjectList.delete
    navis.interfaces.blender.ObjectList.to_json


.. _api_interfaces.cytoscape:

Cytoscape API
+++++++++++++
Functions to use `Cytoscape <https://cytoscape.org/>`_ via the cyREST API.

Not imported at top level! Must be imported explicitly::

    from navis.interfaces import cytoscape

.. autosummary::
    :toctree: generated/

    navis.interfaces.cytoscape.generate_network
    navis.interfaces.cytoscape.get_client


.. _api_interfaces.microns:

Allen MICrONS datasets
+++++++++++++++++++++++
Functions to fetch neurons (including synapses) from the Allen Institute's
`MICrONS <https://www.microns-explorer.org/>`_ EM datasets.

Requires ``caveclient`` as additional dependencies::

    pip3 install caveclient -U

Please see ``caveclient's`` `docs <https://caveclient.readthedocs.io>`_ for
details on how to retrieve and set credentials.

Not imported at top level! Must be imported explicitly::

    from navis.interfaces import microns

.. autosummary::
    :toctree: generated/

    navis.interfaces.microns.fetch_neurons

.. _api_interfaces.r:

R interface
+++++++++++
Bundle of functions to use R natverse libraries.

Not imported at top level! Must be imported explicitly::

    from navis.interfaces import r

.. autosummary::
    :toctree: generated/

    navis.interfaces.r.data2py
    navis.interfaces.r.get_neuropil
    navis.interfaces.r.init_rcatmaid
    navis.interfaces.r.load_rda
    navis.interfaces.r.nblast
    navis.interfaces.r.nblast_allbyall
    navis.interfaces.r.NBLASTresults
    navis.interfaces.r.neuron2py
    navis.interfaces.r.neuron2r
    navis.interfaces.r.xform_brain
    navis.interfaces.r.mirror_brain

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
   * - NBLAST (e.g. :func:`navis.nblast`)
     - no
     - no
     - no
     - yes
   * - :func:`navis.xform_brain`
     - yes
     - yes
     - yes
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
