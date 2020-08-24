.. _whats_new:

What's new?
===========

.. list-table::
   :widths: 7 7 86
   :header-rows: 1

   * - Version
     - Date
     -
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
       - improved :func:`navis.stitch_neurons`: much faster now if you have iGraph
       - fixed errors when using multiprocessing (e.g. in ``NeuronList.apply``)
       - fixed bugs in :func:`navis.downsample_neuron`
   * - 0.1.10
     - 24/02/20
     - - fixed bugs in Blender interface introduced in 0.1.9
   * - 0.1.9
     - 24/02/20
     - - removed hard-coded swapping and translation of axes in the Blender interface
       - fixed bugs in stitch_neurons
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
