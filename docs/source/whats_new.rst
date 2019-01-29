.. _whats_new:

What's new?
===========

.. list-table::
   :widths: 7 7 86
   :header-rows: 1

   * - Version
     - Date
     -
   * - 0.92
     - 06/11/18
     - - new pymaid.Volume methods: ``to_csv`` and ``from_csv``
       - new functions: :func:`~pymaid.add_meta_annotations`, :func:`~pymaid.remove_meta_annotations`, :func:`~pymaid.get_annotated`
       - some under-the-hood changes following change in CATMAID's API
       - general bug fixes and improvements
   * - 0.91
     - 31/10/18
     - - new CatmaidInstance attributes to get info on your server: ``catmaid_version``, ``available_projects`` and ``image_stacks``
       - new functions: :func:`~pymaid.shorten_name`, :func:`~pymaid.get_user_stats`, :func:`~pymaid.intersection_matrix`, :func:`~pymaid.get_node_location`
       - various improvements and bugfixes
   * - 0.90
     - 20/09/18
     - - vispy 3d viewer overhaul: prettier, better picking, new shortcuts
       - indexing of :class:`~pymaid.CatmaidNeuronList` via ``.skid[]`` now returns results in order of query
       - new function: :func:`~pymaid.get_treenodes_by_tag`
       - new function: :func:`~pymaid.connection_density`
       - improved :func:`~pymaid.split_axon_dendrite`
       - improved :func:`~pymaid.to_swc` and :func:`~pymaid.from_swc`
       - improved :ref:`neuronlist math and comparisons <neuronlist_math>`
       - :func:`~pymaid.plot2d` and :func:`~pymaid.plot3d` now accept lists of colors
       - :func:`~pymaid.has_soma` is now much faster
       - faster neuron import in :ref:`blender_3d`
       - improved docstrings
       - various bugfixes
   * - 0.89
     - 14/08/18
     - - new function: :func:`~pymaid.cytoscape.watch_network` constantly pushes updates Cytoscape
       - new function: :func:`~pymaid.get_nth_partners` returns neurons connected via n hops
       - by default, :func:`~pymaid.plot3d` now chooses the backend automatically: vispy for terminal sessions, plotly for Jupyter notebook/lab
       - :func:`~pymaid.get_skids_by_annotation` now accepts negative search criteria
       - :func:`~pymaid.from_swc` now imports multiple SWCs at a time
       - major improvements to caching system
       - by default, progress bars will now vanish after completion
       - followed changes in CATMAID API regarding treenode tables
       - various bugfixes
   * - 0.88
     - 29/07/18
     - - data caching for faster queries, see :doc:`caching demo <data_caching>`
       - new function: :func:`~pymaid.smooth_neuron`
       - :func:`~pymaid.resample_neuron` now resamples radius too
       - :func:`~pymaid.guess_radius` interpolation now takes distance along spines into account
       - :func:`~pymaid.despike_neuron` is now able to catch spikes that consist of multiple nodes
       - :func:`~pymaid.calc_cable` is now deprecated
       - general improvements to docstrings
   * - 0.87
     - 20/07/18
     - - :func:`~pymaid.get_team_contributions` now takes link creation into account
       - :func:`~pymaid.get_time_invested` should be way faster now
       - :func:`~pymaid.geodesic_matrix` now returns a SparseDataFrame to save memory
       - added :func:`pymaid.CatmaidNeuron.to_dataframe` method
       - general improvements and docstrings
   * - 0.86
     - 16/07/18
     - - arithmetric operations with CatmaidNeuron/Lists will now warn if skeleton IDs match but neuron objects are not identical. See :doc:`here <neuronlist_math>` for explanation.
       - fixed a bug when using regex to query for neurons that led to duplicate skeleton IDs being returned
   * - 0.85
     - 13/07/18
     - - fixed a series of critical bugs in :func:`~pymaid.plot3d`, :func:`pymaid.Volume.combine`, :func:`~pymaid.cut_neuron`, :func:`pymaid.CatmaidNeuronList.remove_duplicates`,  :func:`~pymaid.get_skid_from_treenode` and :func:`~pymaid.neuron2json`
       - :func:`~pymaid.cut_neuron` now accepts multiple cut nodes
       - improved depth coloring in :func:`~pymaid.plot2d`
       - added depth coloring to :func:`~pymaid.plot2d` with method '3d' - see :doc:`here <depth_coloring>` for examples
