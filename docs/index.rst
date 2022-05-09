**NAVis** - **N**\ euron **A**\ nalysis and **Vis**\ ualization
===============================================================

.. raw:: html

   <div class="container-fluid">
      <div class="row">
         <div class="col-lg-6">

NAVis is a Python library for analysis and visualization of neuron
morphology. It stands on the shoulders of the excellent
`natverse <http://natverse.org>`_ for R.

For a brief introduction to the library, please see
:ref:`Quickstart <quickstart>`. Visit the :ref:`installation page<installing>`
to learn how to install the package. You can browse the
:ref:`example gallery<example_gallery>` and :ref:`API reference <api>` to see
what you can do with navis.

NAVis is designed to be highly extensible. Make sure to check out the other
libraries in the navis :ref:`ecosystem <other_libs>`.

NAVis is licensed under the GNU GPL v3+ license. The source code is hosted
at `Github <https://github.com/navis-org/navis>`_. Feedback, feature requests,
bug reports and general questions are very welcome and best placed in a
`Github issue <https://github.com/navis-org/navis/issues>`_.

.. raw:: html

         </div>
         <div class="col-lg-6">
            <div class="panel panel-default">
               <div class="panel-heading">
                  <h3 class="panel-title">Features</h3>
               </div>
               <div class="panel-body">

* works in Jupyter notebooks, from terminal or as script
* supports various neuron types: skeletons, meshes, dotprops, voxels
* process neurons: smoothing, resampling, skeletonization, meshing, ...
* virtual neuron surgery: cutting, stitching, pruning, rerooting, ...
* morphometrics: Strahler analysis, cable length, volume, tortuosity, ...
* cluster by morphology (e.g. NBLAST, persistence, form factor) or connectivity
* 2D (matplotlib) and 3D (vispy & plotly) plotting
* transform neurons between template brains
* Python bindings for R natverse library
* load neurons directly from the Allen's `MICrONS <https://www.microns-explorer.org/>`_ datasets, `neuromorpho <http://neuromorpho.org>`_ or :ref:`neuPrint<neuprint_intro>`
* simulate neurons and networks using `NEURON`
* interface with :ref:`Blender 3D<blender3d>`
* import-export from/to SWC, NRRD, Neuroglancer's precomputed format and more
* scalable thanks to out-of-the-box support for multiprocessing
* highly extensible - see the :ref:`ecosystem <other_libs>`

.. raw:: html

               </div>
            </div>
         </div>
      </div>
   </div>
