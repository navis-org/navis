**NAVis** - **N**\ euron **A**\ nalysis and **Vis**\ ualization
===============================================================

.. grid:: 1 1 1 2

    .. grid-item::
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

    .. grid-item::

        .. card-carousel:: 2

            .. card:: :octicon:`code-square` Polyglot

                Work with all kinds of data: skeletons, meshes, dotprops, images.

            .. card:: :octicon:`beaker` Morphometrics

                Calculate Strahler indices, cable length, volume, tortuosity and more.

            .. card:: :octicon:`paintbrush` Plotting

                Generate beautiful scientific 2D (matplotlib) and 3D (vispy or
                plotly) figures.

            .. card:: :octicon:`gear` Processing

                Smoothing, resampling, skeletonization, meshing and more!

            .. card:: :octicon:`rocket` Fast

                Scalable thanks to out-of-the-box support for multiprocessing.

            .. card:: :octicon:`versions` NBLAST

                Cluster your neurons by morphology.

        .. card-carousel:: 2

            .. card:: :octicon:`paper-airplane` Transform

                Fully featured transform system to move neurons between brain spaces.

            .. card:: :octicon:`file-binary` Import/Export

                Read and write from/to SWC, NRRD, Neuroglancer's precomputed format,
                OBJ, STL and more!

            .. card:: :octicon:`globe` Online

                Download neurons straight from Allen's
                `MICrONS <https://www.microns-explorer.org/>`_ datasets,
                `neuromorpho <http://neuromorpho.org>`_ or :ref:`neuPrint<neuprint_intro>`.

            .. card:: :octicon:`link-external` Interfaces

                Load neurons into Blender 3D, simulate neurons and networks using
                NEURON, or use the R natverse library.

            .. card:: :octicon:`person` Have it your way

                Designed to work in Jupyter notebooks, from terminal or as a script.

            .. card:: :octicon:`person` Extensible
                :link: other_libs
                :link-type: ref

                Write your own library built on top of navis functions.
