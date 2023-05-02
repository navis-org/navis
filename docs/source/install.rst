.. _installing:

Install
=======

NAVis currently requires Python 3.8 or later. Below instructions assume that
you have already installed Python and its package manager ``pip``.

.. topic:: By the way

   You can use NAVis without having to install anything on your local machine!
   Follow this `link <https://colab.research.google.com/github/navis-org/navis/blob/master/examples/colab.ipynb>`_
   to open an example notebook in Google's Colaboratory.


From pip, with "batteries included"
-----------------------------------

Install navis including optional dependencies plus a GUI backend for the
terminal (see bottom of the page for details):

::

   pip3 install navis[all] -U


From pip, as minimal install
----------------------------

On certain systems/setups (e.g. on the new M1 Macs with ARM CPU architecture)
you might run into issues during the full install. In that case
you can try installing a "minimal" version by dropping the ``[all]``:

::

   pip3 install navis -U

Please see below for a list of the dropped optional dependencies in case you
want to try installing them manually!


Development branch from Github
------------------------------

To install the most recent development branch from Github:

::

    pip3 install git+https://github.com/navis-org/navis@master


Optional Dependencies
---------------------

If you installed navis using ``pip``, mandatory dependencies will have been
installed automatically. Unless you used the "batteries included" ``[all]``
option, there are a few optional dependencies that e.g. provide
speed-boosts in certain situations or are required only for certain functions.

These extras can be installed directly, or along with navis with

::

   pip3 install navis[extra1,extra2]


The user-facing extras, the dependencies they install,
and how to install those dependencies directly, are below.


.. _pykd:

``kdtree``: `pykdtree <https://github.com/storpipfugl/pykdtree>`_
  Faster than scipy's cKDTree implementation. If available, will be used to
  speed up e.g. NBLAST. **Important**: on Linux I found that I need to set
  a ``OMP_NUM_THREADS=4`` environment variable (see also ``pykdtree`` docs).
  Otherwise it's actually slower than scipy's KDTree.

  ::

    pip3 install pykdtree

.. _pathos:

``pathos``: `pathos <https://github.com/uqfoundation/pathos>`_
  Pathos is a multiprocessing library. NAVis uses it to parallelize functions
  across lists of neurons.

  ::

    pip3 install pathos

.. _pyoc:

``octree``: `PyOctree <https://pypi.python.org/pypi/pyoctree/>`_
  Slower alternative to ncollpyde.

  ::

    pip3 install pyoctree

.. _rpy:

``r``: `Rpy2 <https://rpy2.readthedocs.io/en/version_2.8.x/overview.html#installation>`_ (``rpy2``)
  Provides interface with R. This allows you to use e.g. the
  `natverse <https://natverse.org>`_  R packages. Note that
  this package is not installed automatically as it would fail
  if R is not already installed on the system. You have to
  install Rpy2 manually!

  ::

    pip3 install rpy2

.. _shapely:

``shapely``: `Shapely <https://shapely.readthedocs.io/en/latest/>`_ (``shapely``)
  This is used to get 2D outlines of navis.Volumes.

  ::

    pip3 install shapely

.. _vispy:

``vispy-*`` backends: `vispy <https://vispy.org>`_
  For 3D visualisation.

  Vispy provides a high-performance viewer for 3D visualisation of neurons.
  Vispy itself has a choice of backends: the one which works for you will depend on
  your operating system, hardware, other installed packages, and how you're using navis.
  The default, supplied with navis' ``vispy-default`` extra, is pyside6;
  this works best when called from an ``ipython`` console.
  Each of vispy's backends, listed
  `here <https://vispy.org/installation.html#backend-requirements>`_,
  can be installed through vispy and its extras, or navis' `vispy-*` extras.

  ::

    pip3 install navis[vispy-pyqt5]
    # or
    pip3 install vispy[pyqt5]

.. _hash:

``hash``: `xxhash <https://cyan4973.github.io/xxHash/>`_
  For speeding up some lookup tables.

.. _flybrains:

``flybrains``: `flybrains <https://github.com/navis-org/navis-flybrains>`_
  Transforming data between some template *Drosophila* brains.

.. _cloudvolume:

``cloudvolume``: `cloud-volume <https://github.com/seung-lab/cloud-volume>`_
  Reading and writing images, meshes, and skeletons in Neuroglancer precomputed format.
  This is required required for e.g. the MICrONs interface.

.. _meshes:

``meshes``: `open3d <https://pypi.org/project/open3d/>`_, `pyfqmr <https://github.com/Kramer84/pyfqmr-Fast-quadric-Mesh-Reduction>`_
  Assorted functionality associated with meshes. ``pyfqmr`` in particular is
  highly recommended if you want to downsample meshes.
