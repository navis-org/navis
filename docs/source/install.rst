.. _installing:

Install
=======

NAVis currently requires Python 3.9 or later. Below instructions assume that
you have already installed Python and its package manager ``pip``.

.. topic:: By the way

   You can use NAVis without having to install anything on your local machine!
   Follow this `link <https://colab.research.google.com/github/navis-org/navis/blob/master/examples/colab.ipynb>`_
   to open an example notebook in Google's Colaboratory.


From pip, with "batteries included"
-----------------------------------

Install navis including optional dependencies plus a GUI backend for plotting (see bottom of the page for details):

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

Performance
~~~~~~~~~~~

Dependencies that aren't strictly necessary but will speed up certain operations:

.. _fastcore:

``fastcore``: `navis-fastcore <https://github.com/schlegelp/fastcore-rs>`_
  ``navis-fastcore`` re-implements a bunch of low-level functions in Rust
  and wraps them in Python. ``navis`` will use ``fastcore`` under-the-hood
  if it is available. This is a highly recommended extra as it can
  speed up operations such as geodesic distances, Strahler Index, pruning
  and other downstream functions by several orders of magnitude.

  ::

    pip3 install navis-fastcore


.. _pykd:

``kdtree``: `pykdtree <https://github.com/storpipfugl/pykdtree>`_
  Faster than scipy's cKDTree implementation. If available, will be used to
  speed up e.g. NBLAST.

  ::

    pip3 install pykdtree

.. _pathos:

``pathos``: `pathos <https://github.com/uqfoundation/pathos>`_
  Pathos is a multiprocessing library. NAVis uses it to parallelize functions
  across lists of neurons.

  ::

    pip3 install pathos

.. _hash:

  ``hash``: `xxhash <https://cyan4973.github.io/xxHash/>`_
  For speeding up some lookup tables.

    ::

      pip3 install xxhash

.. _meshes:

``meshes``: `open3d <https://pypi.org/project/open3d/>`_, `pyfqmr <https://github.com/Kramer84/pyfqmr-Fast-quadric-Mesh-Reduction>`_
  Assorted functionality associated with meshes. ``pyfqmr`` in particular is
  highly recommended if you want to downsample meshes.

  ::

    pip3 install open3d pyfqmr

.. _viz_install:

Visualization
~~~~~~~~~~~~~

``navis`` supports various different backends for 2D and 3D visualization. For 2D visualizations we
use ``matplotlib`` by default which is installed automatically. For 3D visualizations, you can use
``octarine3d``, ``vispy``, ``plotly`` or ``k3d`` backends.

.. _octarine:

``octarine3d``: `octarine3d <https://schlegelp.github.io/octarine/>`_ (``octarine3d``)
  For 3D visualisation in terminal and Jupyter notebooks.

  Octarine3d is a modern, high-performance, WGPU-based viewer for 3D visualisation of neurons.
  It is the default 3D viewer for navis. It is recommended to install it as it is the fastest, most
  feature-rich viewer available. By default, ``navis[all]`` will install ``octarine3d`` with
  standard windows manager ``pyside6`` and Jupyter notebook manager ``jupyter_rfb``. It will also
  install the ``navis-octarine-plugin`` which is required to use octarine3d as a viewer for navis.
  This is equivalent to the following command:

  ::

    pip3 install octarine3d[all] octarine-navis-plugin

  Please see ``octarine3`` `installataion instructions <https://schlegelp.github.io/octarine/install/>`_
  for information on how to choose a different backend.

  Note: older systems (pre ~2018) might not support WGPU, in which case you might want to fall back
  to the ``vispy`` backend.

.. _vispy:

``vispy-*`` backends: `vispy <https://vispy.org>`_
  For 3D visualisation in terminal and Jupyter notebooks.

  Vispy provides a high-performance, OpenGL-based viewer for 3D data visualisation.
  Vispy itself has a choice of window managers: the one which works for you will depend on
  your operating system, hardware, other installed packages, and how you're using navis.
  The default, supplied with navis' ``vispy-default`` extra, is ``pyside6`` (for use from the console)
  and ``jupyter_rfb`` (for use in Jupyter notebooks).
  Each of vispy's backends, listed
  `here <https://vispy.org/installation.html#backend-requirements>`_,
  can be installed through vispy and its extras, or navis' `vispy-*` extras.

  ::

    pip3 install navis[vispy-pyqt5]
    # or
    pip3 install vispy[pyqt5]


.. _plotly:

``plotly``: `plotly <https://plotly.com/python/>`_
  For 3D visualisation in Jupyter notebooks.

  ::

    pip3 install plotly

.. _k3d:

``k3d``: `k3d <https://k3d-jupyter.org/>`_
  For 3D visualisation in Jupyter notebooks.

  ::

    pip3 install k3d


Miscellaneous
~~~~~~~~~~~~~

.. _rpy:

``r``: `Rpy2 <https://rpy2.readthedocs.io/en/version_2.8.x/overview.html#installation>`_ (``rpy2``)
  Provides interface with R. This allows you to use e.g. the
  `natverse <https://natverse.org>`_ R packages. Note that
  this package is not installed automatically as it would fail
  if R is not already installed on the system. You have to
  install Rpy2 manually!

  ::

    pip3 install rpy2

.. _shapely:

``shapely``: `Shapely <https://shapely.readthedocs.io/en/latest/>`_ (``shapely``)
  This is used to get 2D outlines of :class:`navis.Volumes` when plotting in 2D
  with ``volume_outlines=True``.

  ::

    pip3 install shapely

.. _flybrains:

``flybrains``: `flybrains <https://github.com/navis-org/navis-flybrains>`_
  Transforming data between some template *Drosophila* brains.

.. _cloudvolume:

``cloudvolume``: `cloud-volume <https://github.com/seung-lab/cloud-volume>`_
  Reading and writing images, meshes, and skeletons in Neuroglancer precomputed format.
  This is required required for e.g. the MICrONs interface.

