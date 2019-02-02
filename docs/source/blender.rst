.. _blender3d:

.. role:: red

Blender interface
*****************

Navis comes with an interface to import neurons into
`Blender 3D <https://www.blender.org>`_: :mod:`navis.b3d`

Installation
============

Blender comes with its own Python 3.5 distribution! So you need to install
navis explicitly for this distribution in order to use it within Blender.

There are several ways to install additional packages for Blender's
built-in Python. The easiest way IMHO is this:

1. Find out where Blender's Python lives (this depends on your OS). In
   :red:`Blender's Python console` run this::

    >>> bpy.app.binary_path_python
    '/Applications/Blender/blender.app/Contents/Resources/2.79/python/bin/python3.5m'

2. Download PIP's ``get-pip.py`` from `here <https://pip.pypa.io/en/stable/installing/>`_
   and save to your Downloads directory. Then open a :red:`terminal`, navigate
   to `Downloads` and run ``get-pip.py`` using your Blender's Python::

    cd Downloads
    /Applications/Blender/blender.app/Contents/Resources/2.79/python/bin/python3.5m get-pip.py

3. Now you can use PIP to install navis (or any other package for that
   matter). Please note we have to - again - specify that we want to install
   for Blender's Python::

    /Applications/Blender/blender.app/Contents/Resources/2.79/python/bin/python3.5m -m pip install git+git://github.com/schlegelp/navis@master

4. You should now be all set to use navis in Blender. Check out Quickstart!

.. note::
   If any of the above steps fails with a *Permission* error, try the same
   command with a leading ``sudo``.

Quickstart
==========

:mod:`navis.b3d` provides a simple interface that lets you add, select and
manipulate neurons from within :red:`Blender's Python console`:

First, import and set up navis like you are used to.

>>> import navis
>>> rm = navis.CatmaidInstance('server_url', 'http_user', 'http_pw', 'token')
>>> # Fetch a bunch of neurons
>>> nl = navis.get_neuron('annotation: glomerulus DA1')

Now initialise the interface with Blender and import the neurons.

>>> # The b3d module is not automatically loaded when importing navis
>>> from navis import b3d
>>> # Initialise handler
>>> handler = b3d.handler()
>>> # Load neurons into scene
>>> handler.add(nl)

The interface lets you manipulate neurons in Blender too.

>>> # Colorise neurons
>>> handler.colorize()
>>> # Change thickness of all neurons
>>> handler.neurons.bevel(.02)
>>> # Select subset
>>> subset = handle.select(nl[:10])
>>> # Make subset red
>>> subset.color(1, 0, 0)
>>> # Change color of presynapses to green
>>> handle.presynapses.color(0, 1, 0)
>>> # Show only connectors
>>> handle.connectors.hide_others()
>>> # Clear all objects
>>> handler.clear()

.. note::
   Blender's Python console does not show all outputs. Please check the terminal
   if you experience issues. In Windows simply go to `Help` >> `Toggle System
   Console`. In MacOS, right-click Blender in Finder >> `Show Package Contents`
   >> `MacOS` >> double click on `blender`.

Reference
=========

Objects
-------
.. autosummary::
    :toctree: generated/

    navis.b3d.handler.add
    navis.b3d.handler.clear
    navis.b3d.handler.select
    navis.b3d.handler.hide
    navis.b3d.handler.unhide

Materials
---------
.. autosummary::
    :toctree: generated/

    navis.b3d.handler.color
    navis.b3d.handler.colorize
    navis.b3d.handler.emit
    navis.b3d.handler.use_transparency
    navis.b3d.handler.alpha
    navis.b3d.handler.bevel


Selections
----------
.. autosummary::
    :toctree: generated/

    navis.b3d.handler.select

    navis.b3d.object_list.select
    navis.b3d.object_list.color
    navis.b3d.object_list.colorize
    navis.b3d.object_list.emit
    navis.b3d.object_list.use_transparency
    navis.b3d.object_list.alpha
    navis.b3d.object_list.bevel

    navis.b3d.object_list.hide
    navis.b3d.object_list.unhide
    navis.b3d.object_list.hide_others

    navis.b3d.object_list.delete

    navis.b3d.object_list.to_json


