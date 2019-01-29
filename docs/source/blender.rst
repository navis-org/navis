.. _blender3d:

.. role:: red

Blender interface
*****************

Pymaid comes with an interface to import neurons into
`Blender 3D <https://www.blender.org>`_: :mod:`pymaid.b3d`

Installation
============

Blender comes with its own Python 3.5 distribution! So you need to install
pymaid explicitly for this distribution in order to use it within Blender.

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

3. Now you can use PIP to install pymaid (or any other package for that
   matter). Please note we have to - again - specify that we want to install
   for Blender's Python::

    /Applications/Blender/blender.app/Contents/Resources/2.79/python/bin/python3.5m -m pip install git+git://github.com/schlegelp/pymaid@master

4. You should now be all set to use pymaid in Blender. Check out Quickstart!

.. note::
   If any of the above steps fails with a *Permission* error, try the same
   command with a leading ``sudo``.

Quickstart
==========

:mod:`pymaid.b3d` provides a simple interface that lets you add, select and
manipulate neurons from within :red:`Blender's Python console`:

First, import and set up pymaid like you are used to.

>>> import pymaid
>>> rm = pymaid.CatmaidInstance('server_url', 'http_user', 'http_pw', 'token')
>>> # Fetch a bunch of neurons
>>> nl = pymaid.get_neuron('annotation: glomerulus DA1')

Now initialise the interface with Blender and import the neurons.

>>> # The b3d module is not automatically loaded when importing pymaid
>>> from pymaid import b3d
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

    pymaid.b3d.handler.add
    pymaid.b3d.handler.clear
    pymaid.b3d.handler.select
    pymaid.b3d.handler.hide
    pymaid.b3d.handler.unhide

Materials
---------
.. autosummary::
    :toctree: generated/

    pymaid.b3d.handler.color
    pymaid.b3d.handler.colorize
    pymaid.b3d.handler.emit
    pymaid.b3d.handler.use_transparency
    pymaid.b3d.handler.alpha
    pymaid.b3d.handler.bevel


Selections
----------
.. autosummary::
    :toctree: generated/

    pymaid.b3d.handler.select

    pymaid.b3d.object_list.select
    pymaid.b3d.object_list.color
    pymaid.b3d.object_list.colorize
    pymaid.b3d.object_list.emit
    pymaid.b3d.object_list.use_transparency
    pymaid.b3d.object_list.alpha
    pymaid.b3d.object_list.bevel

    pymaid.b3d.object_list.hide
    pymaid.b3d.object_list.unhide
    pymaid.b3d.object_list.hide_others

    pymaid.b3d.object_list.delete

    pymaid.b3d.object_list.to_json


