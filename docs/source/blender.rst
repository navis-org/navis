.. _blender3d:

.. role:: red

Blender interface
*****************

Navis comes with an interface to import neurons into
`Blender 3D <https://www.blender.org>`_: :mod:`navis.interfaces.blender`

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

:mod:`navis.interfaces.blender` provides a simple interface that lets you add, select and
manipulate neurons from within :red:`Blender's Python console`:

First, import and set up navis like you are used to.

>>> import navis
>>> rm = navis.CatmaidInstance('server_url', 'http_user', 'http_pw', 'token')
>>> # Fetch a bunch of neurons
>>> nl = navis.get_neuron('annotation: glomerulus DA1')

Now initialise the interface with Blender and import the neurons.

>>> # The interfaces.blender module is not automatically loaded when importing navis
>>> from navis import interfaces.blender
>>> # Initialise handler
>>> handler = interfaces.blender.handler()
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

    navis.interfaces.blender.handler.add
    navis.interfaces.blender.handler.clear
    navis.interfaces.blender.handler.select
    navis.interfaces.blender.handler.hide
    navis.interfaces.blender.handler.unhide

Materials
---------
.. autosummary::
    :toctree: generated/

    navis.interfaces.blender.handler.color
    navis.interfaces.blender.handler.colorize
    navis.interfaces.blender.handler.emit
    navis.interfaces.blender.handler.use_transparency
    navis.interfaces.blender.handler.alpha
    navis.interfaces.blender.handler.bevel


Selections
----------
.. autosummary::
    :toctree: generated/

    navis.interfaces.blender.handler.select

    navis.interfaces.blender.object_list.select
    navis.interfaces.blender.object_list.color
    navis.interfaces.blender.object_list.colorize
    navis.interfaces.blender.object_list.emit
    navis.interfaces.blender.object_list.use_transparency
    navis.interfaces.blender.object_list.alpha
    navis.interfaces.blender.object_list.bevel

    navis.interfaces.blender.object_list.hide
    navis.interfaces.blender.object_list.unhide
    navis.interfaces.blender.object_list.hide_others

    navis.interfaces.blender.object_list.delete

    navis.interfaces.blender.object_list.to_json


