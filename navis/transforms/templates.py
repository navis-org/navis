#    This script is part of navis (http://www.github.com/schlegelp/navis).
#    Copyright (C) 2018 Philipp Schlegel
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

"""Functions to work with templates."""

import functools
import math
import numbers
import os
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from scipy.spatial.distance import pdist

from collections import namedtuple
from typing import List, Union, Optional
from typing_extensions import Literal

from .. import config, core, utils

from . import factory

from .base import TransformSequence, BaseTransform, TransOptimizer
from .affine import AffineTransform

__all__ = ['xform_brain', 'mirror_brain']

logger = config.logger

# Defines entry the registry needs to register a transform
transform_reg = namedtuple('Transform',
                           ['source', 'target', 'transform', 'type',
                            'invertible', 'weight'])

# Check for environment variable pointing to registries
_os_transpaths = os.environ.get('NAVIS_TRANSFORMS', '')
try:
    _os_transpaths = [i for i in _os_transpaths.split(';') if len(i) > 0]
except BaseException:
    logger.error('Error parsing the `NAVIS_TRANSFORMS` environment variable')
    _os_transpaths = []


class TemplateRegistry:
    """Tracks template brains, available transforms and produces bridging sequences.

    Parameters
    ----------
    scan_paths :    bool
                    If True will scan paths on initialization.

    """
    def __init__(self, scan_paths: bool = True):
        # Paths to scan for transforms
        self._transpaths = _os_transpaths
        # Transforms
        self._transforms = []
        # Template brains
        self._templates = []

        if scan_paths:
            self.scan_paths()

    def __contains__(self, other) -> bool:
        """Check if transform is in registry.

        Parameters
        ----------
        other :     transform, filepath, tuple
                    Either a transform (e.g. CMTKtransform), a filepath (e.g.
                    to a .list file) or a tuple of ``(source, target, transform)``
                    where ``transform`` can be a transform or a filepath.

        """
        if isinstance(other, (tuple, list)):
            return any([t == other for t in self.transforms])
        else:
            return other in [t.transform for t in self.transforms]

    def __len__(self) -> int:
        return len(self.transforms)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'TemplateRegistry with {len(self)} transforms'

    @property
    def transpaths(self) -> list:
        """Paths searched for transforms.

        Use `.scan_paths` to trigger a scan. Use `.register_path` to add
        more path(s).
        """
        return self._transpaths

    @property
    def templates(self) -> list:
        """Registered template (brains)."""
        return self._templates

    @property
    def transforms(self) -> list:
        """Registered transforms (bridging + mirror)."""
        return self._transforms

    @property
    def bridges(self) -> list:
        """Registered bridging transforms."""
        return [t for t in self.transforms if t.type == 'bridging']

    @property
    def mirrors(self) ->list:
        """Registered mirror transforms."""
        return [t for t in self.transforms if t.type == 'mirror']

    def clear_caches(self):
        """Clear caches of all cached functions."""
        self.bridging_graph.cache_clear()
        self.shortest_bridging_seq.cache_clear()

    def summary(self) -> pd.DataFrame:
        """Generate summary of available transforms."""
        return pd.DataFrame(self.transforms)

    def register_path(self, paths: str, trigger_scan: bool = True):
        """Register path(s) to scan for transforms.

        Parameters
        ----------
        paths :         str | list thereof
                        Paths (or list thereof) to scans for transforms. This
                        is not permanent. For permanent additions set path(s)
                        via the ``NAVIS_TRANSFORMS`` environment variable.
        trigger_scan :  bool
                        If True, a re-scan of all paths will be triggered.

        """
        paths = utils.make_iterable(paths)

        for p in paths:
            # Try not to duplicate paths
            if p not in self.transpaths:
                self._transpaths.append(p)

        if trigger_scan:
            self.scan_paths()

    def register_templatebrain(self, template: 'TemplateBrain',
                               skip_existing=True):
        """Register a template brain.

        This is used, for example, by navis.mirror_brain.

        Parameters
        ----------
        template :      TemplateBrain
                        TemplateBrain to register.
        skip_existing : bool
                        If True, will skip existing template brains.

        """
        utils.eval_param(template,
                         name='template',
                         allowed_types=(TemplateBrain, ))

        if template not in self._templates or not skip_existing:
            self._templates.append(template)

    def register_transform(self, transform: BaseTransform, source: str,
                           target: str, transform_type: str,
                           invertible: bool = True, skip_existing: bool = True,
                           weight: int = 1):
        """Register a transform.

        Parameters
        ----------
        transform :         subclass of BaseTransform | TransformSequence
                            A transform (AffineTransform, CMTKtransform, etc.)
                            or a TransformSequence.
        source :            str
                            Source for forward transform.
        target :            str
                            Target for forward transform. Ignored for mirror
                            transforms.
        transform_type :    "bridging" | "mirror"
                            Type of transform.
        invertible :        bool
                            Whether transform can be inverted via ``__neg__``.
        skip_existing :     bool
                            If True will skip if transform is already in registry.
        weight :            int
                            Giving a transform a higher weight will make it
                            preferable when plotting bridging sequences.

        See Also
        --------
        register_transformfile
                            If you want to register a file instead of an
                            already constructed transform.

        """
        assert transform_type in ('bridging', 'mirror')
        assert isinstance(transform, BaseTransform)

        # Translate into edge
        edge = transform_reg(source=source, target=target, transform=transform,
                             type=transform_type, invertible=invertible,
                             weight=weight)

        # Don't add if already exists
        if not skip_existing or edge not in self:
            self.transforms.append(edge)

        # Clear cached functions
        self.clear_caches()

    def register_transformfile(self, path: str, **kwargs):
        """Parse and register a transform file.

        File/Directory name must follow the a ``{TARGET}_{SOURCE}.{ext}``
        convention (e.g. ``JRC2013_FCWB.list``).

        Parameters
        ----------
        path :          str
                        Path to transform.
        **kwargs
                        Keyword arguments are passed to the constructor of the
                        Transform (e.g. CMTKtransform for `.list` directory).

        See Also
        --------
        register_transform
                        If you want to register an already constructed transform
                        instead of a transform file that still needs to be
                        parsed.

        """
        assert isinstance(path, (str, pathlib.Path))

        path = pathlib.Path(path).expanduser()

        if not path.is_dir() and not path.is_file():
            raise ValueError(f'File/directory "{path}" does not exist')

        # Parse properties
        try:
            if 'mirror' in path.name or 'imgflip' in path.name:
                transform_type = 'mirror'
                source = path.name.split('_')[0]
                target = None
            else:
                transform_type = 'bridging'
                target = path.name.split('_')[0]
                source = path.name.split('_')[1].split('.')[0]

            # Initialize the transform
            transform = factory.factory_methods[path.suffix](path, **kwargs)

            self.register_transform(transform=transform,
                                    source=source,
                                    target=target,
                                    transform_type=transform_type)
        except BaseException as e:
            logger.error(f'Error registering {path} as transform: {str(e)}')

    def scan_paths(self, extra_paths: List[str] = None):
        """Scan registered paths for transforms and add to registry.

        Will skip transforms that already exist in this registry.

        Parameters
        ----------
        extra_paths :   list of str
                        Any Extra paths to search.

        """
        search_paths = self.transpaths

        if isinstance(extra_paths, str):
            extra_paths = [i for i in extra_paths.split(';') if len(i) > 0]
            search_paths = np.append(search_paths, extra_paths)

        for path in search_paths:
            path = pathlib.Path(path).expanduser()
            # Skip if path does not exist
            if not path.is_dir():
                continue

            # Go over the file extensions we can work with (.h5, .list, .json)
            # These file extensions are registered in the
            # `navis.transforms.factory` module
            for ext in factory.factory_methods:
                for hit in path.rglob(f'*{ext}'):
                    if hit.is_dir() or hit.is_file():
                        # Register this file
                        self.register_transformfile(hit)

        # Clear cached functions
        self.clear_caches()

    @functools.lru_cache()
    def bridging_graph(self,
                       reciprocal: Union[Literal[False], int, float] = True) -> nx.DiGraph:
        """Generate networkx Graph describing the bridging paths.

        Parameters
        ----------
        reciprocal :    bool | float
                        If True or float, will add forward and inverse edges for
                        transforms that are invertible. If float, the inverse
                        edges' weights will be scaled by that factor.

        Returns
        -------
        networkx.MultiDiGraph

        """
        # Drop mirror transforms
        bridge = [t for t in self.transforms if t.type == 'bridging']

        # Generate graph
        # Note we are using MultiDi graph here because we might
        # have multiple edges between nodes. For example, there
        # is a JFRC2013DS_JFRC2013 and a JFRC2013_JFRC2013DS
        # bridging registration. If we include the inverse, there
        # will be two edges connecting JFRC2013DS and JFRC2013 in
        # both directions
        G = nx.MultiDiGraph()
        edges = [(t.source, t.target,
                  {'transform': t.transform,
                   'type': str(type(t.transform)).split('.')[-1],
                   'weight': t.weight}) for t in bridge]

        if reciprocal:
            if isinstance(reciprocal, numbers.Number):
                rv_edges = [(t.target, t.source,
                             {'transform': -t.transform,  # note inverse transform!
                              'type': str(type(t.transform)).split('.')[-1],
                              'weight': t.weight * reciprocal}) for t in bridge]
            else:
                rv_edges = [(t.target, t.source,
                             {'transform': -t.transform,  # note inverse transform!
                              'type': str(type(t.transform)).split('.')[-1],
                              'weight': t.weight}) for t in bridge]
            edges += rv_edges

        G.add_edges_from(edges)

        return G

    def find_bridging_path(self, source: str,
                           target: str, reciprocal=True) -> tuple:
        """Find bridging path from source to target.

        Parameters
        ----------
        G :             nx.DiGraph
                        Bridging graph. See also ``bridging_graph``.
        source :        str
                        Source from which to transform to ``target``.
        target :        str
                        Target to which to transform to.
        reciprocal :    bool | float
                        If True or float, will add forward and inverse edges for
                        transforms that are invertible. If float, the inverse
                        edges' weights will be scaled by that factor.

        Returns
        -------
        path :          list
                        Path from source to target: [source, ..., target]
        transforms :    list
                        Transforms as [[path_to_transform, inverse], ...]

        """
        # Generate (or get cached) bridging graph
        G = self.bridging_graph(reciprocal=reciprocal)

        if len(G) == 0:
            raise ValueError('No bridging registrations available')

        if source not in G.nodes:
            raise ValueError(f'Source "{source}" has no known bridging registrations.')
        if target not in G.nodes:
            raise ValueError(f'Target "{target}" has no known bridging registrations.')

        # This will raise a error message if no path is found
        path = nx.shortest_path(G, source, target, weight='weight')

        # `path` holds the sequence of nodes we are traversing but not which
        # transforms (i.e. edges) to use
        transforms = []
        for n1, n2 in zip(path[:-1], path[1:]):
            this_edges = []
            i = 0
            # First collect all edges between those two nodes
            # - this is annoyingly complicated with MultiDiGraphs
            while True:
                try:
                    e = G.edges[(n1, n2, i)]
                except KeyError:
                    break
                this_edges.append([e['transform'], e['weight']])
                i += 1

            # Now find the edge with the highest weight
            # (inverse transforms might have a lower weight)
            this_edges = sorted(this_edges, key=lambda x: x[-1])
            transforms.append(this_edges[-1][0])

        return path, transforms

    @functools.lru_cache()
    def shortest_bridging_seq(self, source: str, target: str,
                              via: Optional[str] = None,
                              inverse_weight: float = .5) -> tuple:
        """Find shortest bridging sequence to get from source to target.

        Parameters
        ----------
        source :            str
                            Source from which to transform to ``target``.
        target :            str
                            Target to which to transform to.
        via :               str | list of str
                            Waystations to traverse on the way from source to
                            target.
        inverse_weight :    float
                            Weight for inverse transforms. If < 1 will prefer
                            forward transforms.

        Returns
        -------
        sequence :          (N, ) array
                            Sequence of registrations that will be traversed.
        transform_seq :     TransformSequence
                            Class that collates the required transforms to get
                            from source to target.

        """
        # Generate sequence of nodes we need to find a path for
        # Minimally it's just from source to target
        nodes = np.array([source, target])

        if via:
            nodes = np.insert(nodes, 1, via)

        seq = [nodes[0]]
        transforms = []
        for n1, n2 in zip(nodes[:-1], nodes[1:]):
            path, tr = self.find_bridging_path(n1, n2, reciprocal=inverse_weight)
            seq = np.append(seq, path[1:])
            transforms = np.append(transforms, tr)

        if any(np.unique(seq, return_counts=True)[1] > 1):
            logger.warning('Bridging sequence contains loop: '
                           f'{"->".join(seq)}')

        # Generate the transform sequence
        transform_seq = TransformSequence(*transforms)

        return seq, transform_seq

    def find_mirror_reg(self, template: str, non_found: str = 'raise') -> tuple:
        """Search for a mirror transformation for given template.

        Typically a mirror transformation specifies a non-rigid transformation
        to correct asymmetries in an image.

        Parameters
        ----------
        template :  str
                    Name of the template to find a mirror transformation for.
        non_found : "raise" | "ignore"
                    What to do if no mirror transformation is found. If "ignore"
                    and no mirror transformation found, will silently return
                    ``None``.

        Returns
        -------
        tuple
                    Named tuple containing a mirror transformation. Will only
                    ever return one - even if multiple are available.

        """
        for tr in self.mirrors:
            if tr.source == template:
                return tr

        if non_found == 'raise':
            raise ValueError(f'No mirror transformation found for {template}')
        return None

    def find_template(self, name: str, non_found: str = 'raise') -> 'TemplateBrain':
        """Search for a given template (brain).

        Parameters
        ----------
        name :      str
                    Name of the template to find a mirror transformation for.
                    Searches against `name` and `label` (short name) properties
                    of registered templates.
        non_found : "raise" | "ignore"
                    What to do if no mirror transformation is found. If "ignore"
                    and no mirror transformation found, will silently return
                    ``None``.

        Returns
        -------
        TemplateBrain

        """
        for tmp in self.templates:
            if getattr(tmp, 'label', None) == name:
                return tmp
            if getattr(tmp, 'name', None) == name:
                return tmp

        if non_found == 'raise':
            raise ValueError(f'No template brain registered that matches "{name}"')
        return None

    def plot_bridging_graph(self, edge_labels: bool = False, **kwargs):
        """Draw bridging graph using networkX.

        Parameters
        ----------
        **kwargs
                    Keyword arguments are passed to ``networkx.draw_networkx``.

        Returns
        -------
        None

        """
        # Get graph
        G = self.bridging_graph(reciprocal=False)

        # Draw nodes and edges
        node_labels = {n: n for n in G.nodes}
        pos = nx.kamada_kawai_layout(G)
        nx.draw_networkx(G, pos=pos, labels=node_labels)

        if edge_labels:
            el = {(e[0], e[1]): G.edges[e]['type'] for e in G.edges}
            nx.draw_networkx_edge_labels(G, pos=pos, ax=plt.gca(),
                                         edge_labels=el)


def xform_brain(x: Union['core.NeuronObject', 'pd.DataFrame', 'np.ndarray'],
                source: str,
                target: str,
                affine_fallback: bool = True,
                caching: Optional[str] = 'medium',
                verbose = True) -> Union['core.NeuronObject',
                                         'pd.DataFrame',
                                         'np.ndarray']:
    """Transform 3D data between template brains.

    This requires the appropriate transforms to be registered with ``navis``.
    See the docs for details.

    Notes
    -----
    For Neurons only: whether there is a change in units during transformation
    (e.g. nm -> um) is inferred by comparing distances between x/y/z coordinates
    before and after transform. This guesstimate is then used to convert
    ``.units`` and node/soma radii. This works reasonably well with base 10
    increments (e.g. nm -> um) but is off with odd changes in units.

    Parameters
    ----------
    x :                 Neuron/List | numpy.ndarray | pandas.DataFrame
                        Data to transform. Dataframe must contain ``['x', 'y', 'z']``
                        columns. Numpy array must be shape ``(N, 3)``.
    source :            str
                        Source template brain that the data currently is in.
    target :            str
                        Target template brain that the data should be transformed
                        into.
    affine_fallback :   bool
                        In same cases the non-rigid transformation of points
                        can fail - for example if points are outside the
                        deformation field. If that happens, they will be
                        returned as ``NaN``. Unless ``affine_fallback`` is
                        ``True``, in which case we will apply only the rigid
                        affine  part of the transformation to at least get close
                        to the correct coordinates.
    caching :           None | 'medium' | 'aggressive'
                        Sets the level of caching:
                          - ``None`` = no upfront cost, lowest memory footprint
                          - ``medium`` = low upfront cost, potentially faster
                          - ``aggressive`` = high upfront cost, most definitely faster
                        As a rule of thumb: "medium" is sufficient for a
                        few, spatially close neurons whereas "aggressive" is
                        suited for very large transforms across the entire brain
                        space. Only applies if input is NeuronList and if
                        transforms include H5 transform.
    verbose :           bool
                        If True, will print some useful info on transform.

    Returns
    -------
    same type as ``x``
                        Copy of input with transformed coordinates.

    Examples
    --------
    This example requires the flybrains library to be installed:
    ``pip3 install flybrains``

    Also, you will need to have the optional transforms installed (one-off):

    >>> import flybrains # doctest: +SKIP
    >>> flybrains.download_jefferislab_transforms() # doctest: +SKIP
    >>> flybrains.download_saalfeldlab_transforms() # doctest: +SKIP

    Once ``flybrains`` is installed and you have downloaded the registrations,
    you can run this:

    >>> import navis
    >>> import flybrains
    >>> # navis example neurons are in raw (8nm voxel) hemibrain (JRCFIB2018F) space
    >>> n = navis.example_neurons(1)
    >>> # Transform to FAFB14 space
    >>> # Note that we need to convert to microns first because that what the transform expects
    >>> xf = navis.xform_brain(n * 8 / 1000, source='JRCFIB2018F', target='FAFB14') # doctest: +SKIP

    """
    if isinstance(x, core.NeuronList):
        if len(x) == 1:
            x = x[0]
        else:
            xf = []
            # Get the transformation sequence
            path, trans_seq = registry.shortest_bridging_seq(source, target)
            if verbose:
                # Make sure to flush to not mess with the progress bar
                print('Transform path:', '->'.join(path), flush=True)

            _caching = (None, "medium", "aggressive")
            if caching not in _caching:
                raise ValueError('Caching must be None, "medium" or '
                                 f'"aggressive", got {caching}')

            with TransOptimizer(trans_seq, mode=caching):
                for i, n in enumerate(config.tqdm(x, desc='Xforming',
                                                  disable=config.pbar_hide,
                                                  leave=config.pbar_leave)):
                    xf.append(xform_brain(n,
                                          source=source,
                                          target=target,
                                          verbose=False,
                                          affine_fallback=affine_fallback))
            return x.__class__(xf)

    if not isinstance(x, (core.BaseNeuron, np.ndarray, pd.DataFrame, core.Volume)):
        raise TypeError(f'Unable to transform data of type "{type(x)}"')

    if isinstance(x, core.BaseNeuron):
        xf = x.copy()
        # We will collate spatial data to reduce overhead from calling
        # R's xform_brain
        if isinstance(xf, core.TreeNeuron):
            xyz = xf.nodes[['x', 'y', 'z']].values
        elif isinstance(xf, core.MeshNeuron):
            xyz = xf.vertices
        elif isinstance(xf, core.Dotprops):
            xyz = xf.points
        else:
            raise TypeError(f"Don't know how to transform neuron of type '{type(xf)}'")

        # Add connectors if they exist
        if xf.has_connnectors:
            xyz = np.vstack([xyz, xf.connectors[['x', 'y', 'z']].values])

        # Do the xform of all spatial data
        xyz_xf = xform_brain(xyz,
                             source=source,
                             target=target,
                             verbose=verbose,
                             affine_fallback=affine_fallback)

        # Guess change in spatial units
        change, magnitude = _guess_change(xyz, xyz_xf)

        # Round change -> this rounds to the first non-zero digit
        # change = np.around(change, decimals=-magnitude)

        # Map xformed coordinates back
        if isinstance(xf, core.TreeNeuron):
            xf.nodes.loc[:, ['x', 'y', 'z']] = xyz_xf[:xf.n_nodes]
            # Fix radius based on our best estimate
            if 'radius' in xf.nodes.columns:
                xf.nodes['radius'] *= 10**magnitude
        elif isinstance(xf, core.Dotprops):
            xf.points = xyz_xf[:xf.points.shape[0]]
            # Set tangent vectors and alpha to None so they will be regenerated
            xf._vect = xf._alpha = None
        elif isinstance(xf, core.MeshNeuron):
            xf.vertices = xyz_xf[:xf.vertices.shape[0]]

        if xf.has_connectors:
            xf.connectors.loc[:, ['x', 'y', 'z']] = xyz_xf[-xf.connectors.shape[0]:]

        # Make an educated guess as to whether the units have changed
        if hasattr(xf, 'units') and magnitude != 0:
            if isinstance(xf.units, (config.ureg.Unit, config.ureg.Quantity)):
                xf.units = (xf.units / 10**magnitude).to_compact()

        # Fix soma radius if applicable
        if hasattr(xf, 'soma_radius') and isinstance(xf.soma_radius, numbers.Number):
            xf.soma_radius *= 10**magnitude

        return xf
    elif isinstance(x, pd.DataFrame):
        if any([c not in x.columns for c in ['x', 'y', 'z']]):
            raise ValueError('DataFrame must have x, y and z columns.')
        x = x.copy()
        x.loc[:, ['x', 'y', 'z']] = xform_brain(x[['x', 'y', 'z']].values.astype(float),
                                                source=source,
                                                target=target,
                                                verbose=verbose,
                                                affine_fallback=affine_fallback)
        return x
    elif isinstance(x, core.Volume):
        x = x.copy()
        x.vertices = xform_brain(x.vertices,
                                 source=source,
                                 target=target,
                                 verbose=verbose,
                                 affine_fallback=affine_fallback)
        return x
    elif x.shape[1] != 3:
        raise ValueError('Array must be of shape (N, 3).')

    if not isinstance(source, str):
        TypeError(f'Expected source of type str, got "{type(source)}"')

    if not isinstance(target, str):
        TypeError(f'Expected target of type str, got "{type(target)}"')

    # Get the transformation sequence
    path, trans_seq = registry.shortest_bridging_seq(source, target)

    if verbose:
        print('Transform path:', '->'.join(path))

    # Apply transform and returned xformed points
    return trans_seq.xform(x, affine_fallback=affine_fallback)


def _guess_change(xyz_before: np.ndarray,
                  xyz_after: np.ndarray,
                  sample: float = .1) -> tuple:
    """Guess change in units during xforming."""
    if isinstance(xyz_before, pd.DataFrame):
        xyz_before = xyz_before[['x', 'y', 'z']].values
    if isinstance(xyz_after, pd.DataFrame):
        xyz_after = xyz_after[['x', 'y', 'z']].values

    # Select the same random sample of points in both spaces
    if sample <= 1:
        sample = int(xyz_before.shape[0] * sample)
    rnd_ix = np.random.choice(xyz_before.shape[0], sample, replace=False)
    sample_bef = xyz_before[rnd_ix, :]
    sample_aft = xyz_after[rnd_ix, :]

    # Get pairwise distance between those points
    dist_pre = pdist(sample_bef)
    dist_post = pdist(sample_aft)

    # Calculate how the distance between nodes changed and get the average
    # Note we are ignoring nans - happens e.g. when points did not transform.
    with np.errstate(divide='ignore', invalid='ignore'):
        change = dist_post / dist_pre
    # Drop infinite values in rare cases where nodes end up on top of another
    mean_change = np.nanmean(change[change < np.inf])

    # Find the order of magnitude
    magnitude = round(math.log10(mean_change))

    return mean_change, magnitude


def mirror_brain(x: Union['core.NeuronObject', 'pd.DataFrame', 'np.ndarray'],
                 template: Union[str, 'TemplateBrain'],
                 mirror_axis: Union[Literal['x'],
                                    Literal['y'],
                                    Literal['z']] = 'x',
                 warp: Union[Literal['auto'], bool] = 'auto',
                 via: Optional[str] = None) -> Union['core.NeuronObject',
                                                     'pd.DataFrame',
                                                     'np.ndarray']:
    """Mirror 3D object (neuron, coordinates) about given axis.

    The way this works is:
     1. Look up the length of the template space along the given axis. For this,
        the template space has to be registered (see docs for details).
     2. Flip object along midpoint of axis using a affine transformation.
     3. (Optional) Apply a warp transform that corrects asymmetries.

    Parameters
    ----------
    x :             Neuron/List | numpy.ndarray | pandas.DataFrame
                    Data to transform. Dataframe must contain ``['x', 'y', 'z']``
                    columns. Numpy array must be shape ``(N, 3)``.
    template :      str | TemplateBrain
                    Source template brain space that the data is in. If string
                    will be searched against registered template brains.
                    Alternatively check out :func:`navis.transforms.mirror`
                    for a lower level interface.
    mirror_axis :   'x' | 'y' | 'z', optional
                    Axis to mirror. Defaults to `x`.
    warp :          bool | "auto" | Transform, optional
                    If 'auto', will check if a mirror transformation exists
                    for the given ``template`` and apply it after the flipping.
                    You can also just pass a Transform or TransformSequence.
    via :           str | None
                    If provided, (e.g. "FCWB") will first transform
                    coordinates into that space, then mirror and transform back.
                    Use this if there is no mirror registration for the original
                    template, or to transform to a symmetrical template in which
                    flipping is sufficient.

    Returns
    -------
    xf
                    Same object type as input (array, neurons, etc) but with
                    transformed coordinates.

    Examples
    --------
    This example requires the flybrains library to be installed:
    ``pip3 install flybrains``

    >>> import navis
    >>> import flybrains
    >>> # navis example neurons are in raw hemibrain (JRCFIB2018Fraw) space
    >>> n = navis.example_neurons(1)
    >>> # Mirror about x axis (this is a simple flip in this case)
    >>> mirrored = navis.mirror_brain(n * 8 / 1000, tem plate='JRCFIB2018F', via='JRC2018F') # doctest: +SKIP
    >>> # We also need to get back to raw coordinates
    >>> mirrored = mirrored / 8 * 1000 # doctest: +SKIP

    See Also
    --------
    :func:`navis.transform.mirror`
                    Lower level function for mirroring. You can use this if
                    you want to mirror data without having a registered
                    template for it.

    """
    utils.eval_param(mirror_axis, name='mirror_axis',
                     allowed_values=('x', 'y', 'z'), on_error='raise')
    utils.eval_param(warp, name='warp',
                     allowed_values=('auto', True, False), on_error='raise')

    # If we go via another brain space
    if via:
        # Xform to "via" space
        xf = xform_brain(x, source=template, target=via)
        # Mirror
        xfm = mirror_brain(xf,
                           template=via,
                           mirror_axis=mirror_axis,
                           warp=warp,
                           via=None)
        # Xform back to original template space
        xfm_inv = xform_brain(xfm, source=via, target=template)
        return xfm_inv

    if isinstance(x, core.NeuronList):
        if len(x) == 1:
            x = x[0]
        else:
            xf = []
            for n in config.tqdm(x, desc='Mirroring',
                                 disable=config.pbar_hide,
                                 leave=config.pbar_leave):
                xf.append(mirror_brain(n,
                                       template=template,
                                       mirror_axis=mirror_axis,
                                       warp=warp))
            return core.NeuronList(xf)

    if not isinstance(x, (core.BaseNeuron, np.ndarray, pd.DataFrame)):
        raise TypeError(f'Unable to transform data of type "{type(x)}"')

    if isinstance(x, core.BaseNeuron):
        x = x.copy()
        if isinstance(x, core.TreeNeuron):
            x.nodes = mirror_brain(x.nodes,
                                   template=template,
                                   mirror_axis=mirror_axis,
                                   warp=warp)
        elif isinstance(x, core.Dotprops):
            x.points = mirror_brain(x.points,
                                    template=template,
                                    mirror_axis=mirror_axis,
                                    warp=warp)
            # Set tangent vectors and alpha to None so they will be regenerated
            x._vect = x._alpha = None
        elif isinstance(x, core.MeshNeuron):
            x.vertices = mirror_brain(x.vertices,
                                      template=template,
                                      mirror_axis=mirror_axis,
                                      warp=warp)
        else:
            raise TypeError(f"Don't know how to transform neuron of type '{type(x)}'")

        if x.has_connectors:
            x.connectors = mirror_brain(x.connectors,
                                        template=template,
                                        mirror_axis=mirror_axis,
                                        warp=warp)
        return x
    elif isinstance(x, pd.DataFrame):
        if any([c not in x.columns for c in ['x', 'y', 'z']]):
            raise ValueError('DataFrame must have x, y and z columns.')
        x = x.copy()
        x.loc[:, ['x', 'y', 'z']] = mirror_brain(x[['x', 'y', 'z']].values.astype(float),
                                                 template=template,
                                                 mirror_axis=mirror_axis,
                                                 warp=warp)
        return x

    # At this point we expect numpy arrays
    x = np.asarray(x)
    if not x.ndim == 2 or x.shape[1] != 3:
        raise ValueError('Array must be of shape (N, 3).')

    if not isinstance(template, str):
        TypeError(f'Expected template of type str, got "{type(template)}"')

    if warp:
        # See if there is a mirror registration
        mirror_trans = registry.find_mirror_reg(template, non_found='ignore')

        # If warp was not "auto" and we didn't find a registration, raise
        if warp != 'auto' and not mirror_trans:
            raise ValueError(f'No mirror transform found for "{template}"')
    else:
        mirror_trans = None

    # Now find the meta info about the template brain
    if isinstance(template, TemplateBrain):
        tb = template
    else:
        tb = registry.find_template(template, non_found='raise')

    # Get the bounding box
    if not hasattr(tb, 'boundingbox'):
        raise ValueError(f'Template "{tb.label}" has no bounding box info.')

    if not isinstance(tb.boundingbox, (list, tuple, np.ndarray)):
        raise TypeError("Expected the template brain's bounding box to be a "
                        f"list, tuple or array - got '{type(tb.boundingbox)}'")

    # Get bounding box of template brain
    bbox = np.asarray(tb.boundingbox)

    # Reshape if flat array
    if bbox.ndim == 1:
        bbox = bbox.reshape(3, 2)

    # Index of mirror axis
    ix = {'x': 0, 'y': 1, 'z': 2}[mirror_axis]

    if bbox.shape == (3, 2):
        # In nat.templatebrains this is using the sum (min+max) but have a
        # suspicion that this should be the difference (max-min)
        mirror_axis_size = bbox[ix, :].sum()
    elif bbox.shape == (2, 3):
        mirror_axis_size = bbox[:, ix].sum()
    else:
        raise ValueError('Expected bounding box to be of shape (3, 2) or (2, 3)'
                         f' got {bbox.shape}')

    return mirror(x, mirror_axis=mirror_axis, mirror_axis_size=mirror_axis_size,
                  warp=mirror_trans)


def mirror(points: np.ndarray, mirror_axis_size: float,
           mirror_axis: str = 'x',
           warp: Optional['BaseTransform'] = None) -> np.ndarray:
    """Mirror 3D coordinates about given axis.

    This is a lower level version of `navis.mirror_brain` that:
     1. Flips object along midpoint of axis using a affine transformation.
     2. (Optional) Applies a warp transform that corrects asymmetries.

    Parameters
    ----------
    points :            (N, 3) numpy array
                        3D coordinates to mirror.
    mirror_axis_size :  int | float
                        A single number specifying the size of the mirror axis.
                        This is used to find the midpoint to mirror about.
    mirror_axis :       'x' | 'y' | 'z', optional
                        Axis to mirror. Defaults to `x`.
    warp :              Transform, optional
                        If provided, will apply this warp transform after the
                        affine flipping. Typically this will be a mirror
                        registration to compensate for left/right asymmetries.

    Returns
    -------
    points_mirrored
                        Mirrored coordinates.

    """
    utils.eval_param(mirror_axis, name='mirror_axis',
                     allowed_values=('x', 'y', 'z'), on_error='raise')

    # At this point we expect numpy arrays
    points = np.asarray(points)
    if not points.ndim == 2 or points.shape[1] != 3:
        raise ValueError('Array must be of shape (N, 3).')

    # Translate mirror axis to index
    mirror_ix = {'x': 0, 'y': 1, 'z': 2}[mirror_axis]

    # Construct homogeneous affine mirroring transform
    mirrormat = np.eye(4, 4)
    mirrormat[mirror_ix, 3] = mirror_axis_size
    mirrormat[mirror_ix, mirror_ix] = -1

    # Turn into affine transform
    flip_transform = AffineTransform(mirrormat)

    # Flip about mirror axis
    points_mirrored = flip_transform.xform(points)

    if isinstance(warp, (BaseTransform, TransformSequence)):
        points_mirrored = warp.xform(points_mirrored)

    return points_mirrored


class TemplateBrain:
    """Generic base class for template brains.

    Minimally, a template should have a `name` and `label` property. For
    mirroring, it also needs a `boundingbox`.

    See `navis-flybrains <https://github.com/schlegelp/navis-flybrains>`_ for
    an example of how to use template brains.
    """

    def __init__(self, **properties):
        """Initialize class."""
        for k, v in properties.items():
            setattr(self, k, v)

    @property
    def mesh(self):
        """Mesh represenation of this brain."""
        if not hasattr(self, '_mesh'):
            name = getattr(self, 'regName', getattr(self, 'name', None))
            raise ValueError(f'{name} does not appear to have a mesh')
        return self._mesh


# Initialize the registry
registry = TemplateRegistry()
