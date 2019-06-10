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

import copy
import io
import numbers
import types
import uuid

import networkx as nx
import numpy as np
import pandas as pd

from io import BufferedIOBase, StringIO

from typing import Union, Callable, List, Sequence, Optional, Dict, overload
from typing_extensions import Literal

from .. import graph, morpho, utils, config, core, sampling, intersection
from .. import io  # type: ignore # double import

__all__ = ['Neuron', 'TreeNeuron']

# Set up logging
logger = config.logger


def Neuron(x: Union[nx.DiGraph, str, pd.DataFrame, 'TreeNeuron'], **metadata):
    """ Constructor for Neuron objects. Depending on the input, either a
    ``TreeNeuron`` or a ``VolumeNeuron`` is returned.

    Parameters
    ----------
    x                   networkx.Graph | str | file-like | pandas.DataFrame | Neuron
                        Data to construct neuron from.
    **metadata
                        Any additional data to attach to neuron.

    See Also
    --------
    :func:`navis.from_swc`
                        Gives you more control over how data is extraced from
                        SWC file.
    """

    if isinstance(x, (nx.Graph, str, pd.DataFrame)):
        return TreeNeuron(x, **metadata)
    # elif isinstance(x, (Volume, trimesh.Trimesh, dict)):
    #    return VolumeNeuron(x, **metadata)
    else:
        raise TypeError(f'Unable to construct neuron from "{type(x)}"')


class TreeNeuron:
    """ Object representing neurons as hierarchical trees.
    """

    # Minimum radius for soma detection - set to None if no tag needed
    soma_detection_radius: Union[float, int] = .5
    # Label for soma detection - set to None if no tag needed
    soma_detection_label: Union[float, int, str] = 1
    # Set default function for soma finding
    _soma: Union[Callable[['TreeNeuron'], Sequence[int]],
                 int] = morpho.find_soma

    nodes: pd.DataFrame
    connectors: pd.DataFrame

    graph: 'nx.DiGraph'
    igraph: 'igraph.Graph'  # type: ignore  # doesn't know iGraph

    dps: pd.DataFrame

    n_branches: int
    n_leafs: int
    cable_length: Union[int, float]

    segments: List[list]
    small_segments: List[list]

    soma: Union[int, str]
    root: np.ndarray

    name: str

    def __init__(self,
                 x: Union[pd.DataFrame,
                          BufferedIOBase,
                          str,
                          'TreeNeuron',
                          nx.DiGraph],
                 **metadata
                 ):
        """ Initialize Skeleton Neuron.

        Parameters
        ----------
        x
                        Data to construct neuron from:
                         - `pandas.DataFrame` is expected to be SWC table
                         - `str` is treated as SWC file name
                         - `BufferedIOBase` e.g. from `open(filename)`
                         - `networkx.DiGraph` parsed by `navis.nx2neuron`
        **metadata
                        Any additional data to attach to neuron.
        """

        if isinstance(x, pd.DataFrame):
            self.nodes = x
        elif isinstance(x, nx.Graph):
            self.nodes = graph.nx2neuron(x)
        elif isinstance(x, BufferedIOBase) or isinstance(x, str):
            x = io.from_swc(x)  # type: ignore
            self.__dict__.update(x.__dict__)
        elif isinstance(x, TreeNeuron):
            self.__dict__.update(x.copy().__dict__)
        else:
            raise TypeError('Unable to construct TreeNeuron from data '
                            'type %s' % str(type(x)))

        for k, v in metadata.items():
            setattr(self, k, v)

        if not getattr(self, 'uuid', None):
            self.uuid = uuid.uuid4()

    def __getattr__(self, key):
        """ We will use this magic method to calculate some attributes
        on-demand. """
        if key == 'igraph':
            self.igraph = self.get_igraph()
            return self.igraph
        elif key == 'graph':
            self.graph = self.get_graph_nx()
            return self.graph
        elif key == 'segments':
            self.segments = self._get_segments(how='length')
            return self.segments
        elif key == 'small_segments':
            self.small_segments = self._get_segments(how='break')
            return self.small_segments
        elif key == 'dps':
            self.dps = self.get_dps()
            return self.dps
        elif key.startswith('has_'):
            key = key[key.index('_'):]
            if hasattr(self, key):
                data = getattr(self, key)
                if isinstance(data, pd.DataFrame) and not data.empty:
                    return True
            return False
        else:
            raise AttributeError('Attribute "%s" not found' % key)

    @property
    def nodes(self) -> pd.DataFrame:
        """ Node table. """
        return self._nodes

    @nodes.setter
    def nodes(self, v):
        self._nodes = utils.validate_table(v,
                                           required=['node_id',
                                                     'parent_id',
                                                     'x', 'y', 'z'],
                                           optional={'radius': 0},
                                           restrict=False)
        graph.classify_nodes(self)

    @property
    def connectors(self) -> pd.DataFrame:
        """ Connector table. If none, will return ``None``. """
        return getattr(self, '_connectors', None)

    @connectors.setter
    def connectors(self, v):
        if isinstance(v, type(None)):
            self.__connectors = None
        else:
            self._connectors = utils.validate_table(v,
                                                    required=['connector_id',
                                                              'node_id',
                                                              'x', 'y', 'z',
                                                              ('type',
                                                               'relation',
                                                               'label')],
                                                    restrict=False)

    @property
    def n_trees(self) -> int:
        """ Number of connected trees in this neuron. """
        return len(self.subtrees)

    @property
    def subtrees(self) -> List[List[int]]:
        """ List of subtrees as node IDs. """
        comp = nx.connected_components(self.graph.to_undirected())
        return list(comp)

    @property
    def simple(self) -> 'TreeNeuron':
        """ Neuron representation consisting only of root, branch points and
        leafs.
        """
        if not hasattr(self, '_simple'):
            self._simple = self.downsample(float('inf'),
                                           preserve_cn_treenodes=False,
                                           inplace=False)
        return self._simple

    @property
    def soma(self) -> Optional[Union[str, int]]:
        """ Search for soma and return node ID(s) of soma or ``None`` if no
        soma - You can set this ``.soma`` property to either a function that
        accepts a TreeNeuron as input or a fix value - Default is
        ``navis.utils.find_soma``.
        """

        if callable(self._soma):
            soma = self._soma.__call__()  # type: ignore  # say int not callable
        else:
            soma = self._soma

        if utils.is_iterable(soma) and not any(soma):
            soma = None

        return soma

    @soma.setter
    def soma(self, value):
        if hasattr(value, '__call__'):
            self._soma = types.MethodType(value, self)
        elif isinstance(value, type(None)):
            self._soma = None
        else:
            if value in self.nodes.node_id.values:
                self._soma = value
            else:
                raise ValueError('Soma must be function, None or a valid node ID.')

    @property
    def root(self) -> Sequence:
        """ Root node(s)."""
        roots = self.nodes[self.nodes.parent_id < 0].node_id.values
        return roots

    @property
    def n_nodes(self) -> int:
        """ Number of nodes."""
        return self.nodes.shape[0]

    @property
    def n_connectors(self) -> int:
        """ Number of connectors."""
        if self.has_connectors:
            return self.connectors.shape[0]
        else:
            return 0

    @property
    def n_branches(self) -> int:
        """ Number of branch points."""
        return self.nodes[self.nodes.type == 'branch'].shape[0]

    @property
    def n_leafs(self) -> int:
        """ Number of leafs."""
        return self.nodes[self.nodes.type == 'end'].shape[0]

    @property
    def cable_length(self) -> Union[int, float]:
        """ Cable length."""
        # Simply sum up edge weight of all graph edges
        if self.igraph and config.use_igraph:
            w = self.igraph.es.get_attribute_values('weight')  # type: ignore # doesn't know iGraph
        else:
            w = nx.get_edge_attributes(self.graph, 'weight').values()
        return np.nansum(list(w))

    @property
    def bbox(self) -> np.ndarray:
        """ Bounding box."""
        return self.nodes.describe().loc[['min', 'max'],
                                         ['x', 'y', 'z']].values.T

    @property
    def sampling_resolution(self) -> float:
        """ Nodes per unit of cable length. """
        return self.n_nodes / self.cable_length

    @property
    def n_skeletons(self) -> int:
        """ Returns number of seperate skeletons in this neuron. """
        return self.nodes[self.nodes.parent_id < 0].shape[0]

    @property
    def type(self) -> str:
        """ Returns 'TreeNeuron'"""
        return 'TreeNeuron'

    def __copy__(self):
        return self.copy(deepcopy=False)

    def __deepcopy__(self):
        return self.copy(deepcopy=True)

    def copy(self, deepcopy: bool = False) -> 'TreeNeuron':
        """Returns a copy of the neuron.

        Parameters
        ----------
        deepcopy :  bool, optional
                    If False, ``.graph`` (NetworkX DiGraph) will be returned
                    as view - changes to nodes/edges can progagate back!
                    ``.igraph`` (iGraph) - if available - will always be
                    deepcopied.

        Returns
        -------
        copy

        """
        # Generate new neuron
        x = Neuron(self.nodes)
        # Remove everything but then new UUID
        x.__dict__ = {'uuid': x.uuid}
        # Override with this neuron's data
        x.__dict__.update({k: copy.copy(v) for k, v in self.__dict__.items() if k != 'uuid'})

        if 'graph' in self.__dict__:
            x.graph = self.graph.copy(as_view=deepcopy is not True)
        if 'igraph' in self.__dict__:
            if self.igraph is not None:
                # This is pretty cheap, so we will always make a deep copy
                x.igraph = self.igraph.copy()

        return x

    def _clear_temp_attr(self, exclude: list = []) -> None:
        """Clear temporary attributes."""
        temp_att = ['igraph', 'graph', 'segments', 'small_segments',
                    'nodes_geodesic_distance_matrix', 'dps',
                    'centrality_method', '_simple']
        for a in [at for at in temp_att if at not in exclude]:
            try:
                delattr(self, a)
                logger.debug(f'Neuron {id(self)}: {a} cleared')
            except BaseException:
                logger.debug(f'Neuron {id(self)}: Unable to clear temporary attribute "{a}"')
                pass

        temp_node_cols = ['flow_centrality', 'strahler_index']

        # Remove type only if we do not classify -> this speeds up things
        # b/c we don't have to recreate the column, just change the values
        # if 'classify_nodes' in exclude:
        #    temp_node_cols.append('type')

        # Remove temporary node values
        self.nodes = self.nodes[[
            c for c in self.nodes.columns if c not in temp_node_cols]]

        if 'classify_nodes' not in exclude:
            # Reclassify nodes
            graph.classify_nodes(self, inplace=True)

    def get_graph_nx(self) -> nx.DiGraph:
        """Calculates networkX representation of neuron.

        Once calculated stored as ``.graph``. Call function again to update
        graph.

        See Also
        --------
        :func:`navis.neuron2nx`
        """
        self.graph = graph.neuron2nx(self)
        return self.graph

    def get_igraph(self) -> 'igraph.Graph':  # type: ignore
        """Calculates iGraph representation of neuron.

        Once calculated stored as ``.igraph``. Call function again to update
        iGraph.

        Important
        ---------
        Returns ``None`` if igraph is not installed!

        See Also
        --------
        :func:`navis.neuron2igraph`
        """
        self.igraph = graph.neuron2igraph(self)
        return self.igraph

    def get_dps(self) -> pd.DataFrame:
        """Calculates/updates dotproduct representation of the neuron.

        Once calculated stored as ``.dps``.

        See Also
        --------
        :func:`navis.to_dotproduct`
        """

        self.dps = graph.neuron2dps(self)
        return self.dps

    def _get_segments(self,
                      how: Union[Literal['length'],
                                 Literal['break']] = 'length'
                      ) -> List[list]:
        """Generate segments for neuron."""
        if how == 'length':
            return graph._generate_segments(self)
        elif how == 'break':
            return graph._break_segments(self)
        else:
            raise ValueError(f'Unknown how: "{how}"')

    def plot2d(self, **kwargs):
        """Plot neuron using :func:`navis.plot2d`.

        Parameters
        ----------
        **kwargs
                Will be passed to :func:`navis.plot2d`.
                See ``help(navis.plot2d)`` for a list of keywords.

        See Also
        --------
        :func:`navis.plot2d`
                    Function called to generate 2d plot.

        """

        from ..plotting import plot2d

        return plot2d(self, **kwargs)

    def plot3d(self, **kwargs):
        """Plot neuron using :func:`navis.plot3d`.

        Parameters
        ----------
        **kwargs
                Keyword arguments. Will be passed to :func:`navis.plot3d`.
                See ``help(navis.plot3d)`` for a list of keywords.

        See Also
        --------
        :func:`navis.plot3d`
                    Function called to generate 3d plot.

        Examples
        --------
        >>> import navis
        >>> nl = navis.example_neurons()
        >>> #Plot with connectors
        >>> nl.plot3d(connectors=True)

        """

        from ..plotting import plot3d

        return plot3d(core.NeuronList(self, make_copy=False), **kwargs)

    @overload
    def resample(self, resample_to: int, inplace: Literal[False]) -> 'TreeNeuron': ...

    @overload
    def resample(self, resample_to: int, inplace: Literal[True]) -> None: ...

    def resample(self, resample_to, inplace=False):
        """Resample the neuron to given resolution [nm].

        Parameters
        ----------
        resample_to :           int
                                Resolution in nanometer to which to resample
                                the neuron.
        inplace :               bool, optional
                                If True, operation will be performed on
                                itself. If False, operation is performed on
                                copy which is then returned.

        See Also
        --------
        :func:`~navis.resample_neuron`
            Base function. See for details and examples.

        """
        if inplace:
            x = self
        else:
            x = self.copy(deepcopy=False)

        sampling.resample_neuron(x, resample_to, inplace=True)

        # No need to call this as base function does this for us
        # x._clear_temp_attr()

        if not inplace:
            return x
        return None

    @overload
    def downsample(self,
                   factor: float,
                   inplace: Literal[False],
                   **kwargs) -> 'TreeNeuron': ...

    @overload
    def downsample(self,
                   factor: float,
                   inplace: Literal[True],
                   **kwargs) -> None: ...

    def downsample(self, factor=5, inplace=False, **kwargs):
        """Downsample the neuron by given factor.

        Parameters
        ----------
        factor :                int, optional
                                Factor by which to downsample the neurons.
                                Default = 5.
        inplace :               bool, optional
                                If True, operation will be performed on
                                itself. If False, operation is performed on
                                copy which is then returned.
        **kwargs
                                Additional arguments passed to
                                :func:`~navis.downsample_neuron`.

        See Also
        --------
        :func:`~navis.downsample_neuron`
            Base function. See for details and examples.

        """
        if inplace:
            x = self
        else:
            x = self.copy(deepcopy=False)

        sampling.downsample_neuron(x, factor, inplace=True, **kwargs)

        # Delete outdated attributes
        x._clear_temp_attr()

        if not inplace:
            return x
        return None

    def reroot(self,
               new_root: Union[int, str],
               inplace: bool = False) -> Optional['TreeNeuron']:
        """ Reroot neuron to given node ID or node tag.

        Parameters
        ----------
        new_root :  int | str
                    Either node ID or node tag.
        inplace :   bool, optional
                    If True, operation will be performed on itself. If False,
                    operation is performed on copy which is then returned.

        See Also
        --------
        :func:`~navis.reroot_neuron`
            Base function. See for details and examples.

        """

        if inplace:
            x = self
        else:
            x = self.copy(deepcopy=False)

        graph.reroot_neuron(x, new_root, inplace=True)

        # Clear temporary attributes is done by morpho.reroot_neuron()
        # x._clear_temp_attr()

        if not inplace:
            return x
        return None

    def prune_distal_to(self,
                        node: Union[str, int],
                        inplace: bool = False) -> Optional['TreeNeuron']:
        """Cut off nodes distal to given nodes.

        Parameters
        ----------
        node :      node ID | node tag
                    Provide either node ID(s) or a unique tag(s)
        inplace :   bool, optional
                    If True, operation will be performed on itself. If False,
                    operation is performed on copy which is then returned.

        See Also
        --------
        :func:`~navis.cut_neuron`
            Base function. See for details and examples.
        """

        if inplace:
            x = self
        else:
            x = self.copy(deepcopy=False)

        node = utils.make_iterable(node, force_type=None)

        for n in node:
            prox = graph.cut_neuron(x, n, ret='proximal')
            # Reinitialise with proximal data
            x.__init__(prox, x.meta_data)  # type: ignore  # Cannot access "__init__" directly
            # Remove potential "left over" attributes (happens if we use a copy)
            x._clear_temp_attr(exclude=['graph', 'igraph', 'type',
                                        'classify_nodes'])

        if not inplace:
            return x
        return None

    def prune_proximal_to(self,
                          node: Union[str, int],
                          inplace: bool = False) -> Optional['TreeNeuron']:
        """Remove nodes proximal to given node. Reroots neuron to cut node.

        Parameters
        ----------
        node :      node_id | node tag
                    Provide either a node ID or a (unique) tag
        inplace :   bool, optional
                    If True, operation will be performed on itself. If False,
                    operation is performed on copy which is then returned.

        See Also
        --------
        :func:`~navis.cut_neuron`
            Base function. See for details and examples.

        """

        if inplace:
            x = self
        else:
            x = self.copy(deepcopy=False)

        node = utils.make_iterable(node, force_type=None)

        for n in node:
            dist = graph.cut_neuron(x, n, ret='distal')
            # Reinitialise with distal data
            x.__init__(dist, x.meta_data)  # type: ignore  # Cannot access "__init__" directly
            # Remove potential "left over" attributes (happens if we use a copy)
            x._clear_temp_attr(exclude=['graph', 'igraph', 'type',
                                        'classify_nodes'])

        # Clear temporary attributes is done by cut_neuron
        # x._clear_temp_attr()

        if not inplace:
            return x
        return None

    def prune_by_strahler(self,
                          to_prune: Union[int, List[int], slice],
                          inplace: bool = False) -> Optional['TreeNeuron']:
        """ Prune neuron based on `Strahler order
        <https://en.wikipedia.org/wiki/Strahler_number>`_.

        Will reroot neuron to soma if possible.

        Parameters
        ----------
        to_prune :  int | list | range | slice
                    Strahler indices to prune. For example:

                    1. ``to_prune=1`` removes all leaf branches
                    2. ``to_prune=[1, 2]`` removes SI 1 and 2
                    3. ``to_prune=range(1, 4)`` removes SI 1, 2 and 3
                    4. ``to_prune=slice(1, -1)`` removes everything but the
                       highest SI
                    5. ``to_prune=slice(-1, None)`` removes only the highest
                       SI

        inplace :   bool, optional
                    If True, operation will be performed on itself. If False,
                    operation is performed on copy which is then returned.

        See Also
        --------
        :func:`~navis.prune_by_strahler`
            This is the base function. See for details and examples.

        """

        if inplace:
            x = self
        else:
            x = self.copy()

        morpho.prune_by_strahler(
            x, to_prune=to_prune, reroot_soma=True, inplace=True)

        # No need to call this as morpho.prune_by_strahler does this already
        # self._clear_temp_attr()

        if not inplace:
            return x
        return None

    def prune_twigs(self,
                    size: float,
                    inplace: bool = False,
                    recursive: Union[int, bool, float] = False
                    ) -> Optional['TreeNeuron']:
        """ Prune terminal twigs under a given size.

        Parameters
        ----------
        size :          int | float
                        Twigs shorter than this will be pruned.
        inplace :       bool, optional
                        If False, pruning is performed on copy of original neuron
                        which is then returned.
        recursive :     int | bool | "inf", optional
                        If `int` will undergo that many rounds of recursive
                        pruning. Use ``float("inf")`` to prune until no more
                        twigs under the given size are left.

        See Also
        --------
        :func:`~navis.prune_twigs`
            This is the base function. See for details and examples.

        """

        if inplace:
            x = self
        else:
            x = self.copy()

        morpho.prune_twigs(x, size=size, inplace=True)

        if not inplace:
            return x
        return None

    def prune_by_longest_neurite(self,
                                 n: int = 1,
                                 reroot_to_soma: bool = False,
                                 inplace: bool = False
                                 ) -> Optional['TreeNeuron']:
        """ Prune neuron down to the longest neurite.

        Parameters
        ----------
        n :                 int, optional
                            Number of longest neurites to preserve.
        reroot_to_soma :    bool, optional
                            If True, will reroot to soma before pruning.
        inplace :           bool, optional
                            If True, operation will be performed on itself.
                            If False, operation is performed on copy which is
                            then returned.

        See Also
        --------
        :func:`~navis.longest_neurite`
            This is the base function. See for details and examples.

        """

        if inplace:
            x = self
        else:
            x = self.copy()

        graph.longest_neurite(
            x, n, inplace=True, reroot_to_soma=reroot_to_soma)

        # Clear temporary attributes
        x._clear_temp_attr()

        if not inplace:
            return x
        return None

    def prune_by_volume(self,
                        v: Union[core.Volume,
                                 List[core.Volume],
                                 Dict[str, core.Volume]],
                        mode: Union[Literal['IN'], Literal['OUT']] = 'IN',
                        prevent_fragments: bool = False,
                        inplace: bool = False
                        ) -> Optional['TreeNeuron']:
        """ Prune neuron by intersection with given volume(s).

        Parameters
        ----------
        v :                 str | navis.Volume | list of either
                            Volume(s) to check for intersection
        mode :              'IN' | 'OUT', optional
                            If 'IN', parts of the neuron inside the volume are
                            kept.
        prevent_fragments : bool, optional
                            If True, will add nodes to ``subset`` required to
                            keep neuron from fragmenting.
        inplace :           bool, optional
                            If True, operation will be performed on itself. If
                            False, operation is performed on copy which is then
                            returned.

        See Also
        --------
        :func:`~navis.in_volume`
            Base function. See for details and examples.
        """

        if inplace:
            x = self
        else:
            x = self.copy()

        intersection.in_volume(x, v, inplace=True,
                               prevent_fragments=prevent_fragments,
                               mode=mode)

        # Clear temporary attributes
        # x._clear_temp_attr()

        if not inplace:
            return x
        return None

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return str(self.summary())

    def _repr_html_(self):
        frame = self.summary().to_frame()
        frame.columns = ['']
        # return self._gen_svg_thumbnail() + frame._repr_html_()
        return frame._repr_html_()

    def _gen_svg_thumbnail(self):
        import matplotlib.pyplot as plt
        # Store some previous states
        prev_level = logger.getEffectiveLevel()
        prev_pbar = config.pbar_hide
        prev_int = plt.isinteractive()

        plt.ioff()  # turn off interactive mode
        logger.setLevel('WARNING')
        config.pbar_hide = True
        fig = plt.figure(figsize=(2, 2))
        ax = fig.add_subplot(111)
        fig, ax = self.plot2d(connectors=False, ax=ax)
        output = StringIO()
        fig.savefig(output, format='svg')

        if prev_int:
            plt.ion()  # turn on interactive mode
        logger.setLevel(prev_level)
        config.pbar_hide = prev_pbar
        _ = plt.clf()
        return output.getvalue()

    def __eq__(self, other):
        """Implements neuron comparison."""
        if isinstance(other, TreeNeuron):
            # Make some morphological comparisons and make sure to go
            # from simple to computationally expensive
            to_comp = ['uuid', 'n_nodes', 'n_connectors', 'soma', 'root',
                       'n_branches', 'n_leafs', 'cable_length']

            # We will do this sequentially and stop as soon as we find a
            # discrepancy -> this saves tons of time!
            for at in to_comp:
                comp = getattr(self, at) == getattr(other, at)
                if isinstance(comp, np.ndarray) and not all(comp):
                    return False
                elif comp is False:
                    return False
            # If all comparisons have passed, return True
            return True
        else:
            return NotImplemented

    def __hash__(self):
        """Generate a hashable value."""
        # We will simply use the neuron's memory address
        return id(self)

    def __add__(self, other):
        """ Implements addition. """
        if isinstance(other, TreeNeuron):
            return core.NeuronList([self, other])
        else:
            return NotImplemented

    def __truediv__(self, other):
        """Implements division for coordinates (nodes, connectors)."""
        if isinstance(other, numbers.Number):
            # If a number, consider this an offset for coordinates
            n = self.copy()
            n.nodes.loc[:, ['x', 'y', 'z', 'radius']] /= other
            if n.has_connectors:
                n.connectors.loc[:, ['x', 'y', 'z']] /= other
            n._clear_temp_attr(exclude=['classify_nodes'])
            return n
        else:
            return NotImplemented

    def __mul__(self, other):
        """Implements multiplication for coordinates (nodes, connectors)."""
        if isinstance(other, numbers.Number):
            # If a number, consider this an offset for coordinates
            n = self.copy()
            n.nodes.loc[:, ['x', 'y', 'z', 'radius']] *= other
            if n.has_connectors:
                n.connectors.loc[:, ['x', 'y', 'z']] *= other
            n._clear_temp_attr(exclude=['classify_nodes'])
            return n
        else:
            return NotImplemented

    def summary(self) -> pd.Series:
        """Get a summary of this neuron."""

        # Set logger to warning only - otherwise you might get tons of
        # "skeleton data not available" messages
        lvl = logger.level
        logger.setLevel('WARNING')

        # Look up these values without requesting them
        props = ['type', 'name', 'n_nodes', 'n_connectors', 'n_branches',
                 'n_leafs', 'cable_length', 'soma']
        s = pd.Series([getattr(self, at, 'NA') for at in props],
                      index=props)

        logger.setLevel(lvl)
        return s

    def to_swc(self,
               filename: Optional[str] = None,
               **kwargs) -> None:
        """ Generate SWC file from this neuron.

        This converts navis nanometer coordinates into microns.

        Parameters
        ----------
        filename :      str | None, optional
                        If ``None``, will use "neuron_{skeletonID}.swc".
        kwargs
                        Additional arguments passed to :func:`~navis.to_swc`.

        Returns
        -------
        Nothing

        See Also
        --------
        :func:`~navis.to_swc`
                See this function for further details.

        """

        return io.to_swc(self, filename, **kwargs)  # type: ignore  # double import of "io"
