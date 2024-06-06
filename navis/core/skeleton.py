#    This script is part of navis (http://www.github.com/navis-org/navis).
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
import functools
import numbers
import pint
import types
import warnings

import networkx as nx
import numpy as np
import pandas as pd
import skeletor as sk

from io import BufferedIOBase

from typing import Union, Callable, List, Sequence, Optional, Dict, overload
from typing_extensions import Literal

from .. import graph, morpho, utils, config, core, sampling, intersection
from .. import io  # type: ignore # double import

from .base import BaseNeuron
from .core_utils import temp_property

try:
    import xxhash
except ImportError:
    xxhash = None

__all__ = ['TreeNeuron']

# Set up logging
logger = config.get_logger(__name__)

# This is to prevent pint to throw a warning about numpy integration
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pint.Quantity([])


def requires_nodes(func):
    """Return ``None`` if neuron has no nodes."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]
        # Return 0
        if isinstance(self.nodes, str) and self.nodes == 'NA':
            return 'NA'
        if not isinstance(self.nodes, pd.DataFrame):
            return None
        return func(*args, **kwargs)
    return wrapper


class TreeNeuron(BaseNeuron):
    """Neuron represented as hierarchical tree (i.e. a skeleton).

    Parameters
    ----------
    x
                    Data to construct neuron from:
                     - ``pandas.DataFrame`` is expected to be SWC table
                     - ``pandas.Series`` is expected to have a DataFrame as
                       ``.nodes`` - additional properties will be attached
                       as meta data
                     - ``str`` filepath is passed to :func:`navis.read_swc`
                     - ``BufferedIOBase`` e.g. from ``open(filename)``
                     - ``networkx.DiGraph`` parsed by `navis.nx2neuron`
                     - ``None`` will initialize an empty neuron
                     - ``skeletor.Skeleton``
                     - ``TreeNeuron`` - in this case we will try to copy every
                       attribute
    units :         str | pint.Units | pint.Quantity
                    Units for coordinates. Defaults to ``None`` (dimensionless).
                    Strings must be parsable by pint: e.g. "nm", "um",
                    "micrometer" or "8 nanometers".
    **metadata
                    Any additional data to attach to neuron.

    """

    nodes: pd.DataFrame

    graph: 'nx.DiGraph'
    igraph: 'igraph.Graph'  # type: ignore  # doesn't know iGraph

    n_branches: int
    n_leafs: int
    cable_length: Union[int, float]

    segments: List[list]
    small_segments: List[list]

    root: np.ndarray

    soma: Optional[Union[int, str]]
    soma_pos: Optional[Sequence]

    #: Minimum radius for soma detection. Set to ``None`` if no tag needed.
    #: Default = 1 micron
    soma_detection_radius: Union[float, int, pint.Quantity] = 1 * config.ureg.um
    #: Label for soma detection. Set to ``None`` if no tag needed. Default = 1.
    soma_detection_label: Union[float, int, str] = 1
    #: Soma radius (e.g. for plotting). If string, must be column in nodes
    #: table. Default = 'radius'.
    soma_radius: Union[float, int, str] = 'radius'
    # Set default function for soma finding. Default = :func:`navis.morpho.find_soma`
    _soma: Union[Callable[['TreeNeuron'], Sequence[int]], int] = morpho.find_soma

    tags: Optional[Dict[str, List[int]]] = None

    #: Attributes to be used when comparing two neurons.
    EQ_ATTRIBUTES = ['n_nodes', 'n_connectors', 'soma', 'root',
                     'n_branches', 'n_leafs', 'cable_length', 'name']

    #: Temporary attributes that need to be regenerated when data changes.
    TEMP_ATTR = ['_igraph', '_graph_nx', '_segments', '_small_segments',
                 '_geodesic_matrix', 'centrality_method', '_simple',
                 '_cable_length', '_memory_usage']

    #: Attributes used for neuron summary
    SUMMARY_PROPS = ['type', 'name', 'n_nodes', 'n_connectors', 'n_branches',
                     'n_leafs', 'cable_length', 'soma', 'units']

    #: Core data table(s) used to calculate hash
    CORE_DATA = ['nodes:node_id,parent_id,x,y,z']

    def __init__(self,
                 x: Union[pd.DataFrame,
                          BufferedIOBase,
                          str,
                          'TreeNeuron',
                          nx.DiGraph],
                 units: Union[pint.Unit, str] = None,
                 **metadata
                 ):
        """Initialize Skeleton Neuron."""
        super().__init__()

        # Lock neuron during construction
        self._lock = 1

        if isinstance(x, pd.DataFrame):
            self.nodes = x
        elif isinstance(x, pd.Series):
            if not hasattr(x, 'nodes'):
                raise ValueError('pandas.Series must have `nodes` entry.')
            elif not isinstance(x.nodes, pd.DataFrame):
                raise TypeError(f'Nodes must be pandas DataFrame, got "{type(x.nodes)}"')
            self.nodes = x.nodes
            metadata.update(x.to_dict())
        elif isinstance(x, nx.Graph):
            self.nodes = graph.nx2neuron(x).nodes
        elif isinstance(x, BufferedIOBase) or isinstance(x, str):
            x = io.read_swc(x)  # type: ignore
            self.__dict__.update(x.__dict__)
        elif isinstance(x, sk.Skeleton):
            self.nodes = x.swc.copy()
            self.vertex_map = x.mesh_map
        elif isinstance(x, TreeNeuron):
            self.__dict__.update(x.copy().__dict__)
            # Try to copy every attribute
            for at in self.__dict__:
                try:
                    setattr(self, at, copy.copy(getattr(self, at)))
                except BaseException:
                    logger.warning(f'Unable to deep-copy attribute "{at}"')
        elif isinstance(x, type(None)):
            # This is a essentially an empty neuron
            pass
        else:
            raise utils.ConstructionError(f'Unable to construct TreeNeuron from "{type(x)}"')

        for k, v in metadata.items():
            try:
                setattr(self, k, v)
            except AttributeError:
                raise AttributeError(f"Unable to set neuron's `{k}` attribute.")

        self.units = units
        self._current_md5 = self.core_md5

        self._lock = 0

    def __getattr__(self, key):
        """We will use this magic method to calculate some attributes on-demand."""
        # Note that we're mixing @property and __getattr__ which causes problems:
        # if a @property raises an Exception, Python falls back to __getattr__
        # and traceback is lost!

        # Last ditch effort - maybe the base class knows the key?
        return super().__getattr__(key)

    def __truediv__(self, other, copy=True):
        """Implement division for coordinates (nodes, connectors)."""
        if isinstance(other, numbers.Number) or utils.is_iterable(other):
            if utils.is_iterable(other):
                # If divisor is isotropic use only single value
                if len(set(other)) == 1:
                    other == other[0]
                elif len(other) != 4:
                    raise ValueError('Division by list/array requires 4 '
                                     'divisors for x/y/z and radius - '
                                     f'got {len(other)}')

            # If a number, consider this an offset for coordinates
            n = self.copy() if copy else self
            n.nodes.loc[:, ['x', 'y', 'z', 'radius']] /= other

            # At this point we can ditch any 4th unit
            if utils.is_iterable(other):
                other = other[:3]
            if n.has_connectors:
                n.connectors.loc[:, ['x', 'y', 'z']] /= other

            if hasattr(n, 'soma_radius'):
                if isinstance(n.soma_radius, numbers.Number):
                    n.soma_radius /= other

            # Convert units
            # Note: .to_compact() throws a RuntimeWarning and returns unchanged
            # values  when `units` is a iterable
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                n.units = (n.units * other).to_compact()

            n._clear_temp_attr(exclude=['classify_nodes'])
            return n
        return NotImplemented

    def __mul__(self, other, copy=True):
        """Implement multiplication for coordinates (nodes, connectors)."""
        if isinstance(other, numbers.Number) or utils.is_iterable(other):
            if utils.is_iterable(other):
                # If multiplicator is isotropic use only single value
                if len(set(other)) == 1:
                    other == other[0]
                elif len(other) != 4:
                    raise ValueError('Multiplication by list/array requires 4'
                                     'multipliers for x/y/z and radius - '
                                     f'got {len(other)}')

            # If a number, consider this an offset for coordinates
            n = self.copy() if copy else self
            n.nodes.loc[:, ['x', 'y', 'z', 'radius']] *= other

            # At this point we can ditch any 4th unit
            if utils.is_iterable(other):
                other = other[:3]
            if n.has_connectors:
                n.connectors.loc[:, ['x', 'y', 'z']] *= other

            if hasattr(n, 'soma_radius'):
                if isinstance(n.soma_radius, numbers.Number):
                    n.soma_radius *= other

            # Convert units
            # Note: .to_compact() throws a RuntimeWarning and returns unchanged
            # values  when `units` is a iterable
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                n.units = (n.units / other).to_compact()

            n._clear_temp_attr(exclude=['classify_nodes'])
            return n
        return NotImplemented

    def __getstate__(self):
        """Get state (used e.g. for pickling)."""
        state = {k: v for k, v in self.__dict__.items() if not callable(v)}

        # Pickling the graphs actually takes longer than regenerating them
        # from scratch
        if '_graph_nx' in state:
            _ = state.pop('_graph_nx')
        if '_igraph' in state:
            _ = state.pop('_igraph')

        return state

    @property
    @requires_nodes
    def edges(self) -> np.ndarray:
        """Edges between nodes.

        See Also
        --------
        edge_coords
                Same but with x/y/z coordinates instead of node IDs.

        """
        not_root = self.nodes[self.nodes.parent_id >= 0]
        return not_root[['node_id', 'parent_id']].values

    @property
    def edge_coords(self) -> np.ndarray:
        """Coordinates of edges between nodes.

        See Also
        --------
        edges
                Same but with node IDs instead of x/y/z coordinates.

        """
        locs = self.nodes.set_index('node_id')[['x', 'y', 'z']]
        edges = self.edges
        edges_co = np.zeros((edges.shape[0], 2, 3))
        edges_co[:, 0, :] = locs.loc[edges[:, 0]].values
        edges_co[:, 1, :] = locs.loc[edges[:, 1]].values
        return edges_co

    @temp_property
    def igraph(self) -> 'igraph.Graph':
        """iGraph representation of this neuron."""
        # If igraph does not exist, create and return
        if not hasattr(self, '_igraph'):
            # This also sets the attribute
            return self.get_igraph()
        return self._igraph

    @temp_property
    def graph(self) -> nx.DiGraph:
        """Networkx Graph representation of this neuron."""
        # If graph does not exist, create and return
        if not hasattr(self, '_graph_nx'):
            # This also sets the attribute
            return self.get_graph_nx()
        return self._graph_nx

    @temp_property
    def geodesic_matrix(self):
        """Matrix with geodesic (along-the-arbor) distance between nodes."""
        # If matrix has not yet been generated or needs update
        if not hasattr(self, '_geodesic_matrix'):
            # (Re-)generate matrix
            self._geodesic_matrix = graph.geodesic_matrix(self)

        return self._geodesic_matrix

    @property
    @requires_nodes
    def leafs(self) -> pd.DataFrame:
        """Leaf node table."""
        return self.nodes[self.nodes['type'] == 'end']

    @property
    @requires_nodes
    def ends(self):
        """End node table (same as leafs)."""
        return self.leafs

    @property
    @requires_nodes
    def branch_points(self):
        """Branch node table."""
        return self.nodes[self.nodes['type'] == 'branch']

    @property
    def nodes(self) -> pd.DataFrame:
        """Node table."""
        return self._get_nodes()

    def _get_nodes(self) -> pd.DataFrame:
        # Redefine this function in subclass to change how nodes are retrieved
        return self._nodes

    @nodes.setter
    def nodes(self, v):
        """Validate and set node table."""
        # We are refering to an extra function to facilitate subclassing:
        # Redefine _set_nodes() to not break property
        self._set_nodes(v)

    def _set_nodes(self, v):
        # Redefine this function in subclass to change validation
        self._nodes = utils.validate_table(v,
                                           required=[('node_id', 'rowId', 'node', 'treenode_id', 'PointNo'),
                                                     ('parent_id', 'link', 'parent', 'Parent'),
                                                     ('x', 'X'),
                                                     ('y', 'Y'),
                                                     ('z', 'Z')],
                                           rename=True,
                                           optional={('radius', 'W'): 0},
                                           restrict=False)

        # Make sure we don't end up with object dtype anywhere as this can
        # cause problems
        for c in ('node_id', 'parent_id'):
            if self._nodes[c].dtype == 'O':
                self._nodes[c] = self._nodes[c].astype(int)

        graph.classify_nodes(self)

    @property
    def n_trees(self) -> int:
        """Count number of connected trees in this neuron."""
        return len(self.subtrees)

    @property
    def is_tree(self) -> bool:
        """Whether neuron is a tree.

        Also returns True if neuron consists of multiple separate trees!

        See also
        --------
        networkx.is_forest()
                    Function used to test whether neuron is a tree.
        :attr:`TreeNeuron.cycles`
                    If your neuron is not a tree, this will help you identify
                    cycles.

        """
        return nx.is_forest(self.graph)

    @property
    def subtrees(self) -> List[List[int]]:
        """List of subtrees. Sorted by size as sets of node IDs."""
        return sorted(graph._connected_components(self),
                      key=lambda x: -len(x))

    @property
    def connectors(self) -> pd.DataFrame:
        """Connector table. If none, will return ``None``."""
        return self._get_connectors()

    def _get_connectors(self) -> pd.DataFrame:
        # Redefine this function in subclass to change how nodes are retrieved
        return getattr(self, '_connectors', None)

    @connectors.setter
    def connectors(self, v):
        """Validate and set connector table."""
        # We are refering to an extra function to facilitate subclassing:
        # Redefine _set_connectors() to not break property
        self._set_connectors(v)

    def _set_connectors(self, v):
        # Redefine this function in subclass to change validation
        if isinstance(v, type(None)):
            self._connectors = None
        else:
            self._connectors = utils.validate_table(v,
                                                    required=[('connector_id', 'id'),
                                                              ('node_id', 'rowId', 'node', 'treenode_id'),
                                                              ('x', 'X'),
                                                              ('y', 'Y'),
                                                              ('z', 'Z'),
                                                              ('type', 'relation', 'label', 'prepost')],
                                                    rename=True,
                                                    restrict=False)

    @property
    @requires_nodes
    def cycles(self) -> Optional[List[int]]:
        """Cycles in neuron (if any).

        See also
        --------
        networkx.find_cycles()
                    Function used to find cycles.

        """
        try:
            c = nx.find_cycle(self.graph,
                              source=self.nodes[self.nodes.type == 'end'].node_id.values)
            return c
        except nx.exception.NetworkXNoCycle:
            return None
        except BaseException:
            raise

    @property
    def simple(self) -> 'TreeNeuron':
        """Simplified representation consisting only of root, branch points and leafs."""
        if not hasattr(self, '_simple'):
            self._simple = self.downsample(float('inf'),
                                           inplace=False)
        return self._simple

    @property
    def soma(self) -> Optional[Union[str, int]]:
        """Search for soma and return node ID(s).

        ``None`` if no soma. You can assign either a function that accepts a
        TreeNeuron as input or a fix value. The default is
        :func:`navis.utils.find_soma`.

        """
        if callable(self._soma):
            soma = self._soma.__call__()  # type: ignore  # say int not callable
        else:
            soma = self._soma

        # Sanity check to make sure that the soma node actually exists
        if isinstance(soma, type(None)):
            # Return immmediately without expensive checks
            return soma
        elif utils.is_iterable(soma):
            if all(pd.isnull(soma)):
                soma = None
            elif not any(self.nodes.node_id.isin(soma)):
                logger.warning(f'Soma(s) {soma} not found in node table.')
                soma = None
        else:
            if soma not in self.nodes.node_id.values:
                logger.warning(f'Soma {soma} not found in node table.')
                soma = None

        return soma

    @soma.setter
    def soma(self, value: Union[Callable, int, None]) -> None:
        """Set soma."""
        if hasattr(value, '__call__'):
            self._soma = types.MethodType(value, self)
        elif isinstance(value, type(None)):
            self._soma = None
        elif isinstance(value, bool) and not value:
            self._soma = None
        else:
            if value in self.nodes.node_id.values:
                self._soma = value
            else:
                raise ValueError('Soma must be function, None or a valid node ID.')

    @property
    def soma_pos(self) -> Optional[Sequence]:
        """Search for soma and return its position.

        Returns ``None`` if no soma. You can also use this to assign a soma by
        position in which case it will snap to the closest node.
        """
        # Sanity check to make sure that the soma node actually exists
        soma = self.soma
        if isinstance(soma, type(None)):
            return None
        elif utils.is_iterable(soma):
            if all(pd.isnull(soma)):
                return None
        else:
            soma = utils.make_iterable(soma)

        return self.nodes.loc[self.nodes.node_id.isin(soma), ['x', 'y', 'z']].values

    @soma_pos.setter
    def soma_pos(self, value: Sequence) -> None:
        """Set soma by position."""
        try:
            value = np.asarray(value).astype(np.float64).reshape(3)
        except BaseException:
            raise ValueError(f'Unable to convert soma position "{value}" '
                             f'to numeric (3, ) numpy array.')

        # Generate tree
        id, dist = self.snap(value, to='nodes')

        # A sanity check
        if dist > (self.sampling_resolution * 10):
            logger.warning(f'New soma position for {self.id} is suspiciously '
                           f'far away from the closest node: {dist}')

        self.soma = id

    @property
    @requires_nodes
    def root(self) -> Sequence:
        """Root node(s)."""
        roots = self.nodes[self.nodes.parent_id < 0].node_id.values
        return roots

    @root.setter
    def root(self, value: Union[int, List[int]]) -> None:
        """Reroot neuron to given node."""
        self.reroot(value, inplace=True)

    @property
    def type(self) -> str:
        """Neuron type."""
        return 'navis.TreeNeuron'

    @property
    @requires_nodes
    def n_branches(self) -> Optional[int]:
        """Number of branch points."""
        return self.nodes[self.nodes.type == 'branch'].shape[0]

    @property
    @requires_nodes
    def n_leafs(self) -> Optional[int]:
        """Number of leaf nodes."""
        return self.nodes[self.nodes.type == 'end'].shape[0]

    @temp_property
    def cable_length(self) -> Union[int, float]:
        """Cable length."""
        if not hasattr(self, '_cable_length'):
            # Simply sum up edge weight of all graph edges
            if config.use_igraph and self.igraph:
                w = self.igraph.es.get_attribute_values('weight')  # type: ignore # doesn't know iGraph
            else:
                w = nx.get_edge_attributes(self.graph, 'weight').values()
            self._cable_length = np.nansum(list(w))
        return self._cable_length

    @property
    def surface_area(self) -> float:
        """Radius-based lateral surface area."""
        if 'radius' not in self.nodes.columns:
            raise ValueError(f'Neuron {self.id} does not have radius information')

        if any(self.nodes.radius < 0):
            logger.warning(f'Neuron {self.id} has negative radii - area will not be correct.')

        if any(self.nodes.radius.isnull()):
            logger.warning(f'Neuron {self.id} has NaN radii - area will not be correct.')

        # Generate radius dict
        radii = self.nodes.set_index('node_id').radius.to_dict()
        # Drop root node(s)
        not_root = self.nodes.parent_id >= 0
        # For each cylinder get the height
        h = morpho.mmetrics.parent_dist(self, root_dist=0)[not_root]

        # Radii for top and bottom of tapered cylinder
        nodes = self.nodes[not_root]
        r1 = nodes.node_id.map(radii).values
        r2 = nodes.parent_id.map(radii).values

        return (np.pi * (r1 + r2) * np.sqrt( (r1-r2)**2 + h**2)).sum()

    @property
    def volume(self) -> float:
        """Radius-based volume."""
        if 'radius' not in self.nodes.columns:
            raise ValueError(f'Neuron {self.id} does not have radius information')

        if any(self.nodes.radius < 0):
            logger.warning(f'Neuron {self.id} has negative radii - volume will not be correct.')

        if any(self.nodes.radius.isnull()):
            logger.warning(f'Neuron {self.id} has NaN radii - volume will not be correct.')

        # Generate radius dict
        radii = self.nodes.set_index('node_id').radius.to_dict()
        # Drop root node(s)
        not_root = self.nodes.parent_id >= 0
        # For each cylinder get the height
        h = morpho.mmetrics.parent_dist(self, root_dist=0)[not_root]

        # Radii for top and bottom of tapered cylinder
        nodes = self.nodes[not_root]
        r1 = nodes.node_id.map(radii).values
        r2 = nodes.parent_id.map(radii).values

        return (1/3 * np.pi * (r1**2 + r1 * r2 + r2**2) * h).sum()

    @property
    def bbox(self) -> np.ndarray:
        """Bounding box (includes connectors)."""
        mn = np.min(self.nodes[['x', 'y', 'z']].values, axis=0)
        mx = np.max(self.nodes[['x', 'y', 'z']].values, axis=0)

        if self.has_connectors:
            cn_mn = np.min(self.connectors[['x', 'y', 'z']].values, axis=0)
            cn_mx = np.max(self.connectors[['x', 'y', 'z']].values, axis=0)

            mn = np.min(np.vstack((mn, cn_mn)), axis=0)
            mx = np.max(np.vstack((mx, cn_mx)), axis=0)

        return np.vstack((mn, mx)).T

    @property
    def sampling_resolution(self) -> float:
        """Average cable length between child -> parent nodes."""
        return self.cable_length / self.n_nodes

    @temp_property
    def segments(self) -> List[list]:
        """Neuron broken down into linear segments (see also `.small_segments`)."""
        # Calculate if required
        if not hasattr(self, '_segments'):
            # This also sets the attribute
            self._segments = self._get_segments(how='length')
        return self._segments

    @temp_property
    def small_segments(self) -> List[list]:
        """Neuron broken down into small linear segments (see also `.segments`)."""
        # Calculate if required
        if not hasattr(self, '_small_segments'):
            # This also sets the attribute
            self._small_segments = self._get_segments(how='break')
        return self._small_segments

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
            raise ValueError(f'Unknown method: "{how}"')

    @property
    def n_skeletons(self) -> int:
        """Number of seperate skeletons in this neuron."""
        return len(self.root)

    def _clear_temp_attr(self, exclude: list = []) -> None:
        """Clear temporary attributes."""
        super()._clear_temp_attr(exclude=exclude)

        # Remove temporary node values
        # temp_node_cols = ['flow_centrality', 'strahler_index', 'SI', 'bending_flow']
        # self._nodes.drop(columns=temp_node_cols, errors='ignore', inplace=True)

        # Remove soma if it was manually assigned and is not present anymore
        if not callable(self._soma) and not isinstance(self._soma, type(None)):
            if utils.is_iterable(self._soma):
                exists = np.isin(self._soma, self.nodes.node_id.values)
                self._soma = np.asarray(self._soma)[exists]
                if not np.any(self._soma):
                    self._soma = None
            elif self._soma not in self.nodes.node_id.values:
                self.soma = None

        if 'classify_nodes' not in exclude:
            # Reclassify nodes
            graph.classify_nodes(self, inplace=True)

    def copy(self, deepcopy: bool = False) -> 'TreeNeuron':
        """Return a copy of the neuron.

        Parameters
        ----------
        deepcopy :  bool, optional
                    If False, ``.graph`` (NetworkX DiGraph) will be returned
                    as view - changes to nodes/edges can progagate back!
                    ``.igraph`` (iGraph) - if available - will always be
                    deepcopied.

        Returns
        -------
        TreeNeuron

        """
        no_copy = ['_lock']
        # Generate new empty neuron
        x = self.__class__(None)
        # Populate with this neuron's data
        x.__dict__.update({k: copy.copy(v) for k, v in self.__dict__.items() if k not in no_copy})

        # Copy graphs only if neuron is not stale
        if not self.is_stale:
            if '_graph_nx' in self.__dict__:
                x._graph_nx = self._graph_nx.copy(as_view=deepcopy is not True)
            if '_igraph' in self.__dict__:
                if self._igraph is not None:
                    # This is pretty cheap, so we will always make a deep copy
                    x._igraph = self._igraph.copy()
        else:
            x._clear_temp_attr()

        return x

    def get_graph_nx(self) -> nx.DiGraph:
        """Calculate and return networkX representation of neuron.

        Once calculated stored as ``.graph``. Call function again to update
        graph.

        See Also
        --------
        :func:`navis.neuron2nx`

        """
        self._graph_nx = graph.neuron2nx(self)
        return self._graph_nx

    def get_igraph(self) -> 'igraph.Graph':  # type: ignore
        """Calculate and return iGraph representation of neuron.

        Once calculated stored as ``.igraph``. Call function again to update
        iGraph.

        Important
        ---------
        Returns ``None`` if igraph is not installed!

        See Also
        --------
        :func:`navis.neuron2igraph`

        """
        self._igraph = graph.neuron2igraph(self, raise_not_installed=False)
        return self._igraph

    @overload
    def resample(self, resample_to: int, inplace: Literal[False]) -> 'TreeNeuron': ...

    @overload
    def resample(self, resample_to: int, inplace: Literal[True]) -> None: ...

    def resample(self, resample_to, inplace=False):
        """Resample neuron to given resolution.

        Parameters
        ----------
        resample_to :           int
                                Resolution to which to resample the neuron.
        inplace :               bool, optional
                                If True, operation will be performed on
                                itself. If False, operation is performed on
                                copy which is then returned.

        See Also
        --------
        :func:`~navis.resample_skeleton`
            Base function. See for details and examples.

        """
        if inplace:
            x = self
        else:
            x = self.copy(deepcopy=False)

        sampling.resample_skeleton(x, resample_to, inplace=True)

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
        """Reroot neuron to given node ID or node tag.

        Parameters
        ----------
        new_root :  int | str
                    Either node ID or node tag.
        inplace :   bool, optional
                    If True, operation will be performed on itself. If False,
                    operation is performed on copy which is then returned.

        See Also
        --------
        :func:`~navis.reroot_skeleton`
            Base function. See for details and examples.

        """
        if inplace:
            x = self
        else:
            x = self.copy(deepcopy=False)

        graph.reroot_skeleton(x, new_root, inplace=True)

        # Clear temporary attributes is done by morpho.reroot_skeleton()
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
        :func:`~navis.cut_skeleton`
            Base function. See for details and examples.

        """
        if inplace:
            x = self
        else:
            x = self.copy(deepcopy=False)

        node = utils.make_iterable(node, force_type=None)

        for n in node:
            prox = graph.cut_skeleton(x, n, ret='proximal')[0]
            # Reinitialise with proximal data
            x.__init__(prox)  # type: ignore  # Cannot access "__init__" directly
            # Remove potential "left over" attributes (happens if we use a copy)
            x._clear_temp_attr()

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
        :func:`~navis.cut_skeleton`
            Base function. See for details and examples.

        """
        if inplace:
            x = self
        else:
            x = self.copy(deepcopy=False)

        node = utils.make_iterable(node, force_type=None)

        for n in node:
            dist = graph.cut_skeleton(x, n, ret='distal')[0]
            # Reinitialise with distal data
            x.__init__(dist)  # type: ignore  # Cannot access "__init__" directly
            # Remove potential "left over" attributes (happens if we use a copy)
            x._clear_temp_attr()

        # Clear temporary attributes is done by cut_skeleton
        # x._clear_temp_attr()

        if not inplace:
            return x
        return None

    def prune_by_strahler(self,
                          to_prune: Union[int, List[int], slice],
                          inplace: bool = False) -> Optional['TreeNeuron']:
        """Prune neuron based on `Strahler order
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
        """Prune terminal twigs under a given size.

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

    def prune_at_depth(self,
                       depth: Union[float, int],
                       source: Optional[int] = None,
                       inplace: bool = False
                       ) -> Optional['TreeNeuron']:
        """Prune all neurites past a given distance from a source.

        Parameters
        ----------
        x :             TreeNeuron | NeuronList
        depth :         int | float
                        Distance from source at which to start pruning.
        source :        int, optional
                        Source node for depth calculation. If ``None``, will use
                        root. If ``x`` is a list of neurons then must provide a
                        source for each neuron.
        inplace :       bool, optional
                        If False, pruning is performed on copy of original neuron
                        which is then returned.

        Returns
        -------
        TreeNeuron/List
                        Pruned neuron(s).

        See Also
        --------
        :func:`~navis.prune_at_depth`
            This is the base function. See for details and examples.

        """
        if inplace:
            x = self
        else:
            x = self.copy()

        morpho.prune_at_depth(x, depth=depth, source=source, inplace=True)

        if not inplace:
            return x
        return None

    def cell_body_fiber(self,
                        reroot_soma: bool = True,
                        inplace: bool = False,
                        ) -> Optional['TreeNeuron']:
        """Prune neuron to its cell body fiber.

        Parameters
        ----------
        reroot_soma :       bool, optional
                            If True, will reroot to soma.
        inplace :           bool, optional
                            If True, operation will be performed on itself.
                            If False, operation is performed on copy which is
                            then returned.

        See Also
        --------
        :func:`~navis.cell_body_fiber`
            This is the base function. See for details and examples.

        """
        if inplace:
            x = self
        else:
            x = self.copy()

        morpho.cell_body_fiber(x, inplace=True, reroot_soma=reroot_soma)

        # Clear temporary attributes
        x._clear_temp_attr()

        if not inplace:
            return x
        return None

    def prune_by_longest_neurite(self,
                                 n: int = 1,
                                 reroot_soma: bool = False,
                                 inplace: bool = False,
                                 ) -> Optional['TreeNeuron']:
        """Prune neuron down to the longest neurite.

        Parameters
        ----------
        n :                 int, optional
                            Number of longest neurites to preserve.
        reroot_soma :       bool, optional
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
            x, n, inplace=True, reroot_soma=reroot_soma)

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
        """Prune neuron by intersection with given volume(s).

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

    def to_swc(self,
               filename: Optional[str] = None,
               **kwargs) -> None:
        """Generate SWC file from this neuron.

        Parameters
        ----------
        filename :      str | None, optional
                        If ``None``, will use "neuron_{id}.swc".
        kwargs
                        Additional arguments passed to :func:`~navis.write_swc`.

        Returns
        -------
        Nothing

        See Also
        --------
        :func:`~navis.write_swc`
                See this function for further details.

        """
        return io.write_swc(self, filename, **kwargs)  # type: ignore  # double import of "io"

    def reload(self,
               inplace: bool = False,
               ) -> Optional['TreeNeuron']:
        """Reload neuron. Must have filepath as ``.origin`` as attribute.

        Returns
        -------
        TreeNeuron
                If ``inplace=False``.

        """
        if not hasattr(self, 'origin'):
            raise AttributeError('To reload TreeNeuron must have `.origin` '
                                 'attribute')

        if self.origin in ('DataFrame', 'string'):
            raise ValueError('Unable to reload TreeNeuron: it appears to have '
                             'been created from string or DataFrame.')

        kwargs = {}
        if hasattr(self, 'soma_label'):
            kwargs['soma_label'] = self.soma_label
        if hasattr(self, 'connector_labels'):
            kwargs['connector_labels'] = self.connector_labels

        x = io.read_swc(self.origin, **kwargs)

        if inplace:
            self.__dict__.update(x.__dict__)
            self._clear_temp_attr()
        else:
            # This makes sure that we keep any additional data stored after
            # this neuron has been loaded
            x2 = self.copy()
            x2.__dict__.update(x.__dict__)
            x2._clear_temp_attr()
            return x

    def snap(self, locs, to='nodes'):
        """Snap xyz location(s) to closest node or synapse.

        Parameters
        ----------
        locs :      (N, 3) array | (3, ) array
                    Either single or multiple XYZ locations.
        to :        "nodes" | "connectors"
                    Whether to snap to nodes or connectors.

        Returns
        -------
        id :        int | list of int
                    ID(s) of the closest node/connector.
        dist :      float | list of float
                    Distance(s) to the closest node/connector.

        Examples
        --------
        >>> import navis
        >>> n = navis.example_neurons(1)
        >>> id, dist = n.snap([0, 0, 0])
        >>> id
        1124

        """
        locs = np.asarray(locs).astype(np.float64)

        is_single = (locs.ndim == 1 and len(locs) == 3)
        is_multi = (locs.ndim == 2 and locs.shape[1] == 3)
        if not is_single and not is_multi:
            raise ValueError('Expected a single (x, y, z) location or a '
                             '(N, 3) array of multiple locations')

        if to not in ['nodes', 'connectors']:
            raise ValueError('`to` must be "nodes" or "connectors", '
                             f'got {to}')

        # Generate tree
        tree = graph.neuron2KDTree(self, data=to)

        # Find the closest node
        dist, ix = tree.query(locs)

        if to == 'nodes':
            id = self.nodes.node_id.values[ix]
        else:
            id = self.connectors.connector_id.values[ix]

        return id, dist
