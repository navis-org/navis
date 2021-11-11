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

import io

import networkx as nx
import pandas as pd
import xml.etree.ElementTree as ET

from typing import Union, Dict, Optional, Any, IO, Iterable
from zipfile import ZipFile

from .. import config, core
from . import base


__all__ = ["read_nmx"]

# Set up logging
logger = config.logger

NODE_COLUMNS = ('node_id', 'label', 'x', 'y', 'z', 'radius', 'parent_id')
DEFAULT_PRECISION = 32
DEFAULT_FMT = "{name}.nmx"


class NMXReader(base.BaseReader):
    def __init__(
        self,
        precision: int = DEFAULT_PRECISION,
        attrs: Optional[Dict[str, Any]] = None
    ):
        super().__init__(fmt='',
                         attrs=attrs,
                         file_ext='.nmx',
                         read_binary=True,
                         name_fallback='NMX')

        int_, float_ = base.parse_precision(precision)
        self._dtypes = {
            'node_id': int_,
            'parent_id': int_,
            'label': 'category',
            'x': float_,
            'y': float_,
            'z': float_,
            'radius': float_,
        }

    def read_nml(
        self, f: IO, attrs: Optional[Dict[str, Any]] = None
    ) -> 'core.TreeNeuron':
        """Read .nml buffer into a TreeNeuron.

        NML files are XML files containing a single neuron.

        Parameters
        ----------
        f :         IO
                    Readable buffer.
        attrs :     dict | None
                    Arbitrary attributes to include in the TreeNeuron.

        Returns
        -------
        core.TreeNeuron
        """
        if isinstance(f, bytes):
            f = f.decode()

        f = io.StringIO(f)
        root = ET.parse(f).getroot()

        # Copy the attributes dict
        for element in root:
            if element.tag == 'thing':
                nodes = pd.DataFrame.from_records([n.attrib for n in element[0]])
                edges = pd.DataFrame.from_records([n.attrib for n in element[1]])
                edges = edges.astype(self._dtypes['node_id'])

                nodes.rename({'id': 'node_id'}, axis=1, inplace=True)
                nodes = nodes.astype({k: v for k, v in self._dtypes.items() if k in nodes.columns})

        G = nx.Graph()
        G.add_edges_from(edges.values)
        tree = nx.bfs_tree(G, list(G.nodes)[0])
        edges = pd.DataFrame(list(tree.edges), columns=['source', 'target'])
        nodes['parent_id'] = edges.set_index('target').reindex(nodes.node_id.values).source.values
        nodes['parent_id'] = nodes.parent_id.fillna(-1).astype(self._dtypes['node_id'])
        nodes.sort_values('node_id', inplace=True)

        return core.TreeNeuron(
            nodes,
            **(self._make_attributes({'name': 'NML', 'origin': 'nml'}, attrs))
        )

    def read_buffer(
        self, f: IO, attrs: Optional[Dict[str, Any]] = None
    ) -> 'core.TreeNeuron':
        """Read .nmx buffer into a TreeNeuron.

        NMX files are zip files containing XML-encoded .nml files containing
        data for a single neuron.

        Parameters
        ----------
        f :         IO
                    Readable buffer (must be bytes).
        attrs :     dict | None
                    Arbitrary attributes to include in the TreeNeuron.

        Returns
        -------
        core.TreeNeuron
        """
        if not isinstance(f.read(0), bytes):
            raise ValueError(f'Expecet bytes, got "{type(f.read(0))}"')

        zip = ZipFile(f)
        for f in zip.filelist:
            if f.filename.endswith('.nml') and 'skeleton' in f.filename:
                attrs['file'] = f.filename
                attrs['id'] = f.filename.split('/')[0]
                return self.read_nml(zip.read(f), attrs=attrs)
        logger.warning(f'Skipped "{f.filename.split("/")[0]}.nmx": failed to '
                       'import skeleton.')


def read_nmx(f: Union[str, pd.DataFrame, Iterable],
             parallel: Union[bool, int] = 'auto',
             precision: int = 32,
             read_meta: bool = True,
             **kwargs) -> 'core.NeuronObject':
    """Read NMX files into Neuron/Lists.

    NMX is an xml-based format used by pyKNOSSOS.
    See e.g. `here <https://doi.org/10.5281/zenodo.58985>`_ for a data dump
    of neurons from Wanner et al. (2016).

    Parameters
    ----------
    f :                 str
                        Filename or folder. If folder, will import all ``.nmx``
                        files.
    delimiter :         str
                        Delimiter to use. Passed to ``pandas.read_csv``.
    parallel :          "auto" | bool | int
                        Defaults to ``auto`` which means only use parallel
                        processing if more than 200 SWC are imported. Spawning
                        and joining processes causes overhead and is
                        considerably slower for imports of small numbers of
                        neurons. Integer will be interpreted as the
                        number of cores (otherwise defaults to
                        ``os.cpu_count() // 2``).
    precision :         int [8, 16, 32, 64] | None
                        Precision for data. Defaults to 32 bit integers/floats.
                        If ``None`` will let pandas infer data types - this
                        typically leads to higher than necessary precision.
    read_meta :         bool
                        If True and SWC header contains a line with JSON-encoded
                        meta data e.g. (``# Meta: {'id': 123}``), these data
                        will be read as neuron properties. `fmt` takes
                        precedene.
    **kwargs
                        Keyword arguments passed to the construction of
                        ``navis.TreeNeuron``. You can use this to e.g. set
                        meta data.

    Returns
    -------
    navis.NeuronList


    """
    reader = NMXReader(precision=precision,
                       attrs=kwargs)
    # Read neurons
    neurons = reader.read_any(f, parallel=parallel, include_subdirs=False)

    # Failed reads will produce empty neurons which we need to remove
    return neurons[neurons.has_nodes]
