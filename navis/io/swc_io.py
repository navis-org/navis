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

import csv
import datetime
import os
import io
import requests

import pandas as pd
import numpy as np

from glob import glob

from textwrap import dedent

import multiprocessing as mp

from typing import Union, Iterable, Dict, Optional, Any

from .. import config, utils, core

# Set up logging
logger = config.logger


def from_swc(f: Union[str, pd.DataFrame, Iterable],
             connector_labels: Optional[Dict[str, Union[str, int]]] = {},
             soma_label: Union[str, int] = 1,
             include_subdirs: bool = False,
             parallel: Union[bool, int] = 'auto',
             **kwargs) -> 'core.NeuronObject':
    """Create Neuron/List from SWC file.

    This import is following format specified
    `here <http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html>`_

    Parameters
    ----------
    f :                 str | pandas.DataFrame | iterable
                        SWC string, URL, filename, folder or DataFrame.
                        If folder, will import all ``.swc`` files.
    connector_labels :  dict, optional
                        If provided will extract connectors from SWC.
                        Dictionary must map type to label:
                        ``{'presynapse': 7, 'postsynapse': 8}``
    include_subdirs :   bool, optional
                        If True and ``f`` is a folder, will also search
                        subdirectories for ``.swc`` files.
    parallel :          "auto" | bool | int,
                        Defaults to ``auto`` which means only use parallel
                        processing if more than 200 SWC are imported. Spawning
                        and joining processes causes overhead and is
                        considerably slower for imports of small numbers of
                        neurons. Integer will be interpreted as the
                        number of cores (otherwise defaults to
                        ``os.cpu_count() - 2``).

    **kwargs
                        Keyword arguments passed to the construction of
                        ``navis.TreeNeuron``. You can use this to e.g. set
                        meta data.

    Returns
    -------
    navis.TreeNeuron
                        Contains SWC file header as ``.swc_header`` attribute.
    navis.NeuronList
                        If import of multiple SWCs will return NeuronList of
                        TreeNeurons.

    See Also
    --------
    :func:`navis.to_swc`
                        Export neurons as SWC files.

    """
    # If is directory, compile list of filenames
    if isinstance(f, str) and os.path.isdir(f):
        if not include_subdirs:
            f = [os.path.join(f, x) for x in os.listdir(f) if
                 os.path.isfile(os.path.join(f, x)) and x.endswith('.swc')]
        else:
            f = [y for x in os.walk(f) for y in glob(os.path.join(x[0], '*.swc'))]

    if utils.is_iterable(f):
        # Do not use if there is only a small batch to import
        if isinstance(parallel, str) and parallel.lower() == 'auto':
            if len(f) < 200:
                parallel = False

        if parallel:
            # Do not swap this as ``isinstance(True, int)`` returns ``True``
            if isinstance(parallel, (bool, str)):
                n_cores = os.cpu_count() - 2
            else:
                n_cores = int(parallel)

            with mp.Pool(processes=n_cores) as pool:
                results = pool.imap(_worker_wrapper, [dict(f=x,
                                                           connector_labels=connector_labels,
                                                           include_subdirs=include_subdirs,
                                                           parallel=False) for x in f],
                                    chunksize=1)
                nl = list(config.tqdm(results,
                                      desc='Importing',
                                      total=len(f),
                                      disable=config.pbar_hide,
                                      leave=config.pbar_leave))

                return core.NeuronList(nl)

        # If not parallel just import the good 'ole way: sequentially
        return core.NeuronList([from_swc(x,
                                         connector_labels=connector_labels,
                                         include_subdirs=include_subdirs,
                                         parallel=parallel,
                                         **kwargs)
                                for x in config.tqdm(f, desc='Importing',
                                                     disable=config.pbar_hide,
                                                     leave=config.pbar_leave)])

    header = []
    attributes = dict(created_at=str(datetime.datetime.now()),
                      connector_labels=connector_labels,
                      soma_label=soma_label)

    if isinstance(f, pd.DataFrame):
        nodes = f
        # Generic name and origin
        attributes['name'] = 'SWC'
        attributes['origin'] = 'DataFrame'
    elif isinstance(f, str):
        try:
            # If file, open it
            if os.path.isfile(f):
                file = open(f, mode='r')
                attributes['name'] = os.path.basename(f).split('.')[0]
                attributes['origin'] = f
            # Check if is url
            elif utils.is_url(f):
                # Fetch data
                r = requests.get(f)
                r.raise_for_status()
                # Decode and turn into a streamable object
                file = io.StringIO(r.content.decode())
                attributes['name'] = f.split('/')[-1]
                attributes['origin'] = f
            else:
                # Assume it's already a SWC string
                file = io.StringIO(f)
                attributes['name'] = 'SWC'
                attributes['origin'] = 'string'

            # Parse header for safekeeping
            line = file.readline()
            while line.startswith('#'):
                header.append(line)
                line = file.readline()

            # Seek back to beginning
            _ = file.seek(0)

            # Load into pandas DataFrame
            nodes = pd.read_csv(file,
                                delimiter=' ',
                                skipinitialspace=True,
                                skiprows=len(header),
                                header=None)
            nodes.columns = ['node_id', 'label', 'x', 'y', 'z',
                             'radius', 'parent_id']
        except BaseException:
            raise
        finally:
            # Make sure we close the stream
            _ = file.close()

        if nodes.empty:
            raise ValueError('No data found in SWC.')
    else:
        raise TypeError('"f" must be filename, SWC string or DataFrame, not '
                        f'{type(f)}')

    # Turn header back into single string
    attributes['swc_header'] = '\n'.join(header)

    # If any invalid nodes are found
    if np.any(nodes[['node_id', 'parent_id', 'x', 'y', 'z']].isnull()):
        # Remove nodes with missing data
        nodes = nodes.loc[~nodes[['node_id', 'parent_id', 'x', 'y', 'z']].isnull().any(axis=1)]

        # Because we removed nodes, we'll have to run a more complicated root
        # detection
        nodes.loc[~nodes.parent_id.isin(nodes.node_id), 'parent_id'] = -1

    # Convert data to respective dtypes
    dtypes = {'node_id': int, 'parent_id': int, 'label': str,
              'x': float, 'y': float, 'z': float, 'radius': float}

    for k, v in dtypes.items():
        if isinstance(v, type):
            nodes[k] = nodes[k].astype(v, errors='ignore')
        else:
            nodes[k] = nodes[k].map(v)
            nodes[k] = nodes[k].astype(object)

    # Take care of connectors
    if connector_labels:
        connectors = pd.DataFrame([], columns=['node_id', 'connector_id',
                                               'type', 'x', 'y', 'z'],
                                  dtype=object)
        for t, l in connector_labels.items():
            cn = nodes[nodes.label == l][['node_id', 'x', 'y', 'z']].copy()
            cn['connector_id'] = None
            cn['type'] = t
            connectors = pd.concat([connectors, cn], axis=0)
    else:
        connectors = None

    # Make sure kwargs override attributes
    _ = attributes.update(kwargs)

    n = core.TreeNeuron(nodes,
                        connectors=connectors,
                        **attributes)

    return n


def to_swc(x: 'core.NeuronObject',
           filename: Optional[str] = None,
           header: Optional[str] = None,
           labels: Union[str, dict, bool] = True,
           export_connectors: bool = False) -> None:
    """Generate SWC file from neuron(s).

    Follows the format specified
    `here <http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html>`_.

    Parameters
    ----------
    x :                 TreeNeuron | NeuronList
                        If multiple neurons, will generate a single SWC file
                        for each neuron (see also ``filename``).
    filename :          None | str | list, optional
                        If ``None``, will use "neuron_{skeletonID}.swc". Pass
                        filenames as list when processing multiple neurons.
    header :            str | None, optional
                        Header for SWC file. If not provided, will use
                        generic header.
    labels :            str | dict | bool, optional
                        Node labels. Can be::

                            str : column name in node table
                            dict: must be of format {node_id: 'label', ...}.
                            bool: if True, will generate automatic labels, if False all nodes have label "0".

    export_connectors : bool, optional
                        If True, will label nodes with pre- ("7") and
                        postsynapse ("8"). Because only one label can be given
                        this might drop synapses (i.e. in case of multiple
                        pre- or postsynapses on a single node)! ``labels``
                        must be ``True`` for this to have any effect.

    Returns
    -------
    Nothing

    See Also
    --------
    :func:`navis.from_swc`
                        Import SWC files.

    """
    if isinstance(x, core.NeuronList):
        if not utils.is_iterable(filename):
            filename = [filename] * len(x)

        # At this point filename is iterable
        filename: Iterable[str]
        for n, f in zip(x, filename):
            to_swc(n, filename=f, labels=labels, header=header,
                   export_synapses=export_connectors)
        return

    if not isinstance(x, core.TreeNeuron):
        raise ValueError(f'Expected TreeNeuron(s), got "{type(x)}"')

    # If not specified, generate generic filename
    if isinstance(filename, type(None)):
        filename = f'neuron_{x.id}.swc'

    # Check if filename is of correct type
    if not isinstance(filename, str):
        raise ValueError(f'Filename must be str or None, got "{type(filename)}"')

    # Make sure file ending is correct
    if os.path.isdir(filename):
        filename += f'neuron_{x.id}.swc'
    elif not filename.endswith('.swc'):
        filename += '.swc'

    # Generate SWC table
    swc = make_swc_table(x, labels=labels, export_connectors=export_connectors)

    # Generate header if not provided
    if not isinstance(header, str):
        header = dedent(f"""\
        # SWC format file
        # based on specifications at http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html
        # Created on {datetime.date.today()} using navis (https://github.com/schlegelp/navis)
        # PointNo Label X Y Z Radius Parent
        # Labels:
        # 0 = undefined, 1 = soma, 5 = fork point, 6 = end point
        """)
        if export_connectors:
            header += dedent("""\
            # 7 = presynapses, 8 = postsynapses
            """)

    with open(filename, 'w') as file:
        # Write header
        file.write(header)

        # Write data
        writer = csv.writer(file, delimiter=' ')
        writer.writerows(swc.astype(str).values)


def make_swc_table(x: 'core.TreeNeuron',
                   labels: Union[str, dict, bool] = None,
                   export_connectors: bool = False,
                   leave_original_id: bool = False) -> pd.DataFrame:
    """Generate a node table compliant with the SWC format.

    Follows the format specified
    `here <http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html>`_.

    Parameters
    ----------
    x :                 TreeNeuron
    labels :            str | dict | bool, optional
                        Node labels. Can be::

                        str : column name in node table
                        dict: must be of format {node_id: 'label', ...}.
                        bool: if True, will generate automatic labels, if False all nodes have label "0".

    export_connectors : bool, optional
                        If True, will label nodes with pre- ("7") and
                        postsynapse ("8"). Because only one label can be given
                        this might drop synapses (i.e. in case of multiple
                        pre- or postsynapses on a single node)! ``labels``
                        must be ``True`` for this to have any effect.
    leave_original_id : bool, optional
                        If True, will keep the original node IDs as index.

    Returns
    -------
    pandas.DataFrame

    """
    # Make copy of nodes and reorder such that the parent comes always before
    # its child(s)
    nodes_ordered = [n for seg in x.segments for n in seg[::-1]]
    this_tn = x.nodes.set_index('node_id', inplace=False).loc[nodes_ordered]

    # Because the last node ID of each segment is a duplicate
    # (except for the first segment ), we have to remove these
    this_tn = this_tn[~this_tn.index.duplicated(keep='first')]

    # Add an index column (must start with "1", not "0")
    this_tn['index'] = list(range(1, this_tn.shape[0] + 1))

    # Make a dictionary node_id -> index
    tn2ix = this_tn['index'].to_dict()

    # Make parent index column
    this_tn['parent_ix'] = this_tn.parent_id.map(lambda x: tn2ix.get(x, -1))

    # Add labels
    if isinstance(labels, dict):
        this_tn['label'] = this_tn.index.map(labels)
    elif isinstance(labels, str):
        this_tn['label'] = this_tn[labels]
    else:
        this_tn['label'] = 0
        # Add end/branch labels
        this_tn.loc[this_tn.type == 'branch', 'label'] = 5
        this_tn.loc[this_tn.type == 'end', 'label'] = 6
        # Add soma label
        if x.soma:
            this_tn.loc[x.soma, 'label'] = 1
        if export_connectors:
            # Add synapse label
            this_tn.loc[x.presynapses.node_id.values, 'label'] = 7
            this_tn.loc[x.postsynapses.node_id.values, 'label'] = 8

    # Generate table consisting of PointNo Label X Y Z Radius Parent
    # .copy() is to prevent pandas' chaining warnings
    swc = this_tn[['index', 'label', 'x', 'y', 'z',
                   'radius', 'parent_ix']].copy()

    # Adjust column titles
    swc.columns = ['PointNo', 'Label', 'X', 'Y', 'Z', 'Radius', 'Parent']

    # Drop index
    if not leave_original_id:
        swc.reset_index(inplace=True, drop=True)

    return swc


def to_float(x: Any) -> Optional[float]:
    """Try to convert to float."""
    try:
        return float(x)
    except BaseException:
        return None


def to_int(x: Any) -> Optional[int]:
    """Try to convert to int."""
    try:
        return int(x)
    except BaseException:
        return None

def _worker_wrapper(kwargs):
    """Helper for importing SWCs using multiple processes."""
    return from_swc(**kwargs)
