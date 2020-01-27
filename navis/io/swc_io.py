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
import requests

import pandas as pd
import numpy as np

from glob import glob

from typing import Union, Iterable, Dict, Optional, Any

from .. import config, utils, core

# Set up logging
logger = config.logger


def from_swc(f: Union[str, pd.DataFrame, Iterable],
             connector_labels: Optional[Dict[str, Union[str, int]]] = {},
             soma_label: Union[str, int] = 1,
             include_subdirs: bool = False,
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

    **kwargs
                        Keyword arguments passed to ``navis.TreeNeuron``

    Returns
    -------
    navis.TreeNeuron
                        Contains SWC file header as ``.swc_header`` attribute.
    navis.NeuronList
                        If import of multiple SWCs, will return NeuronList of
                        TreeNeurons.

    See Also
    --------
    :func:`navis.to_swc`
                        Export neurons as SWC files.

    """
    if utils.is_iterable(f):
        return core.NeuronList([from_swc(x,
                                         connector_labels=connector_labels,
                                         include_subdirs=include_subdirs,
                                         **kwargs)
                                for x in config.tqdm(f, desc='Importing',
                                                     disable=config.pbar_hide,
                                                     leave=config.pbar_leave)])

    header = []
    if isinstance(f, pd.DataFrame):
        nodes = f
        f = 'SWC'
    elif isinstance(f, str) and os.path.isdir(f):
        if not include_subdirs:
            swc = [os.path.join(f, x) for x in os.listdir(f) if
                   os.path.isfile(os.path.join(f, x)) and x.endswith('.swc')]
        else:
            swc = [y for x in os.walk(f) for y in glob(os.path.join(x[0], '*.swc'))]

        return core.NeuronList([from_swc(x,
                                         connector_labels=connector_labels)
                                for x in config.tqdm(swc,
                                                     desc='Reading {}'.format(f.split('/')[-1]),
                                                     disable=config.pbar_hide,
                                                     leave=config.pbar_leave)])
    elif isinstance(f, str):
        data = []
        if os.path.isfile(f):
            with open(f) as file:
                reader = csv.reader(file, delimiter=' ')
                for row in reader:
                    # skip empty rows
                    if not row:
                        continue
                    # skip comments
                    if not row[0].startswith('#'):
                        data.append(row)
                    else:
                        header.append(' '.join(row))

        # If not file, assume it's a SWC string or a URL
        else:
            # Check if is url
            if utils.is_url(f):
                r = requests.get(f)
                r.raise_for_status()

                f = r.content.decode()

            # Note that with .split(), the last row will be empty
            rows = f.split('\n')[:-1]
            for r in rows:
                if not r.startswith('#'):
                    data.append(r.split(' '))
                else:
                    header.append(r)

            # Change f to generic name so that we can use it as name
            f = 'SWC'

        if not data:
            raise ValueError('No data found in SWC.')

        # Remove empty entries and generate nodes DataFrame
        nodes = pd.DataFrame([[e for e in row if e != ''] for row in data],
                             columns=['node_id', 'label', 'x', 'y', 'z',
                                      'radius', 'parent_id'],
                             dtype=object)
    else:
        raise TypeError('"f" must be filename, SWC string or DataFrame, not '
                        f'{type(f)}')

    # Turn header back into single string
    header = '\n'.join(header)

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

    n = core.TreeNeuron(nodes,
                        connectors=connectors,
                        name=kwargs.pop('name',
                                        os.path.basename(f).replace('.swc', '')),
                        filename=os.path.basename(f),
                        pathname=os.path.dirname(f),
                        file=f,
                        header=header,
                        soma_label=soma_label,
                        connector_labels=connector_labels,
                        created_at=str(datetime.datetime.now()),
                        **kwargs)

    return n


def to_swc(x: 'core.NeuronObject',
           filename: Optional[str] = None,
           header: Optional[str] = None,
           labels: bool = True,
           export_synapses: bool = False) -> None:
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
            to_swc(n, f,
                   export_synapses=export_synapses)
        return

    if not isinstance(x, core.TreeNeuron):
        raise ValueError('Can only process TreeNeurons, '
                         'got "{}"'.format(type(x)))

    # If not specified, generate generic filename
    if isinstance(filename, type(None)):
        filename = 'neuron_{}.swc'.format(x.skeleton_id)

    # Check if filename is of correct type
    if not isinstance(filename, str):
        raise ValueError('Filename must be str or None, '
                         'got "{}"'.format(type(filename)))

    # Make sure file ending is correct
    if os.path.isdir(filename):
        filename += 'neuron_{}.swc'.format(x.skeleton_id)
    elif not filename.endswith('.swc'):
        filename += '.swc'

    # Make copy of nodes and reorder such that the parent is always before a
    # node
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
        if export_synapses:
            # Add synapse label
            this_tn.loc[x.presynapses.node_id.values, 'label'] = 7
            this_tn.loc[x.postsynapses.node_id.values, 'label'] = 8

    # Generate table consisting of PointNo Label X Y Z Radius Parent
    # .copy() is to prevent pandas' chaining warnings
    swc = this_tn[['index', 'label', 'x', 'y', 'z',
                   'radius', 'parent_ix']].copy()

    # Adjust column titles
    swc.columns = ['PointNo', 'Label', 'X', 'Y', 'Z', 'Radius', 'Parent']

    with open(filename, 'w') as file:
        # Write header
        if not isinstance(header, str):
            file.write('# SWC format file\n')
            file.write('# based on specifications at http://research.mssm.edu/cnic/swc.html\n')
            file.write('# Created on {} using navis (https://github.com/schlegelp/navis)\n'.format(str(datetime.date.today())))
            file.write('# PointNo Label X Y Z Radius Parent\n')
            file.write('# Labels:\n')
            for l in ['0 = undefined', '1 = soma', '5 = fork point', '6 = end point']:
                file.write('# {}\n'.format(l))
            if export_synapses:
                for l in ['7 = presynapse', '8 = postsynapse']:
                    file.write('# {}\n'.format(l))
        else:
            file.write(header)

        writer = csv.writer(file, delimiter=' ')
        writer.writerows(swc.astype(str).values)


def to_float(x: Any) -> Optional[float]:
    """ Helper to try to convert to float.
    """
    try:
        return float(x)
    except BaseException:
        return None


def to_int(x: Any) -> Optional[int]:
    """ Helper to try to convert to float.
    """
    try:
        return int(x)
    except BaseException:
        return None
