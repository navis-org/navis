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

import pandas as pd

from .. import core, config, utils

# Set up logging
logger = config.logger


def from_swc(f, connector_labels={}, **kwargs):
    """ Creates Neuron/List from SWC file.

    This import is following format specified
    `here <http://research.mssm.edu/cnic/swc.html>`_

    Parameters
    ----------
    f :                 str
                        SWC string, filename or folder. If folder, will import
                        all ``.swc`` files.
    connector_labels :  dict, optional
                        If provided will extract connectors from SWC.
                        Dictionary must map type to label:
                        ``{'presynapse': 7, 'postsynapse': 8}``
    **kwargs
                        Keyword arguments passed to ``navis.TreeNeuron``

    Returns
    -------
    navis.TreeNeuron

    See Also
    --------
    :func:`navis.to_swc`
                        Export neurons as SWC files.

    """
    if os.path.isdir(f):
        swc = [os.path.join(f, x) for x in os.listdir(f) if os.path.isfile(os.path.join(f, x)) and x.endswith('.swc')]
        return core.NeuronList([from_swc(x,
                                         connector_labels=connector_labels)
                                         for x in config.tqdm(swc, desc='Importing',
                                                              disable=config.pbar_hide,
                                                              leave=config.pbar_leave)])

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

    # If not file, assume it's a SWC string
    else:
        # Note that with .split(), the last row will be empty
        for row in f.split('\n')[:-1]:
            if not row.startswith('#'):
                data.append(row.split(' '))

        # Change f to generic name so that we can use it as name
        f = 'SWC'

    if not data:
        raise ValueError('No data found in SWC.')

    # Remove empty entries and generate nodes DataFrame
    nodes = pd.DataFrame([[float(e) for e in row if e != ''] for row in data],
                         columns=['node_id', 'label', 'x', 'y', 'z',
                                  'radius', 'parent_id'],
                         dtype=object)

    # Make sure we are using integers
    nodes.parent_id = nodes.parent_id.astype(int, errors='ignore')
    nodes.node_id = nodes.node_id.astype(int, errors='ignore')
    nodes.parent_id = nodes.parent_id.astype(object)
    nodes.node_id = nodes.node_id.astype(object)

    # Root node will have parent=-1 -> set this to None
    nodes.loc[nodes.parent_id < 0, 'parent_id'] = None

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

    return core.TreeNeuron(nodes,
                           connectors=connectors,
                           name=kwargs.pop('name',
                                            os.path.basename(f).replace('.swc',
                                                                        '')),
                           input_file=f,
                           created_at=str(datetime.datetime.now()),
                           **kwargs)


def to_swc(x, filename=None, export_synapses=False):
    """ Generate SWC file from neuron(s).

    Follows the format specified
    `here <http://research.mssm.edu/cnic/swc.html>`_.

    Parameters
    ----------
    x :                 TreeNeuron | NeuronList
                        If multiple neurons, will generate a single SWC file
                        for each neurons (see also ``filename``).
    filename :          None | str | list, optional
                        If ``None``, will use "neuron_{skeletonID}.swc". Pass
                        filenames as list when processing multiple neurons.
    export_connectors : bool, optional
                        If True, will label nodes with pre- ("7") and
                        postsynapse ("8"). Because only one label can be given
                        this might drop synapses (i.e. in case of multiple
                        pre- or postsynapses on a single treenode)!

    Returns
    -------
    Nothing

    See Also
    --------
    :func:`pymaid.from_swc`
                        Import SWC files.

    """
    if isinstance(x, core.NeuronList):
        if not utils.is_iterable(filename):
            filename = [filename] * len(x)

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
    if os.isdir(filename):
        filename += 'neuron_{}.swc'.format(x.skeleton_id)
    elif not filename.endswith('.swc'):
        filename += '.swc'

    # Make copy of nodes and reorder such that the parent is always before a
    # treenode
    nodes_ordered = [n for seg in x.segments for n in seg[::-1]]
    this_tn = x.nodes.set_index('node_id').loc[nodes_ordered]

    # Because the last treenode ID of each segment is a duplicate
    # (except for the first segment ), we have to remove these
    this_tn = this_tn[~this_tn.index.duplicated(keep='first')]

    # Add an index column (must start with "1", not "0")
    this_tn['index'] = list(range(1, this_tn.shape[0] + 1))

    # Make a dictionary node_id -> index
    tn2ix = this_tn['index'].to_dict()
    tn2ix[None] = -1

    # Make parent index column
    this_tn['parent_ix'] = this_tn.parent_id.map(tn2ix)

    # Set Label column to 0 (undefined)
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

    # Coordinates and radius to microns
    swc.loc[:, ['X', 'Y', 'Z', 'Radius']] /= 1000

    with open(filename, 'w') as file:
        # Write header
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
        #file.write('\n')

        writer = csv.writer(file, delimiter=' ')
        writer.writerows(swc.astype(str).values)