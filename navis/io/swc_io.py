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
from functools import partial
import os
import io
import requests
from pathlib import Path
from textwrap import dedent
import multiprocessing as mp
from typing import List, Type, Union, Iterable, Dict, Optional, Any, TextIO, IO

import pandas as pd
import numpy as np

from .. import config, utils, core

__all__ = ["SwcReader", "read_swc", "write_swc"]

# Set up logging
logger = config.logger

NODE_COLUMNS = ('node_id', 'label', 'x', 'y', 'z', 'radius', 'parent_id')
COMMENT = "#"
DEFAULT_DELIMITER = " "
DEFAULT_PRECISION = 32
DEFAULT_INCLUDE_SUBDIRS = False
AUTO_PARALLEL = "auto"


def merge_dicts(*dicts: Optional[Dict], **kwargs) -> Dict:
    """Merge dicts and kwargs left to right.

    Ignores None arguments.
    """
    # handles nones
    out = dict()
    for d in dicts:
        if d:
            out.update(d)
    out.update(kwargs)
    return out


def swcs_in_dir(
    dpath: Path, include_subdirs: bool = DEFAULT_INCLUDE_SUBDIRS
) -> Iterable[Path]:
    pattern = '*.swc'
    if include_subdirs:
        pattern = os.path.join("**", pattern)
    yield from dpath.glob(pattern)


class SwcReader:
    def __init__(
        self,
        connector_labels: Optional[Dict[str, Union[str, int]]] = None,
        soma_label: Union[str, int] = 1,
        delimiter: str = DEFAULT_DELIMITER,
        precision: int = DEFAULT_PRECISION,
        attrs: Optional[Dict[str, Any]] = None
    ):
        self.connector_labels = connector_labels or dict()
        self.soma_label = soma_label
        self.delimiter = delimiter
        self.attrs = attrs

        int_, float_ = parse_precision(precision)
        self._dtypes = {
            'node_id': int_,
            'parent_id': int_,
            'label': 'category',
            'x': float_,
            'y': float_,
            'z': float_,
            'radius': float_,
        }

    def _make_attributes(
        self, *dicts: Optional[Dict[str, Any]], **kwargs
    ) -> Dict[str, Any]:
        """Combine given attributes with a timestamp
        and those defined on the object.

        Later additions take precedence:

        - created_at (now), connector_labels (from object), soma_label (from object)
        - object-defined attributes
        - additional dicts given as *args
        - additional attributes given as **kwargs

        Returns
        -------
        Dict[str, Any]
            Arbitrary string-keyed attributes.
        """
        return merge_dicts(
            dict(
                created_at=str(datetime.datetime.now()),
                connector_labels=self.connector_labels,
                soma_label=self.soma_label,
            ),
            self.attrs,
            *dicts,
            **kwargs,
        )

    def read_buffer(
        self, f: IO, attrs: Optional[Dict[str, Any]] = None
    ) -> 'core.TreeNeuron':
        """Read buffer into a TreeNeuron.

        Parameters
        ----------
        f : IO
            Readable buffer (if bytes, interpreted as utf-8)
        attrs : dict | None
            Arbitrary attributes to include in the TreeNeuron

        Returns
        -------
        core.TreeNeuron
        """
        if isinstance(f.read(0), bytes):
            f = io.TextIOWrapper(f, encoding="utf-8")

        header_rows = read_header_rows(f)
        nodes = pd.read_csv(
            f,
            delimiter=self.delimiter,
            skipinitialspace=True,
            skiprows=len(header_rows),
            comment=COMMENT,
            header=None,
        )
        nodes.columns = NODE_COLUMNS
        return self.read_dataframe(nodes, merge_dicts({'swc_header': '\n'.join(header_rows)}, attrs))

    def read_file_path(
        self, fpath: os.PathLike, attrs: Optional[Dict[str, Any]] = None
    ) -> 'core.TreeNeuron':
        """Read single SWC file from path into a TreeNeuron.

        Parameters
        ----------
        fpath : str | os.PathLike
            Path to SWC file
        attrs : dict or None
            Arbitrary attributes to include in the TreeNeuron

        Returns
        -------
        core.TreeNeuron
        """
        p = Path(fpath)
        with open(p, "r") as f:
            return self.read_buffer(
                f, merge_dicts({"name": p.stem, "origin": str(p)}, attrs)
            )

    def read_directory(
        self, path: os.PathLike,
        include_subdirs=DEFAULT_INCLUDE_SUBDIRS,
        parallel="auto",
        attrs: Optional[Dict[str, Any]] = None
    ) -> 'core.NeuronList':
        """Read directory of SWC files into a NeuronList.

        Parameters
        ----------
        fpath : str | os.PathLike
            Path to directory containing SWC files
        include_subdirs : bool, optional
            Whether to descend into subdirectories, default False
        parallel : str | bool |
        attrs : dict or None
            Arbitrary attributes to include in the TreeNeurons of the NeuronList

        Returns
        -------
        core.NeuronList
        """
        swcs = list(swcs_in_dir(Path(path), include_subdirs))
        read_fn = partial(self.read_file_path, attrs=attrs)
        neurons = parallel_read(read_fn, swcs, parallel)
        return core.NeuronList(neurons)

    def read_url(
        self, url: str, attrs: Optional[Dict[str, Any]] = None
    ) -> 'core.TreeNeuron':
        """Read SWC file from URL into a TreeNeuron.

        Parameters
        ----------
        url : str
            URL to SWC file
        attrs : dict or None
            Arbitrary attributes to include in the TreeNeuron

        Returns
        -------
        core.TreeNeuron
        """
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            return self.read_buffer(
                r.raw,
                merge_dicts({'name': url.split('/')[1], 'origin': url}, attrs)
            )

    def read_string(
        self, s: str, attrs: Optional[Dict[str, Any]] = None
    ) -> 'core.TreeNeuron':
        """Read single SWC-like string into a TreeNeuron.

        Parameters
        ----------
        s : str
            SWC string
        attrs : dict or None
            Arbitrary attributes to include in the TreeNeuron

        Returns
        -------
        core.TreeNeuron
        """
        sio = io.StringIO(s)
        return self.read_buffer(
            sio,
            merge_dicts({'name': 'SWC', 'origin': 'string'}, attrs)
        )

    def read_dataframe(
        self, nodes: pd.DataFrame, attrs: Optional[Dict[str, Any]] = None
    ) -> 'core.TreeNeuron':
        """Convert a SWC-like DataFrame into a TreeNeuron.

        Parameters
        ----------
        nodes : pandas.DataFrame
        attrs : dict or None
            Arbitrary attributes to include in the TreeNeuron

        Returns
        -------
        core.TreeNeuron
        """
        return core.TreeNeuron(
            sanitise_nodes(
                nodes.astype(self._dtypes, errors='ignore', copy=False)
            ),
            connectors=self._extract_connectors(nodes),
            **(self._make_attributes({'name': 'SWC', 'origin': 'DataFrame'}, attrs))
        )

    def read_any_single(
        self, obj, attrs: Optional[Dict[str, Any]] = None
    ) -> 'core.TreeNeuron':
        """Attempt to convert an arbitrary object into a TreeNeuron.

        Parameters
        ----------
        obj : typing.IO | pandas.DataFrame | str | os.PathLike
            Readable buffer, dataframe, path or URL to single file,
            or SWC string
        attrs : dict or None
            Arbitrary attributes to include in the TreeNeuron

        Returns
        -------
        core.TreeNeuron
        """
        if hasattr(obj, "read"):
            return self.read_buffer(obj, attrs)
        if isinstance(obj, pd.DataFrame):
            return self.read_dataframe(obj, attrs)
        if isinstance(obj, os.PathLike):
            return self.read_file_path(obj, attrs)
        if isinstance(obj, str):
            if os.path.isfile(obj):
                return self.read_file_path(Path(obj), attrs)
            if obj.startswith("http://") or obj.startswith("https://"):
                return self.read_url(obj, attrs)
            return self.read_string(obj, attrs)
        raise ValueError(
            f"Could not read SWC from object of type '{type(obj)}'"
        )

    def read_any_multi(
        self,
        objs,
        include_subdirs=DEFAULT_INCLUDE_SUBDIRS,
        parallel="auto",
        attrs: Optional[Dict[str, Any]] = None,
    ) -> 'core.NeuronList':
        """Attempt to convert an arbitrary object into a NeuronList,
        potentially in parallel.

        Parameters
        ----------
        obj : sequence
            Sequence of anything readable by read_any_single
            or directory path(s)
        include_subdirs : bool
            Whether to include subdirectories of a given directory
        parallel : str | bool | int | None
            "auto" or True for n_cores - 2, otherwise int for number of jobs,
            or falsey for serial.
        attrs : dict or None
            Arbitrary attributes to include in each TreeNeuron of the NeuronList

        Returns
        -------
        core.NeuronList
        """
        if not utils.is_iterable(objs):
            objs = [objs]

        if not objs:
            logger.warning("No SWC found, returning empty NeuronList")
            return core.NeuronList([])

        new_objs = []
        for obj in objs:
            try:
                if os.path.isdir(obj):
                    new_objs.extend(swcs_in_dir(obj, include_subdirs))
                    continue
            except TypeError:
                pass
            new_objs.append(obj)

        if (
            isinstance(parallel, str)
            and parallel.lower() == 'auto'
            and len(new_objs) < 200
        ):
            parallel = False

        read_fn = partial(self.read_any_single, attrs=attrs)
        neurons = parallel_read(read_fn, new_objs, parallel)
        return core.NeuronList(neurons)

    def read_any(
        self,
        obj,
        include_subdirs=DEFAULT_INCLUDE_SUBDIRS,
        parallel="auto",
        attrs: Optional[Dict[str, Any]] = None
    ) -> 'core.NeuronObject':
        """Attempt to read an arbitrary object into a TreeNeuron or NeuronObject.

        Parameters
        ----------
        obj : typing.IO | str | os.PathLike | pandas.DataFrame
            Buffer, path to file or directory, URL, SWC string, or dataframe.

        Returns
        -------
        core.TreeNeuron | core.NeuronObject
        """
        if utils.is_iterable(obj) and not hasattr(obj, "read"):
            return self.read_any_multi(obj, parallel, include_subdirs, attrs)
        else:
            try:
                if os.path.isdir(obj):
                    return self.read_directory(
                        obj, include_subdirs, parallel, attrs
                    )
            except TypeError:
                pass
            return self.read_any_single(obj, attrs)

    def _extract_connectors(
        self, nodes: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """Infer outgoing/incoming connectors from node labels.

        Parameters
        ----------
        nodes : pd.DataFrame

        Returns
        -------
        Optional[pd.DataFrame]
            With columns ``["node_id", "x", "y", "z", "connector_id", "type"]``
        """
        if not self.connector_labels:
            return None

        to_concat = [
            pd.DataFrame(
                [], columns=['node_id', 'connector_id', 'type', 'x', 'y', 'z']
            )
        ]
        for name, val in self.connector_labels.items():
            cn = nodes[nodes.label == val][['node_id', 'x', 'y', 'z']].copy()
            cn['connector_id'] = None
            cn['type'] = name
            to_concat.append(cn)

        return pd.concat(to_concat, axis=0)


def parallel_read(read_fn, objs, parallel="auto") -> List['core.NeuronList']:
    """Get SWCs from some objects with the given reader function,
    potentially in parallel.

    Reader function must be picklable.

    Parameters
    ----------
    read_fn : Callable
    objs : Iterable
    parallel : str | bool | int
        "auto" or True for n_cores - 2, otherwise int for number of jobs, or falsey for serial.

    Returns
    -------
    core.NeuronList
    """
    try:
        length = len(objs)
    except TypeError:
        length = None

    prog = partial(
        config.tqdm,
        desc='Importing',
        total=length,
        disable=config.pbar_hide,
        leave=config.pbar_leave
    )

    if parallel:
        # Do not swap this as ``isinstance(True, int)`` returns ``True``
        if isinstance(parallel, (bool, str)):
            n_cores = max(1, os.cpu_count() - 2)
        else:
            n_cores = int(parallel)

        with mp.Pool(processes=n_cores) as pool:
            results = pool.imap(read_fn, objs)
            neurons = list(prog(results))
    else:
        neurons = [read_fn(obj) for obj in prog(objs)]

    return neurons


def parse_precision(precision: Optional[int]):
    """Convert bit width into int and float dtypes.

    Parameters
    ----------
    precision : int
        16, 32, 64, or None

    Returns
    -------
    tuple
        Integer numpy dtype, float numpy dtype

    Raises
    ------
    ValueError
        Unknown precision.
    """
    INT_DTYPES = {16: np.int16, 32: np.int32, 64: np.int64, None: None}
    FLOAT_DTYPES = {16: np.float16, 32: np.float32, 64: np.float64, None: None}

    try:
        return (INT_DTYPES[precision], FLOAT_DTYPES[precision])
    except KeyError:
        raise ValueError(
            f'Unknown precision {precision}. Expected on of the following: 16, 32 (default), 64 or None'
        )


def sanitise_nodes(nodes: pd.DataFrame) -> pd.DataFrame:
    """
    Check that nodes dataframe is non-empty and is not missing any data.

    Parameters
    ----------
    nodes : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
    """
    if nodes.empty:
        raise ValueError('No data found in SWC.')

    is_na = nodes[['node_id', 'parent_id', 'x', 'y', 'z']].isna().any(axis=1)

    if is_na.any():
        # Remove nodes with missing data
        nodes = nodes.loc[~is_na.any(axis=1)]

        # Because we removed nodes, we'll have to run a more complicated root
        # detection
        nodes.loc[~nodes.parent_id.isin(nodes.node_id), 'parent_id'] = -1

    return nodes


def read_header_rows(f: TextIO):
    f"""Read {COMMENT}-prefixed lines from the start of a buffer,
    then seek back to the start of the buffer.

    Parameters
    ----------
    f : io.TextIO

    Returns
    -------
    list : List of strings
    """
    out = []
    for line in f:
        if not line.startswith(COMMENT):
            break
        out.append(line)

    f.seek(0)
    return out


def read_swc(f: Union[str, pd.DataFrame, Iterable],
             connector_labels: Optional[Dict[str, Union[str, int]]] = {},
             soma_label: Union[str, int] = 1,
             include_subdirs: bool = False,
             delimiter: str = ' ',
             parallel: Union[bool, int] = 'auto',
             precision: int = 32,
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
    delimiter :         str
                        Delimiter to use. Passed to ``pandas.read_csv``.
    parallel :          "auto" | bool | int
                        Defaults to ``auto`` which means only use parallel
                        processing if more than 200 SWC are imported. Spawning
                        and joining processes causes overhead and is
                        considerably slower for imports of small numbers of
                        neurons. Integer will be interpreted as the
                        number of cores (otherwise defaults to
                        ``os.cpu_count() - 2``).
    precision :         int [8, 16, 32, 64] | None
                        Precision for data. Defaults to 32 bit integers/floats.
                        If ``None`` will let pandas infer data types - this
                        typically leads to higher than necessary precision.
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
    :func:`navis.write_swc`
                        Export neurons as SWC files.

    """
    reader = SwcReader(
        connector_labels, soma_label, delimiter, precision, kwargs
    )
    return reader.read_any(f, include_subdirs, parallel)


def write_swc(x: 'core.NeuronObject',
              filepath: Union[str, Path],
              header: Optional[str] = None,
              labels: Union[str, dict, bool] = True,
              export_connectors: bool = False,
              return_node_map : bool = False) -> None:
    """Generate SWC file from neuron(s).

    Follows the format specified
    `here <http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html>`_.

    Parameters
    ----------
    x :                 TreeNeuron | NeuronList
                        If multiple neurons, will generate a single SWC file
                        for each neuron (see also ``filepath``).
    filepath :          str | pathlib.Path | list thereof
                        Destination for the SWC files. See examples for options.
                        If ``x`` is multiple neurons, ``filepath`` must either
                        be a folder, a "formattable" filename or a list of
                        filenames for each neuron in x.
    header :            str | None, optionals
                        Header for SWC file. If not provided, will use generic
                        header.
    labels :            str | dict | bool, optional
                        Node labels. Can be::

                            str : column name in node table
                            dict: must be of format {node_id: 'label', ...}.
                            bool: if True, will generate automatic labels, if False all nodes have label "0".

    export_connectors : bool, optional
                        If True, will label nodes with pre- ("7") and
                        postsynapse ("8"). Because only one label can be given
                        this might drop synapses (i.e. in case of multiple
                        pre- or postsynapses on a single node)! ``labels`` must
                        be ``True`` for this to have any effect.
    return_node_map :   bool
                        If True, will return a dictionary mapping the old node
                        ID to the new reindexed node IDs in the file.

    Returns
    -------
    node_map :          dict
                        Only if ``return_node_map=True``.

    Examples
    --------
    Save a single neuron to a specific file:

    >>> import navis
    >>> n = navis.example_neurons(1, kind='skeleton')
    >>> navis.write_swc(n, tmp_dir / 'my_neuron.swc')

    Save two neurons to specific files:

    >>> import navis
    >>> nl = navis.example_neurons(2, kind='skeleton')
    >>> navis.write_swc(nl, [tmp_dir / 'my_neuron1.swc', tmp_dir / 'my_neuron2.swc'])

    Save multiple neurons to a folder (must exist). Filenames will be
    autogenerated as "{neuron.id}.swc":

    >>> import navis
    >>> nl = navis.example_neurons(5, kind='skeleton')
    >>> navis.write_swc(nl, tmp_dir)

    Save multiple neurons to a folder but modify the pattern for the
    autogenerated filenames:

    >>> import navis
    >>> nl = navis.example_neurons(5, kind='skeleton')
    >>> navis.write_swc(nl, tmp_dir / 'skel-{neuron.name}.swc')

    See Also
    --------
    :func:`navis.read_swc`
                        Import skeleton from SWC files.

    """
    if isinstance(x, core.NeuronList):
        if not utils.is_iterable(filepath):
            # Assume this is a folder if it doesn't end with '.swc'
            is_filename = str(filepath).endswith('.swc')
            is_single = len(x) == 1
            is_formattable = "{" in str(filepath) and "}" in str(filepath)
            if not is_filename or is_single or is_formattable:
                filepath = [filepath] * len(x)
            else:
                raise ValueError('`filepath` must either be a folder, a '
                                 'formattable filepath or a list of filepaths'
                                 'when saving multiple neurons.')

        if len(filepath) != len(x):
            raise ValueError(f'Got {len(filepath)} file names for '
                             f'{len(x)} neurons.')

        # At this point filepath is iterable
        filepath: Iterable[str]
        for n, f in config.tqdm(zip(x, filepath), disable=config.pbar_hide,
                                leave=config.pbar_leave, total=len(x),
                                desc='Saving'):
            write_swc(n, filepath=f, labels=labels, header=header,
                      export_connectors=export_connectors)
        return

    if not isinstance(x, core.TreeNeuron):
        raise ValueError(f'Expected TreeNeuron(s), got "{type(x)}"')

    # try to str.format any path-like
    try:
        as_str = os.fspath(filepath)
    except TypeError:
        raise ValueError(f'`filepath` must be str or pathlib.Path, got "{type(filepath)}"')

    formatted_str = as_str.format(neuron=x)

    # if it was formatted, make sure it's a SWC file
    if formatted_str != as_str and not as_str.endswith(".swc"):
        raise ValueError("Formattable filepaths must end with '.swc'")

    filepath = Path(formatted_str)

    # Expand user - otherwise .exists() might fail
    filepath = filepath.expanduser()

    # If not specified, generate filename
    if not str(filepath).endswith('.swc'):
        filepath = filepath / f'{x.id}.swc'

    # Make sure the parent directory exists
    if not filepath.parent.exists():
        raise ValueError(f'Parent folder {filepath.parent} must exist.')

    # Generate SWC table
    res = make_swc_table(x,
                         labels=labels,
                         export_connectors=export_connectors,
                         return_node_map=return_node_map)

    if return_node_map:
        swc, node_map = res[0], res[1]
    else:
        swc = res

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

    with open(filepath, 'w') as file:
        # Write header
        file.write(header)

        # Write data
        writer = csv.writer(file, delimiter=' ')
        writer.writerows(swc.astype(str).values)

    if return_node_map:
        return node_map


def make_swc_table(x: 'core.TreeNeuron',
                   labels: Union[str, dict, bool] = None,
                   export_connectors: bool = False,
                   return_node_map: bool = False) -> pd.DataFrame:
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
    return_node_map :   bool
                        If True, will return a dictionary mapping the old node
                        ID to the new reindexed node IDs in the file.

    Returns
    -------
    SWC table :         pandas.DataFrame
    node map :          dict
                        Only if ``return_node_map=True``.

    """
    # Work on a copy
    swc = x.nodes.copy()

    # Add labels
    swc['label'] = 0
    if isinstance(labels, dict):
        swc['label'] = swc.index.map(labels)
    elif isinstance(labels, str):
        swc['label'] = swc[labels]
    elif labels:
        # Add end/branch labels
        swc.loc[swc.type == 'branch', 'label'] = 5
        swc.loc[swc.type == 'end', 'label'] = 6
        # Add soma label
        if not isinstance(x.soma, type(None)):
            soma = utils.make_iterable(x.soma)
            swc.loc[soma, 'label'] = 1
        if export_connectors:
            # Add synapse label
            swc.loc[x.presynapses.node_id.values, 'label'] = 7
            swc.loc[x.postsynapses.node_id.values, 'label'] = 8

    # Sort such that the parent is always before the child
    swc.sort_values('parent_id', ascending=True, inplace=True)

    # Reset index
    swc.reset_index(drop=True, inplace=True)

    # Generate mapping
    new_ids = dict(zip(swc.node_id.values, swc.index.values))

    swc['node_id'] = swc.node_id.map(new_ids)
    # Lambda prevents potential issue with missing parents
    swc['parent_id'] = swc.parent_id.map(lambda x: new_ids.get(x, -1))

    # Get things in order
    swc = swc[['node_id', 'label', 'x', 'y', 'z', 'radius', 'parent_id']]

    # Adjust column titles
    swc.columns = ['PointNo', 'Label', 'X', 'Y', 'Z', 'Radius', 'Parent']

    if return_node_map:
        return swc, new_ids

    return swc
