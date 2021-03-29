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

import h5py
import os
import pickle
import pint
import warnings

import multiprocessing as mp
import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from typing import Union, Iterable, Dict, Optional, Any

from .. import config, utils, core


class BaseH5Reader(ABC):
    """Reads neurons from HDF5 files."""

    def __init__(self, filepath: str):
        """Initialize."""
        self.filepath = filepath

    def __enter__(self):
        """Open file on enter."""
        self.f = h5py.File(self.filepath, 'r')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close file on exit."""
        self.f.close()

    def check_compatible(self):
        """Check if this reader is compatible with existing data."""
        fmt = self.f.attrs.get('format_spec')
        if not isinstance(fmt, str):
            raise ValueError('No format specifier found for file '
                             f'{self.filepath}')
        elif fmt != f'hnf_v{self.version}':
            raise ValueError('Unexpected format specifier for file '
                             f'{self.filepath}: "{fmt}"')

    @abstractmethod
    def list_neurons(self):
        """List neurons in file."""
        pass

    @abstractmethod
    def read_neurons(self):
        """Read neurons from file."""
        pass

    def read_dataframe(self, group, subset=None, exclude=None,
                       include_attrs=False, skip_hidden=True):
        """Read dataset within a group into a single pandas DataFrame."""
        # List datasets contained in this group
        datasets = [k for k, v in group.items() if isinstance(v, h5py.Dataset)]

        if isinstance(subset, type(None)):
            subset = datasets
        else:
            subset = utils.make_iterable(subset)

        if isinstance(exclude, type(None)):
            exclude = []
        else:
            exclude = utils.make_iterable(exclude)

        # Read the datasets
        df = pd.DataFrame()
        for col in datasets:
            if col not in subset or col in exclude:
                continue
            # Skip hidden datasets
            if col.startswith('.') and skip_hidden:
                continue

            # This is the dataset
            ds = group[col]

            # Easy-peasy if dataset has 1 dimension
            if ds.ndim == 1:
                df[col] = ds[:]
            # 2-dimensions should not happen but we will simply enumerate cols
            elif ds.ndim == 2:
                for i in range(ds.shape[1]):
                    df[f'{col}_{i}'] = ds[:, i]
            else:
                raise ValueError(f'Dataset {col} has more than two dimensions.')

        if include_attrs:
            for k in group.attrs:
                df.attrs[k] = group.attrs[k]

        return df


class H5ReaderV1(BaseH5Reader):
    """Reads neurons from HDF5 files."""

    version = 1

    def list_neurons(self, from_cache=True):
        """List all neurons in file."""
        if from_cache and hasattr(self, 'neurons'):
            return self.neurons

        # Go over all top level groups
        self.neurons = []
        for id, grp in self.f.items():
            # Skip if not a group
            if not isinstance(grp, h5py.Group):
                continue

            self.neurons.append(id)

        return self.neurons

    def read_neurons(self,
                     subset=None,
                     read='mesh->skeleton->dotprops',
                     strict=False,
                     prefer_raw=False,
                     on_error='stop',
                     progress=True,
                     annotations=False,
                     **kwargs):
        """Read neurons from file."""
        assert isinstance(read, str)

        readers = {'mesh': self.read_meshneuron,
                   'skeleton': self.read_treeneuron,
                   'dotprops': self.read_dotprops}

        # If no subset specified, load all neurons
        if isinstance(subset, type(None)):
            subset = self.list_neurons()
        else:
            subset = [id for id in subset if id in self.f]

        neurons = []
        errors = {}

        # We can not simply use disable=True here because of some odd
        # interactions when this is run in multiple processes
        if progress:
            pbar = config.tqdm(desc='Reading',
                               leave=False,
                               disable=config.pbar_hide,
                               total=len(subset))

        try:
            for id in subset:
                # Go over the requested neuron representations
                for rep in read.split(','):
                    # Strip potential whitespaces
                    rep = rep.strip()

                    # Go over priorities
                    for prio in rep.split('->'):
                        # Strip potential whitespaces
                        prio = prio.strip()
                        # If that neuron type is present
                        if prio in self.f[id]:
                            try:
                                # Read neuron
                                read_func = readers[prio]
                                n = read_func(id,
                                              strict=strict,
                                              prefer_raw=prefer_raw,
                                              **kwargs)

                                # Read annotations
                                if not isinstance(annotations, type(None)):
                                    an = self.read_annotations(id, annotations, **kwargs)
                                    for k, v in an.items():
                                        setattr(n, k, v)

                                # Append neuron and break prio loop
                                neurons.append(n)
                            except BaseException as e:
                                errors[id] = str(e)
                                if on_error in ('stop', 'raise'):
                                    raise e
                                elif on_error == 'warn':
                                    warnings.warn(f'Error parsing {prio} for '
                                                  f'neuron {id}: {e}')

                            break

                if progress:
                    pbar.update()
        except BaseException:
            raise
        finally:
            if progress:
                pbar.close()

        return neurons, errors

    def parse_add_attributes(self, grp, base_grp, neuron,
                             subset=None, exclude=['units_nm']):
        """Parse attributes and associate with neuron."""
        if not isinstance(subset, type(None)):
            subset = utils.make_iterable(subset)

        # First parse base (neuron-level) attributes
        attrs = {}
        for k, v in base_grp.attrs.items():
            if not isinstance(subset, type(None)) and k not in subset:
                continue
            elif k in exclude:
                continue
            attrs[k] = v

        # Now parse attributes specific for this representation
        for k, v in grp.attrs.items():
            if not isinstance(subset, type(None)) and k not in subset:
                continue
            elif k in exclude:
                continue
            attrs[k] = v

        # Now add attributes
        for k, v in attrs.items():
            setattr(neuron, k, v)

    def parse_add_datasets(self, grp, neuron, subset=None, exclude=None,
                           skip_hidden=True):
        """Parse datasets and associate with neuron."""
        if not isinstance(subset, type(None)):
            subset = utils.make_iterable(subset)
        if not isinstance(exclude, type(None)):
            exclude = utils.make_iterable(exclude)

        # Now parse attributes specific for this representation
        for k in grp.keys():
            if not isinstance(subset, type(None)) and k not in subset:
                continue
            elif k.startswith('.') and skip_hidden:
                continue
            elif k in exclude:
                continue
            # Do not remove the [:] as it ensures that we get a numpy array
            setattr(neuron, k, grp[k][:])

    def parse_add_units(self, grp, base_grp, neuron):
        """Parse units and associate with neuron."""
        # Check if we have units
        units = grp.attrs.get('units_nm',
                              base_grp.attrs.get('units_nm',
                                                 None))

        if isinstance(units, str):
            # If string, see if pint can parse it
            neuron.units = units
        elif isinstance(units, np.ndarray):
            # Currently navis .units does support x/y/z units
            # We will use only the first entry for now
            neuron.units = f'{units[0]} nm'
        elif not isinstance(units, type(None)):
            neuron.units = f'{units} nm'

    def read_annotations(self, id, annotations, **kwargs):
        """Read annotations for given neuron from file."""
        # Get the group for this neuron
        neuron_grp = self.f[id]

        an_grp = neuron_grp.get('annotations')
        if not an_grp:
            return {}

        if isinstance(annotations, bool):
            annotations = [k for k, grp in an_grp.items() if isinstance(grp, type(h5py.Group))]
        else:
            annotations = utils.make_iterable(annotations)

        parsed_an = {}
        for an in annotations:
            if an in an_grp:
                parsed_an[an] = self.read_dataframe(an_grp[an],
                                                    include_attrs=True)

        return parsed_an

    def read_treeneuron(self, id, strict=False, prefer_raw=False, **kwargs):
        """Read given TreeNeuron from file."""
        # Get the group for this neuron
        neuron_grp = self.f[id]

        # See if this neuron has a skeleton
        if 'skeleton' not in neuron_grp:
            raise ValueError(f'Neuron {id} has no skeleton')
        sk_grp = neuron_grp['skeleton']

        if '.serialized_navis' in sk_grp and not prefer_raw:
            return pickle.loads(sk_grp['.serialized_navis'][()])

        # Parse node table
        nodes = self.read_dataframe(sk_grp,
                                    subset=['node_id', 'parent_id',
                                            'x', 'y', 'z',
                                            'radius'] if strict else None)

        n = core.TreeNeuron(nodes, id=id)

        # Check if we have units
        self.parse_add_units(sk_grp, neuron_grp, n)

        # Parse attributes
        self.parse_add_attributes(sk_grp, neuron_grp, n,
                                  subset=['soma'] if strict else None)

        return n

    def read_dotprops(self, id, strict=False, prefer_raw=False, **kwargs):
        """Read given Dotprops from file."""
        # Get the group for this neuron
        neuron_grp = self.f[id]

        # See if this neuron has dotprops
        if 'dotprops' not in neuron_grp:
            raise ValueError(f'Neuron {id} has no dotprops')
        dp_grp = neuron_grp['dotprops']

        if '.serialized_navis' in dp_grp and not prefer_raw:
            return pickle.loads(dp_grp['.serialized_navis'][()])

        # Parse dotprop arrays
        points = dp_grp['points']
        vect = dp_grp['vect']
        alpha = dp_grp['alpha']
        k = dp_grp.attrs['k']

        n = core.Dotprops(points=points,
                          k=k,
                          vect=vect,
                          alpha=alpha,
                          id=id,
                          name=neuron_grp.attrs.get('neuron_name'))

        # Check if we have units
        self.parse_add_units(dp_grp, neuron_grp, n)

        # Parse additional attributes
        self.parse_add_attributes(dp_grp, neuron_grp, n,
                                  subset=['soma'] if strict else None,
                                  exclude=['units_nm', 'k'])

        # Parse additional datatsets
        if not strict:
            self.parse_add_datasets(dp_grp, n,
                                    exclude=['points', 'vect', 'alpha'])

        return n

    def read_meshneuron(self, id, strict=False, prefer_raw=False, **kwargs):
        """Read given MeshNeuron from file."""
        # Get the group for this neuron
        neuron_grp = self.f[id]

        # See if this neuron has a mesh
        if 'mesh' not in neuron_grp:
            raise ValueError(f'Neuron {id} has no mesh')
        me_grp = neuron_grp['mesh']

        if '.serialized_navis' in me_grp and not prefer_raw:
            return pickle.loads(me_grp['.serialized_navis'][()])

        # Parse node table -> do not remove the [:] as it ensures that we get
        # a numpy array
        verts = me_grp['vertices'][:]
        faces = me_grp['faces'][:]

        n = core.MeshNeuron({'vertices': verts, 'faces': faces},
                            id=id,
                            name=neuron_grp.attrs.get('neuron_name'))

        if 'skeleton_map' in me_grp:
            n.skeleton_map = me_grp['skeleton_map']

        # Check if we have units
        self.parse_add_units(me_grp, neuron_grp, n)

        # Parse additional attributes
        self.parse_add_attributes(me_grp, neuron_grp, n,
                                  subset=['soma'] if strict else None,
                                  exclude=['units_nm'])

        # Parse additional datatsets
        if not strict:
            self.parse_add_datasets(me_grp, n,
                                    exclude=['vertices',
                                             'faces',
                                             'skeleton_map'])

        return n


class BaseH5Writer(ABC):
    """Writes neurons to HDF5 files."""

    version = None

    def __init__(self, filepath: str,  mode='a', **kwargs):
        """Initialize.

        Parameters
        ----------
        filepath :      str
                        Path to HDF5 file.
        mode :          "a" | "w" |
                        Mode in which to open H5 file::

                            w	Create file, truncate if exists
                            a	Read/write if exists, create otherwise

        **kwargs
                        Passed to ``h5py.File``.
        """
        self.filepath = filepath
        self.mode = mode
        self.kwargs = kwargs

    def __enter__(self):
        """Open file on enter."""
        self.f = h5py.File(self.filepath, self.mode, **self.kwargs)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close file on exit."""
        self.f.close()

    def check_compatible(self):
        """Check if this writer is compatible with existing data."""
        fmt = self.f.attrs.get('format_spec')
        if fmt:
            if not fmt.startswith('hnf_'):
                raise ValueError('Unexpected format specifier for file '
                                 f'{self.filepath}: "{fmt}"')

            ver = fmt.split('_')[-1]
            if ver != 'v1':
                raise ValueError(f'File {self.filepath} appears to contain '
                                 f'data from an incompatible version: "{ver}"')

    def write_dataframe(self, df, group, subset=None, exclude=None, overwrite=False):
        """Write dataframe to group."""
        assert isinstance(df, pd.DataFrame)

        if isinstance(subset, type(None)):
            subset = df.columns
        else:
            subset = utils.make_iterable(subset)

        if isinstance(exclude, type(None)):
            exclude = []
        else:
            exclude = utils.make_iterable(exclude)

        # Remove datasets if it already exists
        for c in df.columns:
            if c not in subset or c in exclude:
                continue

            if c in group:
                if not overwrite:
                    raise ValueError(f'Dataset {c} already exists in group "{group}"')
                del group[c]

            # Convert categoricals
            if isinstance(df[c].dtype, pd.CategoricalDtype):
                data = np.asarray(df[c])
            else:
                data = df[c].values

            # HDF5 does not like numpy strings ("<U4") or object
            if data.dtype.type in (np.str_, np.object_):
                data = data.astype("S")

            group.create_dataset(c, data=data, compression='gzip')

    def write_base_info(self):
        """Add version and url to root path."""
        self.f.attrs['format_spec'] = f'hnf_v{self.version}'
        self.f.attrs['format_url'] = 'https://github.com/schlegelp/navis'

    @abstractmethod
    def write_neurons(self, neuron: 'core.NeuronObject'):
        """Write Neuron to file."""
        pass


class H5WriterV1(BaseH5Writer):
    """Implements v1 of the HDF schema.

    See <url> for format specs.
    """

    version = 1

    def write_neurons(self, neuron, serialized=True, raw=False,
                      overwrite=True, annotations=None, **kwargs):
        """Write neuron to file."""
        if isinstance(neuron, core.NeuronList):
            for n in config.tqdm(neuron, desc='Writing',
                                 leave=False,
                                 disable=config.pbar_hide):
                self.write_neurons(n, overwrite=overwrite,
                                   annotations=annotations, **kwargs)
            return

        if isinstance(neuron, core.TreeNeuron):
            self.write_treeneuron(neuron, serialized=serialized, raw=raw,
                                  overwrite=overwrite, **kwargs)
        elif isinstance(neuron, core.MeshNeuron):
            self.write_meshneuron(neuron, serialized=serialized, raw=raw,
                                  overwrite=overwrite, **kwargs)
        elif isinstance(neuron, core.Dotprops):
            self.write_dotprops(neuron, serialized=serialized, raw=raw,
                                overwrite=overwrite, **kwargs)
        else:
            raise TypeError(f'Unable to write object of type "{type(neuron)}"'
                            'to HDF5 file.')

        # Write annotations
        if not isinstance(annotations, type(None)):
            _ = self.write_annotations(neuron, annotations,
                                       overwrite=overwrite,
                                       **kwargs)

    def write_annotations(self, neuron, annotations, overwrite=True, **kwargs):
        """Write annotations for given neuron to file."""
        # Get this neuron's group
        neuron_grp = self.get_neuron_group(neuron)
        # Create annotation group if it does not exist
        an_grp = neuron_grp.require_group('annotations')

        annotations = utils.make_iterable(annotations)

        for an in annotations:
            data = getattr(neuron, an)
            if isinstance(data, pd.DataFrame):
                data_grp = an_grp.require_group(an)
                self.write_dataframe(data, data_grp, overwrite=overwrite)
            else:
                raise ValueError(f'Unable to write "{an}" of type '
                                 f'"({type(data)})" to HDF5 file.')

    def get_neuron_group(self, neuron):
        """Get group for neuron or create if it does not exists."""
        # Can only use strings as group names
        id = str(neuron.id)

        grp = self.f.require_group(id)

        # Write some basic info
        if hasattr(neuron, 'name'):
            grp.attrs['neuron_name'] = neuron.name

        return grp

    def write_treeneuron(self, neuron,
                         serialized=True, raw=False,
                         overwrite=True, **kwargs):
        """Write TreeNeuron to file."""
        assert isinstance(neuron, core.TreeNeuron)

        # Get the group for this neuron -> this also write BaseInfo
        neuron_grp = self.get_neuron_group(neuron)

        # See if this neuron already has a skeleton
        if 'skeleton' in neuron_grp:
            if not overwrite:
                raise ValueError('File already contains a skeleton for neuron '
                                 f'{neuron.id}')
            del neuron_grp['skeleton']

        sk_grp = neuron_grp.require_group('skeleton')

        if serialized:
            sk_grp.create_dataset('.serialized_navis',
                                  data=np.void(pickle.dumps(neuron)))

        if raw:
            # Write info
            units = neuron_nm_units(neuron)
            if units:
                sk_grp.attrs['units_nm'] = units

            if neuron.has_soma:
                sk_grp.attrs['soma'] = neuron.soma

            # Write node table
            self.write_dataframe(neuron.nodes,
                                 group=sk_grp,
                                 exclude=['type'],
                                 overwrite=overwrite)

    def write_dotprops(self, neuron,
                       serialized=True, raw=False,
                       overwrite=True, **kwargs):
        """Write Dotprops to file."""
        assert isinstance(neuron, core.Dotprops)

        # Get the group for this neuron -> this also write BaseInfo
        neuron_grp = self.get_neuron_group(neuron)

        # See if this neuron already has dotprops
        if 'dotprops' in neuron_grp:
            if not overwrite:
                raise ValueError('File already contains dotprops for neuron '
                                 f'{neuron.id}')
            del neuron_grp['dotprops']

        dp_grp = neuron_grp.require_group('dotprops')

        if serialized:
            dp_grp.create_dataset('.serialized_navis',
                                  data=np.void(pickle.dumps(neuron)))

        if raw:
            # Write info
            dp_grp.attrs['k'] = neuron.k
            units = neuron_nm_units(neuron)
            if units:
                dp_grp.attrs['units_nm'] = units
            if neuron.has_soma:
                dp_grp.attrs['soma'] = neuron.soma

            # Write data
            for d in ['points', 'vect', 'alpha']:
                # The neuron really ought to have these but just in case
                if not hasattr(neuron, d):
                    continue

                data = getattr(neuron, d)
                dp_grp.create_dataset(d, data=data, compression='gzip')

    def write_meshneuron(self, neuron,
                         serialized=True, raw=False,
                         overwrite=True, **kwargs):
        """Write MeshNeuron to file."""
        assert isinstance(neuron, core.MeshNeuron)

        # Get the group for this neuron -> this also write BaseInfo
        neuron_grp = self.get_neuron_group(neuron)

        # See if this neuron already has a mesh
        if 'mesh' in neuron_grp:
            if not overwrite:
                raise ValueError('File already contains a mesh for neuron '
                                 f'{neuron.id}')
            del neuron_grp['mesh']

        me_grp = neuron_grp.require_group('mesh')

        if serialized:
            me_grp.create_dataset('.serialized_navis',
                                  data=np.void(pickle.dumps(neuron)))

        if raw:
            # Write info
            units = neuron_nm_units(neuron)
            if units:
                me_grp.attrs['units_nm'] = units
            if neuron.has_soma:
                me_grp.attrs['soma'] = neuron.soma

            # Write data
            for d in ['vertices', 'faces', 'skeleton_map']:
                if not hasattr(neuron, d):
                    continue

                data = getattr(neuron, d)
                me_grp.create_dataset(d, data=data, compression='gzip')


def read_h5(filepath: str,
            read='mesh->skeleton->dotprops',
            subset=None,
            prefer_raw=False,
            annotations=True,
            strict=False,
            reader='auto',
            on_error='stop',
            ret_errors=False,
            parallel='auto') -> 'core.NeuronObject':
    """Read Neuron/List from Hdf5 file.

    This import is following the schema specified
    `here <http://www.>`_

    Parameters
    ----------
    filepath :          filepath
                        Path to HDF5 file.
    read :              str
                        The HDF5 file might contain skeleton, dotprops and/or
                        mesh representations for any given neuron. This
                        parameter determines which one are returned. Some
                        illustrative examples:

                          - 'mesh', 'skeleton' or 'dotprops' will return only
                            the given representation
                          - 'mesh->skeleton->dotprops' will return a mesh if the
                            neuron has one, a skeleton if it does not and
                            dotprops if it has neither mesh nor skeleton
                          - 'mesh,skeleton,dotprops' will return all available
                            representations
                          - 'mesh,dotprops' will only return meshes and dotprops
                          - 'mesh,skeleton->dotprops' will return the mesh
                            and a skeleton or alternatively the dotprops

                        Note that neurons which have none of the requested
                        representations are silently skipped!
    subset :            list of IDs | slice
                        If provided, will read only a subset of neurons from the
                        file. IDs that don't exist are silently ignored. Also
                        note that due to HDF5 restrictions numeric IDs will be
                        converted to strings.
    prefer_raw :        bool
                        If True and a neuron has is saved as both serialized and
                        raw data, will load the neuron from the raw data.
    parallel :          "auto" | bool | int
                        Defaults to ``auto`` which means only use parallel
                        processing if more than 200 neurons are imported.
                        Spawning and joining processes causes overhead and is
                        considerably slower for imports of small numbers of
                        neurons. Integer will be interpreted as the
                        number of cores (otherwise defaults to
                        ``os.cpu_count() - 2``).
    on_error :          "stop" | "warn" | "ignore"
                        What to do if a neuron can not be parsed: "stop" and
                        raise an exception, "warn" and keep going or silently
                        "ignore" and skip.
    ret_errors :        bool
                        If True, will also return a list of errors encountered
                        while parsing the neurons.

    Only relevant for raw data:

    annotations :       bool | str | list of str
                        Whether to load annotations associated with the
                        neuron(s):

                         - ``True`` reads all annotations
                         - ``False`` reads no annotations
                         - e.g. ``["connenctors"]`` reads only "connectors"

                        Non-existing annotations are silently ignored!
    strict :            bool
                        If True, will read only the attributes/columns which
                        are absolutely required to construct the respective
                        neuron representation. This is useful if you either want
                        to keep memory usage low or if any additional attributes
                        are causing troubles. If False (default), will read
                        every attribute and dataframe column and attach it to
                        the neuron.
    reader :            "auto" | subclass of BaseH5Reader
                        Which reader to use to parse the given format. By
                        default ("auto") will try to pick the correct parser
                        for you depending on the `format_spec` attribute in
                        the HDF5 file. You can also directly provide a subclass
                        of BaseH5Reader that is capable of reading neurons from
                        the file.

    Returns
    -------
    neurons :           navis.NeuronList

    errors :            dict
                        If ``ret_errors=True`` return dictionary with errors:
                        ``{id: "error"}``.

    Examples
    --------
    See :func:`navis.write_h5` for examples.


    See Also
    --------
    :func:`navis.write_h5`
                        Write neurons to HDF5 file.
    :func:`navis.io.inspect_h5`
                        Extract meta data (format, number of neurons,
                        available annotations and representations) from
                        HDF5 file. This is useful if you don't know what's
                        actually contained within the HDF5 file.

    """
    utils.eval_param(read, name='read', allowed_types=(str, ))
    utils.eval_param(on_error, name='on_error',
                     allowed_values=('stop', 'warn', 'ignore'))

    # Make sure the read string is "correct"
    for rep in read.split(','):
        rep = rep.strip()
        for prio in rep.split('->'):
            prio = prio.strip()
            if prio not in ('mesh', 'skeleton', 'dotprops'):
                raise ValueError(f'Unexpected representation in `read` parameter: {prio}')

    # Get info for this file
    filepath = os.path.expanduser(filepath)
    info = inspect_h5(filepath, inspect_neurons=True, inspect_annotations=False)

    # Get a reader for these specs
    if reader == 'auto':
        if info['format_spec'] not in READERS:
            raise TypeError(f'No reader for HDF5 format {info["format_spec"]}')
        reader = READERS[info['format_spec']]
    elif not isinstance(reader, BaseH5Reader):
        raise TypeError('If provided, the reader must be a subclass of '
                        f'BaseH5Reader - got "{type(reader)}"')

    # By default only use parallel if there are more than 200 neurons
    if parallel == 'auto':
        if len(info['neurons']) > 200:
            parallel = True
        else:
            parallel = False

    if not parallel:
        # This opens the file
        with reader(filepath) as r:
            nl, errors = r.read_neurons(subset=subset,
                                        read=read,
                                        strict=strict,
                                        on_error=on_error,
                                        annotations=annotations)
    else:
        # Do not swap this as ``isinstance(True, int)`` returns ``True``
        if isinstance(parallel, (bool, str)):
            n_cores = os.cpu_count() - 2
        else:
            n_cores = int(parallel)

        # If subset not specified, fetch all neurons
        if isinstance(subset, type(None)):
            subset = list(info['neurons'])
        elif isinstance(subset, slice):
            subset = list(info['neurons'])[subset]
        else:
            # Make sure it's an iterable and strings
            subset = utils.make_iterable(subset).astype(str)

        # Just to leave note that I tried splitting the array into
        # `n_cores` chunks but that caused massive memory usage in the
        # spawned processes without being any faster - reading and returning
        # one neuron at a time seems to be the most efficient way
        reader = READERS[info['format_spec']]
        with mp.Pool(processes=n_cores) as pool:
            futures = pool.imap(_h5_reader_worker, [dict(reader=reader,
                                                         filepath=filepath,
                                                         read=read,
                                                         strict=strict,
                                                         prefer_raw=prefer_raw,
                                                         on_error=on_error,
                                                         annotations=annotations,
                                                         subset=[x]) for x in subset],
                                chunksize=1)

            # Wait for results and show progress bar althewhile
            # Do not close the pool before doing this
            res = list(config.tqdm(futures,
                                   desc='Reading',
                                   total=len(subset)))

        # Unpack results
        nl = []
        errors = {}
        for n, e in res:
            nl += n
            errors.update(e)

        # Warnings will not have propagated
        if on_error == 'warn':
            for e in errors:
                warnings.warn(f"Error reading neuron {e}: {errors[e]}")

    if ret_errors:
        return core.NeuronList(nl), errors
    else:
        return core.NeuronList(nl)


def _h5_reader_worker(kwargs):
    """Trigger reading neurons from H5 file."""
    reader = kwargs['reader']
    filepath = kwargs['filepath']
    read = kwargs['read']
    subset = kwargs['subset']
    annotations = kwargs['annotations']
    strict = kwargs['strict']
    on_error = kwargs['on_error']
    prefer_raw = kwargs['prefer_raw']
    # This opens the file
    with reader(filepath) as r:
        return r.read_neurons(subset=subset,
                              read=read,
                              strict=strict,
                              on_error=on_error,
                              prefer_raw=prefer_raw,
                              progress=False,
                              annotations=annotations)


def write_h5(n: 'core.NeuronObject',
             filepath: str,
             serialized: bool = True,
             raw: bool = False,
             annotations: Optional[Union[str, list]] = None,
             format: str = 'latest',
             append: bool = True,
             overwrite_neurons: bool = False) -> 'core.NeuronObject':
    """Write Neuron/List to Hdf5 file.

    Parameters
    ----------
    n :                 Neuron | NeuronList
                        Neuron(s) to write to file.
    filepath :          str
                        Path to HDF5 file. Will be created if it does not
                        exist. If it does exist, we will add data to it.
    serialized :        bool
                        Whether to write a serialized (pickled) version of the
                        neuron to file.
    raw :               bool
                        Whether to write the neurons' raw data to file. This
                        is required to re-generate neurons from tools other
                        than `navis` (e.g. R's `nat`). This follows the schema
                        specified `here <https://github.com/flyconnectome/hnf>`_.
    append :            bool
                        If file already exists, whether to append data or to
                        overwrite the entire file.
    overwrite_neurons : bool
                        If a given neuron already exists in the h5 file whether
                        to overwrite it or throw an exception.

    Only relevant if ``raw=True``:

    annotations :       str | list thereof, optional
                        Whether to write annotations (e.g. "connectors")
                        associated with the neuron(s) to file. Annotations
                        must be pandas DataFrames. If a neuron does not contain
                        a given annotation, it is silently skipped.
    format :            "latest" | "v1"
                        Which version of the format specs to use. By default
                        use latest. Note that we don't allow mixing format
                        specs in the same HDF5 file. So if you want to write
                        to a file which already contains data in a given
                        format, you have to use that format.

    Returns
    -------
    Nothing

    Examples
    --------
    >>> import navis
    >>> # First get mesh, skeleton and dotprop representations for some neurons
    >>> sk = navis.example_neurons(5, kind='skeleton')
    >>> me = navis.example_neurons(5, kind='mesh')
    >>> dp = navis.make_dotprops(sk, k=5)
    >>> # Write them to a file
    >>> navis.write_h5(sk + me + dp, '~/test.h5', overwrite_neurons=True)
    >>> # Read back from file
    >>> nl = navis.read_h5('~/test.h5')

    See Also
    --------
    :func:`navis.read_h5`
                        Read neurons from h5 file.

    """
    if not serialized and not raw:
        raise ValueError('`serialized` and `raw` must not both be False.')

    utils.eval_param(format, name='format',
                     allowed_values=tuple(WRITERS.keys()))

    filepath = os.path.expanduser(filepath)

    # Get the writer for the specified format
    writer = WRITERS[format]

    # This opens the file
    with writer(filepath, mode='a' if append else 'w') as w:
        w.write_base_info()
        w.write_neurons(n,
                        raw=raw,
                        serialized=serialized,
                        overwrite=overwrite_neurons,
                        annotations=annotations)


def inspect_h5(filepath, inspect_neurons=True, inspect_annotations=True):
    """Extract basic info from Hdf5 file.

    Parameters
    ----------
    filepath :              str
                            Path to HDF5 file.
    inspect_neurons :       bool
                            If True, will return info about the neurons contained.
    inspect_annotations :   bool
                            If True, include info about annotations associated
                            with each neuron.

    Returns
    -------
    dict
                        Returns a dictionary with basic info about the file.
                        An example::

                         {
                          'format_spec': 'hnf_v1', # format specifier
                          'neurons': {
                                      'someID': {'skeleton': True,
                                                 'mesh': False,
                                                 'dotprops': True,
                                                 'annotations': ['connectors']},
                                      'someID2': {'skeleton': False,
                                                  'mesh': False,
                                                  'dotprops': True,
                                                  'annotations': ['connectors']}
                                     }
                            }


    """
    if not isinstance(filepath, str):
        raise TypeError(f'`filepath` must be str, got "{type(filepath)}"')

    if not os.path.isfile(filepath):
        raise ValueError(f'{filepath} does not exist')

    info = dict()
    with h5py.File(filepath, 'r') as f:
        info['format_spec'] = f.attrs.get('format_spec')
        # R strings are automatically stored as vectors
        info['format_spec'] = utils.make_non_iterable(info['format_spec'])

        if inspect_neurons:
            info['neurons'] = {}
            # Go over all top level groups
            for id, grp in f.items():
                # Skip if not a group
                if not isinstance(grp, h5py.Group):
                    continue

                # Do not change this test
                this = {}
                if 'skeleton' in grp:
                    this['skeleton'] = True
                if 'mesh' in grp:
                    this['mesh'] = True
                if 'dotprops' in grp:
                    this['dotprops'] = True

                if this:
                    info['neurons'][id] = this
                    if inspect_annotations:
                        annotations = grp.get('annotations', None)
                        if annotations:
                            info['neurons'][id]['annotations'] = list(annotations.keys())

    return info


def neuron_nm_units(neuron):
    """Return neuron's units in nanometers.

    Returns ``None`` if units are dimensionless.

    """
    units = getattr(neuron, 'units')

    if isinstance(units, type(None)):
        return

    if not isinstance(units, (pint.Quantity, pint.Unit)):
        return None

    # Casting to nm throws an error if dimensionless
    try:
        return units.to('nm').magnitude
    except pint.DimensionalityError:
        return None
    except BaseException:
        raise


WRITERS = {'v1': H5WriterV1,
           'latest': H5WriterV1}

READERS = {'hnf_v1': H5ReaderV1}
