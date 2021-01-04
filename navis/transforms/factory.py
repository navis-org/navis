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

import pathlib
import json

from .affine import AffineTransform
from .base import TransformSequence
from .cmtk import CMTKtransform
from .h5reg import H5transform
from .thinplate import TPStransform


def parse_json(filepath: str, **kwargs):
    """Parse json-encoded transform (experimental).

    Parameters
    ----------
    filepath :      str
                    Filepath to json file to generate transform from.
    **kwargs
                    Keyword arguments passed to the respective transform class.

    Returns
    -------
    transform

    """
    fp = pathlib.Path(filepath)

    with open(fp, 'r') as f:
        data = json.load(f)

    if not isinstance(data, list):
        data = []

    transforms = []
    for reg in data:
        if not isinstance(reg, type(dict)):
            raise TypeError(f'{filepath} expected data as dict or list of '
                            f'dicts, got "{type(reg)}"')

        if reg.get('type', None) == 'tpsreg':
            transforms.append(TPStransform(reg['refmat'].values,
                                           reg['tarmat'].values, **kwargs))
        elif reg.get('type', None) == 'affine':
            transforms.append(AffineTransform(reg['affine_matrix'], **kwargs))
        else:
            raise TypeError(f'{reg} has unknown "type"')

    return TransformSequence(*transforms)


def transform_factory(filepath: str, **kwargs):
    """Generate appropriate transform from file.

    Parameters
    ----------
    filepath :      str
                    Filepath to generate transform from.
    **kwargs
                    Keyword arguments passed to the respective transform class.

    Returns
    -------
    transform

    """
    fp = pathlib.Path(filepath)

    # Check if file/path exists
    if not fp.is_dir() or not fp.is_file():
        raise ValueError(f'{fp} does not appear to exist')

    if fp.endswith('.list'):
        return CMTKtransform(fp, **kwargs)

    if fp.endswith('.h5'):
        return H5transform(fp, **kwargs)

    # Custom transforms
    if fp.endswith('.json'):
        return parse_json(fp, **kwargs)

    raise TypeError(f'Unknown transform format for {filepath}')


# This must comme after defining/importing the functions
factory_methods = {'.list': CMTKtransform,
                   '.h5': H5transform,
                   '.json': parse_json}
