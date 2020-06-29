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


""" This module contains functions to analyse connectivity.
"""

import pandas as pd
import numpy as np

from typing import Union, Optional

from .. import utils, config

# Set up logging
logger = config.logger


def group_matrix(mat: Union[pd.DataFrame, np.ndarray],
                 row_groups: Optional[dict] = {},
                 col_groups: Optional[dict] = {},
                 drop_ungrouped: bool = False,
                 method: str = 'SUM') -> pd.DataFrame:
    """Group adjacency matrix into neuron groups.

    Parameters
    ----------
    mat :               pandas.DataFrame | numpy.array
                        Matrix to group.
    row_groups :        dict, optional
                        Row groups to be formed. Can be either:

                          1. ``{group1: [neuron1, neuron2, ...], ...}``
                          2. ``{neuron1: group1, neuron2:group2, ...}``

                        If grouping numpy arrays, use indices!
    col_groups :        dict, optional
                        Col groups. See ``row_groups`` for details.
    drop_ungrouped :    bool, optional
                        If ungrouped, neurons that are not part of a
                        row/col_group are dropped from the matrix.
    method :            'AVERAGE' | 'MAX' | 'MIN' | 'SUM', optional
                        Method by which values are collapsed into groups.

    Returns
    -------
    pandas.DataFrame

    """
    PERMISSIBLE_METHODS = ['AVERAGE', 'MIN', 'MAX', 'SUM']
    if method not in PERMISSIBLE_METHODS:
        raise ValueError(f'Unknown method "{method}". Please use either '
                         f'{",".join(PERMISSIBLE_METHODS)}')

    if not row_groups and not col_groups:
        logger.warning('No column/row groups provided - skipping.')
        return mat

    # Convert numpy array to DataFrame
    if isinstance(mat, np.ndarray):
        mat = pd.DataFrame(mat)
    # Make copy of original DataFrame
    elif isinstance(mat, pd.DataFrame):
        mat = mat.copy()
    else:
        raise TypeError(f'Expected numpy array or pandas DataFrames, got "{type(mat)}"')

    # Convert to neuron->group format if necessary
    if col_groups and utils.is_iterable(list(col_groups.values())[0]):
        col_groups = {n: g for g in col_groups for n in col_groups[g]}
    if row_groups and utils.is_iterable(list(row_groups.values())[0]):
        row_groups = {n: g for g in row_groups for n in row_groups[g]}

    # Make sure everything is string
    mat.index = mat.index.astype(str)
    mat.columns = mat.columns.astype(str)
    col_groups = {str(k): str(v) for k, v in col_groups.items()}  # type: ignore # redefinition error
    row_groups = {str(k): str(v) for k, v in row_groups.items()}  # type: ignore # redefinition error

    if row_groups:
        # Drop non-grouped values if applicable
        if drop_ungrouped:
            mat = mat.loc[mat.index.isin(row_groups.keys())]

        # Add temporary grouping column
        mat['row_groups'] = [row_groups.get(s, s) for s in mat.index]

        if method == 'AVERAGE':
            mat = mat.groupby('row_groups').mean()
        elif method == 'MAX':
            mat = mat.groupby('row_groups').max()
        elif method == 'MIN':
            mat = mat.groupby('row_groups').min()
        elif method == 'SUM':
            mat = mat.groupby('row_groups').sum()

    if col_groups:
        # Transpose for grouping
        mat = mat.T

        # Drop non-grouped values if applicable
        if drop_ungrouped:
            mat = mat.loc[mat.index.isin(col_groups.keys())]

        # Add temporary grouping column
        mat['col_groups'] = [col_groups.get(s, s) for s in mat.index]

        if method == 'AVERAGE':
            mat = mat.groupby('col_groups').mean()
        elif method == 'MAX':
            mat = mat.groupby('col_groups').max()
        elif method == 'MIN':
            mat = mat.groupby('col_groups').min()
        elif method == 'SUM':
            mat = mat.groupby('col_groups').sum()

        # Transpose back
        mat = mat.T

    # Preserve datatype
    mat.datatype = 'adjacency_matrix'
    # Add flag that this matrix has been grouped
    mat.is_grouped = True

    return mat
