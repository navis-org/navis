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

import numpy as np
import pandas as pd

from typing import Union
from typing_extensions import Literal


def sparseness(x: Union[pd.DataFrame, np.ndarray],
               which: Union[Literal['LTS'],
                            Literal['LTK'],
                            Literal['activity_ratio']] = 'LTS') -> Union[pd.Series, np.ndarray]:
    r"""Calculate sparseness.

    Sparseness comes in three flavors:

    **Lifetime kurtosis (LTK)** quantifies the widths of tuning curves
    (according to Muench & Galizia, 2016):

    .. math::

        S = \Bigg\{ \frac{1}{N} \sum^N_{i=1} \Big[ \frac{r_i - \overline{r}}{\sigma_r} \Big] ^4  \Bigg\} - 3

    where :math:`N` is the number of observations, :math:`r_i` the value of
    observation :math:`i`, and :math:`\overline{r}` and
    :math:`\sigma_r` the mean and the standard deviation of the observations'
    values, respectively. LTK is assuming a normal, or at least symmetric
    distribution.

    **Lifetime sparseness (LTS)** quantifies selectivity
    (Bhandawat et al., 2007):

    .. math::

        S = \frac{1}{1-1/N} \Bigg[1- \frac{\big(\sum^N_{j=1} r_j / N\big)^2}{\sum^N_{j=1} r_j^2 / N} \Bigg]

    where :math:`N` is the number of observations, and :math:`r_j` is the
    value of an observation.

    **Activity ratio** describes distributions with heavy tails (Rolls and
    Tovee, 1995).


    Notes
    -----
    ``NaN`` values will be ignored. You can use that to e.g. ignore zero
    values in a large connectivity matrix by changing these values to ``NaN``
    before passing it to ``navis.sparseness``.


    Parameters
    ----------
    x :         DataFrame | array-like
                (N, M) dataset with N (rows) observations for M (columns)
                neurons. One-dimensional data will be converted to two
                dimensions (N rows, 1 column).
    which :     "LTS" | "LTK" | "activity_ratio"
                Determines whether lifetime sparseness (LTS) or lifetime
                kurtosis (LTK) is returned.

    Returns
    -------
    sparseness
                ``pandas.Series`` if input was pandas DataFrame, else
                ``numpy.array``.

    Examples
    --------
    Calculate sparseness of olfactory inputs to group of neurons:

    >>> import navis
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> # Get ORN response matrix from DoOR database
    >>> url = 'https://raw.githubusercontent.com/ropensci/DoOR.data/master/data/door_response_matrix.csv'
    >>> adj = pd.read_csv(url, delimiter=';')
    >>> # Calculate lifetime sparseness
    >>> S = navis.sparseness(adj, which='LTS')
    >>> # Plot distribution
    >>> ax = S.plot.hist(bins=np.arange(0, 1, .1))
    >>> _ = ax.set_xlabel('LTS')
    >>> plt.show()                                              # doctest: +SKIP

    """
    if not isinstance(x, (pd.DataFrame, np.ndarray)):
        x = np.array(x)

    # Make sure we are working with 2 dimensional data
    if isinstance(x, np.ndarray) and x.ndim == 1:
        x = x.reshape(x.shape[0], 1)

    N = np.sum(~np.isnan(x), axis=0)

    if which == 'LTK':
        return np.nansum(((x - np.nanmean(x, axis=0)) / np.nanstd(x, axis=0)) ** 4, axis=0) / N - 3
    elif which == 'LTS':
        return 1 / (1 - (1 / N)) * (1 - np.nansum(x / N, axis=0) ** 2 / np.nansum(x**2 / N, axis=0))
    elif which == 'activity_ratio':
        a = (np.nansum(x, axis=0) / N) ** 2 / (np.nansum(x, axis=0) ** 2 / N)
        return 1 - a
    else:
        raise ValueError('Parameter "which" must be either "LTS", "LTK" or '
                         '"activity_ratio"')
