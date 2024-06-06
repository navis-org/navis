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


""" This module contains functions to analyse neuron's form factors."""

import os
import scipy

import pandas as pd
import multiprocessing as mp
import numpy as np

from functools import partial
from typing import Union, Optional, Sequence, List, Dict, overload
from typing_extensions import Literal

from .. import config, core, utils

# Set up logging
logger = config.get_logger(__name__)

__all__ = sorted(['form_factor'])


def form_factor(x: Union['core.TreeNeuron', 'core.MeshNeuron'],
                start: int = -3,
                stop: int = 3,
                num: int = 601,
                parallel: bool = False,
                n_cores: int = os.cpu_count() // 2,
                progress=True):
    """Calculate form factor for given neuron.

    The form factor F(q) is a Fourier transform of density-density correlation
    of particles used to classify objects in polymer physics. Based on Choi et
    al., 2022 (bioRxiv). Code adapted from github.com/kirichoi/FqClustering.

    Parameters
    ----------
    x :         TreeNeuron | Meshneuron | Dotprops | NeuronList
                Neurons to calculate form factor for. A few notes:
                  - data should be in micron - if not, you might want to adjust
                    start/stop/min!
                  - since this is all about density, it may make sense to
                    resample neurons
    start/stop/num : int
                Start/stop/num describe the (log) space over which to calculate
                the form factor. Effectively determining the resolution.
                Assuming ``x`` is in microns the defaults mean we pay attention
                to densities between 1 nm (1e-3 microns) and 1 mm (1e+3 microns).
                The x-value corresponding to the form factor(s) in ``Fq`` will
                be ``np.logspace(start, stop, num)``.
    parallel :  bool
                Whether to use multiple cores when ``x`` is a NeuronList.
    n_cores :   bool
                Number of cores to use when ``x`` is a NeuronList and
                ``parallel=True``. Even on a single core this function makes
                heavy use of numpy which itself uses multiple threads - it is
                therefore not advisable to use all your cores as this would
                create a bottleneck.
    progress :  bool
                Whether to show a progress bar.

    Returns
    -------
    Fq :        np.ndarray
                For single neurons: ``(num,)`` array
                For Neuronlists: ``(len(x), num)`` array

    References
    ----------
    Polymer physics-based classification of neurons
    Kiri Choi, Won Kyu Kim, Changbong Hyeon
    bioRxiv 2022.04.07.487455; doi: https://doi.org/10.1101/2022.04.07.487455

    Examples
    --------
    >>> import navis
    >>> nl = navis.example_neurons(3)
    >>> nl = nl.convert_units('microns')
    >>> # Resample to 1 node / micron
    >>> rs = navis.resample_skeleton(nl, '1 micron')
    >>> # Calculate form factor
    >>> Fq = navis.form_factor(rs, start=-3, stop=3, num=301,
    ...                        parallel=True, n_cores=3)
    >>> # Plot
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> x = np.logspace(-3, 3,  301)
    >>> fig, ax = plt.subplots()
    >>> for i in range(len(Fq)):
    ...     _ = ax.plot(x, Fq[i])
    >>> # Make log-log
    >>> ax.set_xscale('log')
    >>> ax.set_yscale('log')
    >>> plt.show()                                              # doctest: +SKIP
    >>> # Cluster
    >>> from scipy.spatial.distance import pdist
    >>> from scipy.cluster.hierarchy import dendrogram, linkage
    >>> dists = pdist(Fq)
    >>> Z = linkage(dists, method='ward')
    >>> dn = dendrogram(Z)                                      # doctest: +SKIP

    """
    if isinstance(x, core.NeuronList):
        pbar = partial(
            config.tqdm,
            desc='Calc. form factor',
            total=len(x),
            disable=config.pbar_hide or not progress,
            leave=config.pbar_leave
        )
        _calc_form_factor = partial(form_factor, progress=False,
                                    start=start, stop=stop, num=num)

        if parallel:
            with mp.Pool(processes=n_cores) as pool:
                results = pool.imap(_calc_form_factor, x)
                Fq = list(pbar(results))
        else:
            Fq = [_calc_form_factor(n) for n in pbar(x)]

        return np.vstack(Fq)

    utils.eval_param(x, name='x', allowed_types=(core.TreeNeuron,
                                                 core.Dotprops,
                                                 core.MeshNeuron))

    if isinstance(x, core.TreeNeuron):
        coor = x.nodes[['x', 'y', 'z']].values
    elif isinstance(x, core.MeshNeuron):
        coor = x.vertices
    elif isinstance(x, core.Dotprops):
        coor = x.points

    ucoor = np.unique(coor, axis=0)
    lenucoor = len(ucoor)

    q_range = np.logspace(start, stop, num)
    Fq = np.empty(len(q_range))
    ccdisttri = scipy.spatial.distance.pdist(ucoor)

    for q in config.trange(len(q_range),
                           desc='Calc. form factor',
                           disable=config.pbar_hide or not progress,
                           leave=config.pbar_leave):
        qrvec = q_range[q] * ccdisttri
        Fq[q] = np.divide(np.divide(2 * np.sum(np.sin(qrvec) / qrvec), lenucoor), lenucoor) + 1 / lenucoor

    return Fq
