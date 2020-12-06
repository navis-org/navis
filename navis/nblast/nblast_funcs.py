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

"""Module contains functions implementing NBLAST."""

import numbers
import os

import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from typing import Union, Optional
from concurrent.futures import ProcessPoolExecutor

from ..core import NeuronList, Dotprops, make_dotprops
from .. import config

__all__ = ['nblast', 'nblast_allbyall']

fp = os.path.dirname(__file__)
smat_path = os.path.join(fp, 'score_mats')

logger = config.logger


class ScoringFunction:
    """Class representing scoring function."""

    def __init__(self, smat):
        if isinstance(smat, type(None)):
            self.scoring_function = self.pass_through
        elif isinstance(smat, (pd.DataFrame, str)):
            self.parse_matrix(smat)
            self.scoring_function = self.score_lookup
        else:
            raise TypeError

    def __call__(self, dist, dot):
        return self.scoring_function(dist, dot)

    def pass_through(self, dist, dot):
        """Pass-through scores if no scoring matrix."""
        return dist * dot

    def score_lookup(self, dist, dot):
        return self.cells[
                          np.digitize(dist, self.dist_bins),
                          np.digitize(dot, self.dot_bins),
                         ]

    def parse_matrix(self, smat):
        """Parse matrix."""
        if isinstance(smat, str):
            smat = pd.read_csv(smat, index_col=0)

        if not isinstance(smat, pd.DataFrame):
            raise TypeError(f'Excepted filepath or DataFrame, got "{type(smat)}"')

        self.cells = smat.to_numpy()

        self.dist_thresholds = [self.parse_interval(s) for s in smat.index]
        # Make sure right bin is open
        self.dist_thresholds[-1] = np.inf
        self.dist_bins = np.array(self.dist_thresholds, float)

        self.dot_thresholds = [self.parse_interval(s) for s in smat.columns]
        # Make sure right bin is open
        self.dot_thresholds[-1] = np.inf
        self.dot_bins = np.array(self.dot_thresholds, float)

    def parse_interval(self, s):
        """Strip brackets and parse right interval.

        Example
        -------
        >>> parse_intervals("(0,0.1]")
        0.1
        """
        return float(s.strip("([])").split(",")[-1])


class NBlaster:
    """Implements version 2 of the NBLAST algorithm.

    Please note that some properties are computed on initialization and
    changing parameters (e.g. ``use_alpha``) at a later stage will mess things
    up!

    Parameters
    ----------
    use_alpha :     bool
                    Whether or not to use alpha values for the scoring.
                    If True, the dotproduct of nearest neighbor vectors will
                    be scaled by ``sqrt(alpha1 * alpha2)``.
    normalzed :     bool
                    If True, will normalize scores by the best possible score
                    (i.e. self-self) of the query neuron.
    smat :          str | pd.DataFrame
                    Score matrix. If 'auto' (default), will use scoring matrices
                    from FCWB. Same behaviour as in R's nat.nblast
                    implementation. If ``smat=None`` the scores will be
                    generated as the product of the distances and the dotproduct
                    of the vectors of nearest-neighbor pairs.
    progress :      bool
                    If True, will show a progress bar.

    """

    def __init__(self, use_alpha=False, normalized=True, smat='auto', progress=True):
        """Initialize class."""
        self.use_alpha = use_alpha
        self.normalized = normalized
        self.progress = progress

        if smat == 'auto':
            if self.use_alpha:
                smat = pd.read_csv(f'{smat_path}/smat_alpha_fcwb.csv',
                                   index_col=0)
            else:
                smat = pd.read_csv(f'{smat_path}/smat_fcwb.csv',
                                   index_col=0)

        self.score_fn = ScoringFunction(smat)

        self.self_hits = []
        self.dotprops = []

    def append(self, dotprops):
        """Append dotprops."""
        if isinstance(dotprops, (NeuronList, list)):
            for n in dotprops:
                self.append(n)
            return

        if not isinstance(dotprops, Dotprops):
            raise ValueError(f'Expected Dotprops, got "{type(dotprops)}"')

        self.dotprops.append(dotprops)
        # Calculate score for self hit
        self.self_hits.append(self.calc_self_hit(dotprops))

    def calc_self_hit(self, dotprops):
        """Non-normalized value for self hit."""
        if not self.use_alpha:
            return len(dotprops.points) * self.score_fn(0, 1.0)
        else:
            dists = np.repeat(0, len(dotprops.points))
            alpha = dotprops.alpha * dotprops.alpha
            dots = np.repeat(1, len(dotprops.points)) * np.sqrt(alpha)
            return self.score_fn(dists, dots).sum()

    def single_query_target(self, q_idx, t_idx, mean_score=False):
        """Query single target against single target."""
        # Take a short-cut if this is a self-self comparison
        if q_idx == t_idx:
            if self.normalized:
                return 1
            return self.self_hits[q_idx]

        # Run nearest-neighbor search for query against target
        data = self.dotprops[q_idx].dist_dots(self.dotprops[t_idx],
                                              alpha=self.use_alpha)
        if self.use_alpha:
            dists, dots, alpha = data
            dots *= np.sqrt(alpha)
        else:
            dists, dots = data

        score = self.score_fn(dists, dots).sum()

        # Normalize against best hit
        if self.normalized:
            score /= self.self_hits[q_idx]

        # For the mean score we also have to produce the reverse score
        if mean_score:
            reverse = self.single_query_target(t_idx, q_idx, mean_score=False)
            score = (score + reverse) / 2

        return score

    def multi_query_target(self, q_idx, t_idx, mean_scores=False):
        """NBLAST multiple queries against multiple targets."""
        rows = []
        for q in tqdm(q_idx,
                      desc='Blasting',
                      leave=False,
                      position=getattr(self, 'pbar_position', 0),
                      disable=not self.progress):
            rows.append([])
            for t in t_idx:
                score = self.single_query_target(q, t, mean_score=mean_scores)
                rows[-1].append(score)

        # Generate results
        res = pd.DataFrame(rows)
        res.columns = [self.dotprops[t].id for t in t_idx]
        res.index = [self.dotprops[q].id for q in q_idx]

        return res

    def all_by_all(self, mean_scores=False):
        """NBLAST all-by-all neurons."""
        res = self.multi_query_target(range(len(self.dotprops)),
                                      range(len(self.dotprops)),
                                      mean_scores=False)

        # For all-by-all NBLAST we can get the mean score by
        # transposing the scores
        if mean_scores:
            res = (res + res.T) / 2

        return res


def nblast(query: Union['core.TreeNeuron', 'core.NeuronList', 'core.Dotprops'],
           target: Optional[str] = None,
           mean_scores: bool = False,
           normalized: bool = True,
           use_alpha: bool = False,
           n_cores: int = os.cpu_count() - 2,
           progress: bool = True,
           k: int = 20,
           resample: Optional[int] = None) -> pd.DataFrame:
    """NBLAST query against target neurons.

    This implements the NBLAST algorithm from Costa et al. (2016) (see
    references) and mirror the implementation in R's ``nat.nblast``
    (https://github.com/natverse/nat.nblast).

    Parameters
    ----------
    query,target :  Dotprops | TreeNeuron | MeshNeuron | NeuronList
                    Query neuron(s) to NBLAST against the targets. Units should
                    be in microns as NBLAST is optimized for that and have
                    similar sampling resolutions. Non-Dotprops will be converted
                    to Dotprops (see parameters below).
    mean_scores :   bool
                    If True, will run query->target NBLAST followed by a
                    target->query NBLAST and return the mean scores.
    n_cores :       int, optional
                    Max number of cores to use for nblasting. Default is
                    ``os.cpu_count() - 2``. This should ideally be an even
                    number as that allows optimally splitting queries onto
                    individual processes.
    use_alpha :     bool, optional
                    Emphasizes neurons' straight parts (backbone) over parts
                    that have lots of branches.
    normalized :    bool, optional
                    Whether to return normalized NBLAST scores.
    progress :      bool
                    Whether to show progress bars.

    Dotprop-conversion (Only relevant if input data is not already ``Dotprops``)

    k :             int, optional
                    Number of nearest neighbors to use for dotprops generation.
    resample :      float | int | bool, optional
                    Resampling before dotprops generation.

    Returns
    -------
    scores :        pandas.DataFrame
                    Matrix with NBLAST scores. Rows are query neurons, columns
                    are targets.

    References
    ----------
    Costa M, Manton JD, Ostrovsky AD, Prohaska S, Jefferis GS. NBLAST: Rapid,
    Sensitive Comparison of Neuronal Structure and Construction of Neuron
    Family Databases. Neuron. 2016 Jul 20;91(2):293-311.
    doi: 10.1016/j.neuron.2016.06.012.

    Examples
    --------
    >>> import navis
    >>> nl = navis.example_neurons(n=5)
    >>> nl.units
    array([8, 8, 8, 8, 8]) <Unit('nanometer')>
    >>> # Convert to microns
    >>> nl_um = nl / (1000 / 8)
    >>> # Run the nblast
    >>> scores = navis.nblast(nl_um[:3], nl_um[3:])

    See Also
    --------
    :func:`navis.nblast_allbyall`
                A more efficient way than ``nblast(query=x, target=x)``.

    """
    # Check if query or targets are in microns
    # Note this test can return `None` if it can't be determined
    if check_microns(query) is False:
        logger.warning('NBLAST is optimized for data in microns and it looks '
                       'like your queries are not in microns.')
    if check_microns(target) is False:
        logger.warning('NBLAST is optimized for data in microns and it looks '
                       'like your targets are not in microns.')

    if not isinstance(n_cores, numbers.Number) or n_cores < 1:
        raise TypeError('`n_cores` must be an integer > 0')

    n_cores = int(n_cores)
    if n_cores > 1 and n_cores % 2:
        logger.warning('NBLAST is most efficient if `n_cores` is an even number')

    # Turn query into dotprops
    query_dps = force_dotprops(query, resample=resample, k=k, progress=progress)
    target_dps = force_dotprops(target, resample=resample, k=k, progress=progress)

    # Make sure we're working on NeuronList
    query_dps = NeuronList(query_dps)
    target_dps = NeuronList(target_dps)

    # Find an optimal partition that minimizes the number of neurons
    # we have to send to each process
    n_rows, n_cols = find_optimal_partition(n_cores, query_dps, target_dps)

    nblasters = []
    for q in np.array_split(query_dps, n_rows):
        for t in np.array_split(target_dps, n_cols):
            # Initialize NBlaster
            this = NBlaster(use_alpha=use_alpha,
                            normalized=normalized,
                            progress=progress)
            # Add queries and targets
            for n in q:
                this.append(n)
            for n in t:
                this.append(n)
            # Keep track of indices of queries and targets
            this.queries = np.arange(len(q))
            this.targets = np.arange(len(t)) + len(q)
            this.pbar_position = len(nblasters)

            nblasters.append(this)

    # If only one core, we don't need to break out the multiprocessing
    if n_cores == 1:
        return this.multi_query_target(this.queries,
                                       this.targets,
                                       mean_scores=mean_scores)

    with ProcessPoolExecutor(max_workers=len(nblasters)) as pool:
        # Each nblaster is passed to it's own process
        futures = [pool.submit(this.multi_query_target,
                               q_idx=this.queries,
                               t_idx=this.targets,
                               mean_scores=mean_scores) for this in nblasters]

        results = [f.result() for f in futures]

    scores = pd.DataFrame(np.zeros((len(query_dps), len(target_dps))),
                          index=query_dps.id, columns=target_dps.id)

    for res in results:
        scores.loc[res.index, res.columns] = res.values

    return scores


def nblast_allbyall(x: NeuronList,
                    normalized: bool = True,
                    use_alpha: bool = False,
                    n_cores: int = os.cpu_count() - 2,
                    progress: bool = True,
                    k: int = 20,
                    resample: Optional[int] = None) -> pd.DataFrame:
    """All-by-all NBLAST of inputs neurons.

    This is a more efficient way than running ``nblast(query=x, target=x)``.

    Parameters
    ----------
    query,target :  Dotprops | TreeNeuron | MeshNeuron | NeuronList
                    Query neuron(s) to NBLAST against the targets. Units should
                    be in microns as NBLAST is optimized for that and have
                    similar sampling resolutions. Non-Dotprops will be converted
                    to Dotprops (see parameters below).
    n_cores :       int, optional
                    Max number of cores to use for nblasting. Default is
                    ``os.cpu_count() - 2``. This should ideally be an even
                    number as that allows optimally splitting queries onto
                    individual processes.
    use_alpha :     bool, optional
                    Emphasizes neurons' straight parts (backbone) over parts
                    that have lots of branches.
    normalized :    bool, optional
                    Whether to return normalized NBLAST scores.
    progress :      bool
                    Whether to show progress bars.

    Dotprop-conversion (Only relevant if input data is not already ``Dotprops``)

    k :             int, optional
                    Number of nearest neighbors to use for dotprops generation.
    resample :      float | int | bool, optional
                    Resampling before dotprops generation.

    Returns
    -------
    scores :        pandas.DataFrame
                    Matrix with NBLAST scores.

    References
    ----------
    Costa M, Manton JD, Ostrovsky AD, Prohaska S, Jefferis GS. NBLAST: Rapid,
    Sensitive Comparison of Neuronal Structure and Construction of Neuron
    Family Databases. Neuron. 2016 Jul 20;91(2):293-311.
    doi: 10.1016/j.neuron.2016.06.012.

    Examples
    --------
    >>> import navis
    >>> nl = navis.example_neurons(n=5)
    >>> nl.units
    array([8, 8, 8, 8, 8]) <Unit('nanometer')>
    >>> # Convert to microns
    >>> nl_um = nl / (1000 / 8)
    >>> # Run the nblast
    >>> scores = navis.nblast_allbyall(nl_um)

    See Also
    --------
    :func:`navis.nblast`
                For generic query -> target nblasts.

    """
    # Check if query or targets are in microns
    # Note this test can return `None` if it can't be determined
    if check_microns(x) is False:
        logger.warning('NBLAST is optimized for data in microns and it looks '
                       'like your neurons are not in microns.')

    if not isinstance(n_cores, numbers.Number) or n_cores < 1:
        raise TypeError('`n_cores` must be an integer > 0')

    n_cores = int(n_cores)
    if n_cores > 1 and n_cores % 2:
        logger.warning('NBLAST is most efficient if `n_cores` is an even number')

    # Turn neurons into dotprops
    dps = force_dotprops(x, resample=resample, k=k, progress=progress)

    # Make sure we're working on NeuronList
    dps = NeuronList(dps)

    # Find an optimal partition that minimizes the number of neurons
    # we have to send to each process
    n_rows, n_cols = find_optimal_partition(n_cores, dps, dps)

    nblasters = []
    for qix in np.array_split(np.arange(len(dps)), n_rows):
        for tix in np.array_split(np.arange(len(dps)), n_cols):
            # Initialize NBlaster
            this = NBlaster(use_alpha=use_alpha,
                            normalized=normalized,
                            progress=progress)

            # Make sure we don't add the same neuron twice
            # Map indices to neurons
            to_add = list(set(qix) | set(tix))

            ixmap = {}
            for i, ix in enumerate(to_add):
                this.append(dps[ix])
                ixmap[ix] = i

            # Keep track of indices of queries and targets
            this.queries = [ixmap[ix] for ix in qix]
            this.targets = [ixmap[ix] for ix in tix]
            this.pbar_position = len(nblasters)

            nblasters.append(this)

    # If only one core, we don't need to break out the multiprocessing
    if n_cores == 1:
        return this.all_by_all()

    with ProcessPoolExecutor(max_workers=len(nblasters)) as pool:
        # Each nblaster is passed to it's own process
        futures = [pool.submit(this.multi_query_target,
                               q_idx=this.queries,
                               t_idx=this.targets,
                               mean_scores=False) for this in nblasters]

        results = [f.result() for f in futures]

    scores = pd.DataFrame(np.zeros((len(dps), len(dps))),
                          index=dps.id, columns=dps.id)

    for res in results:
        scores.loc[res.index, res.columns] = res.values

    return scores


def find_optimal_partition(n_cores, q, t):
    """Find an optimal partition for given NBLAST query."""
    # Find an optimal partition that minimizes the number of neurons
    # we have to send to each process
    neurons_per_query = []
    for n_rows in range(1, n_cores + 1):
        # Skip splits we can't make it
        if n_cores % n_rows:
            continue
        if n_rows > len(q):
            continue

        n_cols = min(int(n_cores / n_rows), len(t))

        n_queries = len(q) / n_rows
        n_targets = len(t) / n_cols

        neurons_per_query.append([n_rows, n_cols, n_queries + n_targets])

    # Find the optimal partition
    neurons_per_query = np.array(neurons_per_query)
    n_rows, n_cols = neurons_per_query[np.argmin(neurons_per_query[:, 2]), :2]

    return int(n_rows), int(n_cols)


def force_dotprops(x, k, resample, progress=False):
    """Force data into Dotprops."""
    if isinstance(x, (NeuronList, list)):
        dp = [force_dotprops(n, k, resample) for n in tqdm(x,
                                                           desc='Dotprops',
                                                           disable=not progress,
                                                           leave=False)]
        return NeuronList(dp)

    # Try converting non-Dotprops
    if not isinstance(x, Dotprops):
        return make_dotprops(x, k=k, resample=resample)

    # Return Dotprops
    return x


def check_microns(x):
    """Check if neuron data is in microns.

    Returns either [True, None (=unclear), False]
    """
    if isinstance(x, NeuronList):
        check = np.array([check_microns(n) for n in x])
        if np.all(check):
            return True
        # Do NOT change the "check == False" to "check is False" here!
        elif np.any(check == False):
            return False
        return None

    u = getattr(x, 'units', None)
    if isinstance(u, (config.ureg.Quantity, config.ureg.Unit)):
        if not u.unitless:
            return u.to_compact().units == config.ureg.Unit('um')

    return None
