
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

"""Module contains functions implementing NBLAST."""

import numbers
import os

import numpy as np
import pandas as pd

from concurrent.futures import ProcessPoolExecutor
from typing import Union, Optional
from typing_extensions import Literal

from .. import utils, config
from ..core import NeuronList, Dotprops, make_dotprops
from .base import Blaster

__all__ = ['nblast', 'nblast_smart', 'nblast_allbyall', 'sim_to_dist']

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

    @property
    def max_dist(self):
        """Max distance considered by the scoring matrix.

        Returns ``None`` if pass through.
        """
        if self.scoring_function == self.pass_through:
            return None
        # The last bin is always `np.inf`, so we need the second last bin
        return self.dist_bins[-2]

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
        >>> parse_intervals("(0,0.1]")                          # doctest: +SKIP
        0.1
        """
        return float(s.strip("([])").split(",")[-1])


class NBlaster(Blaster):
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
    normalized :    bool
                    If True, will normalize scores by the best possible score
                    (i.e. self-self) of the query neuron.
    smat :          str | pd.DataFrame
                    Score matrix. If 'auto' (default), will use scoring matrices
                    from FCWB. Same behaviour as in R's nat.nblast
                    implementation. If ``smat=None`` the scores will be
                    generated as the product of the distances and the dotproduct
                    of the vectors of nearest-neighbor pairs.
    limit_dist :    float | "auto" | None
                    Sets the max distance for the nearest neighbor search
                    (`distance_upper_bound`). Typically this should be the
                    highest distance considered by the scoring function. If
                    "auto", will extract that value from the scoring matrix.
    progress :      bool
                    If True, will show a progress bar.

    """

    def __init__(self, use_alpha=False, normalized=True, smat='auto',
                 limit_dist=None, progress=True):
        """Initialize class."""
        super().__init__(progress=progress)
        self.use_alpha = use_alpha
        self.normalized = normalized
        self.desc = "NBlasting"

        if smat == 'auto':
            if self.use_alpha:
                smat = pd.read_csv(f'{smat_path}/smat_alpha_fcwb.csv',
                                   index_col=0)
            else:
                smat = pd.read_csv(f'{smat_path}/smat_fcwb.csv',
                                   index_col=0)

        self.score_fn = ScoringFunction(smat)

        if limit_dist == 'auto':
            self.distance_upper_bound = self.score_fn.max_dist
        else:
            self.distance_upper_bound = limit_dist

    def append(self, dotprops):
        """Append dotprops."""
        if isinstance(dotprops, (NeuronList, list)):
            for n in dotprops:
                self.append(n)
            return

        if not isinstance(dotprops, Dotprops):
            raise ValueError(f'Expected Dotprops, got "{type(dotprops)}"')

        self.neurons.append(dotprops)
        self.ids.append(dotprops.id)
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

    def single_query_target(self, q_idx, t_idx, scores='forward'):
        """Query single target against single target."""
        # Take a short-cut if this is a self-self comparison
        if q_idx == t_idx:
            if self.normalized:
                return 1
            return self.self_hits[q_idx]

        # Run nearest-neighbor search for query against target
        data = self.neurons[q_idx].dist_dots(self.neurons[t_idx],
                                             alpha=self.use_alpha,
                                             distance_upper_bound=self.distance_upper_bound)
        if self.use_alpha:
            dists, dots, alpha = data
            dots *= np.sqrt(alpha)
        else:
            dists, dots = data

        scr = self.score_fn(dists, dots).sum()

        # Normalize against best hit
        if self.normalized:
            scr /= self.self_hits[q_idx]

        # For the mean score we also have to produce the reverse score
        if scores in ('mean', 'min', 'max'):
            reverse = self.single_query_target(t_idx, q_idx, scores='forward')
            if scores == 'mean':
                scr = (scr + reverse) / 2
            elif scores == 'min':
                scr = min(scr, reverse)
            elif scores == 'max':
                scr = max(scr, reverse)

        return scr


def nblast_smart(query: Union[Dotprops, NeuronList],
                 target: Optional[str] = None,
                 t: Union[int, float] = 90,
                 criterion: Union[Literal['percentile'],
                                  Literal['score'],
                                  Literal['N']] = 'percentile',
                 scores: Union[Literal['forward'],
                               Literal['mean'],
                               Literal['min'],
                               Literal['max']] = 'forward',
                 return_mask: bool = False,
                 normalized: bool = True,
                 use_alpha: bool = False,
                 smat: Optional[Union[str, pd.DataFrame]] = 'auto',
                 limit_dist: Optional[Union[Literal['auto'], int, float]] = 'auto',
                 n_cores: int = os.cpu_count() // 2,
                 progress: bool = True) -> pd.DataFrame:
    """Smart(er) NBLAST query against target neurons.

    In contrast to :func:`navis.nblast` this function will first run a
    "pre-NBLAST" in which only 10% of the query dotprops' points are used.
    Using those initial scores, we select for each query the highest scoring
    targets and run the full NBLAST only on those query-target pairs (see
    ``t`` and ``criterion`` for fine-tuning).

    Parameters
    ----------
    query :         Dotprops | NeuronList
                    Query neuron(s) to NBLAST against the targets. Neurons
                    should be in microns as NBLAST is optimized for that and
                    have similar sampling resolutions.
    target :        Dotprops | NeuronList, optional
                    Target neuron(s) to NBLAST against. Neurons should be in
                    microns as NBLAST is optimized for that and have
                    similar sampling resolutions. If not provided, will NBLAST
                    queries against themselves.
    t :             int | float
                    Determines for which pairs we will run a full NBLAST. See
                    ``criterion`` parameter for details.
    criterion :     "percentile" | "score" | "N"
                    Criterion for selecting query-target pairs for full NBLAST:

                      - "percentile" runs full NBLAST on the ``t``-th percentile
                      - "score" runs full NBLAST on all scores above ``t``
                      - "N" runs full NBLAST on top ``t`` targets

    return_mask :   bool
                    If True, will also return a boolean mask that shows which
                    scores are based on a full NBLAST and which ones only on
                    the pre-NBLAST.
    scores :        'forward' | 'mean' | 'min' | 'max'
                    Determines the final scores:

                      - 'forward' (default) returns query->target scores
                      - 'mean' returns the mean of query->target and
                        target->query scores
                      - 'min' returns the minium between query->target and
                        target->query scores
                      - 'max' returns the maximum between query->target and
                        target->query scores

    n_cores :       int, optional
                    Max number of cores to use for nblasting. Default is
                    ``os.cpu_count() // 2``. This should ideally be an even
                    number as that allows optimally splitting queries onto
                    individual processes.
    use_alpha :     bool, optional
                    Emphasizes neurons' straight parts (backbone) over parts
                    that have lots of branches.
    normalized :    bool, optional
                    Whether to return normalized NBLAST scores.
    smat :          str | pd.DataFrame
                    Score matrix. If 'auto' (default), will use scoring matrices
                    from FCWB. Same behaviour as in R's nat.nblast
                    implementation. If ``smat=None`` the scores will be
                    generated as the product of the distances and the dotproduct
                    of the vectors of nearest-neighbor pairs.
    limit_dist :    float | "auto" | None
                    Sets the max distance for the nearest neighbor search
                    (`distance_upper_bound`). Typically this should be the
                    highest distance considered by the scoring function. If
                    "auto", will extract that value from the scoring matrix.
    progress :      bool
                    Whether to show progress bars.

    Returns
    -------
    scores :        pandas.DataFrame
                    Matrix with NBLAST scores. Rows are query neurons, columns
                    are targets.
    mask :          np.ndarray
                    Only if ``return_mask=True``: a boolean mask with same shape
                    as ``scores`` that shows which scores are based on a full
                    NBLAST and which ones only on the pre-NBLAST.

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
    <Quantity([8 8 8 8 8], 'nanometer')>
    >>> # Convert to microns
    >>> nl_um = nl * (8 / 1000)
    >>> # Convert to dotprops
    >>> dps = navis.make_dotprops(nl_um)
    >>> # Run a NBLAST where only the top target from the pre-NBLAST is run
    >>> # through a full NBLAST
    >>> scores = navis.nblast_smart(dps[:3], dps[3:], t=1, criterion='N')

    See Also
    --------
    :func:`navis.nblast`
                The conventional full NBLAST.
    :func:`navis.nblast_allbyall`
                A more efficient way than ``nblast(query=x, target=x)``.
    :func:`navis.synblast`
                A synapse-based variant of NBLAST.

    """
    utils.eval_param(criterion, name='criterion',
                     allowed_values=("percentile", "score", "N"))

    if isinstance(target, type(None)):
        target = query

    try:
        t = int(t)
    except BaseException:
        raise TypeError(f'`t` must be (convertable to) integer - got "{type(t)}"')

    if criterion == 'percentile':
        if (t <= 0 or t >= 100):
            raise ValueError('Expected `t` to be integer between 0 and 100 for '
                             f'criterion "percentile", got {t}')
    elif criterion == 'N':
        if (t < 0 or t >= len(target)):
            raise ValueError('`t` must be between 0 and the total number of'
                             f'targets ({len(target)} for criterion "N", '
                             f'got {t}')

    # Make sure we're working on NeuronLists
    query_dps = NeuronList(query)
    target_dps = NeuronList(target)

    # Run NBLAST preflight checks
    nblast_preflight(query_dps, target_dps, n_cores, req_unique_ids=True)

    # Find an optimal partition that minimizes the number of neurons
    # we have to send to each process
    n_rows, n_cols = find_optimal_partition(n_cores, query_dps, target_dps)

    # Make simplified dotprops
    query_dps_simp = query_dps.downsample(10, inplace=False)
    target_dps_simp = target_dps.downsample(10, inplace=False)

    # First we NBLAST the highly simplified dotprops against another
    nblasters = []
    with config.tqdm(desc='Prep. pre-NBLAST',
                     total=n_rows * n_cols,
                     leave=False,
                     disable=not progress) as pbar:
        for qq in np.array_split(query_dps_simp, n_rows):
            for tt in np.array_split(target_dps_simp, n_cols):
                # Initialize NBlaster
                this = NBlaster(use_alpha=use_alpha,
                                normalized=normalized,
                                smat=smat,
                                limit_dist=limit_dist,
                                progress=progress)
                # Add queries and targets
                for n in qq:
                    this.append(n)
                for n in tt:
                    this.append(n)
                # Keep track of indices of queries and targets
                this.queries = np.arange(len(qq))
                this.targets = np.arange(len(tt)) + len(qq)
                this.pbar_position = len(nblasters)
                this.desc = 'Pre(NBLASTING)'

                nblasters.append(this)
                pbar.update()

    # If only one core, we don't need to break out the multiprocessing
    if n_cores == 1:
        scr = this.multi_query_target(this.queries,
                                      this.targets,
                                      scores=scores)
    else:
        with ProcessPoolExecutor(max_workers=len(nblasters)) as pool:
            # Each nblaster is passed to it's own process
            futures = [pool.submit(this.multi_query_target,
                                   q_idx=this.queries,
                                   t_idx=this.targets,
                                   scores=scores) for this in nblasters]

            results = [f.result() for f in futures]

        scr = pd.DataFrame(np.zeros((len(query_dps), len(target_dps))),
                           index=query_dps.id, columns=target_dps.id)

        for res in results:
            scr.loc[res.index, res.columns] = res.values

    # Now select targets of interest for each query
    if criterion == 'percentile':
        # Generate a mask for the scores we want to recalculate from full dotprops
        sel = np.percentile(scr, q=t, axis=1)
        mask = scr >= sel.reshape(-1, 1)
    elif criterion == 'score':
        # Generate a mask for the scores we want to recalculate from full dotprops
        sel = np.full(scr.shape[0], fill_value=t)
        mask = scr >= sel.reshape(-1, 1)
    else:
        # Sort such that the top hit is to the left
        srt = np.argsort(scr.values, axis=1)[:, ::-1]
        # Generate the mask
        mask = pd.DataFrame(np.zeros(scr.shape), dtype=bool,
                            columns=scr.columns, index=scr.index)
        _ = np.arange(mask.shape[0])
        for N in range(t):
            mask.values[_, srt[:, N]] = True

    # Now re-generate the NBLASTERs with the full dotprops
    nblasters = []
    with config.tqdm(desc='Preparing full NBLASTs',
                     total=n_rows * n_cols,
                     leave=False,
                     disable=not progress) as pbar:
        for q in np.array_split(np.arange(len(query_dps)), n_rows):
            for t in np.array_split(np.arange(len(target_dps)), n_cols):
                # Initialize NBlaster
                this = NBlaster(use_alpha=use_alpha,
                                normalized=normalized,
                                smat=smat,
                                limit_dist=limit_dist,
                                progress=progress)
                # Add queries and targets
                for n in query_dps[q]:
                    this.append(n)
                for n in target_dps[t]:
                    this.append(n)

                # Find the pairs to NBLAST in this part of the matrix
                submask = mask.loc[query_dps[q].id,
                                   target_dps[t].id]
                # `pairs` is an array of `[[query, target], [...]]` pairs
                this.pairs = np.vstack(np.where(submask)).T

                # Offset the query indices
                this.pairs[:, 1] += len(q)

                # Track this NBLASTER's mask relative to the original big one
                this.mask = np.zeros(mask.shape, dtype=bool)
                this.mask[q[0]:q[-1]+1, t[0]:t[-1]+1] = submask

                # Make sure position of progress bar checks out
                this.pbar_position = len(nblasters)

                nblasters.append(this)
                pbar.update()

    # If only one core, we don't need to break out the multiprocessing
    if n_cores == 1:
        scr[mask] = this.pair_query_target(this.pairs, scores=scores)
    else:
        with ProcessPoolExecutor(max_workers=len(nblasters)) as pool:
            # Each nblaster is passed to its own process
            futures = [pool.submit(this.pair_query_target,
                                   pairs=this.pairs,
                                   scores=scores) for this in nblasters]

            results = [f.result() for f in futures]

        for res, nbl in zip(results, nblasters):
            scr[nbl.mask] = res

    if return_mask:
        return scr, mask

    return scr


def nblast(query: Union[Dotprops, NeuronList],
           target: Optional[str] = None,
           scores: Union[Literal['forward'],
                         Literal['mean'],
                         Literal['min'],
                         Literal['max']] = 'forward',
           normalized: bool = True,
           use_alpha: bool = False,
           smat: Optional[Union[str, pd.DataFrame]] = 'auto',
           limit_dist: Optional[Union[Literal['auto'], int, float]] = None,
           n_cores: int = os.cpu_count() // 2,
           batch_size: Optional[int] = None,
           progress: bool = True) -> pd.DataFrame:
    """NBLAST query against target neurons.

    This implements the NBLAST algorithm from Costa et al. (2016) (see
    references) and mirror the implementation in R's ``nat.nblast``
    (https://github.com/natverse/nat.nblast).

    Parameters
    ----------
    query :         Dotprops | NeuronList
                    Query neuron(s) to NBLAST against the targets. Neurons
                    should be in microns as NBLAST is optimized for that and
                    have similar sampling resolutions.
    target :        Dotprops | NeuronList, optional
                    Target neuron(s) to NBLAST against. Neurons should be in
                    microns as NBLAST is optimized for that and have
                    similar sampling resolutions. If not provided, will NBLAST
                    queries against themselves.
    scores :        'forward' | 'mean' | 'min' | 'max'
                    Determines the final scores:

                      - 'forward' (default) returns query->target scores
                      - 'mean' returns the mean of query->target and
                        target->query scores
                      - 'min' returns the minium between query->target and
                        target->query scores
                      - 'max' returns the maximum between query->target and
                        target->query scores

    use_alpha :     bool, optional
                    Emphasizes neurons' straight parts (backbone) over parts
                    that have lots of branches.
    normalized :    bool, optional
                    Whether to return normalized NBLAST scores.
    smat :          str | pd.DataFrame
                    Score matrix. If 'auto' (default), will use scoring matrices
                    from FCWB. Same behaviour as in R's nat.nblast
                    implementation. If ``smat=None`` the scores will be
                    generated as the product of the distances and the dotproduct
                    of the vectors of nearest-neighbor pairs.
    limit_dist :    float | "auto" | None
                    Sets the max distance for the nearest neighbor search
                    (`distance_upper_bound`). Typically this should be the
                    highest distance considered by the scoring function. If
                    "auto", will extract that value from the scoring matrix.
    n_cores :       int, optional
                    Max number of cores to use for nblasting. Default is
                    ``os.cpu_count() - 2``. This should ideally be an even
                    number as that allows optimally splitting queries onto
                    individual processes.
    batch_size :    int, optional
                    Set the number of Dotprops per process. This can be useful
                    to reduce the memory footprint at the cost of longer run
                    times. By default (``None``), queries and targets will be
                    evenly distributed across n_cores. Ignored if ``n_cores=1``.
    progress :      bool
                    Whether to show progress bars.

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
    <Quantity([8 8 8 8 8], 'nanometer')>
    >>> # Convert to microns
    >>> nl_um = nl * (8 / 1000)
    >>> # Convert to dotprops
    >>> dps = navis.make_dotprops(nl_um)
    >>> # Run the nblast
    >>> scores = navis.nblast(dps[:3], dps[3:])

    See Also
    --------
    :func:`navis.nblast_allbyall`
                A more efficient way than ``nblast(query=x, target=x)``.
    :func:`navis.nblast_smart`
                A smart(er) NBLAST suited for very large NBLAST.
    :func:`navis.synblast`
                A synapse-based variant of NBLAST.

    """
    # Make sure we're working on NeuronLists
    query_dps = NeuronList(query)
    target_dps = NeuronList(target)

    # Run NBLAST preflight checks
    nblast_preflight(query_dps, target_dps, n_cores,
                     batch_size=batch_size, req_unique_ids=True)

    # Find an optimal partition that minimizes the number of neurons
    # we have to send to each process
    if not batch_size or n_cores == 1:
        n_rows, n_cols = find_optimal_partition(n_cores, query_dps, target_dps)
    else:
        n_rows, n_cols = find_batch_partition(batch_size, query_dps, target_dps)

    nblasters = []
    with config.tqdm(desc='Preparing',
                     total=n_rows * n_cols,
                     leave=False,
                     disable=not progress) as pbar:
        for q in np.array_split(query_dps, n_rows):
            for t in np.array_split(target_dps, n_cols):
                # Initialize NBlaster
                this = NBlaster(use_alpha=use_alpha,
                                normalized=normalized,
                                smat=smat,
                                limit_dist=limit_dist,
                                progress=progress)

                # Use better description if we process in batches
                if batch_size:
                    this.desc += f' (batch {len(nblasters) + 1}/{n_rows * n_cols})'

                # Add queries and targets
                for n in q:
                    this.append(n)
                for n in t:
                    this.append(n)
                # Keep track of indices of queries and targets
                this.queries = np.arange(len(q))
                this.targets = np.arange(len(t)) + len(q)
                this.pbar_position = len(nblasters) % n_cores

                nblasters.append(this)
                pbar.update(1)

    # If only one core, we don't need to break out the multiprocessing
    if n_cores == 1:
        return this.multi_query_target(this.queries,
                                       this.targets,
                                       scores=scores)

    with ProcessPoolExecutor(max_workers=n_cores) as pool:
        # Each nblaster is passed to its own process
        futures = [pool.submit(this.multi_query_target,
                               q_idx=this.queries,
                               t_idx=this.targets,
                               scores=scores) for this in nblasters]

        results = [f.result() for f in futures]

    scores = pd.DataFrame(np.zeros((len(query_dps), len(target_dps))),
                          index=query_dps.id, columns=target_dps.id)

    for res in results:
        scores.loc[res.index, res.columns] = res.values

    return scores


def nblast_allbyall(x: NeuronList,
                    normalized: bool = True,
                    use_alpha: bool = False,
                    smat: Optional[Union[str, pd.DataFrame]] = 'auto',
                    limit_dist: Optional[Union[Literal['auto'], int, float]] = None,
                    n_cores: int = os.cpu_count() // 2,
                    progress: bool = True) -> pd.DataFrame:
    """All-by-all NBLAST of inputs neurons.

    A slightly more efficient way of running ``nblast(query=x, target=x)``.

    Parameters
    ----------
    x :             Dotprops | NeuronList
                    Neuron(s) to NBLAST against each other. Neurons should
                    be in microns as NBLAST is optimized for that and have
                    similar sampling resolutions.
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
    smat :          str | pd.DataFrame, optional
                    Score matrix. If 'auto' (default), will use scoring matrices
                    from FCWB. Same behaviour as in R's nat.nblast
                    implementation. If ``smat=None`` the scores will be
                    generated as the product of the distances and the dotproduct
                    of the vectors of nearest-neighbor pairs.
    limit_dist :    float | "auto" | None
                    Sets the max distance for the nearest neighbor search
                    (`distance_upper_bound`). Typically this should be the
                    highest distance considered by the scoring function. If
                    "auto", will extract that value from the scoring matrix.
    progress :      bool
                    Whether to show progress bars.

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
    <Quantity([8 8 8 8 8], 'nanometer')>
    >>> # Convert to microns
    >>> nl_um = nl * (8 / 1000)
    >>> # Make dotprops
    >>> dps = navis.make_dotprops(nl_um)
    >>> # Run the nblast
    >>> scores = navis.nblast_allbyall(dps)

    See Also
    --------
    :func:`navis.nblast`
                For generic query -> target nblasts.

    """
    # Make sure we're working on NeuronLists
    dps = NeuronList(x)

    # Run NBLAST preflight checks
    # Note that we are passing the same dotprops twice to avoid having to
    # change the function's signature. Should have little to no overhead.
    nblast_preflight(dps, dps, n_cores, req_unique_ids=True)

    # Find an optimal partition that minimizes the number of neurons
    # we have to send to each process
    n_rows, n_cols = find_optimal_partition(n_cores, dps, dps)

    nblasters = []
    with config.tqdm(desc='Preparing',
                     total=n_rows * n_cols,
                     leave=False,
                     disable=not progress) as pbar:
        for qix in np.array_split(np.arange(len(dps)), n_rows):
            for tix in np.array_split(np.arange(len(dps)), n_cols):
                # Initialize NBlaster
                this = NBlaster(use_alpha=use_alpha,
                                normalized=normalized,
                                smat=smat,
                                limit_dist=limit_dist,
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
                pbar.update()

    # If only one core, we don't need to break out the multiprocessing
    if n_cores == 1:
        return this.all_by_all()

    with ProcessPoolExecutor(max_workers=len(nblasters)) as pool:
        # Each nblaster is passed to it's own process
        futures = [pool.submit(this.multi_query_target,
                               q_idx=this.queries,
                               t_idx=this.targets,
                               scores='forward') for this in nblasters]

        results = [f.result() for f in futures]

    scores = pd.DataFrame(np.zeros((len(dps), len(dps))),
                          index=dps.id, columns=dps.id)

    for res in results:
        scores.loc[res.index, res.columns] = res.values

    return scores


def find_batch_partition(N, q, t):
    """Find partition such that each batch contains at most `N` neurons."""
    import math
    # Number of rows (i.e. how many splits for queries)
    n_rows = max(math.floor(len(q) / (N / 2)), 1)

    # Given the number of queries per batch, how many targets can we have
    remaining = N - (len(q) / n_rows)

    n_cols = max(math.floor(len(t) / remaining), 1)

    return int(n_rows), int(n_cols)


def find_optimal_partition(n_cores, q, t):
    """Find an optimal partition for given NBLAST query."""
    # Find an optimal partition that minimizes the number of neurons
    # we have to send to each process
    neurons_per_query = []
    for n_rows in range(1, n_cores + 1):
        # Skip splits we can't make
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
        dp = [force_dotprops(n, k, resample) for n in config.tqdm(x,
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
    microns = [config.ureg.Unit('microns'),
               config.ureg.Unit('um'),
               config.ureg.Unit('micrometer'),
               config.ureg.Unit('dimensionless')]
    if not isinstance(x, NeuronList):
        x = NeuronList(x)

    # For very large NeuronLists converting the unit string to pint units is
    # the time consuming step. Here we will first reduce to unique units:
    unit_str = []
    for n in x:
        if isinstance(n._unit_str, str):
            unit_str.append(n._unit_str)
        else:
            unit_str += list(n._unit_str)
    unit_str = np.unique(unit_str)

    any_not_microns = False
    all_units = True
    for u in unit_str:
        # If not a unit (i.e. `None`)
        if not u:
            all_units = False
            continue

        # Convert to proper unit
        u = config.ureg(u).to_compact().units

        if u not in microns:
            any_not_microns = True

    if any_not_microns:
        return False
    elif all_units:
        return True
    return None


def align_dtypes(*x, downcast=False, inplace=True):
    """Align data types of dotprops.

    Parameters
    ----------
    *x :        Dotprops | NeuronLists thereof
    downcast :  bool
                If True, will downcast all points to the lowest precision
                dtype.
    inplace :   bool
                If True, will modify the original neuron objects. If False, will
                make a copy before changing dtypes.

    Returns
    -------
    *x
                Input data with aligned dtypes.

    """
    dtypes = get_dtypes(x)

    if len(dtypes) == 1:
        return x

    if not inplace:
        for i in range(x):
            x[i] = x[i].copy()

    if downcast:
        target = lowest_type(*x)
    else:
        target = np.result_type(*dtypes)

    for n in x:
        if isinstance(n, NeuronList):
            for i in range(len(n)):
                n[i].points = n[i].points.astype(target, copy=False)
        elif isinstance(n, Dotprops):
            n.points = n.points.astype(target, copy=False)
        else:
            raise TypeError(f'Unable to process "{type(n)}"')

    return x


def get_dtypes(*x):
    """Collect data types of dotprops points."""
    dtypes = set()
    for n in x:
        if isinstance(n, NeuronList):
            dtypes = dtypes | get_dtypes(n)
        elif isinstance(n, Dotprops):
            dtypes.add(n.points.dtype)
        else:
            raise TypeError(f'Unable to process "{type(n)}"')
    return dtypes


def lowest_type(*x):
    """Find the lowest data type."""
    dtypes = get_dtypes(x)

    if len(dtypes) == 1:
        return dtypes[0]

    lowest = dtypes[0]
    for dt in dtypes[1:]:
        lowest = demote_types(lowest, dt)

    return lowest


def demote_types(a, b):
    """Determine the lower of two dtypes."""
    if isinstance(a, np.ndarray):
        a = a.dtype
    if isinstance(b, np.ndarray):
        b = b.dtype

    # No change is same
    if a == b:
        return a

    # First, get the "higher" type
    higher = np.promote_types(a, b)
    # Now get the one that's not higher
    if a != higher:
        return a
    return b


def sim_to_dist(x):
    """Convert similarity scores to distances.

    Parameters
    ----------
    x :     (M, M) np.ndarray | pandas.DataFrame
            Similarity score matrix to invert.

    Returns
    -------
    distances

    """
    if not isinstance(x, (np.ndarray, pd.DataFrame)):
        raise TypeError(f'Expected numpy array or pandas DataFrame, got "{type(x)}"')

    if isinstance(x, pd.DataFrame):
        mx = x.values.max()
    else:
        mx = x.max()

    return (x - mx) * -1


def nblast_preflight(query, target, n_cores, batch_size=None,
                     req_unique_ids=False, req_dotprops=True):
    """Run preflight checks for NBLAST."""
    if req_dotprops:
        if query.types != (Dotprops, ):
            raise TypeError(f'`query` must be Dotprop(s), got "{query.types}". '
                            'Use `navis.make_dotprops` to convert neurons.')

        if target.types != (Dotprops, ):
            raise TypeError(f'`target` must be Dotprop(s), got "{target.types}". '
                            'Use `navis.make_dotprops` to convert neurons.')

    if req_unique_ids:
        # At the moment, neurons need to have a unique ID for things to work
        if query.is_degenerated:
            raise ValueError('Queries have non-unique IDs.')
        if target.is_degenerated:
            raise ValueError('Targets have non-unique IDs.')

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
    elif n_cores < 1:
        raise ValueError('`n_cores` must not be smaller than 1')
    elif n_cores > os.cpu_count():
        logger.warning('`n_cores` should not larger than the number of '
                       'available cores')

    if batch_size is not None:
        if batch_size <= 0:
            raise ValueError('`batch_size` must be >= 1 or `None`.')
