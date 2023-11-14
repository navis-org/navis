
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

import time
import numbers
import os
import operator
from functools import partial

import numpy as np
import pandas as pd
import multiprocessing as mp

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Dict, Union, Optional
from typing_extensions import Literal

from navis.nbl.smat import Lookup2d, smat_fcwb, _nblast_v1_scoring

from .. import utils, config
from ..core import NeuronList, Dotprops, make_dotprops
from .base import Blaster, NestedIndices

__all__ = ['nblast', 'nblast_smart', 'nblast_allbyall', 'sim_to_dist']

fp = os.path.dirname(__file__)
smat_path = os.path.join(fp, 'score_mats')

logger = config.get_logger(__name__)

# This multiplier controls job size for NBLASTs (only relevant for
# multiprocessing and if progress=True).
# Larger multiplier = larger job sizes = fewer jobs = slower updates & less overhead
# Smaller multiplier = smaller job sizes = more jobs = faster updates & more overhead
JOB_SIZE_MULTIPLIER = 1
JOB_MAX_TIME_SECONDS = 60 * 30

# This controls how many threads we allow pykdtree to use during multi-core
# NBLAST
OMP_NUM_THREADS_LIMIT = 1

ALLOWED_SCORES = ('forward', 'mean', 'min', 'max', 'both')


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
    smat :          navis.nbl.smat.Lookup2d | pd.DataFrame | str | Callable
                    How to convert the point match pairs into an NBLAST score,
                    usually by a lookup table:
                     - if 'auto' (default), will use the "official" NBLAST scoring
                       matrices based on FCWB data. Same behaviour as in R's
                       nat.nblast implementation.
                     - if ``smat='v1'`` it uses the analytic formulation of the
                       NBLAST scoring from Kohl et. al (2013). You can adjust parameter
                       ``sigma_scaling`` (default to 10) using ``smat_kwargs``.
                     - DataFrames will be used to build a ``Lookup2d``
                     - if ``Callable`` given, it passes distance and dot products as
                       first and second argument respectively
                     - if ``smat=None`` the scores will be generated as the
                       product of the distances and the dotproduct of the vectors
                       of nearest-neighbor pairs
    smat_kwargs:    Dictionary with additional parameters passed to scoring
                    functions. For example: ``smat_kwargs["sigma_scoring"] = 10``.
    limit_dist :    float | "auto" | None
                    Sets the max distance for the nearest neighbor search
                    (`distance_upper_bound`). Typically this should be the
                    highest distance considered by the scoring function. If
                    "auto", will extract that value from the first axis of the
                    scoring matrix.
    progress :      bool
                    If True, will show a progress bar.

    """

    def __init__(self, use_alpha=False, normalized=True, smat='auto',
                 limit_dist=None, approx_nn=False, dtype=np.float64,
                 progress=True, smat_kwargs=dict()):
        """Initialize class."""
        super().__init__(progress=progress, dtype=dtype)
        self.use_alpha = use_alpha
        self.normalized = normalized
        self.approx_nn = approx_nn
        self.desc = "NBlasting"

        if smat is None:
            self.score_fn = operator.mul
        elif smat == 'auto':
            self.score_fn = smat_fcwb(self.use_alpha)
        elif smat == 'v1':
            self.score_fn = partial(
                _nblast_v1_scoring, sigma_scoring = smat_kwargs.get('sigma_scoring', 10)
            )
        elif isinstance(smat, pd.DataFrame):
            self.score_fn = Lookup2d.from_dataframe(smat)
        else:
            self.score_fn = smat

        if limit_dist == "auto":
            try:
                if self.score_fn.axes[0].boundaries[-1] != np.inf:
                    self.distance_upper_bound = self.score_fn.axes[0].boundaries[-1]
                else:
                    # If the right boundary is open (i.e. infinity), we will use
                    # the second highest boundary plus a 5% offset
                    self.distance_upper_bound = self.score_fn.axes[0].boundaries[-2] * 1.05
            except AttributeError:
                logger.warning("Could not infer distance upper bound from scoring function")
                self.distance_upper_bound = None
        else:
            self.distance_upper_bound = limit_dist

    def append(self, dotprops: Dotprops, self_hit: Optional[float] = None) -> NestedIndices:
        """Append dotprops.

        Returns the numerical index appended dotprops.
        If dotprops is a (possibly nested) sequence of dotprops,
        return a (possibly nested) list of indices.

        Note that `self_hit` is ignored (and hence calculated from scratch)
        when `dotprops` is a nested list of dotprops.
        """
        if isinstance(dotprops, Dotprops):
            return self._append_dotprops(dotprops, self_hit=self_hit)

        try:
            return [self.append(n) for n in dotprops]
        except TypeError:  # i.e. not iterable
            raise ValueError(f"Expected Dotprops or iterable thereof; got {type(dotprops)}")

    def _append_dotprops(self, dotprops: Dotprops, self_hit: Optional[float] = None) -> int:
        next_id = len(self)
        self.neurons.append(dotprops)
        self.ids.append(dotprops.id)
        # Calculate score for self hit
        if not self_hit:
            self.self_hits.append(self.calc_self_hit(dotprops))
        else:
            self.self_hits.append(self_hit)
        return next_id

    def calc_self_hit(self, dotprops):
        """Non-normalized value for self hit."""
        if not self.use_alpha:
            return len(dotprops.points) * self.score_fn(0, 1.0)
        else:
            dists = np.repeat(0, len(dotprops.points))
            alpha = dotprops.alpha * dotprops.alpha
            dots = np.repeat(1, len(dotprops.points)) * np.sqrt(alpha)
            return self.score_fn(dists, dots).sum()

    def single_query_target(self, q_idx: int, t_idx: int, scores='forward'):
        """Query single target against single target."""
        # Take a short-cut if this is a self-self comparison
        if q_idx == t_idx:
            if self.normalized:
                return 1
            return self.self_hits[q_idx]

        # Run nearest-neighbor search for query against target
        data = self.neurons[q_idx].dist_dots(self.neurons[t_idx],
                                             alpha=self.use_alpha,
                                             # eps=0.1 means we accept 10% inaccuracy
                                             eps=.1 if self.approx_nn else 0,
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
        if scores in ('mean', 'min', 'max', 'both'):
            reverse = self.single_query_target(t_idx, q_idx, scores='forward')
            if scores == 'mean':
                scr = (scr + reverse) / 2
            elif scores == 'min':
                scr = min(scr, reverse)
            elif scores == 'max':
                scr = max(scr, reverse)
            elif scores == 'both':
                # If both scores are requested
                scr = [scr, reverse]

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
                 approx_nn: bool = False,
                 precision: Union[int, str, np.dtype] = 64,
                 n_cores: int = os.cpu_count() // 2,
                 progress: bool = True,
                 smat_kwargs: Optional[Dict] = dict()) -> pd.DataFrame:
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
    use_alpha :     bool, optional
                    Emphasizes neurons' straight parts (backbone) over parts
                    that have lots of branches.
    normalized :    bool, optional
                    Whether to return normalized NBLAST scores.
    smat :          str | pd.DataFrame | Callable
                    Score matrix. If 'auto' (default), will use scoring matrices
                    from FCWB. Same behaviour as in R's nat.nblast
                    implementation.
                    If ``smat='v1'`` it uses the analytic formulation of the
                    NBLAST scoring from Kohl et. al (2013). You can adjust parameter
                    ``sigma_scaling`` (default to 10) using ``smat_kwargs``.
                    If ``smat=None`` the scores will be
                    generated as the product of the distances and the dotproduct
                    of the vectors of nearest-neighbor pairs.
                    If ``Callable`` given, it passes distance and dot products as
                    first and second argument respectively.
    smat_kwargs:    Dictionary with additional parameters passed to scoring
                    functions.
    limit_dist :    float | "auto" | None
                    Sets the max distance for the nearest neighbor search
                    (`distance_upper_bound`). Typically this should be the
                    highest distance considered by the scoring function. If
                    "auto", will extract that value from the scoring matrix.
                    While this can give a ~2X speed up, it will introduce slight
                    inaccuracies because we won't have a vector component for
                    points without a nearest neighbour within the distance
                    limits. The impact depends on the scoring function but with
                    the default FCWB ``smat``, this is typically limited to the
                    third decimal (0.0086 +/- 0.0027 for an all-by-all of 1k
                    neurons).
    approx_nn :     bool
                    If True, will use approximate nearest neighbors. This gives
                    a >2X speed up but also produces only approximate scores.
                    Impact depends on the use case - testing highly recommended!
    precision :     int [16, 32, 64] | str [e.g. "float64"] | np.dtype
                    Precision for scores. Defaults to 64 bit (double) floats.
                    This is useful to reduce the memory footprint for very large
                    matrices. In real-world scenarios 32 bit (single)- and
                    depending on the purpose even 16 bit (half) - are typically
                    sufficient.
    n_cores :       int, optional
                    Max number of cores to use for nblasting. Default is
                    ``os.cpu_count() // 2``. This should ideally be an even
                    number as that allows optimally splitting queries onto
                    individual processes.
    progress :      bool
                    Whether to show progress bars.

    Returns
    -------
    scores :        pandas.DataFrame
                    Matrix with NBLAST scores. Rows are query neurons, columns
                    are targets. The order is the same as in ``query``/``target``
                    and the labels are based on the neurons' ``.id`` property.
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
    utils.eval_param(scores, name='scores', allowed_values=ALLOWED_SCORES)

    # We will make a couple tweaks for speed things up if this is
    # an all-by-all NBLAST
    aba = False
    pre_scores = scores
    if isinstance(target, type(None)):
        target = query
        aba = True
        # For all-by-all's we can compute only forward scores and
        # produce the mean later
        if scores == 'mean':
            pre_scores = 'forward'

    try:
        t = int(t)
    except BaseException:
        raise TypeError(f'`t` must be (convertable to) integer - got "{type(t)}"')

    if criterion == 'percentile':
        if (t <= 0 or t >= 100):
            raise ValueError('Expected `t` to be integer between 0 and 100 for '
                             f'criterion "percentile", got {t}')
    elif criterion == 'N':
        if (t < 0 or t > len(target)):
            raise ValueError('`t` must be between 0 and the total number of '
                             f'targets ({len(target)}) for criterion "N", '
                             f'got {t}')

    # Make sure we're working on NeuronLists
    query_dps = NeuronList(query)
    target_dps = NeuronList(target)

    # Run NBLAST preflight checks
    nblast_preflight(query_dps, target_dps, n_cores,
                     req_unique_ids=True,
                     req_microns=isinstance(smat, str) and smat=='auto')

    # Make simplified dotprops
    query_dps_simp = query_dps.downsample(10, inplace=False)
    if not aba:
        target_dps_simp = target_dps.downsample(10, inplace=False)
    else:
        target_dps_simp = query_dps_simp

    # Find a partition that produces batches that each run in approximately
    # 10 seconds
    if n_cores and n_cores > 1:
        if progress:
            # If progress bar, we need to make smaller mini batches.
            # These mini jobs must not be too small - otherwise the overhead
            # from spawning and sending results between processes slows things
            # down dramatically. Hence we want to make sure that each job runs
            # for >10s. The run time depends on the system and how big the neurons
            # are. Here, we run a quick test and try to extrapolate from there
            n_rows, n_cols = find_batch_partition(query_dps_simp, target_dps_simp,
                                                  T=10 * JOB_SIZE_MULTIPLIER)
        else:
            # If no progress bar needed, we can just split neurons evenly across
            # all available cores
            n_rows, n_cols = find_optimal_partition(n_cores, query_dps_simp, target_dps_simp)
    else:
        n_rows = n_cols = 1

    # Calculate self-hits once for all neurons
    nb = NBlaster(use_alpha=use_alpha,
                  normalized=normalized,
                  smat=smat,
                  limit_dist=limit_dist,
                  dtype=precision,
                  approx_nn=approx_nn,
                  progress=progress,
                  smat_kwargs=smat_kwargs)
    query_self_hits = np.array([nb.calc_self_hit(n) for n in query_dps_simp])
    target_self_hits = np.array([nb.calc_self_hit(n) for n in target_dps_simp])

    # This makes sure we don't run into multiple layers of concurrency
    with set_omp_flag(limits=OMP_NUM_THREADS_LIMIT if n_cores and (n_cores > 1) else None):
        # Initialize a pool of workers
        # Note that we're forcing "spawn" instead of "fork" (default on linux)!
        # This is to reduce the memory footprint since "fork" appears to inherit all
        # variables (including all neurons) while "spawn" appears to get only
        # what's required to run the job?
        with ProcessPoolExecutor(max_workers=n_cores,
                                 mp_context=mp.get_context('spawn')) as pool:
            with config.tqdm(desc='Prep. pre-NBLAST',
                             total=n_rows * n_cols,
                             leave=False,
                             disable=not progress) as pbar:
                futures = {}
                nblasters = []
                for qix in np.array_split(np.arange(len(query_dps_simp)), n_rows):
                    for tix in np.array_split(np.arange(len(target_dps_simp)), n_cols):
                        # Initialize NBlaster
                        this = NBlaster(use_alpha=use_alpha,
                                        normalized=normalized,
                                        smat=smat,
                                        limit_dist=limit_dist,
                                        dtype=precision,
                                        approx_nn=approx_nn,
                                        progress=progress,
                                        smat_kwargs=smat_kwargs)

                        # Add queries and targets
                        for i, ix in enumerate(qix):
                            this.append(query_dps_simp[ix], query_self_hits[ix])
                        for i, ix in enumerate(tix):
                            this.append(target_dps_simp[ix], target_self_hits[ix])

                        # Keep track of indices of queries and targets
                        this.queries = np.arange(len(qix))
                        this.targets = np.arange(len(tix)) + len(qix)
                        this.queries_ix = qix  # this facilitates filling in the big matrix later
                        this.targets_ix = tix  # this facilitates filling in the big matrix later
                        this.pbar_position = len(nblasters) if not utils.is_jupyter() else None

                        nblasters.append(this)
                        pbar.update()

                        # If multiple cores requested, submit job to the pool right away
                        if n_cores and n_cores > 1 and (n_cols > 1 or n_rows > 1):
                            this.progress=False  # no progress bar for individual NBLASTERs
                            futures[pool.submit(this.multi_query_target,
                                                q_idx=this.queries,
                                                t_idx=this.targets,
                                                scores=pre_scores)] = this

            # Collect results
            if futures and len(futures) > 1:
                # Prepare empty score matrix
                scr = pd.DataFrame(np.empty((len(query_dps_simp),
                                             len(target_dps_simp)),
                                            dtype=this.dtype),
                                      index=query_dps_simp.id,
                                      columns=target_dps_simp.id)
                scr.index.name = 'query'
                scr.columns.name = 'target'

                # Collect results
                # We're dropping the "N / N_total" bit from the progress bar because
                # it's not helpful here
                fmt = ('{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}]')
                for f in config.tqdm(as_completed(futures),
                                     desc='Pre-NBLASTs',
                                     bar_format=fmt,
                                     total=len(futures),
                                     smoothing=0,
                                     disable=not progress,
                                     leave=False):
                    res = f.result()
                    this = futures[f]
                    # Fill-in big score matrix
                    scr.iloc[this.queries_ix, this.targets_ix] = res.values
            else:
                scr = this.multi_query_target(this.queries,
                                              this.targets,
                                              scores=scores)

    # If this is an all-by-all and we would have computed only forward scores
    # during pre-NBLAST
    if aba and scores == 'mean':
        scr = (scr + scr.T.values) / 2

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
        mask = pd.DataFrame(np.zeros(scr.shape, dtype=bool),
                            columns=scr.columns, index=scr.index)
        _ = np.arange(mask.shape[0])
        for N in range(t):
            mask.values[_, srt[:, N]] = True

    # Calculate self-hits for full neurons
    query_self_hits = np.array([nb.calc_self_hit(n) for n in query_dps])
    target_self_hits = np.array([nb.calc_self_hit(n) for n in target_dps])

    # This makes sure we don't run into multiple layers of concurrency
    with set_omp_flag(limits=OMP_NUM_THREADS_LIMIT if n_cores and (n_cores > 1) else None):
        # Initialize a pool of workers
        # Note that we're forcing "spawn" instead of "fork" (default on linux)!
        # This is to reduce the memory footprint since "fork" appears to inherit all
        # variables (including all neurons) while "spawn" appears to get only
        # what's required to run the job?
        with ProcessPoolExecutor(max_workers=n_cores,
                                 mp_context=mp.get_context('spawn')) as pool:
            with config.tqdm(desc='Prep. full NBLAST',
                             total=n_rows * n_cols,
                             leave=False,
                             disable=not progress) as pbar:
                futures = {}
                nblasters = []
                for qix in np.array_split(np.arange(len(query_dps)), n_rows):
                    for tix in np.array_split(np.arange(len(target_dps)), n_cols):
                        # Initialize NBlaster
                        this = NBlaster(use_alpha=use_alpha,
                                        normalized=normalized,
                                        smat=smat,
                                        limit_dist=limit_dist,
                                        dtype=precision,
                                        approx_nn=approx_nn,
                                        progress=progress,
                                        smat_kwargs=smat_kwargs)
                        # Add queries and targets
                        for i, ix in enumerate(qix):
                            this.append(query_dps[ix], query_self_hits[ix])
                        for i, ix in enumerate(tix):
                            this.append(target_dps[ix], target_self_hits[ix])

                        # Find the pairs to NBLAST in this part of the matrix
                        submask = mask.loc[query_dps[qix].id,
                                           target_dps[tix].id]
                        # `pairs` is an array of `[[query, target], [...]]` pairs
                        this.pairs = np.vstack(np.where(submask)).T

                        # Offset the query indices
                        this.pairs[:, 1] += len(qix)

                        # Track this NBLASTER's mask relative to the original big one
                        this.mask = np.zeros(mask.shape, dtype=bool)
                        this.mask[qix[0]:qix[-1]+1, tix[0]:tix[-1]+1] = submask

                        # Make sure position of progress bar checks out
                        this.pbar_position = len(nblasters) if not utils.is_jupyter() else None
                        this.desc = 'Full NBLAST'

                        nblasters.append(this)
                        pbar.update()

                        # If multiple cores requested, submit job to the pool right away
                        if n_cores and n_cores > 1 and (n_cols > 1 or n_rows > 1):
                            this.progress=False  # no progress bar for individual NBLASTERs
                            futures[pool.submit(this.pair_query_target,
                                                pairs=this.pairs,
                                                scores=scores)] = this

            # Collect results
            if futures and len(futures) > 1:
                # Collect results
                # We're dropping the "N / N_total" bit from the progress bar because
                # it's not helpful here
                fmt = ('{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}]')
                for f in config.tqdm(as_completed(futures),
                                     desc='NBLASTing',
                                     bar_format=fmt,
                                     total=len(futures),
                                     smoothing=0,
                                     disable=not progress,
                                     leave=False):
                    res = f.result()
                    this = futures[f]

                    # Fill-in big score matrix
                    scr[this.mask] = res
            else:
                scr[mask] = this.pair_query_target(this.pairs, scores=scores)

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
           smat: Optional[Union[str, pd.DataFrame, Callable]] = 'auto',
           limit_dist: Optional[Union[Literal['auto'], int, float]] = None,
           approx_nn: bool = False,
           precision: Union[int, str, np.dtype] = 64,
           n_cores: int = os.cpu_count() // 2,
           progress: bool = True,
           smat_kwargs: Optional[Dict] = dict()) -> pd.DataFrame:
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
    scores :        'forward' | 'mean' | 'min' | 'max' | 'both'
                    Determines the final scores:

                      - 'forward' (default) returns query->target scores
                      - 'mean' returns the mean of query->target and
                        target->query scores
                      - 'min' returns the minium between query->target and
                        target->query scores
                      - 'max' returns the maximum between query->target and
                        target->query scores
                      - 'both' will return foward and reverse scores as
                        multi-index DataFrame

    use_alpha :     bool, optional
                    Emphasizes neurons' straight parts (backbone) over parts
                    that have lots of branches.
    normalized :    bool, optional
                    Whether to return normalized NBLAST scores.
    smat :          str | pd.DataFrame | Callable
                    Score matrix. If 'auto' (default), will use scoring matrices
                    from FCWB. Same behaviour as in R's nat.nblast
                    implementation.
                    If ``smat='v1'`` it uses the analytic formulation of the
                    NBLAST scoring from Kohl et. al (2013). You can adjust parameter
                    ``sigma_scaling`` (default to 10) using ``smat_kwargs``.
                    If ``Callable`` given, it passes distance and dot products as
                    first and second argument respectively.
                    If ``smat=None`` the scores will be
                    generated as the product of the distances and the dotproduct
                    of the vectors of nearest-neighbor pairs.
    limit_dist :    float | "auto" | None
                    Sets the max distance for the nearest neighbor search
                    (`distance_upper_bound`). Typically this should be the
                    highest distance considered by the scoring function. If
                    "auto", will extract that value from the scoring matrix.
                    While this can give a ~2X speed up, it will introduce slight
                    inaccuracies because we won't have a vector component for
                    points without a nearest neighbour within the distance
                    limits. The impact depends on the scoring function but with
                    the default FCWB ``smat``, this is typically limited to the
                    third decimal (0.0086 +/- 0.0027 for an all-by-all of 1k
                    neurons).
    approx_nn :     bool
                    If True, will use approximate nearest neighbors. This gives
                    a >2X speed up but also produces only approximate scores.
                    Impact depends on the use case - testing highly recommended!
    n_cores :       int, optional
                    Max number of cores to use for nblasting. Default is
                    ``os.cpu_count() // 2``. This should ideally be an even
                    number as that allows optimally splitting queries onto
                    individual processes.
    precision :     int [16, 32, 64] | str [e.g. "float64"] | np.dtype
                    Precision for scores. Defaults to 64 bit (double) floats.
                    This is useful to reduce the memory footprint for very large
                    matrices. In real-world scenarios 32 bit (single)- and
                    depending on the purpose even 16 bit (half) - are typically
                    sufficient.
    progress :      bool
                    Whether to show progress bars. This may cause some overhead,
                    so switch off if you don't really need it.
    smat_kwargs:    Dictionary with additional parameters passed to scoring
                    functions.

    Returns
    -------
    scores :        pandas.DataFrame
                    Matrix with NBLAST scores. Rows are query neurons, columns
                    are targets. The order is the same as in ``query``/``target``
                    and the labels are based on the neurons' ``.id`` property.

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
    utils.eval_param(scores, name='scores', allowed_values=ALLOWED_SCORES)

    if isinstance(target, type(None)):
        target = query

    # Make sure we're working on NeuronLists
    query_dps = NeuronList(query)
    target_dps = NeuronList(target)

    # Run NBLAST preflight checks
    nblast_preflight(query_dps, target_dps, n_cores,
                     req_unique_ids=True,
                     req_microns=isinstance(smat, str) and smat=='auto')

    # Find a partition that produces batches that each run in approximately
    # 10 seconds
    if n_cores and n_cores > 1:
        if progress:
            # If progress bar, we need to make smaller mini batches.
            # These mini jobs must not be too small - otherwise the overhead
            # from spawning and sending results between processes slows things
            # down dramatically. Hence we want to make sure that each job runs
            # for >10s. The run time depends on the system and how big the neurons
            # are. Here, we run a quick test and try to extrapolate from there
            n_rows, n_cols = find_batch_partition(query_dps, target_dps,
                                                  T=10 * JOB_SIZE_MULTIPLIER)
        else:
            # If no progress bar needed, we could just split neurons evenly across
            # all available cores but that can lead to one core lagging behind
            # and finishing much later than all the others. To avoid this, we
            # should aim for each batch to finish in a certain amount of time
            n_rows, n_cols = find_batch_partition(query_dps, target_dps,
                                                  T=JOB_MAX_TIME_SECONDS)
            if (n_rows * n_cols) < n_cores:
                n_rows, n_cols = find_optimal_partition(n_cores, query_dps, target_dps)
    else:
        n_rows = n_cols = 1

    # Calculate self-hits once for all neurons
    nb = NBlaster(use_alpha=use_alpha,
                  normalized=normalized,
                  smat=smat,
                  limit_dist=limit_dist,
                  dtype=precision,
                  approx_nn=approx_nn,
                  progress=progress,
                  smat_kwargs=smat_kwargs)
    query_self_hits = np.array([nb.calc_self_hit(n) for n in query_dps])
    target_self_hits = np.array([nb.calc_self_hit(n) for n in target_dps])

    # This makes sure we don't run into multiple layers of concurrency
    with set_omp_flag(limits=OMP_NUM_THREADS_LIMIT if n_cores and (n_cores > 1) else None):
        # Initialize a pool of workers
        # Note that we're forcing "spawn" instead of "fork" (default on linux)!
        # This is to reduce the memory footprint since "fork" appears to inherit all
        # variables (including all neurons) while "spawn" appears to get only
        # what's required to run the job?
        with ProcessPoolExecutor(max_workers=n_cores,
                                 mp_context=mp.get_context('spawn')) as pool:
            with config.tqdm(desc='Preparing',
                             total=n_rows * n_cols,
                             leave=False,
                             disable=not progress) as pbar:
                futures = {}
                nblasters = []
                for qix in np.array_split(np.arange(len(query_dps)), n_rows):
                    for tix in np.array_split(np.arange(len(target_dps)), n_cols):
                        # Initialize NBlaster
                        this = NBlaster(use_alpha=use_alpha,
                                        normalized=normalized,
                                        smat=smat,
                                        limit_dist=limit_dist,
                                        dtype=precision,
                                        approx_nn=approx_nn,
                                        progress=progress,
                                        smat_kwargs=smat_kwargs)

                        # Add queries and targets
                        for i, ix in enumerate(qix):
                            this.append(query_dps[ix], query_self_hits[ix])
                        for i, ix in enumerate(tix):
                            this.append(target_dps[ix], target_self_hits[ix])

                        # Keep track of indices of queries and targets
                        this.queries = np.arange(len(qix))
                        this.targets = np.arange(len(tix)) + len(qix)
                        this.queries_ix = qix  # this facilitates filling in the big matrix later
                        this.targets_ix = tix  # this facilitates filling in the big matrix later
                        this.pbar_position = len(nblasters) if not utils.is_jupyter() else None

                        nblasters.append(this)
                        pbar.update()

                        # If multiple cores requested, submit job to the pool right away
                        if n_cores and n_cores > 1 and (n_cols > 1 or n_rows > 1):
                            this.progress=False  # no progress bar for individual NBLASTERs
                            futures[pool.submit(this.multi_query_target,
                                                q_idx=this.queries,
                                                t_idx=this.targets,
                                                scores=scores)] = this

            # Collect results
            if futures and len(futures) > 1:
                # Prepare empty score matrix
                scores = pd.DataFrame(np.empty((len(query_dps), len(target_dps)),
                                               dtype=this.dtype),
                                      index=query_dps.id, columns=target_dps.id)
                scores.index.name = 'query'
                scores.columns.name = 'target'

                # Collect results
                # We're dropping the "N / N_total" bit from the progress bar because
                # it's not helpful here
                fmt = ('{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}]')
                for f in config.tqdm(as_completed(futures),
                                     desc='NBLASTing',
                                     bar_format=fmt,
                                     total=len(futures),
                                     smoothing=0,
                                     disable=not progress,
                                     leave=False):
                    res = f.result()
                    this = futures[f]
                    # Fill-in big score matrix
                    scores.iloc[this.queries_ix, this.targets_ix] = res.values
            else:
                scores = this.multi_query_target(this.queries,
                                                 this.targets,
                                                 scores=scores)

    return scores


def nblast_allbyall(x: NeuronList,
                    normalized: bool = True,
                    use_alpha: bool = False,
                    smat: Optional[Union[str, pd.DataFrame, Callable]] = 'auto',
                    limit_dist: Optional[Union[Literal['auto'], int, float]] = None,
                    approx_nn: bool = False,
                    precision: Union[int, str, np.dtype] = 64,
                    n_cores: int = os.cpu_count() // 2,
                    progress: bool = True,
                    smat_kwargs: Optional[Dict] = dict()) -> pd.DataFrame:
    """All-by-all NBLAST of inputs neurons.

    A more efficient way than running ``nblast(query=x, target=x)``.

    Parameters
    ----------
    x :             Dotprops | NeuronList
                    Neuron(s) to NBLAST against each other. Neurons should
                    be in microns as NBLAST is optimized for that and have
                    similar sampling resolutions.
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
    smat :          str | pd.DataFrame | Callable, optional
                    Score matrix/function:
                     - If ``smat='auto'`` (default), will use scoring matrices
                       based on flycircuit data. Same behaviour as in R's
                       nat.nblast implementation.
                     - For ``smat='v1'``, uses the analytic formulation of the
                       NBLAST scoring from Kohl et. al (2013). You can adjust
                       parameter ``sigma_scaling`` (default to 10) using ``smat_kwargs``.
                     - For ``smat=None`` the scores will be generated as the product
                       of the distances and the dotproduct of the vectors of
                       nearest-neighbor pairs.
                     - If function, must consume distance and dot products as
                       first and second argument, respectively and return float.
    limit_dist :    float | "auto" | None
                    Sets the max distance for the nearest neighbor search
                    (`distance_upper_bound`). Typically this should be the
                    highest distance considered by the scoring function. If
                    "auto", will extract that value from the scoring matrix.
                    While this can give a ~2X speed up, it will introduce slight
                    inaccuracies because we won't have a vector component for
                    points without a nearest neighbour within the distance
                    limits. The impact depends on the scoring function but with
                    the default FCWB ``smat``, this is typically limited to the
                    third decimal (0.0086 +/- 0.0027 for an all-by-all of 1k
                    neurons).
    approx_nn :     bool
                    If True, will use approximate nearest neighbors. This gives
                    a >2X speed up but also produces only approximate scores.
                    Impact depends on the use case - testing highly recommended!
    precision :     int [16, 32, 64] | str [e.g. "float64"] | np.dtype
                    Precision for scores. Defaults to 64 bit (double) floats.
                    This is useful to reduce the memory footprint for very large
                    matrices. In real-world scenarios 32 bit (single)- and
                    depending on the purpose even 16 bit (half) - are typically
                    sufficient.
    progress :      bool
                    Whether to show progress bars. This cause may some overhead,
                    so switch off if you don't really need it.
    smat_kwargs:    Dictionary with additional parameters passed to scoring
                    functions.

    Returns
    -------
    scores :        pandas.DataFrame
                    Matrix with NBLAST scores. Rows are query neurons, columns
                    are targets. The order is the same as in ``x``
                    and the labels are based on the neurons' ``.id`` property.

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
    # Check if pykdtree flag needed to be set
    #if n_cores and n_cores > 1:
    #    check_pykdtree_flag()

    # Make sure we're working on NeuronLists
    dps = NeuronList(x)

    # Run NBLAST preflight checks
    # Note that we are passing the same dotprops twice to avoid having to
    # change the function's signature. Should have little to no overhead.
    nblast_preflight(dps, dps, n_cores,
                     req_unique_ids=True,
                     req_microns=isinstance(smat, str) and smat=='auto')

    # Find a partition that produces batches that each run in approximately
    # 10 seconds
    if n_cores and n_cores > 1:
        if progress:
            # If progress bar, we need to make smaller mini batches.
            # These mini jobs must not be too small - otherwise the overhead
            # from spawning and sending results between processes slows things
            # down dramatically. Hence we want to make sure that each job runs
            # for >10s. The run time depends on the system and how big the neurons
            # are. Here, we run a quick test and try to extrapolate from there:
            n_rows, n_cols = find_batch_partition(dps, dps,
                                                  T=10 * JOB_SIZE_MULTIPLIER)
        else:
            # If no progress bar needed, we can just split neurons evenly across
            # all available cores
            n_rows, n_cols = find_optimal_partition(n_cores, dps, dps)
    else:
        n_rows = n_cols = 1

    # Calculate self-hits once for all neurons
    nb = NBlaster(use_alpha=use_alpha,
                  normalized=normalized,
                  smat=smat,
                  limit_dist=limit_dist,
                  dtype=precision,
                  approx_nn=approx_nn,
                  progress=progress,
                  smat_kwargs=smat_kwargs)
    self_hits = np.array([nb.calc_self_hit(n) for n in dps])

    # This makes sure we don't run into multiple layers of concurrency
    with set_omp_flag(limits=OMP_NUM_THREADS_LIMIT if n_cores and (n_cores > 1) else None):
        # Initialize a pool of workers
        # Note that we're forcing "spawn" instead of "fork" (default on linux)!
        # This is to reduce the memory footprint since "fork" appears to inherit all
        # variables (including all neurons) while "spawn" appears to get only
        # what's required to run the job?
        with ProcessPoolExecutor(max_workers=n_cores,
                                 mp_context=mp.get_context('spawn')) as pool:
            with config.tqdm(desc='Preparing',
                             total=n_rows * n_cols,
                             leave=False,
                             disable=not progress) as pbar:
                futures = {}
                nblasters = []
                for qix in np.array_split(np.arange(len(dps)), n_rows):
                    for tix in np.array_split(np.arange(len(dps)), n_cols):
                        # Initialize NBlaster
                        this = NBlaster(use_alpha=use_alpha,
                                        normalized=normalized,
                                        smat=smat,
                                        limit_dist=limit_dist,
                                        dtype=precision,
                                        approx_nn=approx_nn,
                                        progress=progress,
                                        smat_kwargs=smat_kwargs)

                        # Make sure we don't add the same neuron twice
                        # Map indices to neurons
                        to_add = list(set(qix) | set(tix))

                        # Add neurons
                        ixmap = {}
                        for i, ix in enumerate(to_add):
                            this.append(dps[ix], self_hits[ix])
                            ixmap[ix] = i

                        # Keep track of indices of queries and targets
                        this.queries = [ixmap[ix] for ix in qix]
                        this.targets = [ixmap[ix] for ix in tix]
                        this.queries_ix = qix  # this facilitates filling in the big matrix later
                        this.targets_ix = tix  # this facilitates filling in the big matrix later
                        this.pbar_position = len(nblasters) if not utils.is_jupyter() else None

                        nblasters.append(this)
                        pbar.update()

                        # If multiple cores requested, submit job to the pool right away
                        if n_cores and n_cores > 1 and (n_cols > 1 or n_rows > 1):
                            this.progress=False  # no progress bar for individual NBLASTERs
                            futures[pool.submit(this.multi_query_target,
                                                q_idx=this.queries,
                                                t_idx=this.targets,
                                                scores='forward')] = this

            # Collect results
            if futures and len(futures) > 1:
                # Prepare empty score matrix
                scores = pd.DataFrame(np.empty((len(dps), len(dps)),
                                               dtype=this.dtype),
                                      index=dps.id, columns=dps.id)
                scores.index.name = 'query'
                scores.columns.name = 'target'

                # Collect results
                # We're dropping the "N / N_total" bit from the progress bar because
                # it's not helpful here
                fmt = ('{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}]')
                for f in config.tqdm(as_completed(futures),
                                     desc='NBLASTing',
                                     bar_format=fmt,
                                     total=len(futures),
                                     smoothing=0,
                                     disable=not progress,
                                     leave=False):
                    res = f.result()
                    this = futures[f]
                    # Fill-in big score matrix
                    scores.iloc[this.queries_ix, this.targets_ix] = res.values
            else:
                scores = this.all_by_all()

    return scores


def test_single_query_time(q, t, it=100):
    """Test average time of a single NBLAST query."""
    # Get a median-sized query and target
    q_ix = np.argsort(q.n_points)[len(q)//2]
    t_ix = np.argsort(t.n_points)[len(t)//2]

    # Run a quick single query benchmark
    timings = []
    for i in range(it):  # Run N tests
        s = time.time()
        _ = t[t_ix].dist_dots(q[q_ix])  # Dist dot (ignore scoring / normalizing)
        timings.append(time.time() - s)
    return np.mean(timings)  # seconds per medium sized query


def find_batch_partition(q, t, T=10, n_cores=None):
    """Find partitions such that each batch takes about `T` seconds.

    Parameters
    ----------
    q,t :       NeuronList of Dotprops
                Query and targets, respectively.
    T :         int
                Time (in seconds) to aim for.
    n_cores :   int, optional
                Number of cores that will be used. If provided, will try to
                make sure that (n_rows * n_cols) is a multiple of n_cores by
                increasing the number of rows (thereby decreasing the time
                per batch).

    Returns
    -------
    n_rows, n_cols

    """
    # Test a single query
    time_per_query = test_single_query_time(q, t)

    # Number of queries per job such that each job runs in `T` second
    queries_per_batch = T / time_per_query

    # Number of neurons per batch
    neurons_per_batch  = max(1, int(np.sqrt(queries_per_batch)))

    n_rows = max(1, len(q) // neurons_per_batch)
    n_cols = max(1, len(t) // neurons_per_batch)

    if n_cores and ((n_rows * n_cols) > n_cores):
        while (n_rows * n_cols) % n_cores:
            n_rows += 1

    return n_rows, n_cols


def find_optimal_partition(N_cores, q, t):
    """Find an optimal partition for given NBLAST query.

    Parameters
    ----------
    N_cores :   int
                Number of available cores.
    q,t :       NeuronList of Dotprops
                Query and targets, respectively.

    Returns
    -------
    n_rows, n_cols

    """
    neurons_per_query = []
    for n_rows in range(1, N_cores + 1):
        # Skip splits we can't make
        if N_cores % n_rows:
            continue
        if n_rows > len(q):
            continue

        n_cols = min(int(N_cores / n_rows), len(t))

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
                     req_unique_ids=False, req_dotprops=True, req_points=True,
                     req_microns=True):
    """Run preflight checks for NBLAST."""
    if req_dotprops:
        if query.types != (Dotprops, ):
            raise TypeError(f'`query` must be Dotprop(s), got "{query.types}". '
                            'Use `navis.make_dotprops` to convert neurons.')

        if target.types != (Dotprops, ):
            raise TypeError(f'`target` must be Dotprop(s), got "{target.types}". '
                            'Use `navis.make_dotprops` to convert neurons.')

        if req_points:
            no_points = query.n_points == 0
            if any(no_points):
                raise ValueError('Some query dotprops appear to have no points: '
                                 f'{query.id[no_points]}')
            no_points = target.n_points == 0
            if any(no_points):
                raise ValueError('Some target dotprops appear to have no points: '
                                 f'{target.id[no_points]}')

    if req_unique_ids:
        # At the moment, neurons need to have a unique ID for things to work
        if query.is_degenerated:
            raise ValueError('Queries have non-unique IDs.')
        if target.is_degenerated:
            raise ValueError('Targets have non-unique IDs.')

    # Check if query or targets are in microns
    # Note this test can return `None` if it can't be determined
    if req_microns:
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


def eval_limit_dist(x):
    """Evaluate `limit_dist` parameter."""
    if x == 'auto':
        return
    if isinstance(x, type(None)):
        return
    if isinstance(x, numbers.Number):
        return

    raise ValueError(f'`limit_dist` must be None, "auto" or float, got {x}' )


def check_pykdtree_flag():
    """Check if pykdtree is used and if the OMP_NUM_THREADS flag is set.

    The issue is that on Linux pykdtree uses threads by default which causes
    issues when we're also using multiple cores (= multiple layers of concurrency).
    """
    # This is only relevant for Linux (unless someone compiled pykdtree
    # themselves using a compiler that supports openmp)
    from sys import platform
    if platform not in ("linux", "linux2"):
        return

    # See if pykdtree is present
    try:
        import pykdtree
    except ImportError:
        # If not present, just return
        return

    import os
    if os.environ.get('OMP_NUM_THREADS', None) != "1":
        msg = ('`OMP_NUM_THREADS` environment variable not set to 1. This may '
               'result in multiple layers of concurrency which in turn will '
               'slow down NBLAST when using multiple cores. '
               'See also https://github.com/navis-org/navis/issues/49')
        logger.warning(msg)


def set_omp_flag(limits=1):
    """Set OMP_NUM_THREADS flag to given value.

    This is to avoid pykdtree causing multiple layers of concurrency which
    will over-subcribe and slow down NBLAST on multi-core systems.

    Use as context manager!
    """
    class OMPSetter:
        def __init__(self, num_threads):
            assert isinstance(num_threads, (int, type(None)))
            self.num_threads = num_threads

        def __enter__(self):
            if self.num_threads is None:
                return
            # Track old value (if there is one)
            self.old_value = os.environ.get('OMP_NUM_THREADS', None)
            # Set flag
            os.environ['OMP_NUM_THREADS'] = str(self.num_threads)

        def __exit__(self, *args, **kwargs):
            if self.num_threads is None:
                return

            # Reset flag
            if self.old_value:
                os.environ['OMP_NUM_THREADS'] = str(self.old_value)
            else:
                os.environ.pop('OMP_NUM_THREADS', None)

    return OMPSetter(limits)
