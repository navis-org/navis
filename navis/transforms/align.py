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
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.

import warnings
import os

import numpy as np
import pandas as pd
import multiprocessing as mp

from threadpoolctl import ThreadpoolController
from functools import partial
from typing import Callable, Dict, Union, Optional, List
from typing_extensions import Literal
from concurrent.futures import ProcessPoolExecutor, as_completed

from .. import config, core, utils
from ..nbl.smat import smat_fcwb
from ..nbl.base import Blaster, NestedIndices
from ..nbl.nblast_funcs import nblast_preflight, find_batch_partition, find_optimal_partition


logger = config.logger


# This multiplier controls job size for NBLASTs (only relevant for
# multiprocessing and if progress=True).
# Larger multiplier = larger job sizes = fewer jobs = slower updates & less overhead
# Smaller multiplier = smaller job sizes = more jobs = faster updates & more overhead
JOB_SIZE_MULTIPLIER = 1


class NBlasterAlign(Blaster):
    """Implements a version of NBLAST were neurons are first aligned.

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

    def __init__(self,
                 align_func, two_way_align=True,
                 use_alpha=False, normalized=True, smat='auto',
                 limit_dist=None, approx_nn=False, dtype=np.float64,
                 progress=True,
                 smat_kwargs=dict(),
                 align_kwargs=dict(),
                 dotprop_kwargs=dict(),
                 ):
        """Initialize class."""
        super().__init__(progress=progress, dtype=dtype)
        self.align_func = align_func
        self.two_way_align = two_way_align
        self.use_alpha = use_alpha
        self.normalized = normalized
        self.approx_nn = approx_nn
        self.dotprop_kwargs = dotprop_kwargs
        self.align_kwargs = align_kwargs
        self.desc = "NBlasting"
        self.self_hits = {}
        self.dotprops = {}
        self.neurons = []

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

    def append(self, neuron: core.BaseNeuron) -> NestedIndices:
        """Append neurons.

        Returns the numerical index appended neurons.
        If neurons is a (possibly nested) sequence of neurons,
        return a (possibly nested) list of indices.

        Note that `self_hit` is ignored (and hence calculated from scratch)
        when `neurons` is a nested list of dotprops.
        """
        if isinstance(neuron, core.BaseNeuron):
            return self._append_neuron(neuron)

        try:
            return [self.append(n) for n in neuron]
        except TypeError:  # i.e. not iterable
            raise ValueError(f"Expected Neuron or iterable thereof; got {type(neuron)}")

    def _append_neuron(self, neuron: core.BaseNeuron) -> int:
        next_id = len(self)
        self.neurons.append(neuron)
        self.ids.append(neuron.id)
        return next_id

    def get_dotprop(self, ix):
        if ix not in self.dotprops:
            self.dotprops[ix] = core.make_dotprops(self.neurons[ix],
                                                   **self.dotprop_kwargs)
        return self.dotprops[ix]

    def get_self_hit(self, ix):
        if ix not in self.self_hits:
            self.self_hits[ix] = self.calc_self_hit(self.get_dotprop(ix))
        return self.self_hits[ix]

    def calc_self_hit(self, dotprops):
        """Non-normalized value for self hit."""
        if not self.use_alpha:
            return len(dotprops.points) * self.score_fn(0, 1.0)
        else:
            dists = np.repeat(0, len(dotprops.points))
            alpha = dotprops.alpha * dotprops.alpha
            dots = np.repeat(1, len(dotprops.points)) * np.sqrt(alpha)
            return self.score_fn(dists, dots).sum()

    def score_single(self, q_dp, t_dp, q_idx):
        """Calculate score for single query/target dotprop pair."""
        # Run nearest-neighbor search for query against target
        data = q_dp.dist_dots(t_dp,
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
            scr /= self.get_self_hit(q_idx)

        return scr

    def single_query_target(self, q_idx: int, t_idx: int, scores='forward',
                            allow_rev_align=True):
        """Query single query against single target."""
        # Take a short-cut if this is a self-self comparison
        if q_idx == t_idx:
            if self.normalized:
                return 1
            return self.get_self_hit(q_idx)

        # Align the query to the target
        q_xf = self.align_func(self.neurons[q_idx],
                               target=self.neurons[t_idx],
                               progress=False,
                               **self.align_kwargs)[0][0]

        # The query must always be made into new dotprops
        q_dp = core.make_dotprops(q_xf, **self.dotprop_kwargs)

        # The target dotprop has to be compute only once
        t_dp = self.get_dotprop(t_idx)

        scr = self.score_single(q_dp, t_dp, q_idx)

        # For the mean score we also have to produce the reverse score
        if scores in ('mean', 'min', 'max', 'both'):
            reverse = self.score_single(t_dp, q_dp, t_idx)
            if scores == 'mean':
                scr = (scr + reverse) / 2
            elif scores == 'min':
                scr = min(scr, reverse)
            elif scores == 'max':
                scr = max(scr, reverse)
            elif scores == 'both':
                # If both scores are requested
                scr = [scr, reverse]

        if self.two_way_align and allow_rev_align:
            rev = self.single_query_target(t_idx, q_idx, scores=scores,
                                           allow_rev_align=False)
            scr = (scr + rev) / 2

        return scr


def nblast_align(query: Union[core.BaseNeuron, core.NeuronList],
                 target: Optional[str] = None,
                 align_method: Union[Literal['rigid'],
                                     Literal['deform'],
                                     Literal['pca']] = 'rigid',
                 two_way_align: bool = True,
                 scores: Union[Literal['forward'],
                               Literal['mean'],
                               Literal['min'],
                               Literal['max']] = 'mean',
                 normalized: bool = True,
                 use_alpha: bool = False,
                 smat: Optional[Union[str, pd.DataFrame, Callable]] = 'auto',
                 limit_dist: Optional[Union[Literal['auto'], int, float]] = None,
                 approx_nn: bool = False,
                 precision: Union[int, str, np.dtype] = 64,
                 n_cores: int = os.cpu_count() // 2,
                 progress: bool = True,
                 dotprop_kwargs: Optional[Dict] = dict(),
                 align_kwargs: Optional[Dict] = dict(),
                 smat_kwargs: Optional[Dict] = dict()) -> pd.DataFrame:
    """Run NBLAST on pairwise-aligned neurons.

    Requires the `pycpd` library at least version 2.0.1 which at the time of
    writing is only available from Github (not PyPI):

      https://github.com/siavashk/pycpd

    Parameters
    ----------
    query :         TreeNeuron | NeuronList
                    Query neuron(s) to NBLAST against the targets. Neurons
                    should be in microns as NBLAST is optimized for that and
                    have similar sampling resolutions.
    target :        TreeNeuron | NeuronList, optional
                    Target neuron(s) to NBLAST against. Neurons should be in
                    microns as NBLAST is optimized for that and have
                    similar sampling resolutions. If not provided, will NBLAST
                    queries against themselves.
    align_method :  "rigid" | "deform" | "pca"
                    Which method to use for alignment. Maps to the respective
                    ``navis.align_{method}`` function.
    two_way_align : bool
                    If True, will run the alignment + NBLAST in both,
                    query->target as well as query->target direction. This is
                    highly recommended because it reduces the chance that a
                    single bad alignment will mess up your scores.
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
                    that have lots of branches for the NBLAST.
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
                    individual processes. Also note that due to multiple layers
                    of concurrency using all available cores might not be the
                    fastest option.
    precision :     int [16, 32, 64] | str [e.g. "float64"] | np.dtype
                    Precision for scores. Defaults to 64 bit (double) floats.
                    This is useful to reduce the memory footprint for very large
                    matrices. In real-world scenarios 32 bit (single)- and
                    depending on the purpose even 16 bit (half) - are typically
                    sufficient.
    progress :      bool
                    Whether to show progress bars. This may cause some overhead,
                    so switch off if you don't really need it.
    smat_kwargs :   dict, optional
                    Dictionary with additional parameters passed to scoring
                    functions.
    align_kwargs :  dict, optional
                    Dictionary with additional parameters passed to alignment
                    function.
    dotprop_kwargs : dict, optional
                    Dictionary with additional parameters passed to
                    ``navis.make_dotprops``.


    Returns
    -------
    scores :        pandas.DataFrame
                    Matrix with NBLAST scores. Rows are query neurons, columns
                    are targets. The order is the same as in ``query``/``target``
                    and the labels are based on the neurons' ``.id`` property.
                    Important to note that even when ``q == t`` and with
                    ``scores=mean`` the matrix will not be symmetrical because
                    we run separate alignments for the forward and the reverse
                    comparisons.

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
    >>> # Run the align nblast
    >>> scores = navis.nblast_align(nl_um[:3], nl_um[3:],
    ...                          dotprop_kwargs=dict(k=5))

    See Also
    --------
    :func:`navis.nblast`
                The vanilla version of NBLAST.
    :func:`navis.nblast_allbyall`
                A more efficient way than ``nblast(query=x, target=x)``.
    :func:`navis.nblast_smart`
                A smart(er) NBLAST suited for very large NBLAST.
    :func:`navis.synblast`
                A synapse-based variant of NBLAST.

    """
    if isinstance(target, type(None)):
        target = query

    # Make sure we're working on NeuronLists
    query = core.NeuronList(query)
    target = core.NeuronList(target)

    if not callable(align_method):
        align_func = {'rigid': align_rigid,
                      'deform': align_deform,
                      'pca': align_pca}[align_method]
    else:
        align_func = align_method

    # Run NBLAST preflight checks
    nblast_preflight(query, target, n_cores,
                     req_dotprops=False,
                     req_unique_ids=True,
                     req_microns=isinstance(smat, str) and smat=='auto')

    # Find a partition that produces batches that each run in approximately
    # 10 seconds
    if n_cores and n_cores > 1:
        if progress:
            # If progress bar, we need to make smaller mini batches.
            # These mini jobs must not be too small - otherwise the overhead
            # from spawning and sending results between processes slows things
            # down dramatically. Here we hardcode such that we get updates
            # at most every 1%
            n_rows = max(1, len(query) // 10)
            n_cols = max(1, len(target) // 10)
        else:
            n_rows, n_cols = find_optimal_partition(n_cores, query, target)
    else:
        n_rows = n_cols = 1

    # This makes sure we don't run into multiple layers of concurrency
    controller = ThreadpoolController()
    with controller.limit(limits=1 if (n_cores and n_cores > 1) else None,
                          user_api=None # None means all APIs
                          ):
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
                for qix in np.array_split(np.arange(len(query)), n_rows):
                    for tix in np.array_split(np.arange(len(target)), n_cols):
                        # Initialize NBlaster
                        this = NBlasterAlign(align_func=align_func,
                                             two_way_align=two_way_align,
                                             use_alpha=use_alpha,
                                             normalized=normalized,
                                             smat=smat,
                                             limit_dist=limit_dist,
                                             dtype=precision,
                                             approx_nn=approx_nn,
                                             progress=progress,
                                             align_kwargs=align_kwargs,
                                             dotprop_kwargs=dotprop_kwargs,
                                             smat_kwargs=smat_kwargs)

                        # Add queries and targets
                        for i, ix in enumerate(qix):
                            this.append(query[ix])
                        for i, ix in enumerate(tix):
                            this.append(target[ix])

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
                scores = pd.DataFrame(np.empty((len(query), len(target)),
                                               dtype=this.dtype),
                                      index=query.id, columns=target.id)
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


# This is the old reference implementation - will keep for a little longer
def _nblast_align(q, t=None, method='rigid', scores='mean', normalized=True,
                 progress=True, max_threads=None, dotprop_kwargs={}, **kwargs):
    """Run NBLAST on pairwise-aligned neurons.

    Requires the `pycpd` library.

    Parameters
    ----------
    q :             NeuronList
                    Query neurons.
    t :             NeuronList, optional
                    Target neurons. If ``None``, will run an all-by-all NBLAST of
                    ``x``.
    method :        "rigid" | "deform" | "pca"
                    Which method to use for alignment. Maps to the respective
                    ``navis.align_{method}`` function.
    scores :        "mean" | "forward"
                    Which NBLAST scores to generate.
    max_threads :   int, optional
                    Use this to set the number of threads numpy is allowed to
                    use for the registration. If ``None`` will use system
                    defaults which is typically the number of CPUs.
    **kwargs
                    Keyword arguments are passed through to the respective
                    alignment function.

    Returns
    -------
    scores :    pandas.DataFrame
                DataFrame with the NBLAST scores. Important to note that even
                when ``q == t`` and with ``scores=mean`` the matrix will not be
                symmetrical because we run separate alignments for the forward
                and the reverse comparisons.

    """
    # Notes:
    # We could massively speed things up by daisy-chaining the transforms, i.e.
    # calculcate transform from t[0] -> t[1], t[0] -> t[2], etc.
    # Then we only have to calculate a transform from each query to t[0] (i.e.
    # q[0] -> t[1], q[1] -> t[1]) and from there use the existing transforms
    # While this does make a difference it appears to get neurons at least
    # "in the right ball-park". So perhaps this should be an option of a
    # quick & dirty aligned NBLAST?
    # The problem is that if the q -> t[0] transform is bad all subsequent
    # transforms will be bad. Same if t[0] just doesn't map well onto the rest
    # of the targets (e.g. if it's really big or really small)

    # Alternatively we could "warm-up" by first aligning both query and target
    # neurons to a single neuron to bring them into roughly the same space. This
    # make subsequent pairwise transforms converge ~20% faster. I haven't tested
    # how it affects the transforms though.

    squared = False
    if t is None:
        t = q
        squared = True

    utils.eval_param(q, name='q', allowed_types=(core.NeuronList, ))
    utils.eval_param(t, name='t', allowed_types=(core.NeuronList, ))
    utils.eval_param(method, name='method',
                     allowed_values=('rigid', 'deform', 'pca'))
    utils.eval_param(scores, name='scores',
                     allowed_values=('forward', 'mean'))

    func = {'rigid': align_rigid,
            'deform': align_deform,
            'pca': align_pca}[method]

    score_fn = smat_fcwb(False)
    self_hits = {}

    controller = ThreadpoolController()

    sc = np.zeros((len(q), len(t)), dtype=np.float64)

    with controller.limit(limits=max_threads, user_api='blas'):
        for i, n2 in config.tqdm(enumerate(t),
                                 desc='NBLASTing',
                                 total=len(t),
                                 disable=not progress or (len(t) ==1)):
            dp2 = core.make_dotprops(n2, k=5)
            for k, n1 in enumerate(q):
                if n1 == n2:
                    xf = n1
                else:
                    xf = func(n1, target=n2, **kwargs, progress=False)[0][0]
                dp1 = core.make_dotprops(xf, k=5)

                dists, dots = dp1.dist_dots(dp2, alpha=False)
                scr = score_fn(dists, dots).sum()

                if normalized:
                    if dp1 not in self_hits:
                        self_hits[dp1] = len(dp1.points) * score_fn(0, 1.0)
                    scr /= self_hits[dp1]

                if scores == 'mean':
                    dists, dots = dp2.dist_dots(dp1, alpha=False)
                    reverse = score_fn(dists, dots).sum()
                    if normalized:
                        if dp2 not in self_hits:
                            self_hits[dp2] = len(dp2.points) * score_fn(0, 1.0)
                        reverse /= self_hits[dp2]

                    scr = (scr + reverse) / 2

                sc[k, i] = scr

    return pd.DataFrame(sc, index=q.id, columns=t.id)


def align_pairwise(x, y=None, method='rigid', progress=True, **kwargs):
    """Run a pairwise alignment between given neurons.

    Requires the `pycpd` library.

    Parameters
    ----------
    x :         navis.NeuronList
                Neurons to align to other neurons.
    y :         navis.NeuronList, optional
                The neurons to align to. If ``None``, will run pairwise
                alignment of ``x`` vs ``x``.
    method :    "rigid" | "deform" | "pca"
                Which method to use for alignment. Maps to the respective
                ``navis.align_{method}`` function.
    **kwargs
                Keyword arguments are passed through to the respective
                alignment function.

    Returns
    -------
    np.ndarray
                Array of shape (x, y) with the pairwise-aligned neurons.

    """
    squared = False
    if y is None:
        y = x
        squared = True

    utils.eval_param(x, name='x', allowed_types=(core.NeuronList, ))
    utils.eval_param(y, name='y', allowed_types=(core.NeuronList, ))
    utils.eval_param(method, name='method',
                     allowed_values=('rigid', 'deform', 'pca'))

    func = {'rigid': align_rigid,
            'deform': align_deform,
            'pca': align_pca}[method]

    aligned = []
    for n1 in config.tqdm(x,
                          desc='Aligning',
                          disable=not progress or len(x) == 1):
        aligned.append([])
        for n2 in y:
            if n1 == n2:
                xf = n1
            else:
                xf = func(n1, target=n2, **kwargs, progress=False)[0][0]
            aligned[-1].append(xf)

    return np.array(aligned)


def align_rigid(x, target=None, scale=False, verbose=False, progress=True):
    """Align neurons using a rigid registration.

    Requires the `pycpd` library.

    Parameters
    ----------
    x :             navis.NeuronList
                    Neurons to align.
    target :        navis.Neuron | np.ndarray
                    The neuron that all neurons in `x` will be aligned to.
                    If `None`, neurons will be aligned to the first neuron `x`!
    scale :         bool
                    If True, will also scale the neuron.
    progress :      bool
                    Whether to show a progress bar.

    Returns
    -------
    xf :    navis.NeuronList
            The aligned neurons.
    pca
            The pycpd registration objects.

    """
    try:
        from pycpd import RigidRegistration as Registration
    except ImportError:
        raise ImportError('`rigid_align()` requires the `pycpd` library:\n'
                          '  pip3 install pycpd -U')

    if isinstance(x, core.BaseNeuron):
        x = core.NeuronList(x)

    assert isinstance(x, core.NeuronList)

    if target is None:
        target = x[0]

    target_co = _extract_coords(target)

    xf = x.copy()
    regs = []
    for n in config.tqdm(xf,
                         disable=(not progress) or (len(xf) == 1),
                         desc='Aligning'):
        if n == target:
            continue
        # `w` is used to account for outliers -> higher w = more forgiving
        # the default is w=0 which can lead to failure to converge on a solution
        # in particular when scale=False
        # Our work-around here is to start at w=0 and incrementally increase w
        # if we fail to converge
        # Also note that pycpd ignores the `scale` in earlier versions. The
        # version on PyPI is currently outdated. From what I understand we need
        # the Github version.
        w = 0
        converged = False
        while w <= 0.001:
            try:
                reg = Registration(X=target_co,
                                   Y=_extract_coords(n),
                                   scale=scale,
                                   s=1,
                                   w=w
                                   )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    TY, params = reg.register()
                _set_coords(n, TY)
                converged = True
                break
            except np.linalg.LinAlgError:
                if w == 0:
                    w += 0.000000001
                else:
                    w *= 10

        if verbose and not converged:
            logger.info(f'Registration of {n.id} onto {target.id} did not converge')

        regs.append(reg)

    return xf, regs


def align_deform(x, target=None, progress=True, **kwargs):
    """Align neurons using a deformable registration.

    Requires the `pycpd` library. Note that it's often beneficial to first
    run a rough affine alignment via `rigid_align`.

    Parameters
    ----------
    x :             navis.NeuronList
                    Neurons to align.
    target :        navis.Neuron | np.ndarray
                    The neuron that all neurons in `x` will be aligned to.
                    If `None`, neurons will be aligned to the first neuron `x`!
    progress :      bool
                    Whether to show a progress bar.
    **kwargs
                    Additional keyword-argumens are passed through to
                    pycpd.DeformableRegistration. In brief: lower `alpha` and
                    higher `beta` typically make for more fitting deform. I have
                    gone as far as alpha=.01, beta=10000.

    Returns
    -------
    xf :    navis.NeuronList
            The aligned neurons.
    pca
            The pycpd registration objects.

    """
    try:
        from pycpd import DeformableRegistration as Registration
    except ImportError:
        raise ImportError('`deform_align()` requires the `pycpd` library:\n'
                          '  pip3 install pycpd -U')

    if isinstance(x, core.BaseNeuron):
        x = core.NeuronList(x)

    assert isinstance(x, core.NeuronList)

    if target is None:
        target = x[0]

    target_co = _extract_coords(target)

    # pycpd's deformable registration is very sensitive to the scale of the
    # data. We will hence normalize the neurons to be within the -1 to 1 range
    scale_factor = 0
    for n in x:
        co = _extract_coords(n)
        mx = np.abs(co).max()
        scale_factor = mx if mx > scale_factor else scale_factor

    xf = x / scale_factor
    target_co = target_co / scale_factor
    regs = []
    for n in config.tqdm(xf, disable=not progress, desc='Aligning'):
        if n == target:
            continue
        reg = Registration(X=target_co, Y=_extract_coords(n), **kwargs)
        TY, params = reg.register()
        _set_coords(n, TY)
        regs.append(reg)

    return xf * scale_factor, regs


def align_pca(x, individually=True):
    """Align neurons along their first principal components.

    This will in effect turn the neurons into a 1-dimensional line.
    Requires the `scikit-learn` library.

    Parameters
    ----------
    x :             navis.NeuronList | np.ndarray
                    The neurons to align.
    individually :  bool
                    Whether to align neurons along their individual or
                    collective first principical component.

    Returns
    -------
    xf :    navis.NeuronList
            The PCA-aligned neurons.
    pca
            The scikit-learn PCA object(s)

    """
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        raise ImportError('`pca_align()` requires the `scikit-learn` library:\n'
                          '  pip3 install scikit-learn -U')

    if isinstance(x, core.BaseNeuron):
        x = core.NeuronList(x)

    assert isinstance(x, core.NeuronList)

    pcas = []
    if not individually:
        # Collect coordinates
        co = [_extract_coords(n) for n in x]
        n_points = [len(c) for c in co]  # track how many points per neuron
        co = np.vstack(co)

        pca = PCA(n_components=1)
        co_new = pca.fit_transform(X=co)

        xf = x.copy()
        i = 0
        for n, le in zip(xf, n_points):
            _set_coords(n, co_new[i: i + le])
            i += le
        pcas.append(pca)
    else:
        xf = x.copy()
        for n in xf:
            pca = PCA(n_components=1)
            _set_coords(n, pca.fit_transform(X=_extract_coords(n)))
            pcas.append(pca)
    return xf, pcas


def _extract_coords(n):
    """Extract xyz coordinates from given object."""
    if isinstance(n, np.ndarray):
        return n
    elif isinstance(n, core.MeshNeuron):
        return n.vertices
    elif isinstance(n, core.TreeNeuron):
        return n.nodes[['x', 'y', 'z']].values
    elif isinstance(n, core.Dotprops):
        return n.points
    else:
        raise TypeError(f'Unable to extract coordinates from {type(n)}')


def _set_coords(n, new_co):
    """Set new xyz coordinates for given object."""
    if new_co.ndim == 2 and new_co.shape[1] == 1:
        new_co = new_co.flatten()

    if new_co.ndim == 2:
        if isinstance(n, core.MeshNeuron):
            n.vertices = new_co
        elif isinstance(n, core.TreeNeuron):
            n.nodes[['x', 'y', 'z']] = new_co
        elif isinstance(n, core.Dotprops):
            n.points = new_co
        else:
            raise TypeError(f'Unable to extract coordinates from {type(n)}')
    # If this is a single vector
    else:
        if isinstance(n, core.MeshNeuron):
            for i in range(3):
                n.vertices[:, i] = new_co
        elif isinstance(n, core.TreeNeuron):
            for i in 'xyz':
                n.nodes[i] = new_co
        elif isinstance(n, core.Dotprops):
            for i in range(3):
                n.points[:, i] = new_co
        else:
            raise TypeError(f'Unable to extract coordinates from {type(n)}')
