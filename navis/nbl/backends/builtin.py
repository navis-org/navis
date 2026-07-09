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

"""Built-in NBLAST backend.

This is the reference implementation: a pure-Python/numpy scoring engine
(`NBlaster`, see ``nblast_funcs.py``) dispatched across cores with a
``ProcessPoolExecutor``. All partitioning, pool orchestration and result
stitching that used to be duplicated across the public ``nblast*`` functions
now lives here, once.

The single point where the parallelism is applied is :meth:`BuiltinBackend._map`.
A backend that only wants a *different dispatcher* (e.g. joblib, threads or a
serial debug mode) can subclass ``BuiltinBackend`` and override just that method
- it reuses all the partitioning and stitching.
"""

import multiprocessing as mp

import numpy as np
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed

from ... import config, utils
from .base import NblastBackend

logger = config.get_logger(__name__)


def _run_job(blaster):
    """Execute a single block's work.

    This is a module-level function (rather than a closure/lambda) so it is
    picklable and can be dispatched to a ``ProcessPoolExecutor``. The block's
    parameters travel on the (picklable) blaster instance itself.
    """
    op = blaster._op
    if op == 'multi_query_target':
        return blaster.multi_query_target(blaster.queries, blaster.targets,
                                          scores=blaster._scores)
    elif op == 'pair_query_target':
        return blaster.pair_query_target(blaster.pairs, scores=blaster._scores)
    raise ValueError(f"Unknown block operation '{op}'")


class BuiltinBackend(NblastBackend):
    """The built-in multiprocessing NBLAST backend."""

    name = "builtin"
    priority = 0

    def available(self):
        # Always available - it's pure navis
        return True

    def unsupported(self, operation, **params):
        # The built-in backend supports every operation and parameter
        return super().unsupported(operation)

    # ------------------------------------------------------------------ #
    # Shared helpers
    # ------------------------------------------------------------------ #
    def _map(self, jobs, n_cores, progress, desc="NBLASTing"):
        """Run `jobs` and yield ``(job, result)`` tuples.

        Each *job* is an ``NBlaster`` carrying the picklable attributes read by
        :func:`_run_job` (``_op``, ``_scores`` and the relevant index arrays).
        With a single job (or a single core) work runs inline; otherwise it is
        spread across a spawn-based process pool.

        Override this method to swap in a different dispatcher (joblib, threads,
        serial, ...) while reusing the partitioning and stitching logic.
        """
        from ..nblast_funcs import set_omp_flag, OMP_NUM_THREADS_LIMIT

        multicore = bool(n_cores and n_cores > 1 and len(jobs) > 1)

        # Avoid multiple layers of concurrency (see pykdtree/OMP notes)
        with set_omp_flag(limits=OMP_NUM_THREADS_LIMIT if (n_cores and n_cores > 1) else None):
            if not multicore:
                for this in jobs:
                    yield this, _run_job(this)
                return

            # Note that we're forcing "spawn" instead of "fork" (default on
            # linux) to reduce the memory footprint: "fork" appears to inherit
            # all variables (including all neurons) while "spawn" gets only
            # what's required to run the job.
            with ProcessPoolExecutor(max_workers=n_cores,
                                     mp_context=mp.get_context('spawn')) as pool:
                futures = {}
                for this in jobs:
                    this.progress = False  # no per-block progress bars
                    futures[pool.submit(_run_job, this)] = this

                # We drop the "N / N_total" bit from the progress bar because
                # it's not helpful here.
                fmt = '{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}]'
                for f in config.tqdm(as_completed(futures),
                                     desc=desc,
                                     bar_format=fmt,
                                     total=len(futures),
                                     smoothing=0,
                                     disable=not progress,
                                     leave=False):
                    yield futures[f], f.result()

    def _make_blaster(self, use_alpha, normalized, smat, limit_dist, precision,
                      approx_nn, progress, smat_kwargs):
        from ..nblast_funcs import NBlaster
        return NBlaster(use_alpha=use_alpha,
                        normalized=normalized,
                        smat=smat,
                        limit_dist=limit_dist,
                        dtype=precision,
                        approx_nn=approx_nn,
                        progress=progress,
                        smat_kwargs=smat_kwargs)

    def _partition(self, q, t, n_cores, progress, aba=False):
        """Find (n_rows, n_cols) partition of the query/target matrix."""
        from ..nblast_funcs import (find_batch_partition, find_optimal_partition,
                                     JOB_SIZE_MULTIPLIER, JOB_MAX_TIME_SECONDS)

        if not (n_cores and n_cores > 1):
            return 1, 1

        if progress:
            # If progress bar, we need to make smaller mini batches. These mini
            # jobs must not be too small - otherwise the overhead from spawning
            # and sending results between processes slows things down
            # dramatically. Hence we want each job to run for >10s. The run time
            # depends on the system and how big the neurons are, so we run a
            # quick test and extrapolate.
            return find_batch_partition(q, t, T=10 * JOB_SIZE_MULTIPLIER)

        # No progress bar: for a plain query->target NBLAST we aim for each
        # batch to finish in a certain amount of time (to avoid stragglers);
        # for all-by-all we just split evenly across cores.
        if aba:
            return find_optimal_partition(n_cores, q, t)

        n_rows, n_cols = find_batch_partition(q, t, T=JOB_MAX_TIME_SECONDS)
        if (n_rows * n_cols) < n_cores:
            n_rows, n_cols = find_optimal_partition(n_cores, q, t)
        return n_rows, n_cols

    # ------------------------------------------------------------------ #
    # Operations
    # ------------------------------------------------------------------ #
    def nblast(self, query, target, *, scores, normalized, use_alpha, smat,
               limit_dist, approx_nn, precision, n_cores, progress, smat_kwargs):
        """Query -> target NBLAST."""
        query_dps, target_dps = query, target

        n_rows, n_cols = self._partition(query_dps, target_dps, n_cores, progress)

        # Calculate self-hits once for all neurons
        nb = self._make_blaster(use_alpha, normalized, smat, limit_dist,
                                precision, approx_nn, progress, smat_kwargs)
        query_self_hits = np.array([nb.calc_self_hit(n) for n in query_dps])
        target_self_hits = np.array([nb.calc_self_hit(n) for n in target_dps])

        # Build one blaster per block of the score matrix
        jobs = []
        with config.tqdm(desc='Preparing', total=n_rows * n_cols, leave=False,
                         disable=not progress) as pbar:
            for qix in np.array_split(np.arange(len(query_dps)), n_rows):
                for tix in np.array_split(np.arange(len(target_dps)), n_cols):
                    this = self._make_blaster(use_alpha, normalized, smat,
                                              limit_dist, precision, approx_nn,
                                              progress, smat_kwargs)
                    for ix in qix:
                        this.append(query_dps[ix], query_self_hits[ix])
                    for ix in tix:
                        this.append(target_dps[ix], target_self_hits[ix])

                    this.queries = np.arange(len(qix))
                    this.targets = np.arange(len(tix)) + len(qix)
                    this.queries_ix = qix
                    this.targets_ix = tix
                    this.pbar_position = len(jobs) if not utils.is_jupyter() else None
                    this._op = 'multi_query_target'
                    this._scores = scores

                    jobs.append(this)
                    pbar.update()

        # Single block: return its labeled DataFrame directly
        if len(jobs) == 1:
            (this, res), = self._map(jobs, n_cores, progress)
            return res

        # Multiple blocks: stitch results into the big matrix
        out = pd.DataFrame(np.empty((len(query_dps), len(target_dps)),
                                    dtype=nb.dtype),
                           index=query_dps.id, columns=target_dps.id)
        out.index.name = 'query'
        out.columns.name = 'target'
        for this, res in self._map(jobs, n_cores, progress):
            out.iloc[this.queries_ix, this.targets_ix] = res.values

        return out

    def nblast_allbyall(self, x, *, normalized, use_alpha, smat, limit_dist,
                        approx_nn, precision, n_cores, progress, smat_kwargs):
        """All-by-all NBLAST (always forward scores)."""
        dps = x

        n_rows, n_cols = self._partition(dps, dps, n_cores, progress, aba=True)

        # Calculate self-hits once for all neurons
        nb = self._make_blaster(use_alpha, normalized, smat, limit_dist,
                                precision, approx_nn, progress, smat_kwargs)
        self_hits = np.array([nb.calc_self_hit(n) for n in dps])

        jobs = []
        with config.tqdm(desc='Preparing', total=n_rows * n_cols, leave=False,
                         disable=not progress) as pbar:
            for qix in np.array_split(np.arange(len(dps)), n_rows):
                for tix in np.array_split(np.arange(len(dps)), n_cols):
                    this = self._make_blaster(use_alpha, normalized, smat,
                                              limit_dist, precision, approx_nn,
                                              progress, smat_kwargs)

                    # Make sure we don't add the same neuron twice
                    to_add = list(set(qix) | set(tix))
                    ixmap = {}
                    for i, ix in enumerate(to_add):
                        this.append(dps[ix], self_hits[ix])
                        ixmap[ix] = i

                    this.queries = [ixmap[ix] for ix in qix]
                    this.targets = [ixmap[ix] for ix in tix]
                    this.queries_ix = qix
                    this.targets_ix = tix
                    this.pbar_position = len(jobs) if not utils.is_jupyter() else None
                    this._op = 'multi_query_target'
                    this._scores = 'forward'

                    jobs.append(this)
                    pbar.update()

        if len(jobs) == 1:
            return jobs[0].all_by_all()

        out = pd.DataFrame(np.empty((len(dps), len(dps)), dtype=nb.dtype),
                           index=dps.id, columns=dps.id)
        out.index.name = 'query'
        out.columns.name = 'target'
        for this, res in self._map(jobs, n_cores, progress):
            out.iloc[this.queries_ix, this.targets_ix] = res.values

        return out

    def nblast_smart(self, query, target, *, aba, t, criterion, scores,
                     return_mask, normalized, use_alpha, smat, limit_dist,
                     approx_nn, precision, n_cores, progress, smat_kwargs):
        """Smart(er) NBLAST: pre-NBLAST on simplified dotprops, then full."""
        query_dps, target_dps = query, target

        pre_scores = scores
        # For all-by-all's we can compute only forward scores during the
        # pre-NBLAST and produce the mean later.
        if aba and scores == 'mean':
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
            if (t < 0 or t > len(target_dps)):
                raise ValueError('`t` must be between 0 and the total number of '
                                 f'targets ({len(target_dps)}) for criterion "N", '
                                 f'got {t}')

        # Make simplified dotprops
        query_dps_simp = query_dps.downsample(10, inplace=False)
        if not aba:
            target_dps_simp = target_dps.downsample(10, inplace=False)
        else:
            target_dps_simp = query_dps_simp

        # --- Pre-NBLAST on simplified dotprops --- #
        n_rows, n_cols = self._partition(query_dps_simp, target_dps_simp,
                                         n_cores, progress)

        nb = self._make_blaster(use_alpha, normalized, smat, limit_dist,
                                precision, approx_nn, progress, smat_kwargs)
        query_self_hits = np.array([nb.calc_self_hit(n) for n in query_dps_simp])
        target_self_hits = np.array([nb.calc_self_hit(n) for n in target_dps_simp])

        jobs = []
        with config.tqdm(desc='Prep. pre-NBLAST', total=n_rows * n_cols,
                         leave=False, disable=not progress) as pbar:
            for qix in np.array_split(np.arange(len(query_dps_simp)), n_rows):
                for tix in np.array_split(np.arange(len(target_dps_simp)), n_cols):
                    this = self._make_blaster(use_alpha, normalized, smat,
                                              limit_dist, precision, approx_nn,
                                              progress, smat_kwargs)
                    for ix in qix:
                        this.append(query_dps_simp[ix], query_self_hits[ix])
                    for ix in tix:
                        this.append(target_dps_simp[ix], target_self_hits[ix])

                    this.queries = np.arange(len(qix))
                    this.targets = np.arange(len(tix)) + len(qix)
                    this.queries_ix = qix
                    this.targets_ix = tix
                    this.pbar_position = len(jobs) if not utils.is_jupyter() else None
                    this._op = 'multi_query_target'
                    this._scores = pre_scores

                    jobs.append(this)
                    pbar.update()

        if len(jobs) == 1:
            (this, res), = self._map(jobs, n_cores, progress, desc='Pre-NBLASTs')
            scr = res
        else:
            scr = pd.DataFrame(np.empty((len(query_dps_simp),
                                         len(target_dps_simp)), dtype=nb.dtype),
                               index=query_dps_simp.id, columns=target_dps_simp.id)
            scr.index.name = 'query'
            scr.columns.name = 'target'
            for this, res in self._map(jobs, n_cores, progress, desc='Pre-NBLASTs'):
                scr.iloc[this.queries_ix, this.targets_ix] = res.values

        # If this is an all-by-all and we computed only forward scores
        if aba and scores == 'mean':
            scr = (scr + scr.T.values) / 2

        # Now select targets of interest for each query
        if criterion == 'percentile':
            sel = np.percentile(scr, q=t, axis=1)
            mask = scr >= sel.reshape(-1, 1)
        elif criterion == 'score':
            sel = np.full(scr.shape[0], fill_value=t)
            mask = scr >= sel.reshape(-1, 1)
        else:
            srt = np.argsort(scr.values, axis=1)[:, ::-1]
            mask = pd.DataFrame(np.zeros(scr.shape, dtype=bool),
                                columns=scr.columns, index=scr.index)
            _ = np.arange(mask.shape[0])
            for N in range(t):
                mask.iloc[_, srt[:, N]] = True

        # --- Full NBLAST on the selected pairs --- #
        query_self_hits = np.array([nb.calc_self_hit(n) for n in query_dps])
        target_self_hits = np.array([nb.calc_self_hit(n) for n in target_dps])

        jobs = []
        with config.tqdm(desc='Prep. full NBLAST', total=n_rows * n_cols,
                         leave=False, disable=not progress) as pbar:
            for qix in np.array_split(np.arange(len(query_dps)), n_rows):
                for tix in np.array_split(np.arange(len(target_dps)), n_cols):
                    this = self._make_blaster(use_alpha, normalized, smat,
                                              limit_dist, precision, approx_nn,
                                              progress, smat_kwargs)
                    for ix in qix:
                        this.append(query_dps[ix], query_self_hits[ix])
                    for ix in tix:
                        this.append(target_dps[ix], target_self_hits[ix])

                    # Find the pairs to NBLAST in this part of the matrix
                    submask = mask.loc[query_dps[qix].id, target_dps[tix].id]
                    # `pairs` is an array of `[[query, target], [...]]` pairs
                    this.pairs = np.vstack(np.where(submask)).T
                    # Offset the target indices
                    this.pairs[:, 1] += len(qix)

                    # Track this blaster's mask relative to the original big one
                    this.mask = np.zeros(mask.shape, dtype=bool)
                    this.mask[qix[0]:qix[-1] + 1, tix[0]:tix[-1] + 1] = submask

                    this.pbar_position = len(jobs) if not utils.is_jupyter() else None
                    this.desc = 'Full NBLAST'
                    this._op = 'pair_query_target'
                    this._scores = scores

                    jobs.append(this)
                    pbar.update()

        if len(jobs) == 1:
            (this, res), = self._map(jobs, n_cores, progress)
            scr[mask] = res
        else:
            for this, res in self._map(jobs, n_cores, progress):
                scr[this.mask] = res

        if return_mask:
            return scr, mask

        return scr

    def synblast(self, query, target, *, by_type, cn_types, scores, normalized,
                 smat, n_cores, progress):
        """Synapse-based NBLAST (SynBLAST)."""
        from ..synblast_funcs import SynBlaster, find_batch_partition
        from ..nblast_funcs import find_optimal_partition

        def get_connectors(n):
            if cn_types is not None:
                return n.connectors[n.connectors['type'].isin(cn_types)]
            return n.connectors

        # Find a partition that produces batches that each run in ~10s
        if n_cores and n_cores > 1:
            if progress:
                n_rows, n_cols = find_batch_partition(query, target, T=10)
            else:
                n_rows, n_cols = find_optimal_partition(n_cores, query, target)
        else:
            n_rows = n_cols = 1

        # Calculate self-hits once for all neurons
        nb = SynBlaster(normalized=normalized, by_type=by_type, smat=smat,
                        progress=progress)
        query_self_hits = np.array([nb.calc_self_hit(get_connectors(n)) for n in query])
        target_self_hits = np.array([nb.calc_self_hit(get_connectors(n)) for n in target])

        jobs = []
        with config.tqdm(desc='Preparing', total=n_rows * n_cols, leave=False,
                         disable=not progress) as pbar:
            for qix in np.array_split(np.arange(len(query)), n_rows):
                for tix in np.array_split(np.arange(len(target)), n_cols):
                    this = SynBlaster(normalized=normalized, by_type=by_type,
                                      smat=smat, progress=progress)
                    for ix in qix:
                        n = query[ix]
                        this.append(get_connectors(n), id=n.id,
                                    self_hit=query_self_hits[ix])
                    for ix in tix:
                        n = target[ix]
                        this.append(get_connectors(n), id=n.id,
                                    self_hit=target_self_hits[ix])

                    this.queries = np.arange(len(qix))
                    this.targets = np.arange(len(tix)) + len(qix)
                    this.queries_ix = qix
                    this.targets_ix = tix
                    this.pbar_position = len(jobs) if not utils.is_jupyter() else None
                    this._op = 'multi_query_target'
                    this._scores = scores

                    jobs.append(this)
                    pbar.update()

        if len(jobs) == 1:
            (this, res), = self._map(jobs, n_cores, progress)
            return res

        out = pd.DataFrame(np.empty((len(query), len(target)), dtype=nb.dtype),
                           index=query.id, columns=target.id)
        out.index.name = 'query'
        out.columns.name = 'target'
        for this, res in self._map(jobs, n_cores, progress):
            out.iloc[this.queries_ix, this.targets_ix] = res.values

        return out
