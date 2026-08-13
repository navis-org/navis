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
(`NBlaster`, see ``nblast_funcs.py``) that cuts the query x target matrix into
blocks and runs them independently. All partitioning and result stitching that
used to be duplicated across the public ``nblast*`` functions now lives here,
once.

*Where* the blocks run is not this module's business. :meth:`BuiltinBackend._map`
hands them to whatever [`navis.set_parallel_backend`][] points at, so the same
NBLAST runs on this machine's cores, on a dask cluster or as a SLURM array
without anything here changing::

    with navis.set_parallel_backend(dask_client):
        navis.nblast(query, target, backend='builtin')

A block is already sized for an expensive transport: `_partition` picks the
grid from a per-block *runtime* budget, so each block is seconds to minutes of
work regardless of how many there are. That is why blocks are handed over one
per unit of work rather than bundled the way single neurons are.
"""

import numpy as np
import pandas as pd

from ... import config, utils
from .base import NblastBackend

logger = config.get_logger(__name__)


def _run_job(blaster):
    """Execute a single block's work. Runs in the worker.

    This is a module-level function (rather than a closure/lambda) so it is
    picklable by reference, i.e. cheap to ship and usable on backends that
    serialise with plain `pickle`. The block's parameters travel on the
    (picklable) blaster instance itself.

    Capping the worker's own threading - pykdtree's above all, which would
    otherwise compete with the parallelism that put us here - is the
    dispatcher's job (`navis/compute/threads.py`), not this function's.
    """
    op = blaster._op
    if op == 'multi_query_target':
        return blaster.multi_query_target(blaster.queries, blaster.targets,
                                          scores=blaster._scores)
    elif op == 'pair_query_target':
        return blaster.pair_query_target(blaster.pairs,
                                         scores=blaster._scores)
    raise ValueError(f"Unknown block operation '{op}'")


def _empty_scores(query_ids, target_ids, dtype, scores='forward'):
    """Allocate the full matrix that the blocks get written into."""
    if scores == 'both':
        # `multi_query_target` stacks the two directions per query, so the
        # matrix is twice as tall as it has queries.
        index = pd.MultiIndex.from_product([query_ids, ['forward', 'reverse']],
                                           names=['query', 'score'])
    else:
        index = pd.Index(query_ids, name='query')

    return pd.DataFrame(np.empty((len(index), len(target_ids)), dtype=dtype),
                        index=index,
                        columns=pd.Index(target_ids, name='target'))


def _row_positions(queries_ix, scores):
    """Where a block's rows belong in the matrix `_empty_scores` allocated.

    With `scores='both'` each query owns two adjacent rows - forward then
    reverse - which is the order `multi_query_target` emits them in.
    """
    if scores != 'both':
        return queries_ix
    queries_ix = np.asarray(queries_ix)
    return np.stack([queries_ix * 2, queries_ix * 2 + 1], axis=1).ravel()


class BuiltinBackend(NblastBackend):
    """The built-in NBLAST backend: navis' own scoring engine, in blocks."""

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
    def _dispatcher(self, n_cores):
        """Where this NBLAST's blocks will run, and how many workers that is.

        Resolved once per operation, because `_partition` and `_map` have to
        agree: the grid is sized against the number of workers, and only the
        backend knows what that really is (`n_cores` describes *this* machine
        and says nothing about how big a cluster is).

        Deliberately resolved without a task count - how many blocks there are
        is not known until the grid has been chosen, and choosing the grid
        needs this. `_map` applies the "not worth dispatching" check itself.
        """
        from ... import compute

        backend = compute.resolve_backend(n_workers=n_cores)
        return backend, backend.worker_count(n_cores)

    def _map(self, jobs, n_cores, progress, desc="NBLASTing", *, backend):
        """Run `jobs` and yield ``(job, result)`` tuples as they complete.

        Each *job* is an ``NBlaster`` carrying the picklable attributes read by
        :func:`_run_job` (``_op``, ``_scores`` and the relevant index arrays).
        A single block runs inline; anything more goes to the parallel backend.

        Results are yielded as they arrive rather than collected, so a caller
        can write each block into the output matrix and let it go. Collecting
        them first would hold a second copy of the whole matrix.

        One block per unit of work: `chunksize=1` opts out of the bundling that
        the cluster backends apply to single neurons, because a block is
        already the unit the partitioner sized for a transport.
        """
        from ...compute import imap_tasks

        # A lone block is not worth a round trip, whatever the backend
        if not (backend.concurrent and len(jobs) > 1):
            for this in jobs:
                yield this, _run_job(this)
            return

        for this in jobs:
            this.progress = False   # no per-block progress bars

        tasks = [(_run_job, (this,), {}) for this in jobs]

        # We drop the "N / N_total" bit from the progress bar because it's
        # not helpful here. Hence our own bar rather than the dispatcher's.
        fmt = '{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}]'
        with config.tqdm(total=len(jobs), desc=desc, bar_format=fmt,
                         smoothing=0, disable=not progress,
                         leave=False) as pbar:
            # No `threads=`: the default budget - the machine divided between
            # the workers - is the right one here. NBLAST used to pin one
            # thread per worker, from a time when nothing capped native
            # threading at all and pykdtree's OpenMP would otherwise claim
            # every core in every worker. Dividing the machine already prevents
            # that, and pinning on top of it left NBLAST running at a fraction
            # of the cores it had asked for.
            for index, res in imap_tasks(tasks, backend=backend, chunksize=1,
                                         n_workers=n_cores, disable=True):
                pbar.update()
                yield jobs[index], res

    def _stitch(self, jobs, n_cores, progress, *, backend, query_ids,
                target_ids, dtype, scores='forward', desc="NBLASTing"):
        """Run `jobs` and assemble their blocks into one labelled matrix.

        A single block already *is* the whole matrix, labels and all, so it is
        handed straight back rather than allocated a second time and copied
        into.
        """
        if len(jobs) == 1:
            (_, res), = self._map(jobs, n_cores, progress, desc=desc,
                                  backend=backend)
            return res

        out = _empty_scores(query_ids, target_ids, dtype, scores)
        for this, res in self._map(jobs, n_cores, progress, desc=desc,
                                   backend=backend):
            out.iloc[_row_positions(this.queries_ix, scores),
                     this.targets_ix] = res.values
        return out

    def _make_blaster(self, use_alpha, normalized, smat, max_dist, precision,
                      approx_nn, progress, smat_kwargs):
        from ..nblast_funcs import NBlaster
        return NBlaster(use_alpha=use_alpha,
                        normalized=normalized,
                        smat=smat,
                        max_dist=max_dist,
                        dtype=precision,
                        approx_nn=approx_nn,
                        progress=progress,
                        smat_kwargs=smat_kwargs)

    def _partition(self, q, t, n_workers, progress, estimate_fn=None):
        """Find (n_rows, n_cols) partition of the query/target matrix.

        Estimates a target block count from a per-block runtime budget, then
        hands it to `partition_grid`, which balances the grid, floors it at
        `MIN_BLOCKS_PER_CORE` blocks per worker and rounds up to full waves so
        no worker sits idle. `estimate_fn(q, t, T=...)` supplies the count; it
        defaults to the dotprop estimator, and SynBLAST passes its connector
        one - everything else about the partition is shared.

        `n_workers` comes from the backend (see `_dispatcher`), not from
        `n_cores`: the latter is half *this* machine's cores by default and
        says nothing about the size of a cluster. It is 1 whenever nothing will
        actually run side by side, and `partition_grid` turns that into a
        single block - splitting the matrix up would then buy nothing and cost
        a copy of each neuron per block it appears in.
        """
        from ..nblast_funcs import (estimate_target_blocks, partition_grid,
                                     JOB_SIZE_MULTIPLIER, JOB_MAX_TIME_SECONDS)

        if n_workers <= 1:
            return 1, 1

        if estimate_fn is None:
            estimate_fn = estimate_target_blocks

        # Aim for each block to run for a bounded amount of time. With a progress
        # bar we want short (~10s) blocks so the bar moves; without one we allow
        # much longer blocks (less overhead). All-by-all shares this path:
        # partition_grid derives the shape from the query/target counts either way.
        T = 10 * JOB_SIZE_MULTIPLIER if progress else JOB_MAX_TIME_SECONDS
        target_blocks = estimate_fn(q, t, T=T)

        return partition_grid(n_workers, len(q), len(t),
                              target_blocks=target_blocks)

    # ------------------------------------------------------------------ #
    # Operations
    # ------------------------------------------------------------------ #
    def nblast(self, query, target, *, scores, normalized, use_alpha, smat,
               max_dist, approx_nn, precision, n_cores, progress, smat_kwargs):
        """Query -> target NBLAST."""
        query_dps, target_dps = query, target

        be, n_workers = self._dispatcher(n_cores)
        n_rows, n_cols = self._partition(query_dps, target_dps, n_workers,
                                         progress)

        # Calculate self-hits once for all neurons
        nb = self._make_blaster(use_alpha, normalized, smat, max_dist,
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
                                              max_dist, precision, approx_nn,
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

        return self._stitch(jobs, n_cores, progress, backend=be,
                            query_ids=query_dps.id, target_ids=target_dps.id,
                            dtype=nb.dtype, scores=scores)

    def nblast_allbyall(self, x, *, normalized, use_alpha, smat, max_dist,
                        approx_nn, precision, n_cores, progress, smat_kwargs):
        """All-by-all NBLAST (always forward scores)."""
        dps = x

        be, n_workers = self._dispatcher(n_cores)
        n_rows, n_cols = self._partition(dps, dps, n_workers, progress)

        # Calculate self-hits once for all neurons
        nb = self._make_blaster(use_alpha, normalized, smat, max_dist,
                                precision, approx_nn, progress, smat_kwargs)
        self_hits = np.array([nb.calc_self_hit(n) for n in dps])

        jobs = []
        with config.tqdm(desc='Preparing', total=n_rows * n_cols, leave=False,
                         disable=not progress) as pbar:
            for qix in np.array_split(np.arange(len(dps)), n_rows):
                for tix in np.array_split(np.arange(len(dps)), n_cols):
                    this = self._make_blaster(use_alpha, normalized, smat,
                                              max_dist, precision, approx_nn,
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

        # Not `_stitch`'s single-block path: `all_by_all` is a different
        # computation (it walks the block's own neurons, not a query/target
        # split), and that is what this operation has always returned.
        if len(jobs) == 1:
            return jobs[0].all_by_all()

        return self._stitch(jobs, n_cores, progress, backend=be,
                            query_ids=dps.id, target_ids=dps.id,
                            dtype=nb.dtype)

    def nblast_smart(self, query, target, *, aba, t, criterion, scores,
                     return_mask, normalized, use_alpha, smat, max_dist,
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
        be, n_workers = self._dispatcher(n_cores)
        n_rows, n_cols = self._partition(query_dps_simp, target_dps_simp,
                                         n_workers, progress)

        nb = self._make_blaster(use_alpha, normalized, smat, max_dist,
                                precision, approx_nn, progress, smat_kwargs)
        query_self_hits = np.array([nb.calc_self_hit(n) for n in query_dps_simp])
        target_self_hits = np.array([nb.calc_self_hit(n) for n in target_dps_simp])

        jobs = []
        with config.tqdm(desc='Prep. pre-NBLAST', total=n_rows * n_cols,
                         leave=False, disable=not progress) as pbar:
            for qix in np.array_split(np.arange(len(query_dps_simp)), n_rows):
                for tix in np.array_split(np.arange(len(target_dps_simp)), n_cols):
                    this = self._make_blaster(use_alpha, normalized, smat,
                                              max_dist, precision, approx_nn,
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

        scr = self._stitch(jobs, n_cores, progress, backend=be,
                           query_ids=query_dps_simp.id,
                           target_ids=target_dps_simp.id, dtype=nb.dtype,
                           desc='Pre-NBLASTs')

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
            # Build the mask with numpy fancy indexing (pointwise). Using
            # `DataFrame.iloc[rows, cols]` here would do cross-product (block)
            # indexing and mark far too many cells.
            mask_arr = np.zeros(scr.shape, dtype=bool)
            _ = np.arange(mask_arr.shape[0])
            for N in range(t):
                mask_arr[_, srt[:, N]] = True
            mask = pd.DataFrame(mask_arr, columns=scr.columns, index=scr.index)

        # --- Full NBLAST on the selected pairs --- #
        query_self_hits = np.array([nb.calc_self_hit(n) for n in query_dps])
        target_self_hits = np.array([nb.calc_self_hit(n) for n in target_dps])

        jobs = []
        with config.tqdm(desc='Prep. full NBLAST', total=n_rows * n_cols,
                         leave=False, disable=not progress) as pbar:
            for qix in np.array_split(np.arange(len(query_dps)), n_rows):
                for tix in np.array_split(np.arange(len(target_dps)), n_cols):
                    this = self._make_blaster(use_alpha, normalized, smat,
                                              max_dist, precision, approx_nn,
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

        # Not `_stitch`: these blocks scored individual *pairs*, so each writes
        # into the cells its own mask selected rather than a rectangle.
        for this, res in self._map(jobs, n_cores, progress, backend=be):
            scr[this.mask] = res

        if return_mask:
            return scr, mask

        return scr

    def synblast(self, query, target, *, by_type, cn_types, scores, normalized,
                 smat, n_cores, progress):
        """Synapse-based NBLAST (SynBLAST)."""
        from ..synblast_funcs import SynBlaster, estimate_target_blocks

        def get_connectors(n):
            if cn_types is not None:
                return n.connectors[n.connectors['type'].isin(cn_types)]
            return n.connectors

        # Same partitioning as NBLAST, but timed on connector queries.
        be, n_workers = self._dispatcher(n_cores)
        n_rows, n_cols = self._partition(query, target, n_workers, progress,
                                         estimate_fn=estimate_target_blocks)

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

        return self._stitch(jobs, n_cores, progress, backend=be,
                            query_ids=query.id, target_ids=target.id,
                            dtype=nb.dtype)
