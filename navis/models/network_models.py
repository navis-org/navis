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
"""Module contains network models."""

import multiprocessing as mp
import numpy as np
import pandas as pd
import warnings

from typing import Iterable, Union, Optional, Callable

from .. import config
from ..compute.dispatch import default_n_workers, worker_initializer

# Set up logging
logger = config.get_logger(__name__)

__all__ = ['BayesianTraversalModel', 'TraversalModel', 'linear_activation_p',
           'random_linear_activation_function']


class BaseNetworkModel:
    """Base model for network simulations."""

    def __init__(self, edges: pd.DataFrame, source: str, target: str):
        """Initialize model."""
        assert isinstance(edges, pd.DataFrame), f'edges must be pandas DataFrame, got "{type(edges)}"'
        assert source in edges.columns, f'edges DataFrame must contain "{source}" column'
        assert target in edges.columns, f'edges DataFrame must contain "{target}" column'
        self.edges = edges
        self.source = source
        self.target = target

        if (self.edges.dtypes[source] == object) or (self.edges.dtypes[target] == object):
            logger.warning('Looks like sources and/or targets in your edge list '
                           'might be strings? This can massively slow down '
                           'computations. If at all possible try to use numeric IDs.')

    @property
    def n_nodes(self) -> int:
        """Return unique nodes in network."""
        return np.unique(self.edges[[self.source, self.target]].values.flatten()).shape[0]

    @property
    def has_results(self) -> bool:
        """Check if model has results."""
        if isinstance(getattr(self, 'results', None), pd.DataFrame):
            return True
        return False

    def run_parallel(self,
                     n_cores: Optional[int] = None,
                     iterations: int = 100,
                     **kwargs: dict) -> None:
        """Run model using parallel processes."""
        n_cores = n_cores or default_n_workers()

        # Note that we initialize each process by making "edges" a global argument
        with mp.Pool(processes=n_cores,
                     initializer=worker_initializer(n_cores),
                     initargs=(self.initializer,)) as pool:

            # Each process will take about the same amount of time
            # So we want each process to take a single batch of iterations/n_cores runs
            kwargs['iterations'] = int(iterations/n_cores)
            calls = [{**kwargs, **{'position': i}} for i in range(int(n_cores))]

            # Generate processes - note the use of chunksize 1 because we have
            # already chunked the iterations such that each worker process
            # runs exactly once
            p = pool.imap_unordered(self._worker_wrapper, calls, chunksize=1)

            # Wait for processes to complete
            res = list(p)

        # Combine results
        self.results = pd.concat(res, axis=0)
        self.iterations = iterations

    def _worker_wrapper(self, kwargs: dict):
        return self.run(**kwargs)

    def initializer(self):
        pass

    def run(self):
        pass


class TraversalModel(BaseNetworkModel):
    """Model for traversing a network starting with given seed nodes.

    What this does:

      1. Grab all already visited nodes (starting with `seeds` in step 1)
      2. Find all downstream nodes of these
      3. Probabilistically traverse based on the weight of the connecting edges
      4. Add those newly visited nodes to the pool & repeat from beginning
      5. Stop when every (connected) neuron was visited or we reached `max_steps`

    Parameters
    ----------
    edges :             pandas.DataFrame
                        DataFrame representing an edge list. Must minimally have
                        a `source` and `target` column.
    seeds :             iterable
                        Seed nodes for traversal. Nodes that aren't found in
                        `edges['source']` will be (silently) removed.
    weights :           str, optional
                        Name of a column in `edges` used as weights. If not
                        provided, all edges will be given a weight of 1. If using
                        the default activation function the weights need to be
                        between 0 and 1.
    max_steps :         int
                        Limits the number of steps for each iteration.
    traversal_func :    callable, optional
                        Function that determines whether a given edge will be
                        traversed or not in a given step. Must take numpy array
                        (N, 1) of edge weights and return an array with
                        True/False of equal size. Defaults to
                        [`navis.models.network_models.random_linear_activation_function`][]
                        which will linearly scale probability of traversal
                        from 0 to 100% between edges weights 0 to 0.3.

    Examples
    --------
    >>> from navis.models import TraversalModel
    >>> import networkx as nx
    >>> import numpy as np
    >>> # Generate a random graph
    >>> G = nx.fast_gnp_random_graph(1000, .2, directed=True)
    >>> # Turn into edge list
    >>> edges = nx.to_pandas_edgelist(G)
    >>> # Add random edge weights
    >>> edges['weight'] = np.random.random(edges.shape[0])
    >>> # Initialize model
    >>> model = TraversalModel(edges, seeds=list(G.nodes)[:10])
    >>> # Run model on 2 cores
    >>> model.run_parallel(n_cores=2, iterations=100)
    >>> # Get a summary
    >>> model.summary.tail()                                    # doctest: +SKIP
          layer_min  layer_max  layer_mean  layer_median
    node
    995           2          2        2.00             2
    996           2          3        2.33             2
    997           2          2        2.00             2
    998           2          2        2.00             2
    999           2          2        2.00             2

    Above Graph was traversed quickly (3 steps max). Let's adjust the
    traversal function:

    >>> from navis.models import random_linear_activation_function
    >>> # Use a lower probability for activation
    >>> def my_act(x):
    ...     return random_linear_activation_function(x, max_w=10)
    >>> model = TraversalModel(edges, seeds=list(G.nodes)[:10],
    ...                        traversal_func=my_act)
    >>> res = model.run(iterations=100)
    >>> res.tail()                                              # doctest: +SKIP
          layer_min  layer_max  layer_mean  layer_median
    node
    995           2          4       3.210           3.0
    996           2          4       3.280           3.0
    997           2          4       3.260           3.0
    998           2          4       3.320           3.0
    999           2          4       3.195           3.0

    """

    def __init__(self,
                 edges: pd.DataFrame,
                 seeds: Iterable[Union[str, int]],
                 source: str = 'source',
                 target: str = 'target',
                 weights: Optional[str] = 'weight',
                 max_steps: int = 15,
                 traversal_func: Optional[Callable] = None):
        """Initialize model."""
        super().__init__(edges=edges, source=source, target=target)

        if not weights:
            edges['weight'] = 1
            weights = 'weight'

        assert weights in edges.columns, f'"{weights}" must be column in edge list'

        # Remove seeds that don't exist
        self.seeds = edges[edges[self.source].isin(seeds)][self.source].unique()

        if len(self.seeds) == 0:
            raise ValueError('None of the seeds where among edge list sources.')

        self.weights = weights
        self.max_steps = max_steps

        if isinstance(traversal_func, type(None)):
            self.traversal_func = random_linear_activation_function
        elif callable(traversal_func):
            self.traversal_func = traversal_func
        else:
            raise ValueError('`traversal_func` must be None or a callable')

    @property
    def summary(self) -> pd.DataFrame:
        """Per-node summary."""
        return getattr(self, '_summary', self.make_summary())

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = f'{self.__class__}: {self.edges.shape[0]} edges; {self.n_nodes}' \
            f' unique nodes; {len(self.seeds)} seeds;' \
            f' traversal_func {self.traversal_func}.'
        if self.has_results:
            s += f' Model ran with {self.iterations} iterations.'
        else:
            s += ' Model has not yet been run.'
        return s

    def make_summary(self) -> pd.DataFrame:
        """Generate summary."""
        if not self.has_results:
            logger.error('Must run simulation first.')

        summary = self.results.groupby('node',
                                       as_index=False).steps.agg(['min',
                                                                  'max',
                                                                  'mean',
                                                                  'median'])

        summary.rename({'min': 'layer_min',
                        'mean': 'layer_mean',
                        'max': 'layer_max',
                        'median': 'layer_median'}, axis=1, inplace=True)

        self._summary = summary
        return self._summary

    def run(self, iterations: int = 100, return_iterations=False, **kwargs) -> pd.DataFrame:
        """Run model (single process).

        Use `.run_parallel` to use parallel processes.

        """
        # For some reason this is required for progress bars in Jupyter to show
        print(' ', end='', flush=True)
        # For faster access, use the raw array
        # Note: we're splitting the columns in case we have different datatypes
        # (e.g. int64 for IDs and float for weights)
        sources = self.edges[self.source].values
        targets = self.edges[self.target].values
        weights = self.edges[self.weights].values

        # For some reason the progress bar does not show unless we have a print here
        all_enc_nodes = None
        for it in config.trange(1, iterations + 1,
                                disable=config.pbar_hide,
                                leave=config.pbar_leave,
                                position=kwargs.get('position', 0)):
            # Set seeds as encountered in step 1
            enc_nodes = self.seeds
            enc_steps = np.repeat(1, len(self.seeds))
            if return_iterations:
                enc_it = np.repeat(it, len(self.seeds))

            # Start with all edges
            this_weights = weights
            this_sources = sources
            this_targets = targets
            for i in range(2, self.max_steps + 1):
                # Which edges have their presynaptic node already traversed?
                pre_trav = np.isin(this_sources, enc_nodes)
                # Among those, which edges have the postsynaptic node traversed?
                post_trav = np.isin(this_targets[pre_trav], enc_nodes)

                # Combine conditions to find edges where the presynaptic node
                # has been traversed but not the postsynaptic node
                pre_not_post = np.where(pre_trav)[0][~post_trav]
                out_targets = this_targets[pre_not_post]
                out_weights = this_weights[pre_not_post]

                # Drop edges that have already been traversed - speeds up things
                pre_and_post = np.where(pre_trav)[0][post_trav]
                this_targets = np.delete(this_targets, pre_and_post, axis=0)
                this_sources = np.delete(this_sources, pre_and_post, axis=0)
                this_weights = np.delete(this_weights, pre_and_post, axis=0)

                # Stop if we traversed the entire (reachable) graph
                if out_targets.size == 0:
                    break

                # Edges traversed in this round
                trav = self.traversal_func(out_weights)
                if not trav.sum():
                    continue

                trav_targets = out_targets[trav]

                # Keep track
                new_trav = np.unique(trav_targets)

                # Store results
                enc_nodes = np.concatenate((enc_nodes, new_trav))
                enc_steps = np.concatenate((enc_steps, np.repeat(i, len(new_trav))))
                if return_iterations:
                    enc_it = np.concatenate((enc_it, np.repeat(it, len(new_trav))))

            # Save this round of traversal
            if all_enc_nodes is None:
                all_enc_nodes = enc_nodes
                all_enc_steps = enc_steps
                if return_iterations:
                    all_enc_it = enc_it
            else:
                all_enc_nodes = np.concatenate((all_enc_nodes, enc_nodes))
                all_enc_steps = np.concatenate((all_enc_steps, enc_steps))
                if return_iterations:
                    all_enc_it = np.concatenate((all_enc_it, enc_it))

        self.iterations = iterations

        # Combine results into DataFrame
        self.results = pd.DataFrame()
        self.results['steps'] = all_enc_steps
        self.results['node'] = all_enc_nodes
        if return_iterations:
            self.results['iteration'] = all_enc_it

        return self.results


class BayesianTraversalModel(TraversalModel):
    """Model for traversing a network starting with given seed nodes.

    This model is a Bayes net version of
    [`navis.models.network_models.TraversalModel`][] that propagates
    traversal probabilities through the network and converges to a
    distribution of time of traversal for each node, rather than
    stochastically sampling.

    Unlike `TraversalModel`, this model should only be run once.
    Note alse that `traversal_func` should be a function returing
    probabilities of traversal, rather than a random boolean of traversal.

    Parameters
    ----------
    edges :             pandas.DataFrame
                        DataFrame representing an edge list. Must minimally have
                        a `source` and `target` column.
    seeds :             iterable
                        Seed nodes for traversal. Nodes that aren't found in
                        `edges['source']` will be (silently) removed.
    weights :           str, optional
                        Name of a column in `edges` used as weights. If not
                        provided, all edges will be given a weight of 1. If using
                        the default activation function the weights need to be
                        between 0 and 1.
    max_steps :         int
                        Limits the number of steps for each iteration.
    traversal_func :    callable, optional
                        Function returning probability whether a given edge will be
                        traversed or not in a given step. Must take numpy array
                        (N, 1) of edge weights and return an array with
                        probabilities of equal size. Defaults to
                        [`navis.models.network_models.linear_activation_p`][]
                        which will linearly scale probability of traversal
                        from 0 to 100% between edges weights 0 to 0.3.
    method :            "conditioned" | "legacy"
                        How an edge's traversal is accumulated over time (see
                        Notes). "legacy" reproduces the model as it was before
                        navis 1.12.

    Notes
    -----
    This model is a fast, deterministic approximation of the Monte-Carlo
    [`navis.models.network_models.TraversalModel`][].

    With `method='conditioned'` (the default) it propagates the full delivery
    distribution of every inbound edge (the parent's traversal-time
    distribution convolved with the edge's per-step firing probability) and is
    *exact* for tree-like graphs and for single points of reconvergence (e.g. diamonds). 
    It does, however, still assume that a node's parents are traversed *independently*,
    and is only an approximation when two or more parents share correlated upstream ancestry:
    on the minimal shared-ancestry graph `0 -> 1 -> {2, 3} -> 4` with a
    per-step traversal probability of 0.5, parents 2 and 3 share node 1's
    random activation time and node 4 is reported ~0.27 layers early.

    With `method='legacy'` it additionally treats an edge's firing at each
    step as independent of whether that same edge already failed to fire at
    earlier steps. That over-counts traversal, on the chain `0 -> 1 -> 2` 
    with a per-step traversal probability of 0.5, node 2 is reported ~0.23 layers early. 
    Use it only to reproduce older results.

    Note also that, as with `TraversalModel`, results near the `max_steps`
    horizon are affected by truncation - increase `max_steps` if a node's
    traversal-time distribution has not effectively converged to 1.

    Examples
    --------
    >>> from navis.models import BayesianTraversalModel
    >>> import networkx as nx
    >>> import numpy as np
    >>> # Generate a random graph
    >>> G = nx.fast_gnp_random_graph(1000, .2, directed=True)
    >>> # Turn into edge list
    >>> edges = nx.to_pandas_edgelist(G)
    >>> # Add random edge weights
    >>> edges['weight'] = np.random.random(edges.shape[0])
    >>> # Initialize model
    >>> model = BayesianTraversalModel(edges, seeds=list(G.nodes)[:10])
    >>> # Run model
    >>> res = model.run()
    >>> # Get a summary
    >>> model.summary.tail()                                    # doctest: +SKIP
          layer_min  layer_max  layer_mean  layer_median
    node
    995           2          2        2.00             2
    996           2          3        2.33             2
    997           2          2        2.00             2
    998           2          2        2.00             2
    999           2          2        2.00             2

    Above Graph was traversed quickly (3 steps max). Let's adjust the
    traversal function:

    >>> from navis.models import linear_activation_p
    >>> # Use a lower probability for activation
    >>> def my_act(x):
    ...     return linear_activation_p(x, max_w=10)
    >>> model = BayesianTraversalModel(edges, seeds=list(G.nodes)[:10],
    ...                                traversal_func=my_act)
    >>> res = model.run()
    >>> res.tail()                                              # doctest: +SKIP
          layer_min  layer_max  layer_mean  layer_median
    node
    995           2          4       3.210           3.0
    996           2          4       3.280           3.0
    997           2          4       3.260           3.0
    998           2          4       3.320           3.0
    999           2          4       3.195           3.0

    On a chain the default method is exact, whereas `method='legacy'` reports
    traversal too early:

    >>> import pandas as pd
    >>> # A chain 0 -> 1 -> 2; weight .15 means p = 0.5 of traversal per step
    >>> edges = pd.DataFrame({'source': [0, 1], 'target': [1, 2],
    ...                       'weight': [.15, .15]})
    >>> means = {}
    >>> for method in ('conditioned', 'legacy'):
    ...     model = BayesianTraversalModel(edges, seeds=[0], max_steps=30,
    ...                                    method=method)
    ...     _ = model.run()
    ...     means[method] = round(float(model.summary.loc[2, 'layer_mean']), 2)
    >>> means
    {'conditioned': 5.0, 'legacy': 4.77}

    """

    def __init__(self,
                 *args,
                 traversal_func: Optional[Callable] = None,
                 method: str = 'conditioned',
                 **kwargs):
        """Initialize model."""
        super().__init__(*args, **kwargs)

        if isinstance(traversal_func, type(None)):
            self.traversal_func = linear_activation_p
        elif callable(traversal_func):
            self.traversal_func = traversal_func
        else:
            raise ValueError('`traversal_func` must be None or a callable')

        if method not in ('conditioned', 'legacy'):
            raise ValueError('`method` must be "conditioned" or "legacy", '
                             f'got "{method}"')
        self.method = method

    def make_summary(self) -> pd.DataFrame:
        """Generate summary."""
        if not self.has_results:
            logger.error('Must run simulation first.')

        cmfs = np.stack(self.results.cmf)
        layer_min = (cmfs != 0).argmax(axis=1)
        valid = np.any(cmfs != 0, axis=1)
        layer_max = (cmfs == 1.).argmax(axis=1)
        layer_max[~np.any(cmfs == 1., axis=1)] = cmfs.shape[1]
        layer_median = (cmfs >= .5).argmax(axis=1).astype(float)
        pmfs = np.diff(cmfs, axis=1, prepend=0.)
        layer_pmfs = pmfs * np.arange(pmfs.shape[1])
        layer_mean = np.sum(layer_pmfs, axis=1)

        summary = pd.DataFrame({
                'layer_min': layer_min + 1,
                'layer_max': layer_max + 1,
                'layer_mean': layer_mean + 1,
                'layer_median': layer_median + 1,
            }, index=self.results.node)

        # Discard nodes that are never activated
        summary = summary[valid]

        self._summary = summary
        return self._summary

    def run(self, **kwargs) -> pd.DataFrame:

        # For some reason this is required for progress bars in Jupyter to show
        print(' ', end='', flush=True)

        # Raw edge arrays plus the weight-to-probability mapping.
        src_raw = self.edges[self.source].values
        tgt_raw = self.edges[self.target].values
        w_raw = self.edges[self.weights].values.astype(np.float64, copy=True)
        probs = self.traversal_func(w_raw).astype(np.float64, copy=False)

        id_type = src_raw.dtype

        # Change node IDs into indices in [0, len(nodes)).
        ids, edges_flat = np.unique(np.stack([src_raw, tgt_raw], axis=1),
                                    return_inverse=True)
        ids = ids.astype(id_type)
        edges_flat = edges_flat.reshape(-1, 2)
        src = edges_flat[:, 0].astype(np.int64)
        tgt = edges_flat[:, 1].astype(np.int64)

        N = ids.size
        T = self.max_steps

        # CSR by target: a contiguous block of inbound edges per target node.
        order_t = np.argsort(tgt, kind='stable')
        src_by_tgt = src[order_t]
        w_by_tgt = probs[order_t]
        tgt_starts = np.zeros(N + 1, dtype=np.int64)
        np.cumsum(np.bincount(tgt, minlength=N), out=tgt_starts[1:])

        # CSR by source: a contiguous block of outbound edges per source node.
        order_s = np.argsort(src, kind='stable')
        tgt_by_src = tgt[order_s]
        src_starts = np.zeros(N + 1, dtype=np.int64)
        np.cumsum(np.bincount(src, minlength=N), out=src_starts[1:])

        cmfs = np.zeros((N, T), dtype=np.float64)
        seed_idx = np.searchsorted(ids, self.seeds)
        cmfs[seed_idx, :] = 1.

        # Initial frontier: everything directly downstream of a seed.
        seed_mask = np.zeros(N, dtype=bool)
        seed_mask[seed_idx] = True
        changed = np.unique(tgt[seed_mask[src]])

        iter_n = 0

        with config.tqdm(
                total=None,
                unit='upd',
                disable=config.pbar_hide,
                leave=config.pbar_leave,
                position=kwargs.get('position', 0)) as pbar:
            while changed.size:
                iter_n += 1

                # Gather the inbound edges of every node in the frontier into
                # one flat, group-contiguous array. Every frontier node is the
                # target of at least one edge, so no group is empty and the
                # group offsets below are strictly increasing.
                starts = tgt_starts[changed]
                ends = tgt_starts[changed + 1]
                counts = ends - starts
                edge_pos = _repeat_ranges(starts, ends)
                if edge_pos.size == 0:
                    break
                pre_flat = src_by_tgt[edge_pos]
                w_flat = w_by_tgt[edge_pos]
                pre_cmf = cmfs[pre_flat, :]

                # Offset of each frontier node's edge group in the flat array.
                group_starts = np.empty(changed.size, dtype=np.int64)
                group_starts[0] = 0
                np.cumsum(counts[:-1], out=group_starts[1:])

                old_cmf = cmfs[changed, :]
                if self.method == 'conditioned':
                    # Probability that each parent *first* activates at a
                    # given step.
                    pmf = np.diff(pre_cmf, axis=1, prepend=0.)

                    # For each inbound edge, compute the probability that it
                    # has delivered traversal to its target by a given step. An
                    # edge whose parent became active at step ``k`` fires with
                    # probability ``p`` at each subsequent step, so it has
                    # delivered by step ``t`` with probability
                    # ``1 - (1-p)^(t-k)``. The delivery CMF is therefore the
                    # parent's activation-time PMF convolved with this
                    # geometric edge-firing distribution. This correctly models
                    # the fact that all of a single edge's per-step firings are
                    # driven by the *same* (monotone) parent-activation event,
                    # rather than treating them as independent across time (the
                    # latter over-counts traversal and made nodes appear to be
                    # reached too early - see #194). The convolution is
                    # evaluated via the recurrence
                    # ``c[t] = (1-p) * c[t-1] + pmf[t]``, with
                    # ``deliver = parent_cmf - c``.
                    q = 1. - w_flat
                    c = np.empty_like(pmf)
                    c[:, 0] = pmf[:, 0]
                    for t in range(1, T):
                        c[:, t] = q * c[:, t - 1] + pmf[:, t]
                    deliver = pre_cmf - c

                    # Combine inbound edges assuming independence *across
                    # parents*. This is exact for tree-like graphs and single
                    # points of reconvergence (e.g. diamonds); see the Notes in
                    # the class docstring for the residual approximation.
                    # ``reduceat`` gives the exact per-group product, without
                    # the roundoff of an exp-of-sum-of-logs. Take the maximum
                    # with the previous CMF as it is monotonic and to preserve
                    # fixed seed traversal.
                    new_cmf = np.maximum(
                        old_cmf,
                        1. - np.multiply.reduceat(1. - deliver,
                                                  group_starts, axis=0))
                else:
                    # Traversal probability for each inbound edge at each time.
                    posteriors = pre_cmf * w_flat[:, None]

                    # At each time, the probability that at least one inbound
                    # edge is traversed.
                    new_pmf = 1. - np.multiply.reduceat(1. - posteriors,
                                                        group_starts, axis=0)

                    # Offset the time-cumulative probability by 1 to account
                    # for traversal iteration. Use maximum of previous CMF as
                    # it is monotonic and to include fixed seed traversal.
                    new_cmf = old_cmf.copy()
                    if T > 1:
                        new_cmf[:, 1:] = np.maximum(
                            old_cmf[:, 1:],
                            1. - np.cumprod(1. - new_pmf[:, :-1], axis=1))
                np.clip(new_cmf, 0., 1., out=new_cmf)

                actually_changed = ~np.all(np.isclose(old_cmf, new_cmf), axis=1)
                if not actually_changed.any():
                    break
                cmfs[changed[actually_changed], :] = new_cmf[actually_changed]

                # Notify the nodes downstream of those that actually changed.
                updated = changed[actually_changed]
                out_pos = _repeat_ranges(src_starts[updated],
                                         src_starts[updated + 1])
                next_changed = np.unique(tgt_by_src[out_pos])

                pbar.update(int(actually_changed.sum()))
                pbar.set_postfix(iter=iter_n, frontier=int(changed.size),
                                 refresh=False)
                changed = next_changed

        self.iterations = 1
        self.results = pd.DataFrame({'node': ids, 'cmf': list(cmfs)})
        return self.results

    def run_parallel(self, *args, **kwargs) -> None:
        warnings.warn(f"{self.__class__.__name__} should not be run in parallel. Falling back to run.")
        self.run(**kwargs)


def _repeat_ranges(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Flatten per-row ranges `[starts[i], ends[i])` into one int64 array.

    Vectorised equivalent of
    `np.concatenate([np.arange(s, e) for s, e in zip(starts, ends)])`. Used by
    [`navis.models.network_models.BayesianTraversalModel.run`][] to gather the
    edges of the current frontier without a Python loop.
    """
    counts = ends - starts
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    cumc = np.cumsum(counts)
    group_offsets = starts - np.concatenate(([0], cumc[:-1]))
    return np.arange(total, dtype=np.int64) + np.repeat(group_offsets, counts)


def linear_activation_p(
    w: np.ndarray,
    min_w: float = 0,
    max_w: float = .3,
) -> np.ndarray:
    """Linear activation probability.

    Parameters
    ----------
    w :     np.ndarray
            (N, 1) array containing the edge weights.
    min_w : float
            Value of `w` at which probability of activation is 0%.
    max_w : float
            Value of `w` at which probability of activation is 100%.

    Returns
    -------
    np.ndarray
            Probability of activation for each edge.

    """
    return np.clip((w - min_w) / (max_w - min_w), 0., 1.)


def random_activation_function(
    w: np.ndarray,
    func: Callable,
) -> np.ndarray:
    """Sample random activation for edges given a weight to probability function.

    Parameters
    ----------
    w :     np.ndarray
            (N, 1) array containing the edge weights.
    func :  callable
            A function returning activation probabilities given a set of weights.

    Returns
    -------
    np.ndarray
            True or False values for each edge.

    """
    # Generate a random number between 0 and 1 for each connection
    r = np.random.rand(w.shape[0])

    # Normalize weights to probabilities
    w_norm = func(w)

    # Test active
    act = w_norm >= r

    return act


def random_linear_activation_function(w: np.ndarray,
                                      min_w: float = 0,
                                      max_w: float = .3) -> np.ndarray:
    """Random linear activation function.

    Parameters
    ----------
    w :     np.ndarray
            (N, 1) array containing the edge weights.
    min_w : float
            Value of `w` at which probability of activation is 0%.
    max_w : float
            Value of `w` at which probability of activation is 100%.

    Returns
    -------
    np.ndarray
            True or False values for each edge.

    """
    return random_activation_function(w, lambda w: linear_activation_p(w, min_w, max_w))
