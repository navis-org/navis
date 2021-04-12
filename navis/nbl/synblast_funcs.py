
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

"""Module contains functions implementing SyNBLAST."""

import numbers
import os

import numpy as np
import pandas as pd

from concurrent.futures import ProcessPoolExecutor
from scipy.spatial import cKDTree
from typing import Union, Optional
from typing_extensions import Literal

from .. import config, utils
from ..core import NeuronList, BaseNeuron

from .nblast_funcs import check_microns, find_optimal_partition, ScoringFunction

__all__ = ['synblast']

fp = os.path.dirname(__file__)
smat_path = os.path.join(fp, 'score_mats')

logger = config.logger


class SynBlaster:
    """Implements a synapsed-based NBLAST algorithm.

    Please note that some properties are computed on initialization and
    changing parameters at a later stage will mess things up!

    TODOs
    -----
    - implement per-type comparisons
    - implement `use_alpha` as average synapse density (i.e. emphasise areas where
      a neuron has lots of synapses)

    Parameters
    ----------
    normalzed :     bool
                    If True, will normalize scores by the best possible score
                    (i.e. self-self) of the query neuron.
    by_type :       bool
                    If True will only compare synapses with the same value in
                    the "type" column.
    smat :          str | pd.DataFrame
                    Score matrix. If 'auto' (default), will use scoring matrices
                    from FCWB. Same behaviour as in R's nat.nblast
                    implementation. If ``smat=None`` the scores will be
                    generated as the product of the distances and the dotproduct
                    of the vectors of nearest-neighbor pairs.
    progress :      bool
                    If True, will show a progress bar.

    """

    def __init__(self, normalized=True, by_type=True,
                 smat='auto', progress=True):
        """Initialize class."""
        self.normalized = normalized
        self.progress = progress
        self.by_type = by_type

        if smat == 'auto':
            smat = pd.read_csv(f'{smat_path}/smat_fcwb.csv',
                               index_col=0)

        self.score_fn = ScoringFunction(smat)

        self.self_hits = []
        self.trees = []
        self.ids = []

    def append(self, neuron, id=None):
        """Append neurons/connector tables."""
        if isinstance(neuron, (NeuronList, list)):
            for n in neuron:
                self.append(n, id=id)
            return

        if isinstance(neuron, BaseNeuron):
            if not neuron.has_connectors:
                raise ValueError('Neuron must have synapses.')
            cn = neuron.connectors
        elif isinstance(neuron, pd.DataFrame):
            cn = neuron
        else:
            raise ValueError('Expected a Neuron(s) or single DataFrame, '
                             f'got "{type(neuron)}"')

        if isinstance(id, type(None)):
            if isinstance(neuron, pd.DataFrame):
                raise ValueError('Must provide ID explicitly when append '
                                 'DataFrame.')
            else:
                id = neuron.id

        self.ids.append(id)
        self.trees.append({})
        if not self.by_type:
            data = cn[['x', 'y', 'z']].values
            # Generate the KDTree
            self.trees[-1]['all'] = cKDTree(data)
        else:
            if 'type' not in cn.columns:
                raise ValueError('Connector tables must have a "type" column '
                                 'if `by_type=True`')
            for ty in cn['type'].unique():
                data = cn.loc[cn['type'] == ty, ['x', 'y', 'z']].values
                # Generate the KDTree
                self.trees[-1][ty] = cKDTree(data)

        # Calculate score for self hit
        self.self_hits.append(self.calc_self_hit(cn))

    def calc_self_hit(self, cn):
        """Non-normalized value for self hit."""
        return cn.shape[0] * self.score_fn(0, 1)

    def single_query_target(self, q_idx, t_idx, scores='forward'):
        """Query single target against single target."""
        # Take a short-cut if this is a self-self comparison
        if q_idx == t_idx:
            if self.normalized:
                return 1
            return self.self_hits[q_idx]

        # Run nearest-neighbor search for query against target
        t_trees = self.trees[t_idx]
        q_trees = self.trees[q_idx]

        dists = []
        # Go over all connector types (e.g. "pre" and "post") present in the
        # query neuron. If `by_type=False`, there will be a single tree simply
        # called "all"
        for ty, qt in q_trees.items():
            # If target does not have this type of connectors
            if ty not in t_trees:
                # Note that this infinite distance will simply get the worst
                # score possible in the scoring function
                dists = np.append(dists, [np.inf] * qt.data.shape[0])
            else:
                tt = t_trees[ty]
                dists = np.append(dists, tt.query(qt.data)[0])

        # We use the same scoring function as for normal NBLAST but ignore the
        # vector dotproduct component
        scr = self.score_fn(dists, 1).sum()

        # Normalize against best possible hit (self hit)
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

    def pair_query_target(self, pairs, scores='forward'):
        """SynBLAST multiple pairs.

        Parameters
        ----------
        pairs :             tuples
                            Tuples of (query_ix, target_ix) to query.
        scores :            "forward" | "mean" | "min" | "max"
                            Which scores to return.

        """
        scr = []
        for p in config.tqdm(pairs,
                             desc='SynBlasting pairs',
                             leave=False,
                             position=getattr(self, 'pbar_position', 0),
                             disable=not self.progress):
            scr.append(self.single_query_target(p[0], p[1], scores=scores))

        return scr

    def multi_query_target(self, q_idx, t_idx, scores='forward'):
        """SynBLAST multiple queries against multiple targets.

        Parameters
        ----------
        q_idx,t_idx :       iterable
                            Iterable of query/target neurons indices to
                            SynNBLAST.
        scores :            "forward" | "mean" | "min" | "max"
                            Which scores to return.

        """
        rows = []
        for q in config.tqdm(q_idx,
                             desc='SynBlasting',
                             leave=False,
                             position=getattr(self, 'pbar_position', 0),
                             disable=not self.progress):
            rows.append([])
            for t in t_idx:
                score = self.single_query_target(q, t, scores=scores)
                rows[-1].append(score)

        # Generate results
        res = pd.DataFrame(rows)
        res.columns = [self.ids[t] for t in t_idx]
        res.index = [self.ids[q] for q in q_idx]

        return res

    def all_by_all(self, scores='forward'):
        """NBLAST all-by-all neurons."""
        res = self.multi_query_target(range(len(self.trees)),
                                      range(len(self.trees)),
                                      scores='forward')

        # For all-by-all SyNBLAST we can get the mean score by
        # transposing the scores
        if scores == 'mean':
            res = (res + res.T) / 2
        elif scores == 'min':
            res.loc[:, :] = np.dstack((res, res.T)).min(axis=2)
        elif scores == 'max':
            res.loc[:, :] = np.dstack((res, res.T)).max(axis=2)

        return res


def synblast(query: Union['core.BaseNeuron', 'core.NeuronList'],
             target: Optional[str] = None,
             by_type: bool = False,
             cn_types: Optional[list] = None,
             scores: Union[Literal['forward'],
                           Literal['mean'],
                           Literal['min'],
                           Literal['max']] = 'forward',
             normalized: bool = True,
             n_cores: int = os.cpu_count() // 2,
             progress: bool = True) -> pd.DataFrame:
    """Synapsed-based variant of NBLAST.

    The gist is this: for each synapse in the query neuron, we find the closest
    synapse in the target neuron (can be restricted by synapse types). Those
    distances are then scored similar to nearest-neighbor pairs in NBLAST but
    without the vector (dotproduct) component.

    Parameters
    ----------
    query,target :  Neuron/List
                    Query neuron(s) to SynBLAST against the targets. Units should
                    be in microns as NBLAST is optimized for that and have
                    similar sampling resolutions. Neurons must have non-empty
                    connector tables.
    by_type :       bool
                    If True, will use the "type" column in the connector tables
                    to only compare e.g. pre- with pre- and post- with
                    postsynapses.
    cn_types :      str | list, optional
                    Use this to restrict synblast to specific types of
                    connectors (e.g. "pre"synapses only).
    scores :        'forward' | 'mean' | 'min' | 'max'
                    Determines the final scores:

                        - 'forward' (default) returns query->target scores
                        - 'mean' returns the mean of query->target and target->query scores
                        - 'min' returns the minium between query->target and target->query scores
                        - 'max' returns the maximum between query->target and target->query scores

    n_cores :       int, optional
                    Max number of cores to use for nblasting. Default is
                    ``os.cpu_count() // 2``. This should ideally be an even
                    number as that allows optimally splitting queries onto
                    individual processes.
    normalized :    bool, optional
                    Whether to return normalized SynBLAST scores.
    progress :      bool
                    Whether to show progress bars.

    Returns
    -------
    scores :        pandas.DataFrame
                    Matrix with SynBLAST scores. Rows are query neurons, columns
                    are targets.

    Examples
    --------
    >>> import navis
    >>> nl = navis.example_neurons(n=5)
    >>> nl.units
    <Quantity([8 8 8 8 8], 'nanometer')>
    >>> # Convert to microns
    >>> nl_um = nl * (8 / 1000)
    >>> # Run type-agnostic SyNBLAST
    >>> scores = navis.synblast(nl_um[:3], nl_um[3:], progress=False)
    >>> # Run type-sensitive (i.e. pre vs pre and post vs post) SyNBLAST
    >>> scores = navis.synblast(nl_um[:3], nl_um[3:], by_type=True, progress=False)

    See Also
    --------
    :func:`navis.nblast`
                The original morphology-based NBLAST.

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
    elif n_cores < 1:
        raise ValueError('`n_cores` must not be smaller than 1')
    elif n_cores > os.cpu_count():
        logger.warning('`n_cores` should not larger than the number of '
                       'available cores')

    # Make sure we're working on NeuronList
    query = NeuronList(query)
    target = NeuronList(target)

    # Make sure all neurons have connectors
    if not all(query.has_connectors):
        raise ValueError('Some query neurons appear to not have a connector table.')
    if not all(target.has_connectors):
        raise ValueError('Some target neurons appear to not have a connector table.')

    if not isinstance(cn_types, type(None)):
        cn_types = utils.make_iterable(cn_types)

    if not isinstance(cn_types, type(None)) or by_type:
        if any(['type' not in n.connectors.columns for n in query]):
            raise ValueError('Connector tables must have a "type" column if '
                             '`by_type=True` or `cn_types` is not `None`.')

    # Find an optimal partition that minimizes the number of neurons
    # we have to send to each process
    n_rows, n_cols = find_optimal_partition(n_cores, query, target)

    nblasters = []
    for q in np.array_split(query, n_rows):
        for t in np.array_split(target, n_cols):
            # Initialize SynNBlaster
            this = SynBlaster(normalized=normalized,
                              by_type=by_type,
                              progress=progress)
            # Add queries and targets
            for nl in [q, t]:
                for n in nl:
                    if not isinstance(cn_types, type(None)):
                        cn = n.connectors[n.connectors['type'].isin(cn_types)]
                    else:
                        cn = n.connectors

                    this.append(cn, id=n.id)

            # Keep track of indices of queries and targets
            this.queries = np.arange(len(q))
            this.targets = np.arange(len(t)) + len(q)
            this.pbar_position = len(nblasters)

            nblasters.append(this)

    # If only one core, we don't need to break out the multiprocessing
    if n_cores == 1:
        return this.multi_query_target(this.queries,
                                       this.targets,
                                       scores=scores)

    with ProcessPoolExecutor(max_workers=len(nblasters)) as pool:
        # Each nblaster is passed to its own process
        futures = [pool.submit(this.multi_query_target,
                               q_idx=this.queries,
                               t_idx=this.targets,
                               scores=scores) for this in nblasters]

        results = [f.result() for f in futures]

    scores = pd.DataFrame(np.zeros((len(query), len(target))),
                          index=query.id, columns=target.id)

    for res in results:
        scores.loc[res.index, res.columns] = res.values

    return scores
