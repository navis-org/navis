
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

"""Module contains functions implementing SyNBLAST."""

import os
import operator
import time

import numpy as np
import pandas as pd

from typing import Union, Optional
from typing_extensions import Literal

from .. import config, utils
from ..core import NeuronList, BaseNeuron

from .base import Blaster, NestedIndices
from .smat import Lookup2d
from .backends import resolve_backend

from .nblast_funcs import nblast_preflight, smat_fcwb

try:
    from pykdtree.kdtree import KDTree
except ModuleNotFoundError:
    from scipy.spatial import cKDTree as KDTree

__all__ = ['synblast']

fp = os.path.dirname(__file__)
smat_path = os.path.join(fp, 'score_mats')

logger = config.get_logger(__name__)


# This multiplier controls job size for syNBLASTs (only relevant for
# multiprocessing and if progress=True).
# Larger multiplier = larger job sizes = fewer jobs = slower updates & less overhead
# Smaller multiplier = smaller job sizes = more jobs = faster updates & more overhead
JOB_SIZE_MULTIPLIER = 1


class SynBlaster(Blaster):
    """Implements a synapsed-based NBLAST algorithm.

    Please note that some properties are computed on initialization and
    changing parameters at a later stage will mess things up!

    TODOs
    -----
    - implement `use_alpha` as average synapse density (i.e. emphasize areas
      where a neuron has lots of synapses)

    Parameters
    ----------
    normalized :    bool
                    If True, will normalize scores by the best possible score
                    (i.e. self-self) of the query neuron.
    by_type :       bool
                    If True will only compare synapses with the same value in
                    the "type" column.
    smat :          navis.nbl.smat.Lookup2d | pd.DataFrame | str
                    How to convert the point match pairs into an NBLAST score,
                    usually by a lookup table.
                    If 'auto' (default), will use scoring matrices
                    from FCWB. Same behaviour as in R's nat.nblast
                    implementation.
                    Dataframes will be used to build a `Lookup2d`.
                    If `smat=None` the scores will be
                    generated as the product of the distances and the dotproduct
                    of the vectors of nearest-neighbor pairs.
    progress :      bool
                    If True, will show a progress bar.

    """

    def __init__(self, normalized=True, by_type=True,
                 smat='auto', progress=True):
        """Initialize class."""
        super().__init__(progress=progress)
        self.normalized = normalized
        self.by_type = by_type

        if smat is None:
            self.score_fn = operator.mul
        elif smat == 'auto':
            self.score_fn = smat_fcwb()
        elif isinstance(smat, pd.DataFrame):
            self.score_fn = Lookup2d.from_dataframe(smat)
        else:
            self.score_fn = smat

        self.ids = []

    def append(self, neuron, id=None, self_hit=None) -> NestedIndices:
        """Append neurons/connector tables, returning numerical indices of added objects.

        Note that `self_hit` is ignored and hence calculated from scratch
        when `neuron` is a (nested) list of neurons.
        """
        if isinstance(neuron, pd.DataFrame):
            return self._append_connectors(neuron, id, self_hit=self_hit)

        if isinstance(neuron, BaseNeuron):
            if not neuron.has_connectors:
                raise ValueError('Neuron must have synapses')
            return self._append_connectors(neuron.connectors, neuron.id, self_hit=self_hit)

        try:
            return [self.append(n) for n in neuron]
        except TypeError:
            raise ValueError(
                "Expected a dataframe, or a Neuron or sequence thereof; got "
                f"{type(neuron)}"
            )

    def _append_connectors(self, connectors: pd.DataFrame, id, self_hit=None) -> int:
        if id is None:
            raise ValueError("Explicit non-None id required for appending connectors")

        next_idx = len(self)
        self.ids.append(id)
        self.neurons.append({})
        if not self.by_type:
            data = connectors[['x', 'y', 'z']].values
            # Generate the KDTree
            self.neurons[-1]['all'] = data # KDTree(data)
        else:
            if 'type' not in connectors.columns:
                raise ValueError('Connector tables must have a "type" column '
                                 'if `by_type=True`')
            for ty in connectors['type'].unique():
                data = connectors.loc[connectors['type'] == ty, ['x', 'y', 'z']].values
                # Generate the KDTree
                self.neurons[-1][ty] = data # KDTree(data)

        # Calculate score for self hit if required
        if not self_hit:
            self_hit = self.calc_self_hit(connectors)
        self.self_hits.append(self_hit)

        return next_idx

    def _build_trees(self):
        """Build KDTree for all neurons."""
        for i, n in enumerate(self.neurons):
            for ty, data in n.items():
                n[ty] = KDTree(data)

    def calc_self_hit(self, cn):
        """Non-normalized value for self hit."""
        return cn.shape[0] * self.score_fn(0, 1)

    def single_query_target(self, q_idx: int, t_idx: int, scores='forward'):
        """Query single target against single target."""
        # Take a short-cut if this is a self-self comparison
        if q_idx == t_idx:
            if self.normalized:
                return 1
            return self.self_hits[q_idx]

        # Run nearest-neighbor search for query against target
        t_trees = self.neurons[t_idx]
        q_trees = self.neurons[q_idx]

        dists = []
        # Go over all connector types (e.g. "pre" and "post") present in the
        # query neuron. If `by_type=False`, there will be a single tree simply
        # called "all"
        for ty, qt in q_trees.items():
            # If target does not have this type of connectors
            if ty not in t_trees:
                # Note that this infinite distance will simply get the worst
                # score possible in the scoring function
                n_pts = qt.n if isinstance(qt, KDTree) else qt.shape[0]
                dists = np.append(dists, [np.inf] * n_pts)
            else:
                # Note: we're building the trees lazily here once we actually need them.
                # The main reason is that pykdtree is not picklable and hence
                # we can't pass it to the worker processes.
                if not isinstance(t_trees[ty], KDTree):
                    t_trees[ty] = KDTree(t_trees[ty])

                tt = t_trees[ty]
                if isinstance(qt, KDTree):
                    data = qt.data
                else:
                    data = qt

                # pykdtree tracks data as flat array
                if data.ndim == 1:
                    data = data.reshape((qt.n, qt.ndim))
                dists = np.append(dists, tt.query(data)[0])

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


def synblast(query: Union['BaseNeuron', 'NeuronList'],
             target: Union['BaseNeuron', 'NeuronList'],
             by_type: bool = False,
             cn_types: Optional[list] = None,
             scores: Union[Literal['forward'],
                           Literal['mean'],
                           Literal['min'],
                           Literal['max']] = 'forward',
             normalized: bool = True,
             smat: Optional[Union[str, pd.DataFrame]] = 'auto',
             n_cores: int = os.cpu_count() // 2,
             progress: bool = True,
             backend: Optional[str] = None) -> pd.DataFrame:
    """Synapsed-based variant of NBLAST.

    The gist is this: for each synapse in the query neuron, we find the closest
    synapse in the target neuron (can be restricted by synapse types). Those
    distances are then scored similar to nearest-neighbor pairs in NBLAST but
    without the vector component.

    Parameters
    ----------
    query,target :  Neuron/List
                    Query neuron(s) to SynBLAST against the targets. Units should
                    be in microns as NBLAST is optimized for that and have
                    similar sampling resolutions. Neurons must have (non-empty)
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
                    `os.cpu_count() // 2`. This should ideally be an even
                    number as that allows optimally splitting queries onto
                    individual processes.
    normalized :    bool, optional
                    Whether to return normalized SynBLAST scores.
    smat :          str | pd.DataFrame, optional
                    Score matrix. If 'auto' (default), will use scoring matrices
                    from FCWB. Same behaviour as in R's nat.nblast
                    implementation. If `smat=None` the scores will be
                    generated as the product of the distances and the dotproduct
                    of the vectors of nearest-neighbor pairs.
    progress :      bool
                    Whether to show progress bars. This may cause some overhead,
                    so switch off if you don't really need it.
    backend :       str, optional
                    Which NBLAST backend to use. If `None` (default), uses
                    `navis.config.default_nblast_backend` ("builtin"). Set that
                    to "auto" to pick the fastest available backend. Note that
                    only the "builtin" backend implements SynBLAST.

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
    [`navis.nblast`][]
                The original morphology-based NBLAST.

    """
    # Make sure we're working on NeuronList
    query = NeuronList(query)
    target = NeuronList(target)

    # Run pre-flight checks
    nblast_preflight(query, target, n_cores,
                     req_unique_ids=True, req_dotprops=False,
                     req_microns=isinstance(smat, str) and smat == 'auto')

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

    # Select the backend that will run this SynBLAST
    be = resolve_backend("synblast",
                         backend or config.default_nblast_backend,
                         scores=scores, smat=smat)

    return be.synblast(query, target,
                       by_type=by_type,
                       cn_types=cn_types,
                       scores=scores,
                       normalized=normalized,
                       smat=smat,
                       n_cores=n_cores,
                       progress=progress)


def find_batch_partition(q, t, T=10):
    """Find partitions such that each batch takes about `T` seconds."""
    # Get a median-sized query and target
    q_ix = np.argsort(q.n_connectors)[len(q)//2]
    t_ix = np.argsort(t.n_connectors)[len(t)//2]

    # Generate the KDTree
    tree = KDTree(q[q_ix].connectors[['x', 'y', 'z']].values)

    # Run a quick single query benchmark
    timings = []
    for i in range(10):  # Run 10 tests
        s = time.time()
        _ = tree.query(t[t_ix].connectors[['x', 'y', 'z']].values)  #  ignoring scoring / normalizing
        timings.append(time.time() - s)
    time_per_query = min(timings)  # seconds per medium sized query

    # Number of queries per job such that each job runs in `T` second
    queries_per_batch = T / time_per_query

    # Number of neurons per batch
    neurons_per_batch  = max(1, int(np.sqrt(queries_per_batch)))

    n_rows = len(q) // neurons_per_batch
    n_cols = len(t) // neurons_per_batch

    return max(1, n_rows), max(1, n_cols)
