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

import scipy.optimize
import uuid

import networkx as nx
import numpy as np
import pandas as pd

from typing import Optional

from .. import utils, config


def bipartite_match(scores: pd.DataFrame,
                    one_to_one: bool = True,
                    threshold: Optional[float] = None,
                    labels: Optional[dict] = None,
                    known_matches: Optional[dict] = None,
                    duplicate_penalty: float = 0
                    ) -> dict:
    """Generate a bipartite matching that maximizes the scores.

    Parameters
    ----------
    scores :            pandas.Dataframe
                        *Similarity* scores (e.g. from NBLAST) to use for
                        bipartite matching. Index and columns will be used as
                        node IDs and must hence be unique.
    one_to_one :        bool
                        If True, we will enforce 1:1 matches. This will result
                        in unmatched nodes if ``scores`` is not quadratic or
                        if constraints (below) disallow certain matches. If
                        False, we will iteratively go through rounds of matching
                        in which we duplicate nodes opposite the axis with
                        missing nodes. A simple example to illustrate:

                        Let's assume we want to find a bipartite matching
                        of a 3x5 matrix without constraints. In a first 1:1
                        matching we can only determine 3 pairs leaving 2 nodes
                        unmatched. For those two nodes, we now duplicate their
                        best matches hence creating a 5x5 matrix and re-run
                        the matching. Without additional constraints this will
                        now return 5 pairs of one-to-many matches. With
                        additional constraints, we might have to undergo several
                        rounds of this.

                        Note this is not guaranteed to work if other
                        constraints (``threshold`` or ``restrict_labels``)
                        simply don't allow to match up every node.

    Constraints

    threshold :         int | float, optional
                        Similarities lower than this threshold will be
                        considered to weak for matching and ignored.
    labels :            dict, optional
                        A dictionary mapping node IDs (columns/indices) to
                        label(s). Only nodes that have the exact same set of
                        labels can be matched::

                          {'A': ['uPN', 'adPN'], 'B': ['uPN']}

                        In above example ``A`` can never match with ``B``.

    known_matches :     dict, optional
                        Known matches can be enforced by providing a dictionary
                        mapping node U to V or vice versa::

                            {'A': 'B', 'B': 'C'}

    duplicate_penalty : float [0-1], optional
                        A score penalty for duplicated nodes. Each
                        time a node is duplicated for matching, this penalty
                        will be substracted. For example with
                        ``duplicate_penalty = .1``::

                                               scores
                            original           1
                            first_duplicate    1 - 0.1 = 0.9
                            second_duplicate   0.9 - 0.1 = 0.8

                        Leave at ``0`` for no penalties.

    Returns
    -------
    pandas.DataFrame
                        Ummatched nodes will show up without a counterpart
                        or score.

    Raises
    ------
    ValueError
                        ``scipy.optimize.linear_sum_assignment`` raises an
                        "cost matrix is infeasible" exception when constraints
                        make a matching impossible. Try releasing some
                        constraints.

    """
    assert isinstance(scores, pd.DataFrame)
    assert one_to_one in [False, None, True]

    if not duplicate_penalty:
        duplicate_penalty = 0

    # Make a copy
    scores = scores.copy()

    # Turn such that there are more rows than columns
    if scores.shape[0] < scores.shape[1]:
        scores = scores.T

    # For some odd reason we get funny results if index and column are not str
    scores.index = scores.index.astype(str)
    scores.columns = scores.columns.astype(str)

    # Make sure columns and indices are unique
    if len(set(scores.index) | set(scores.columns)) < (scores.shape[0] + scores.shape[1]):
        raise ValueError('Indices and columns in scores DataFrame must be unique.')

    # Turn into edges
    ix_name = scores.index.name if scores.index.name else 'index'
    edges = scores.reset_index(drop=False).melt(id_vars=ix_name)
    edges.columns = ['U', 'V', 'score']

    # Set distance of below threshold scores to an insanely high value
    # Must not use infinite - otherwise scipy's optimization throws hissy fits
    if not isinstance(threshold, type(None)):
        # Convert threshold to distances
        edges.loc[edges.score < threshold, 'score'] = -np.inf

    # Drop edges between nodes with labels that don't match
    if isinstance(labels, dict):
        # Make sure items are sets
        rl = {k: set(utils.make_iterable(v)) for k, v in labels.items()}

        # Do not change the below! This comparison looks clunky but any other
        # way we run into issues where all of the sudden `np.nan == np.nan = False`
        ul = edges.U.map(lambda x: rl.get(x, None)).values
        vl = edges.V.map(lambda x: rl.get(x, None)).values
        edges.loc[ul != vl, 'score'] = -1e10  # can't use inf here unfortunately

    # Enforce premade matches by adding edges with weight infinity
    if isinstance(known_matches, dict):
        # This is rather slow atm and we should try speeding this up
        match_sets = [{str(k), str(v)} for k, v in known_matches.items()]
        is_match = edges[['U', 'V']].apply(set, axis=1).apply(lambda x: x in match_sets)
        edges.loc[is_match, 'score'] = np.inf

    # Turn back into matrix
    scores_final = edges.pivot(index='U', columns='V', values='score')

    # Make matches
    left_ix, right_ix = scipy.optimize.linear_sum_assignment(scores_final, maximize=True)

    # Drop matches that have effectively minus infinity scores
    is_above = scores_final.values[left_ix, right_ix] > 1e-10
    left_ix, right_ix = left_ix[is_above], right_ix[is_above]

    # Convert to row/column names
    left_id = scores_final.index[left_ix]
    right_id = scores_final.columns[right_ix]

    # Check if we need to rematch -> note we are only checking indices (axis 0)
    missing = np.array(list(set(scores.index) - set(left_id)))

    # We can only rematch neurons that have a possible match to being with
    missing = missing[scores_final.loc[missing].max(axis=1) > 1e-10]

    if not one_to_one and len(missing) > 0:
        while len(missing) > 0:
            # Find the best match for the missing nodes
            best_match = np.argmax(scores_final.loc[missing].values, axis=1)

            # Add virtual clones of these best matches
            clones = scores_final.iloc[:, best_match] - duplicate_penalty
            scores_final = pd.concat([scores_final, clones], axis=1)

            # Make pairs again
            left_ix, right_ix = scipy.optimize.linear_sum_assignment(scores_final, maximize=True)

            # Drop matches that have effectively minus infinity scores
            is_above = scores_final.values[left_ix, right_ix] > 1e-10
            left_ix, right_ix = left_ix[is_above], right_ix[is_above]

            # Convert to row/column names
            left_id = scores_final.index[left_ix]
            right_id = scores_final.columns[right_ix]

            # Check if we need to rematch
            # -> note we are only checking columns now (axis 0)
            missing = np.array(list(set(scores_final.columns) - set(right_id)))

            # We can only rematch neurons that have a possible match to begin with
            missing = missing[scores_final[missing].max(axis=0) > 1e-10]

            # Transpose matrix
            scores_final = scores_final.T

        # One final transpose to get back in original configuration
        scores_final = scores_final.T

    matches = pd.DataFrame()
    matches['U'] = left_id
    matches['V'] = right_id

    # Add scores
    matches['score'] = scores_final.values[left_ix, right_ix]

    # Add neurons that have not been matches
    matched = np.unique(matches[['U', 'V']].values.flatten())
    U_miss = pd.DataFrame([])
    U_miss['U'] = scores_final.index[~scores_final.index.isin(matched)]
    U_miss['V'] = None
    U_miss['score'] = None

    V_miss = pd.DataFrame([])
    V_miss['U'] = None
    V_miss['V'] = scores_final.columns[~scores_final.columns.isin(matched)]
    V_miss['score'] = None

    if not U_miss.empty or not V_miss.empty:
        config.logger.info(f'{U_miss.shape[0] + V_miss.shape[0]} neurons could '
                           'not be matched given the restriction.')

    return pd.concat([matches, U_miss, V_miss], axis=0).reset_index(drop=True)
