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

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

from typing import Union, Optional

from .. import core, plotting, io, utils

__all__ = ['matching_pipeline']

skeleton_source = Union[str, core.NeuronList, callable]


def get_top_N_matches(x: Union[int, str],
                      scores: pd.DataFrame,
                      N: int = 3,
                      raise_missing: bool = False):
    """Get top N matches for a given neuron."""
    matches = pd.DataFrame([])
    if not isinstance(N, slice):
        N = slice(0, N)
    if str(x) in scores.index:
        this = scores.loc[str(x)].sort_values(ascending=False).iloc[N]
        matches['match_id'] = this.index.astype(str)
        matches['score'] = this.values
    elif raise_missing:
        raise ValueError(f'{x} not found in scores.')

    return matches


def get_top_score_matches(x: Union[int, str],
                          scores: pd.DataFrame,
                          min_score: float = .3,
                          raise_missing: bool = False):
    """Get all matches above a given score for a given neuron."""
    if str(x) in scores.index:
        this = scores.loc[str(x)].sort_values(ascending=False)
        this = this[this >= min_score]
        matches = this.to_frame().reset_index()
        matches.columns = ['match_id', 'score']
    elif raise_missing:
        raise ValueError(f'{x} not found in scores.')
    return matches


def matching_pipeline(scores: pd.DataFrame,
                      query_source: skeleton_source,
                      match_source: skeleton_source,
                      N: Union[float, int] = 3,
                      to_match: Optional[list] = None,
                      color_by_score: bool = True,
                      add_plot: core.Volume = None):
    """Pipeline for co-vizualising query neurons and potential matches.

    Parameters
    ----------
    scores :        pandas.DataFrame
                    DataFrame containing _similarity_ scores. Rows are intepreted
                    as query neurons, columns as potential matches.
    query_source :  str | function | NeuronList
                    Source for query neurons. Can be::
                      - str: path to folder with ``.swc`` or ``.obj`` files
                      - function: must accept single ID and return TreeNeuron
                      - NeuronList
    match_source :  str | function | NeuronList
                    Source for potential matches.
    N :             int | float | slice
                    How many matches to show at a time. If >= 1, will show
                    top ``N`` matches. If <= 1, will show all hits with better
                    than ``N`` score.
    to_match :      iterable, optional
                    If given, will match only these neurons. If ``None``, will
                    match all neurons in ``scores.index``.
    color_by_score : bool
                    If True, will color matches by score.
    add_plot :      Volume, optional
                    If provided, will add these Volumes to the plots.

    Returns
    -------
    results :       pandas.DataFrame
                    DataFrame containing the confirmed matches.

    """
    if utils.is_jupyter():
        raise TypeError('Matching pipeline is meant to be run from a terminal '
                        'and does not work in Jupyter environments.')

    # Create from scratch
    v = plotting.Viewer()
    v._cycle_mode = 'hide'

    if add_plot:
        v.add(add_plot)

    if isinstance(to_match, type(None)):
        to_match = scores.index
    else:
        to_match = np.asarray(to_match)
        miss = to_match[~np.isin(to_match, scores.index)]
        if miss.size:
            raise ValueError(f'{len(miss)} IDs not found in scores: {", ".join(miss)}')

    results = []
    for i, x in enumerate(to_match):
        # Grab the matches
        if isinstance(N, slice) or N >= 1:
            matches = get_top_N_matches(x, scores=scores, N=N)
        else:
            matches = get_top_score_matches(x, scores=scores, N=N)
        # Stop if no matches given the criteria
        if matches.empty:
            print(f'No match candidates found for {x}')
            continue

        # Print scores + matches
        print(f'Matches for neuron {x} [{i}/{len(to_match)}]')
        print(matches)
        # Fetch skeletons for query neuron
        if isinstance(query_source, core.NeuronList):
            src = query_source.idx[x]
        elif isinstance(query_source, str):
            src = io.read_swc(os.path.join(query_source, f'{x}.swc'))
        elif callable(query_source):
            src = query_source(x)
        else:
            raise BaseException(f'Unable to use query source of type {type(query_source)}')

        if not isinstance(src, core.NeuronList):
            src = core.NeuronList(src)

        # Fetch skeletons for potential matches
        if isinstance(match_source, core.NeuronList):
            trgt = match_source.idx[matches.match_id.values]
        elif isinstance(match_source, str):
            trgt = io.read_swc([os.path.join(match_source, f'{n}.swc') for n in matches.match_id.values])
        elif callable(match_source):
            trgt = [match_source(n) for n in matches.match_id.values]
        else:
            raise BaseException(f'Unable to use query source of type {type(match_source)}')

        if not isinstance(trgt, core.NeuronList):
            trgt = core.NeuronList(trgt)

        # Make sure the IDs match
        for n, d in zip(trgt, matches.match_id.values):
            n.id = d

        # Generate colors
        if color_by_score:
            this_scores = matches.score.values
            # Normalize if there is more than one neuron to match
            if len(this_scores) > 1:
                this_scores = (this_scores - this_scores.min()) / (this_scores.max() - this_scores.min())
            cmap = plt.get_cmap('cool')
            colors = ['w'] * len(src) + [cmap(s) for s in this_scores]
        else:
            colors = ['w'] * len(src) + sns.color_palette('muted', len(trgt))

        # Plot everything
        v.add([src, trgt], c=colors)
        v.pin_neurons(src)

        # Wait for user to confirm they are happy.
        resp = 'None'
        while resp.lower() not in ['', 'b', 's']:
            resp = input('Enter to Continue | [S]kip | [B]reak')

        if resp.lower() == 's':
            continue

        # Get matches
        matched = matches[matches.match_id.isin(v.visible)].copy()
        if not matched.empty:
            matched['query'] = x
        else:
            matched = pd.DataFrame([[None, None, x]],
                                   columns=['match_id', 'score', 'query'])

        results.append(matched)

        # Clear viewer
        v.remove(src)
        v.remove(trgt)

        if resp.lower() == 'b':
            break

    # Close viewer
    v.close()

    return pd.concat(results, sort=True)
