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
"""Module containing functions and classes to build `NEURON` network models."""

import matplotlib.pyplot as plt
import matplotlib.colors as mcl
import numpy as np
import pandas as pd
import seaborn as sns

import plotly.graph_objects as go

from collections import namedtuple
from matplotlib.collections import LineCollection
from numpy.lib.stride_tricks import sliding_window_view

from ... import config, utils

# We will belay any import error
try:
    import neuron
except ImportError:
    raise ImportError('This interface requires the `neuron` libary to be '
                      'installed:\n pip3 install neuron\n'
                      'See also https://neuron.yale.edu/neuron/')

from neuron.units import ms, mV
neuron.h.load_file('stdrun.hoc')

# Set up logging
logger = config.get_logger(__name__)

__all__ = []

# It looks like there can only ever be one reference to the time
# If we have multiple models, we will each reference them to this variable
main_t = None

Stimulus = namedtuple('Stimulus', ['start', 'stop', 'frequency', 'randomness',
                                   'neurons', 'netstim', 'netcon', 'label'])


"""
Note to self: it would best best if PointNetwork would not actually
build the model until it's executed. That way we can run the entire
simulation in a separate thread (or multiple cores).
"""

class PointNetwork:
    """A Network of Leaky-Integrate-and-Fire (LIF) point processes.

    Examples
    --------
    >>> import navis.interfaces.neuron as nrn
    >>> N = nrn.PointNetwork()
    >>>
    """

    def __init__(self):
        self._neurons = []
        self._neurons_dict = {}
        self._edges = []
        self._stimuli = []
        self._ids = []
        self._labels = []
        self.idx = NetworkIdIndexer(self)

    def __str__(self):
        return f'{type(self).__name__}<neurons={len(self)},edges={len(self.edges)}>'

    def __repr__(self):
        return self.__str__()

    def __contains__(self, id):
        return id in self._ids

    def __getitem__(self, ix):
        """Get point process with given ID."""
        return np.asarray(self.neurons)[ix]

    def __len__(self):
        return len(self._neurons)

    @property
    def edges(self):
        """Edges between nodes of the network.

        Returns
        -------
        list of tuples

                ``[(source_ix, target_ix, weight, NetCon object), ...]``

        """
        return self._edges

    @property
    def neurons(self):
        """Neurons in the network.

        Returns
        -------
        list
                List of ``PointNeurons``.

        """
        return self._neurons

    @property
    def ids(self):
        """IDs of neurons in the network."""
        return self._ids

    @property
    def labels(self):
        """Labels of neurons in the network."""
        return self._labels

    @classmethod
    def from_edge_list(cls, edges, model='IntFire1', source_col='source',
                       target_col='target', weight_col='weight', **props):
        """Generate network from edge list.

        Parameters
        ----------
        edges :         pd.DataFrame
                        Edge list. Must contain 'source', 'target' and 'weight'
                        columns.
        model :         "IntFire1" | "IntFire2" | "IntFire4"
                        The model to use for the integrate-and-fire point processes.
        source_col :    str
                        Name of the column with the source IDs.
        target_col :    str
                        Name of the column with the target IDs.
        weight_col :    str
                        Name of the column with the weights. The important thing
                        to note here is that weight is expected to be in the 0-1
                        range with 1 effectively guaranteeing that a presynaptic
                        spike triggers a postsynaptic spike.
        **props
                        Keyword arguments are passed through to ``add_neurons``.
                        Use to set e.g. labels, threshold or additional
                        model parameters.

        """
        assert isinstance(edges, pd.DataFrame)
        miss = {source_col, target_col, weight_col} - set(edges.columns)
        if miss:
            raise ValueError(f'edge list is missing columns: {miss}')

        # Instantiate
        net = cls()

        # Add neurons
        ids = np.unique(edges[[source_col, target_col]].values)
        net.add_neurons(ids, model=model, **props)

        # Connect neurons
        for (s, t, w) in edges[[source_col, target_col, weight_col]].values:
            net.connect(s, t, w)

        return net

    def add_neurons(self, ids, model='IntFire1', labels=None,
                    skip_existing=False, **props):
        """Add neurons to network.

        Parameters
        ----------
        ids :           list-like
                        Iterable of IDs for which to create neurons.
        model :         "IntFire1" | "IntFire2" | "IntFire4"
                        The model to use for the integrate-and-fire point processes.
        labels :        str | list | dict, optional
                        Labels for neurons. If str will apply the same label to
                        all neurons. If list, must be same length as ``ids``.
                        Dictionary must be ID -> label map.
        skip_existing : bool
                        If True, will skip existing IDs.
        **props
                        Additional parameters used when initializing the point
                        processes. Depends on which model type you are using:
                        e.g. ``refrac`` and ``tau`` for "IntFire1".

        """
        assert model in ["IntFire1", "IntFire2", "IntFire4"]
        model = getattr(neuron.h, model)

        ids = utils.make_iterable(ids)

        if isinstance(labels, dict):
            labels = [labels.get(i, 'NA') for i in ids]
        elif labels:
            labels = utils.make_iterable(labels)
        else:
            labels = ids

        if len(labels) != len(ids):
            raise ValueError('Must provide a label for each neuron')

        for i, l in zip(ids, labels):
            if i in self:
                if skip_existing:
                    continue
                else:
                    raise ValueError(f'Neuron with id {i} already exists. '
                                     'Try using `skip_existing=True`.')

            # Create neuron
            n = PointNeuron(id=i, model=model, label=l, **props)

            self._neurons.append(n)
            self._ids.append(i)
            self._labels.append(l)
            self._neurons_dict[i] = n

    def add_background_noise(self, ids, frequency, randomness=.5,
                             independent=True):
        """Add background noise to given neurons.

        Parameters
        ----------
        ids :           hashable  | iterable
                        IDs of neurons to add noise to.
        frequency :     int | float | iterable
                        Frequency [Hz] of background noise. If iterable must
                        match length of `ids`.
        randomness :    float [0-1]
                        Randomness of spike timings.
        independent :   bool
                        If True (default), each neuron will get its own
                        independent noise stimulus.

        """
        self.add_stimulus(ids=ids, start=0, stop=9999999999,
                          frequency=frequency, randomness=randomness,
                          independent=independent)

    def add_stimulus(self, ids, start, frequency, stop=None, duration=None,
                     randomness=.5, independent=True, label=None, weight=1):
        """Add stimulus to given neurons.

        Important
        ---------
        Stimuli are implemented via a NetStim object which provides the
        specified stimulation (i.e. start, stop, frequency). This NetStim is
        then connected to the neuron(s) via a NetCon. The response of the
        neuron to the stimulus depends heavily on the `weight` of that
        connection: too low and you won't elicit any spikes, too high and you
        will produce higher frequencies than expected. The "correct" weight
        depends on the model & parameters you use for your point processes, and
        I highly recommend you check if you get the expected stimulation.

        Parameters
        ----------
        ids :           int | iterable
                        IDs of neurons to add stimulus to.
        start :         int
                        Start time [ms] for the stimulus. Note that
                        the exact start and stop time will vary depending on
                        `randomness`.
        stop/duration : int
                        Either stop time or duration of stimulus [ms]. Must
                        provide one or the other but not both.
        frequency :     int | iterable
                        Frequency [Hz] of background noise . If iterable must
                        match length of `ids`. Values <= 0 are silently skipped.
        randomness :    float [0-1]
                        Randomness of spike timings.
        independent :   bool
                        If True (default), each neuron will get its own
                        independent stimulus.
        label :         str, optional
                        A label to identify the stimulus.
        weight :        float
                        Weight for the connection between the stimulator and
                        the neuron. This really should be 1 to make sure each
                        spike in the stimulus elicits a spike in the target.

        """
        ids = utils.make_iterable(ids)
        if not utils.is_iterable(frequency):
            frequency = [frequency] * len(ids)
        elif not independent:
            raise ValueError('Stimuli/noises must be independent when '
                             'providing individual frequencies')

        if len(frequency) != len(ids):
            raise ValueError('Must provide either a single frequency for all '
                             'neurons or a frequency for each neuron.')

        if (duration and stop) or (not duration and not stop):
            raise ValueError('Must provide either duration or stop (but not both).')

        if stop:
            duration = stop - start
        else:
            stop = start + duration

        if duration <= 0:
            raise ValueError(f'Duration must be greater than zero.')

        if not independent:
            ns = neuron.h.NetStim()
            ns.interval = frequency[0]
            ns.noise = randomness
            ns.number = int(duration / 1000 * frequency[0])
            ns.start = start

        for i, f in zip(ids, frequency):
            # Skip frequencies lower than 0
            if f <= 0:
                continue

            interval = 1000 / f
            proc = self.idx[i].process

            if independent:
                ns = neuron.h.NetStim()
                ns.interval = interval
                ns.noise = randomness
                ns.number = int(duration / 1000 * f)
                ns.start = start

            nc = neuron.h.NetCon(ns, proc)
            nc.weight[0] = weight
            nc.delay = 0

            self._stimuli.append(Stimulus(start, stop, f, randomness, i, ns, nc, label))

    def clear_stimuli(self):
        """Clear stimuli."""
        self._stimuli = {}

    def connect(self, source, target, weight, delay=5):
        """Connect two neurons.

        Parameters
        ----------
        source :    int | str
                    ID of the source.
        target :    int | str
                    ID of the target
        weight :    float
                    Weight of the edge. The important thing to note here is that
                    the weight is expected to be in the 0-1 range with 1
                    effectively guaranteeing that a presynaptic spike triggers
                    a postsynaptic spike.
        delay :     int
                    Delay in ms between a pre- and a postsynaptic spike.

        """
        # Get the point processes corresponding to source and target
        pre = self.idx[source]
        post = self.idx[target]

        # Connect
        nc = neuron.h.NetCon(pre.process, post.process)
        nc.weight[0] = weight
        nc.delay = delay

        # Keep track
        self._edges.append([source, target, weight, nc])

    def plot_raster(self, subset=None, group=False, stimuli=True, ax=None, label=False,
                    backend='auto', **kwargs):
        """Raster plot of spike timings.

        Parameters
        ----------
        subset :        list-like
                        Subset of IDs to plot. You can also use this to
                        determine the order of appearance.
        ax :            matplotlib axis |  plotly figure, optional
                        Axis/figure to plot onto.
        label :         bool
                        Whether to label individual neurons.
        backend :       "auto" | "plotly" | "matplotlib"
                        Which backend to use. If "auto" will use plotly in
                        Jupyter environments and matplotlib elsewhere.

        """
        if not isinstance(subset, type(None)):
            ids = utils.make_iterable(subset)
            if not len(ids):
                raise ValueError('`ids` must not be empty')
        else:
            ids = self._ids

        # Collect spike timings
        x = []
        y = []
        i = 0
        for id in ids:
            pp = self.idx[id]
            x += list(pp.spk_timings)
            y += [i] * len(pp.spk_timings)
            i += 1

        if not x:
            raise ValueError('No spikes detected.')

        if label:
            ld = dict(zip(self._ids, self._labels))
            labels = [ld[i] for i in ids]
        else:
            labels = None

        # Turn into lines
        x = np.vstack((x, x, [None] * len(x))).T.flatten()
        y = np.array(y)
        y = np.vstack((y, y + .9, [None] * len(y))).T.flatten()

        if backend == 'auto':
            if utils.is_jupyter():
                backend = 'plotly'
            else:
                backend = 'matplotlib'

        if backend == 'plotly':
            return _plot_raster_plotly(x, y, ids, fig=ax, labels=labels, **kwargs)
        elif backend == 'matplotlib':
            ax = _plot_raster_mpl(x, y, ids, ax=ax, labels=labels,
                                  stimuli=self._stimuli if stimuli else None,
                                  **kwargs)
            ax.set_xlim(0, neuron.h.t)
            return ax
        else:
            raise ValueError(f'Unknown backend "{backend}"')

    def plot_traces(self, bin_size=100, subset=None, rolling_window=None,
                    group=False, stimuli=True, ax=None, backend='auto', **kwargs):
        """Plot mean firing rate.

        Parameters
        ----------
        bin_size :          int
                            Size [ms] of the bins over which to average spike
                            frequency.
        subset :            list-like
                            Subset of IDs to plot. You can also use this to
                            determine the order of appearance.
        rolling_window :    int
                            Average firing rates over a given rolling window.
        group :             bool | array
                            Set to True to group traces by label, showing mean
                            firing rate and standard error as envelope. Pass
                            an array of labels for each neuron to group by
                            arbitrary labels.
        ax :                matplotlib axis | plotly figure, optional
                            Axis/figure to plot onto.
        stimuli :           bool
                            Whether to plot stimuli.
        backend :           "auto" | "plotly" | "matplotlib"
                            Which backend to use. If "auto" will use plotly in
                            Jupyter environments and matplotlib elsewhere.

        """
        if not isinstance(subset, type(None)):
            ids = utils.make_iterable(subset)
            if not len(ids):
                raise ValueError('`ids` must not be empty')
        else:
            ids = self._ids

        # Collect spike frequencies
        spks = self.get_spike_counts(bin_size=bin_size, subset=subset,
                                     rolling_window=rolling_window)
        freq = spks * 1000 / bin_size

        if self.labels:
            ld = dict(zip(self._ids, self._labels))
            labels = [f'{i} ({ld[i]})' for i in ids]
        else:
            labels = None

        if backend == 'auto':
            if utils.is_jupyter():
                backend = 'plotly'
            else:
                backend = 'matplotlib'

        if isinstance(group, bool) and group:
            sem = freq.groupby(dict(zip(self._ids, self._labels))).sem()
            freq = freq.groupby(dict(zip(self._ids, self._labels))).mean()
            labels = freq.index.values.tolist()
        elif not isinstance(group, bool):
            sem = freq.groupby(group).sem()
            freq = freq.groupby(group).mean()
            labels = freq.index.values.tolist()
        else:
            sem = None

        if backend == 'plotly':
            return _plot_traces_plotly(freq, fig=ax, labels=labels,
                                       stimuli=self._stimuli if stimuli else None,
                                       env=sem, **kwargs)
        elif backend == 'matplotlib':
            return _plot_traces_mpl(freq, ax=ax, env=sem,
                                    stimuli=self._stimuli if stimuli else None,
                                    **kwargs)
        else:
            raise ValueError(f'Unknown backend "{backend}"')

    def set_labels(self, labels):
        """Set labels for neurons.

        Parameters
        ----------
        labels :    dict | list-like
                    If list, must provide a label for every neuron.

        """
        if isinstance(labels, dict):
            for i, n in enumerate(self.neurons):
                n.label = self._labels[i] = labels.get(n.id, n.id)
        elif utils.is_iterable(labels):
            if len(labels) != len(self.neurons):
                raise ValueError(f'Got {len(labels)} labels for {len(self)} neurons.')
            for i, n in enumerate(self.neurons):
                n.label = self._labels[i] = labels[i]
        else:
            raise TypeError(f'`labels` must be dict or list-like, got "{type(labels)}"')

    def run_simulation(self, duration=25 * ms, v_init=-65 * mV):
        """Run the simulation."""

        # This resets the entire model space not just this neuron!
        neuron.h.finitialize(v_init)
        neuron.h.continuerun(duration)

    def get_spike_counts(self, bin_size=50, subset=None, rolling_window=None,
                         group=False):
        """Get matrix of spike counts.

        Parameters
        ----------
        bin_size :          int | None
                            Size [ms] of the bins over which to count spikes.
                            If None, will simply return total counts.
        rolling_window :    int, optional
                            Average spike counts in a rolling window.
        group :             bool
                            If True, will return the spike counts per unique
                            label.

        Returns
        -------
        pd.DataFrame

        """
        if not isinstance(subset, type(None)):
            ids = utils.make_iterable(subset)
        else:
            ids = self._ids

        end_time = neuron.h.t

        if not end_time:
            raise ValueError('Looks like simulation has not yet been run.')

        if not bin_size:
            bin_size = end_time

        bins = np.arange(0, end_time + bin_size, bin_size)
        counts = np.zeros((len(ids), len(bins) - 1))
        # Collect spike counts
        for i, id in enumerate(ids):
            pp = self.idx[id]
            timings = list(pp.spk_timings)
            if timings:
                hist, _ = np.histogram(timings, bins)
                counts[i, :] = hist

        counts = pd.DataFrame(counts, index=ids, columns=bins[1:])

        if group:
            if not self._labels:
                raise ValueError('Unable to group: Network has no labels.')
            counts = counts.groupby(counts.index.map(dict(zip(self.ids, self._labels)))).sum()

        if rolling_window:
            avg = sliding_window_view(counts, rolling_window, axis=1).mean(axis=2)
            counts.iloc[:, :-(rolling_window - 1)] = avg

        return counts


class PointNeuron:
    def __init__(self, id, model, label=None, **props):
        self.process = model()
        self.id = id
        self.label = label
        for p, v in props.items():
            setattr(self.process, p, v)

        self.record_spikes()

    def __str__(self):
        return f'{type(self).__name__}<id={self.id},label={self.label}>'

    def __repr__(self):
        return self.__str__()

    def record_spikes(self, threshold=10):
        """Set up spike recording for this neuron."""
        self.spk_timings = neuron.h.Vector()
        self.sp_det = neuron.h.NetCon(self.process, None)
        self.sp_det.record(self.spk_timings)

    def record_state(self):
        """Set up recording of state for this neuron."""
        self.state = neuron.h.Vector()
        self.state.record(self.process._ref_m)


class NetworkIdIndexer:
    def __init__(self, network):
        self.network = network

    def __getitem__(self, id):
        neurons = self.network._neurons_dict
        if utils.is_iterable(id):
            return [neurons[i] for i in id]
        else:
            return neurons[id]


def _plot_raster_mpl(x, y, ids, ax=None, labels=None, stimuli=None, **kwargs):
    if not ax:
        fig, ax = plt.subplots(figsize=kwargs.pop('figsize', (12, min(20, len(ids)))))

    DEFAULTS = dict(alpha=.9, rasterized=False, lw=1)
    DEFAULTS.update(kwargs)

    ax.plot(x, y, **DEFAULTS)
    ax.set_ylim(-.5, len(ids) + .5)
    ax.set_xlabel('time [ms]')

    if not isinstance(labels, type(None)):
        ax.set_yticks(np.arange(0, len(ids), 1) + .5)
        ax.set_yticklabels(labels)
    else:
        ax.set_yticks([])
        ax.set_yticklabels([])

    if stimuli:
        y = len(ids) + max(1, len(ids) / 100)
        stimuli = np.unique([(s.start, s.stop) for s in stimuli], axis=0)
        for st in stimuli:
            # Skip background noise
            if st[1] >= 999_999_999:
                continue
            ax.plot([st[0], st[1]], [y, y], lw=4,
                    color=kwargs.get('color', (.5, .5, .5)))

        ax.set_ylim(top=y + 1)

    return ax


def _plot_traces_mpl(freq, ax=None, show=True, env=None, stimuli=None, **kwargs):
    if not ax:
        fig, ax = plt.subplots(figsize=kwargs.pop('figsize', (12, 7)))

    if 'color' in kwargs:
        c = kwargs.pop('color')
        if utils.is_iterable(c) and len(c) == freq.shape[0]:
            colors = np.array([mcl.to_rgb(c) for c in c])
            if colors.max() > 1:
                colors /= 255
        else:
            c = mcl.to_rgb(c)
            if max(c) > 1:
                c = (np.array(c) / 255).astype(int)
            colors = [c] * freq.shape[0]
    else:
        colors = sns.color_palette('tab20', freq.shape[0])

    segs = np.zeros((freq.shape[0], freq.shape[1], 2))
    segs[:, :, 0] = freq.columns.values
    segs[:, :, 1] = freq.values

    DEFAULTS = dict(alpha=.9, rasterized=False, lw=1)
    DEFAULTS.update(kwargs)

    lc = LineCollection(segs, **DEFAULTS, colors=colors)
    ax.add_collection(lc)

    if not isinstance(env, type(None)):
        for i, s in enumerate(env.index.values):
            # If no envelope (single neuron)
            if not env.loc[s].any():
                continue

            y1 = freq.loc[s].values + env.loc[s].values
            y2 = freq.loc[s].values - env.loc[s].values
            ax.fill_between(env.columns.values, y1, y2,
                            facecolor=colors[i],
                            alpha=0.5)

    if stimuli:
        y = freq.max().max() + 20
        stimuli = np.unique([(s.start, s.stop) for s in stimuli], axis=0)
        for st in stimuli:
            # Skip background noise
            if st[1] >= 999_999_999:
                continue
            ax.plot([st[0], st[1]], [y, y], lw=4,
                    color=kwargs.get('color', (.5, .5, .5)))

    ax.set_xlabel('time [ms]')
    ax.set_ylabel('spike rate [Hz]')

    ax.autoscale()

    return ax


def _plot_raster_plotly(x, y, ids, fig=None, labels=None, show=True, **kwargs):
    if not fig:
        fig = go.Figure()

    if labels:
        # Turn into one label for each line
        labels = [labels[i] for i in y[::3]]
        ht = np.vstack((labels, labels, [None] * len(labels))).T.flatten().tolist()
    else:
        ht = None

    fig.add_trace(go.Scattergl(x=x,
                               y=y,
                               mode='lines',
                               hovertext=ht,
                               hoverinfo='text' if labels else 'x+y',
                               line=dict(color=kwargs.get('color', 'black'),
                                         width=kwargs.get('width', 2.5))))

    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                      height=kwargs.get('height', 1000),
                      xaxis_title="time [ms]",
                      plot_bgcolor='rgba(0,0,0,0)')

    if show:
        fig.show()

    return fig


def _plot_traces_plotly(freq, fig=None, labels=None, show=True, stimuli=None,
                        env=None, **kwargs):
    if not fig:
        fig = go.Figure()

    if 'color' in kwargs:
        c = kwargs['color']
        if utils.is_iterable(c) and len(c) == freq.shape[0]:
            colors = np.array([mcl.to_rgb(c) for c in c])
            if colors.max() <= 1 and colors.max() != 0:
                colors *= 255
        else:
            c = mcl.to_rgb(c)
            if max(c) <= 1 and max(c) != 0:
                c = (np.array(c) * 255).astype(int)
            colors = [c] * freq.shape[0]
    else:
        colors = sns.color_palette('tab20', freq.shape[0])
        colors = np.array(colors) * 255

    alpha = kwargs.get('alpha', .5)

    x = freq.columns.values
    for i in range(freq.shape[0]):
        if not any(freq.iloc[i].values > 0):
            continue
        color = colors[i]
        color_str = f'rgba({color[0]},{color[1]},{color[2]},{alpha})'
        fill_color_str = f'rgba({color[0]},{color[1]},{color[2]},{.1})'

        fig.add_trace(go.Scattergl(x=x,
                                   y=freq.iloc[i].values,
                                   mode='lines',
                                   hovertext=labels[i] if labels else None,
                                   hoverinfo='text' if labels else 'x+y',
                                   name=labels[i] if labels else freq.index[i],
                                   legendgroup=labels[i] if labels else freq.index[i],
                                   line=dict(color=color_str,
                                             width=kwargs.get('width', 1.5))))

        if not isinstance(env, type(None)):
            y = (freq.iloc[i].values + env.iloc[i].values).tolist()
            y += (freq.iloc[i].values - env.iloc[i].values).tolist()[::-1]
            fig.add_trace(go.Scattergl(x=x.tolist() + x.tolist()[::-1],
                                       y=y,
                                       fill='toself',
                                       fillcolor=fill_color_str,
                                       name=labels[i] if labels else freq.index[i],
                                       showlegend=False,
                                       hoverinfo='skip',
                                       legendgroup=labels[i] if labels else freq.index[i],
                                       line=dict(color="rgba(255,255,255,0)",
                                                 width=kwargs.get('width', 1.5))))

    if stimuli:
        timings = [(st.start, st.stop) for st in stimuli]
        to_plot = np.unique(timings, axis=0, return_index=True)[1]
        y = freq.max().max() + 20
        for i, ix in enumerate(to_plot):
            st = stimuli[ix]
            # Skip background noise
            if st.stop >= 999_999_999:
                continue
            fig.add_trace(go.Scattergl(x=[st.start, st.stop], y=[y+i, y+i],
                                       mode='lines',
                                       line=dict(color='rgb(155,155,155)',
                                                 width=4),
                                       name=st.label,
                                       hovertext=st.label,
                                       hoverinfo='text',
                                       showlegend=False))

    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                      height=kwargs.get('height', 500),
                      xaxis_title="time [ms]",
                      yaxis_title="spike rate [Hz]",
                      yaxis_gridwidth=.25,
                      xaxis_gridwidth=.25,
                      xaxis_showgrid=False,
                      yaxis_showgrid=False,
                      plot_bgcolor='rgba(0,0,0,0)')

    if show:
        fig.show()

    return fig
