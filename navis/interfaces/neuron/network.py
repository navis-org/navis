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
import numpy as np
import pandas as pd


import plotly.graph_objects as go

from collections import namedtuple

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
logger = config.logger

__all__ = []

# It looks like there can only ever be one reference to the time
# If we have multiple models, we will each reference them to this variable
main_t = None

Stimulus = namedtuple('Stimulus', ['start', 'stop', 'frequency', 'randomness',
                                   'neurons', 'netstim', 'netcon'])


class PointNetwork:
    """A Network in which all neurons are represented as LIF point processes."""

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
        return self._edges

    @property
    def neurons(self):
        return self._neurons

    @property
    def ids(self):
        return self._ids

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
                        Name of the column with the weights.
        **props
                        Additional parameters used when initializing the neurons.
                        Depends on which model type you are using: e.g. ``refrac``
                        and ``tau`` for "IntFire1".

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
        """Add neurons to network."""
        assert model in ["IntFire1", "IntFire2", "IntFire4"]
        model = getattr(neuron.h, model)

        ids = utils.make_iterable(ids)

        if labels:
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
                    raise ValueError(f'Neuron with id {i} already exists.')

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
                     randomness=.5, independent=True):
        """Add stimulus to given neurons.

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
                        match length of `ids`.
        randomness :    float [0-1]
                        Randomness of spike timings.
        independent :   bool
                        If True (default), each neuron will get its own
                        independent stimulus.

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

        if duration <= 0:
            raise ValueError(f'Duration is greater than zero.')

        if not independent:
            ns = neuron.h.NetStim()
            ns.interval = frequency[0]
            ns.noise = randomness
            ns.number = int(duration / 1000 * frequency[0])
            ns.start = start

        for i, f in zip(ids, frequency):
            interval = 1000 / f
            proc = self.idx[i].process

            if independent:
                ns = neuron.h.NetStim()
                ns.interval = interval
                ns.noise = randomness
                ns.number = int(duration / 1000 * f)
                ns.start = start

            nc = neuron.h.NetCon(ns, proc)
            nc.weight[0] = 1
            nc.delay = 0

            self._stimuli.append(Stimulus(start, stop, f, randomness, i, ns, nc))

    def connect(self, source, target, weight, delay=5):
        """Connect two neurons."""
        # Get the point processes corresponding to source and target
        pre = self.idx[source].process
        post = self.idx[target].process

        # Connect
        nc = neuron.h.NetCon(pre, post)
        nc.weight[0] = weight
        nc.delay = delay

        # Keep track
        self._edges.append([source, target, weight, nc])

    def plot_raster(self, subset=None, groups=None, ax=None, label=False,
                    backend='auto', **kwargs):
        """Raster plot of spike timings."""
        if not isinstance(subset, type(None)):
            ids = utils.make_iterable(subset)
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
            return _plot_raster_mpl(x, y, ids, ax=ax, labels=labels, **kwargs)
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

    def record_spikes(self):
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


def _plot_raster_mpl(x, y, ids, ax=None, labels=None, **kwargs):
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
                               line=dict(color='black',
                                         width=kwargs.get('width', 2.5))))

    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                      height=kwargs.get('height', 1000),
                      xaxis_title="time [ms]",
                      plot_bgcolor='rgba(0,0,0,0)')

    if show:
        fig.show()

    return fig
