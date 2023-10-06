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
"""Module containing functions and classes to build `NEURON` compartment models.

Useful resources
----------------
- http://www.inf.ed.ac.uk/teaching/courses/nc/NClab1.pdf

ToDo
----
- connect neurons
- use neuron ID as GID
- [x] add spike recorder
- [x] make a subplot for each recording type (V, current, spikes)
- consider adding 3d points to more accurately represent the neuron

Examples
--------
Initialize and run a simple model. For debugging/testing only

>>> import navis
>>> import navis.interfaces.neuron as nrn
>>> import neuron

>>> # Set finer time steps
>>> neuron.h.dt = 0.025  # .01 ms

>>> # Set the temperature - how much does this matter?
>>> # Default is 6.3 (from HH model)
>>> # neuron.h.celsius = 24

>>> # This is a DA1 PN from the hemibrain dataset
>>> # It's in 8x8x8 nm voxels so we need to convert to convert
>>> n = navis.example_neurons(1) / 125
>>> n.reroot(n.soma, inplace=True)
>>> navis.smooth_skeleton(n, to_smooth='radius', inplace=True, window=3)

>>> # Get dendritic postsynapses
>>> post = n.connectors[n.connectors.type == 'post']
>>> post = post[post.y >= 250]

>>> # Initialize as a DrosophilaPN which automatically assigns a couple
>>> # properties known from the literature.
>>> cmp = nrn.DrosophilaPN(n, res=10)

>>> # Simulate some synaptic inputs on the first 10 input synapse
>>> cmp.add_synaptic_current(post.node_id.unique()[0:10], max_syn_cond=.1,
                             rev_pot=-10)

>>> # Add voltage recording at the soma and some of the synapses
>>> cmp.add_voltage_record(n.soma, label='soma')
>>> cmp.add_voltage_record(post.node_id.unique()[0:3])

>>> # Let's also check out the synaptic current at one of the synapses
>>> cmp.add_current_record(post.node_id.unique()[0])

>>> # Initialize Run for 200ms
>>> print('Running model')
>>> cmp.run_simulation(200, v_init=-60)
>>> print('Done')

>>> # Plot
>>> cmp.plot_results()

Simulate some presynaptic spikes

>>> cmp = nrn.DrosophilaPN(n, res=1)
>>> cmp.add_voltage_record(n.soma, label='soma')
>>> cmp.add_voltage_record(post.node_id.unique()[0:10])
>>> cmp.add_synaptic_input(post.node_id.unique()[0:10], spike_no=5,
                           spike_int=50, spike_noise=1, syn_tau2=1.1,
                           syn_rev_pot=-10, cn_weight=0.04)
>>> cmp.run_simulation(200, v_init=-60)
>>> cmp.plot_results()

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from numbers import Number

from ... import config, core, utils, graph
from .utils import is_NEURON_object, is_section, is_segment

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


class CompartmentModel:
    """Compartment model representing a single neuron in NEURON.

    Parameters
    ----------
    x :         navis.TreeNeuron
                Neuron to generate model for. Has to be in microns!
    res :       int
                Approximate length [um] of segments. This guarantees that
                no section has any segment that is longer than `res` but for
                small branches (i.e. "sections") the segments might be smaller.
                Lower ``res`` = more detailed simulation.

    """

    def __init__(self, x: 'core.TreeNeuron', res=10):
        """Initialize Neuron."""
        utils.eval_param(x, name='x', allowed_types=(core.TreeNeuron, ))

        # Note that we make a copy to make sure that the data underlying the
        # model will not accidentally be changed
        self.skeleton = x.copy()

        # Max section resolution per segment
        self.res = res

        # Some placeholders
        self._sections = []
        self._stimuli = {}
        self._records = {}
        self._synapses = {}

        # Generate the actual model
        self._validate_skeleton()
        self._generate_sections()

    def __repr__(self):
        s = (f'CompartmentModel<id={self.skeleton.label},'
             f'sections={self.n_sections};'
             f'stimuli={self.n_stimuli};'
             f'records={self.n_records}>'
             )
        return s

    @property
    def label(self):
        """Name/label of the neuron."""
        return f'CompartmentModel[{self.skeleton.label}]'

    @property
    def n_records(self):
        """Number of records (across all types) active on this model."""
        return len([r for t in self.records.values() for r in t])

    @property
    def n_sections(self):
        """Number of sections in this model."""
        return len(self.sections)

    @property
    def n_stimuli(self):
        """Number of stimuli active on this model."""
        return len(self.stimuli)

    @property
    def nodes(self) -> pd.DataFrame:
        """Node table of the skeleton."""
        return self.skeleton.nodes

    @property
    def cm(self) -> float:
        """Membran capacity [micro Farads / cm^2] of all sections."""
        return np.array([s.cm for s in self.sections])

    @cm.setter
    def cm(self, value: float):
        """Membran capacity [micro Farads / cm^2] for all sections."""
        for s in self.sections:
            s.cm = value

    @property
    def Ra(self) -> float:
        """Axial resistance [Ohm * cm] of all sections."""
        return np.array([s.Ra for s in self.sections])

    @Ra.setter
    def Ra(self, value: float):
        """Set axial resistance [Ohm * cm] for all sections."""
        for s in self.sections:
            s.Ra = value

    @property
    def records(self) -> dict:
        """Return mapping of node ID(s) to recordings."""
        return self._records

    @property
    def sections(self) -> np.ndarray:
        """List of sections making up this model."""
        return self._sections

    @property
    def stimuli(self) -> dict:
        """Return mapping of node ID(s) to stimuli."""
        return self._stimuli

    @property
    def synapses(self) -> dict:
        """Return mapping of node ID(s) to synapses."""
        return self._synapses

    @property
    def t(self) -> np.ndarray:
        """The global time. Should be the same for all neurons."""
        return main_t

    def _generate_sections(self):
        """Generate sections from the neuron.

        This will automatically be called at initialization and should not be
        called again.

        """
        # First generate sections
        self._sections = []
        nodes = self.skeleton.nodes.set_index('node_id')
        roots = self.skeleton.root
        bp = self.skeleton.branch_points.node_id.values
        G = self.skeleton.graph
        node2sec = {}
        node2pos = {}
        for i, seg in enumerate(self.skeleton.small_segments):
            # Get child -> parent distances in this segment
            dists = np.array([G.edges[(c, p)]['weight']
                              for c, p in zip(seg[:-1], seg[1:])])

            # Invert the sections
            # That's because in navis sections go from tip -> root (i.e.
            # child -> parent) but in neuron section(0) is the base and
            # section(1) is the tip.
            seg = np.asarray(seg)[::-1]
            dists = dists[::-1]

            # Grab the coordinates and radii
            seg_nodes = nodes.loc[seg]
            locs = seg_nodes[['x', 'y', 'z']].values
            radii = seg_nodes.radius.values

            # Generate section
            sec = neuron.h.Section(name=f'segment_{i}')

            # Set 3D points -> this automatically sets length L
            xvec = neuron.h.Vector(locs[:, 0])
            yvec = neuron.h.Vector(locs[:, 1])
            zvec = neuron.h.Vector(locs[:, 2])
            dvec = neuron.h.Vector(radii * 2)
            neuron.h.pt3dadd(xvec, yvec, zvec, dvec, sec=sec)

            # Set number of segments for this section
            # We also will make sure that each section has an odd
            # number of segments
            sec.nseg = 1 + 2 * int(sec.L / (self.res * 2))
            # Keep track of section
            self.sections.append(sec)

            # While we're at it: for each point (except the root of this
            # section) find the relative position within the section

            # Get normalized positions within this segment
            norm_pos = dists.cumsum() / dists.sum()

            # Update positional dictionaries (required for connecting the
            # segments in the next step)
            node2pos.update(dict(zip(seg[1:], norm_pos)))
            node2sec.update({n: i for n in seg[1:]})

            # If this happens to be the segment with the skeleton's root, keep
            # track of it too
            if seg[0] in roots:
                node2pos[seg[0]] = 0
                node2sec[seg[0]] = i

        self._sections = np.array(self.sections)
        self.skeleton.nodes['sec_ix'] = self.skeleton.nodes.node_id.map(node2sec)
        self.skeleton.nodes['sec_pos'] = self.skeleton.nodes.node_id.map(node2pos)

        # Need to grab nodes again after adding `sec_ix` and `sec_pos`
        nodes = self.skeleton.nodes.set_index('node_id')

        # Connect segments
        for i, seg in enumerate(self.skeleton.small_segments):
            # Root is special in that it only needs to be connected if it's also
            # a branch point
            if seg[-1] in roots:
                # Skip if root is not a branch point
                if seg[-1] not in bp:
                    continue
                # If root is also a branch point, it will be part of more than
                # one section but in the positional dicts we will have kept track
                # of only one of them. That's the one we pick as base segment
                if node2sec[seg[-1]] == i:
                    continue

            parent = nodes.loc[seg[-1]]
            parent_sec = self.sections[parent.sec_ix]
            self.sections[i].connect(parent_sec(1))

    def _validate_skeleton(self):
        """Validate skeleton."""
        if self.skeleton.units and not self.skeleton.units.dimensionless:
            not_um = self.skeleton.units.units != config.ureg.Unit('um')
            not_microns = self.skeleton.units.units != config.ureg.Unit('microns')
            if not_um and not_microns:
                logger.warning('Model expects coordinates in microns but '
                               f'neuron has units "{self.skeleton.units}"!')

        if len(self.skeleton.root) > 1:
            logger.warning('Neuron has multiple roots and hence consists of '
                           'multiple disconnected fragments!')

        if 'radius' not in self.skeleton.nodes.columns:
            raise ValueError('Neuron node table must have `radius` column')

        if np.any(self.skeleton.nodes.radius.values <= 0):
            raise ValueError('Neuron node table contains radii <= 0.')

    def add_synaptic_input(self, where, start=5 * ms,
                           spike_no=1, spike_int=10 * ms, spike_noise=0,
                           syn_tau1=.1 * ms, syn_tau2=10 * ms, syn_rev_pot=0,
                           cn_thresh=10, cn_delay=1 * ms, cn_weight=0.05):
        """Add synaptic input to model.

        This uses the Exp2Syn synapse. All targets in `where` are triggered
        by the same NetStim - i.e. they will all receive their spike(s) at the
        same time.

        Parameters
        ----------
        where :         int | list of int
                        Node IDs at which to simulate synaptic input.

        Properties for presynaptic spikes:

        start :         int
                        Onset [ms] of first spike from beginning of simulation.
        spike_no :      int
                        Number of presynaptic spikes to produce.
        spike_int :     int
                        Interval [ms] between consecutive spikes.
        spike_noise :   float [0-1]
                        Fractional randomness in spike timing.

        Synapse properties:

        syn_tau1 :      int
                        Rise time constant [ms].
        syn_tau2 :      int
                        Decay time constant [ms].
        syn_rev_pot :   int
                        Reversal potential (e) [mV].

        Connection properties:

        cn_thresh :     int
                        Presynaptic membrane potential [mV] at which synaptic
                        event is triggered.
        cn_delay :      int
                        Delay [ms] between presynaptic trigger and postsynaptic
                        event.
        cn_weight :     float
                        Weight variable. This bundles a couple of synaptic
                        properties such as e.g. how much transmitter is released
                        or binding affinity at postsynaptic receptors.

        """
        where = utils.make_iterable(where)

        # Make a new stimulator
        stim = neuron.h.NetStim()
        stim.number = spike_no
        stim.start = start
        stim.noise = spike_noise
        stim.interval = spike_int

        # Connect
        self.connect(stim, where, syn_tau1=syn_tau1, syn_tau2=syn_tau2,
                     syn_rev_pot=syn_rev_pot, cn_thresh=cn_thresh,
                     cn_delay=cn_delay, cn_weight=cn_weight)

    def inject_current_pulse(self, where, start=5,
                             duration=1, current=0.1):
        """Add current injection (IClamp) stimulation to model.

        Parameters
        ----------
        where :     int | list of int
                    Node ID(s) at which to stimulate.
        start :     int
                    Onset (delay) [ms] from beginning of simulation.
        duration :  int
                    Duration (dur) [ms] of injection.
        current :   float
                    Amount (i) [nA] of injected current.

        """
        self._add_stimulus('IClamp', where=where, delay=start,
                           dur=duration, amp=current)

    def add_synaptic_current(self, where, start=5, tau=0.1, rev_pot=0,
                             max_syn_cond=0.1):
        """Add synaptic current(s) (AlphaSynapse) to model.

        Parameters
        ----------
        where :         int | list of int
                        Node ID(s) at which to stimulate.
        start :         int
                        Onset [ms] from beginning of simulation.
        tau :           int
                        Decay time constant [ms].
        rev_pot :       int
                        Reverse potential (e) [mV].
        max_syn_cond :  float
                        Max synaptic conductance (gmax) [uS].

        """
        self._add_stimulus('AlphaSynapse', where=where, onset=start,
                           tau=tau, e=rev_pot, gmax=max_syn_cond)

    def _add_stimulus(self, stimulus, where, **kwargs):
        """Add generic stimulus."""
        if not callable(stimulus):
            stimulus = getattr(neuron.h, stimulus)

        where = utils.make_iterable(where)

        nodes = self.nodes.set_index('node_id')
        for node in nodes.loc[where].itertuples():
            sec = self.sections[node.sec_ix](node.sec_pos)
            stim = stimulus(sec)

            for k, v in kwargs.items():
                setattr(stim, k, v)

            self.stimuli[node.Index] = self.stimuli.get(node.Index, []) + [stim]

    def add_voltage_record(self, where, label=None):
        """Add voltage recording to model.

        Parameters
        ----------
        where :     int | list of int
                    Node ID(s) at which to record.
        label :     str, optional
                    If label is given, this recording will be added as
                    ``self.records['v'][label]`` else  ``self.records['v'][node_id]``.

        """
        self._add_record(where, what='v', label=label)

    def add_current_record(self, where, label=None):
        """Add current recording to model.

        This only works if nodes map to sections that have point processes.

        Parameters
        ----------
        where :     int | list of int
                    Node ID(s) at which to record.
        label :     str, optional
                    If label is given, this recording will be added as
                    ``self.records['i'][label]`` else  ``self.records['i'][node_id]``.

        """
        nodes = utils.make_iterable(where)

        # Map nodes to point processes
        secs = self.get_node_segment(nodes)
        where = []
        for n, sec in zip(nodes, secs):
            pp = sec.point_processes()
            if not pp:
                raise TypeError(f'Section for node {n} has no point process '
                                '- unable to add current record')
            elif len(pp) > 1:
                logger.warning(f'Section for node {n} has more than on point '
                               'process. Recording current at first.')
                pp = pp[:1]
            where += pp

        self._add_record(where, what='i', label=label)

    def add_spike_detector(self, where, threshold=20, label=None):
        """Add a spike detector at given node(s).

        Parameters
        ----------
        where :     int | list of int
                    Node ID(s) at which to record.
        threshold : float
                    Threshold in mV for a spike to be counted.
        label :     str, optional
                    If label is given, this recording will be added as
                    ``self.records[label]`` else  ``self.records[node_id]``.

        """
        where = utils.make_iterable(where)

        self.records['spikes'] = self.records.get('spikes', {})
        self._spike_det = getattr(self, '_spike_det', [])
        segments = self.get_node_segment(where)
        sections = self.get_node_section(where)
        for n, sec, seg in zip(where, sections, segments):
            # Generate a NetCon object that has no target
            sp_det = neuron.h.NetCon(seg._ref_v, None, sec=sec)

            # Set threshold
            if threshold:
                sp_det.threshold = threshold

            # Keeping track of this to save it from garbage collector
            self._spike_det.append(sp_det)

            # Create a vector for the spike timings
            vec = neuron.h.Vector()
            # Tell the NetCon object to record into that vector
            sp_det.record(vec)

            if label:
                self.records['spikes'][label] = vec
            else:
                self.records['spikes'][n] = vec

    def _add_record(self, where, what, label=None):
        """Add a recording to given node.

        Parameters
        ----------
        where :     int | list of int | point process | section
                    Node ID(s) (or a section) at which to record.
        what :      str
                    What to record. Can be e.g. `v` or `_ref_v` for Voltage.
        label :     str, optional
                    If label is given, this recording will be added as
                    ``self.records[label]`` else  ``self.records[node_id]``.

        """
        where = utils.make_iterable(where)

        if not isinstance(what, str):
            raise TypeError(f'Required str e.g. "v", got {type(what)}')

        if not what.startswith('_ref_'):
            what = f'_ref_{what}'

        rec_type = what.split('_')[-1]
        if rec_type not in self.records:
            self.records[rec_type] = {}

        # # Get node segments only for nodes
        is_node = ~np.array([is_NEURON_object(w) for w in where])
        node_segs = np.zeros(len(where), dtype=object)
        node_segs[is_node] = self.get_node_segment(where[is_node])

        for i, w in enumerate(where):
            # If this is a neuron object (e.g. segment, section or point
            # process) we assume this does not need mapping
            if is_NEURON_object(w):
                seg = w
            else:
                seg = node_segs[i]

            rec = neuron.h.Vector().record(getattr(seg, what))

            if label:
                self.records[rec_type][label] = rec
            else:
                self.records[rec_type][w] = rec

    def connect(self, pre, where, syn_tau1=.1 * ms, syn_tau2=10 * ms,
                syn_rev_pot=0, cn_thresh=10, cn_delay=1 * ms, cn_weight=0):
        """Connect object to model.

        This uses the Exp2Syn synapse and treats `pre` as the presynaptic
        object.

        Parameters
        ----------
        pre :           NetStim | section
                        The presynaptic object to connect to this neuron.
        where :         int | list of int
                        Node IDs at which to simulate synaptic input.

        Synapse properties:

        syn_tau1 :      int
                        Rise time constant [ms].
        syn_tau2 :      int
                        Decay time constant [ms].
        syn_rev_pot :   int
                        Reversal potential (e) [mV].

        Connection properties:

        cn_thresh :     int
                        Presynaptic membrane potential [mV] at which synaptic
                        event is triggered.
        cn_delay :      int
                        Delay [ms] between presynaptic trigger and postsynaptic
                        event.
        cn_weight :     int
                        Weight variable. This bundles a couple of synaptic
                        properties such as e.g. how much transmitter is released
                        or binding affinity at postsynaptic receptors.

        """
        where = utils.make_iterable(where)

        if not is_NEURON_object(pre):
            raise ValueError(f'Expected NEURON object, got {type(pre)}')

        # Turn section into segment
        if isinstance(pre, neuron.nrn.Section):
            pre = pre()

        # Go over the nodes
        nodes = self.nodes.set_index('node_id')
        for node in nodes.loc[where].itertuples():
            # Generate synapses for the nodes in question
            # Note that we are not reusing existing synapses
            # in case the properties are different
            sec = self.sections[node.sec_ix](node.sec_pos)
            syn = neuron.h.Exp2Syn(sec)
            syn.tau1 = syn_tau1
            syn.tau2 = syn_tau2
            syn.e = syn_rev_pot

            self.synapses[node.Index] = self.synapses.get(node.Index, []) + [syn]

            # Connect spike stimulus and synapse
            if isinstance(pre, neuron.nrn.Segment):
                nc = neuron.h.NetCon(pre._ref_v, syn, sec=pre.sec)
            else:
                nc = neuron.h.NetCon(pre, syn)

            # Set connection parameters
            nc.threshold = cn_thresh
            nc.delay = cn_delay
            nc.weight[0] = cn_weight

            self.stimuli[node.Index] = self.stimuli.get(node.Index, []) + [nc, pre]

    def clear_records(self):
        """Clear records."""
        self._records = {}

    def clear_stimuli(self):
        """Clear stimuli."""
        self._stimuli = {}

    def clear_synapses(self):
        """Clear synapses."""
        self._synapses = {}

    def clear(self):
        """Attempt to remove model from NEURON space.

        This is not guaranteed to work. Check `neuron.h.topology()` to inspect.

        """
        # Basically we have to bring the reference count to zero
        self.clear_records()
        self.clear_stimuli()
        self.clear_synapses()
        for s in self._sections:
            del s
        self._sections = []

    def get_node_section(self, node_ids):
        """Return section(s) for given node(s).

        Parameters
        ----------
        node_ids :  int | list of int
                    Node IDs.

        Returns
        -------
        section(s) :    segment or list of segments
                        Depends on input.

        """
        nodes = self.nodes.set_index('node_id')
        if not utils.is_iterable(node_ids):
            n = nodes.loc[node_ids]
            return self.sections[n.sec_ix]
        else:
            segs = []
            for node in nodes.loc[node_ids].itertuples():
                segs.append(self.sections[node.sec_ix])
            return segs

    def get_node_segment(self, node_ids):
        """Return segment(s) for given node(s).

        Parameters
        ----------
        node_ids :  int | list of int
                    Node IDs.

        Returns
        -------
        segment(s) :    segment or list of segments
                        Depends on input.

        """
        nodes = self.nodes.set_index('node_id')
        if not utils.is_iterable(node_ids):
            n = nodes.loc[node_ids]
            return self.sections[n.sec_ix](n.sec_pos)
        else:
            segs = []
            for node in nodes.loc[node_ids].itertuples():
                segs.append(self.sections[node.sec_ix](node.sec_pos))
            return segs

    def insert(self, mechanism, subset=None, **kwargs):
        """Insert biophysical mechanism for model.

        Parameters
        ----------
        mechanism : str
                    Mechanism to insert - e.g. "hh" for Hodgkin-Huxley kinetics.
        subset :    list of sections | list of int
                    Sections (or indices thereof) to set mechanism for.
                    If ``None`` will add mechanism to all sections.
        **kwargs
                    Use to set properties for mechanism.

        """
        if isinstance(subset, type(None)):
            sections = self.sections
        else:
            subset = utils.make_iterable(subset)

            if all([is_section(s) for s in subset]):
                sections = subset
            elif all([isinstance(s, Number) for s in subset]):
                sections = self.sections[subset]
            else:
                raise TypeError('`subset` must be None, a list of sections or '
                                'a list of section indices')

        for sec in np.unique(sections):
            _ = sec.insert(mechanism)
            for seg in sec:
                mech = getattr(seg, mechanism)
                for k, v in kwargs.items():
                    setattr(mech, k, v)

    def uninsert(self, mechanism, subset=None):
        """Remove biophysical mechanism from model.

        Parameters
        ----------
        mechanism : str
                    Mechanism to remove - e.g. "hh" for Hodgkin-Huxley kinetics.
        subset :    list of sections | list of int
                    Sections (or indices thereof) to set mechanism for.
                    If ``None`` will add mechanism to all sections.

        """
        if isinstance(subset, type(None)):
            sections = self.sections
        else:
            subset = utils.make_iterable(subset)

            if all([is_section(s) for s in subset]):
                sections = subset
            elif all([isinstance(s, Number) for s in subset]):
                sections = self.sections[subset]
            else:
                raise TypeError('`subset` must be None, a list of sections or '
                                'a list of section indices')

        for sec in np.unique(sections):
            if hasattr(sec, mechanism):
                _ = sec.uninsert(mechanism)

    def plot_structure(self):
        """Visualize structure in 3D using matplotlib."""
        _ = neuron.h.PlotShape().plot(plt)

    def run_simulation(self, duration=25 * ms, v_init=-65 * mV):
        """Run the simulation."""
        # Add recording of time
        global main_t
        main_t = neuron.h.Vector().record(neuron.h._ref_t)

        # This resets the entire model space not just this neuron!
        neuron.h.finitialize(v_init)
        neuron.h.continuerun(duration)

    def plot_results(self, axes=None):
        """Plot results.

        Parameters
        ----------
        axes :      matplotlib axes
                    Axes to plot onto. Must have one ax for each recording
                    type (mV, spike count, etc) in `self.records`.

        Returns
        -------
        axes

        """
        if isinstance(self.t, type(None)) or not len(self.t):
            logger.warning('Looks like the simulation has not yet been run.')
            return
        if not self.records:
            logger.warning('Nothing to plot: no recordings found.')
            return

        if not axes:
            fig, axes = plt.subplots(len(self.records), sharex=True)

        # Make sure that even a single ax is a list
        if not isinstance(axes, (np.ndarray, list)):
            axes = [axes] * len(self.records)

        for t, ax in zip(self.records, axes):
            for i, (k, v) in enumerate(self.records[t].items()):
                if not len(v):
                    continue
                v = v.as_numpy()
                # For spikes the vector contains the times
                if t == 'spikes':
                    # Calculate spike rate
                    bins = np.linspace(0, max(self.t), 10)
                    hist, _ = np.histogram(v, bins=bins)
                    width = bins[1] - bins[0]
                    rate = hist * (1000 / width)
                    ax.plot(bins[:-1] + (width / 2), rate, label=k)

                    ax.scatter(v, [-i] * len(v), marker='|', s=100)
                else:
                    ax.plot(self.t, v, label=k)

            ax.set_xlabel('time [ms]')
            ax.set_ylabel(f'{t}')

            ax.legend()
        return axes


class DrosophilaPN(CompartmentModel):
    """Compartment model of an olfactory projection neuron in Drosophila.

    This is a ``CompartmentModel`` that uses passive membrane properties
    from Tobin et al. (2017) as presets:

    - specific axial resistivity (``Ra``) of 266.1 Ohm / cm
    - specific membrane capacitance (``cm``) of 0.8 mF / cm**2
    - specific leakage conductance (``g``) of 1/Rm
    - Rm = specific membran resistance of 20800 Ohm cm**2
    - leakage reverse potential of -60 mV

    Parameters
    ----------
    x :         navis.TreeNeuron
                Neuron to generate model for. Has to be in microns!
    res :       int
                Approximate length [um] of segments. This guarantees that
                no section has any segment that is longer than `res` but for
                small branches (i.e. "sections") the segments might be smaller.
                Lower ``res`` = more detailed simulation.

    """

    def __init__(self, x, res=10):
        super().__init__(x, res=res)

        self.Ra = 266.1  # specific axial resistivity in Ohm cm
        self.cm = 0.8    # specific membrane capacitance in mF / cm**2

        # Add passive membran properties
        self.insert('pas',
                    g=1/20800,  # specific leakage conductance = 1/Rm; Rm = specific membran resistance in Ohm cm**2
                    e=-60,      # leakage reverse potential
                    )
