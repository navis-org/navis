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

"""
A collection of tools to interace with Neuron-related R libraries (e.g. nat,
nblast, elmr).

Notes
-----
See https://github.com/jefferis

"""

import os
import sys
import time

from datetime import datetime
from colorsys import hsv_to_rgb
from typing import Union, Optional, List
from typing_extensions import Literal

import pandas as pd
import numpy as np

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, numpy2ri

from .. import cluster as pyclust
from .. import core, plotting, config, utils

try:
    cl = robjects.r('class')
    names = robjects.r('names')
    attributes = robjects.r('attributes')
except BaseException:
    pass

# Set up logging
logger = config.logger

try:
    nat = importr('nat')
    r_nblast = importr('nat.nblast')
    nat_templatebrains = importr('nat.templatebrains')
    nat_flybrains = importr('nat.flybrains')
    # Even if not used, these packages are important e.g. for template brains!
    flycircuit = importr('flycircuit')
    elmr = importr('elmr')
except BaseException:
    logger.error(
        'R library "nat" not found! Please install from within R.')

__all__ = sorted(['neuron2r', 'neuron2py', 'init_rcatmaid', 'dotprops2py',
                  'data2py', 'NBLASTresults', 'nblast', 'nblast_allbyall',
                  'get_neuropil', 'xform_brain', 'mirror_brain'])

# Do not use this! It will convert stuff in unexpected ways. E.g.
# dps = robjects.r(f'read.neuronlistfh("{datadir}/dpscanon.rds")')
# will turn out as just an array

# numpy2ri.activate()
# pandas2ri.activate()


def init_rcatmaid(**kwargs):
    """ Initialize the R Catmaid package.

    R package by Greg Jefferis: https://github.com/jefferis/rcatmaid

    Parameters
    ----------
    remote_instance :   CATMAID instance
                        From pymaid.CatmaidInstance(). This is used to
                        extract credentials. Overrides other credentials
                        provided!
    server :            str, optional
                        Use this to set server URL if no remote_instance is
                        provided
    authname :          str, optional
                        Use this to set http user if no remote_instance is
                        provided
    authpassword :      str, optional
                        Use this to set http password if no remote_instance
                        is provided
    authtoken :         str, optional
                        Use this to set user token if no remote_instance is
                        provided

    Returns
    -------
    catmaid :           R library
                        R object representing the rcatmaid library

    Examples
    --------
    >>> from navis.interfaces import r
    >>> rcatmaid = r.init_rcatmaid(server='https://your.catmaid.server',
    ...                            authname='http_user',
    ...                            authpassword='http_pw',
    ...                            authtoken='Your token')
    >>> # You can now use rcatmaid functions. For example:
    >>> neuron = rcatmaid.read_neurons_catmaid(16)
    >>> neuron
    R object with classes: ('neuronlist', 'list') mapped to:
    [ListSexpVector]
    16: <class 'rpy2.rinterface.ListSexpVector'>
    <rpy2.rinterface.ListSexpVector object at 0x123d46708> [RTYPES.VECSXP]
    """

    remote_instance = kwargs.get('remote_instance', None)
    server = kwargs.get('server', None)
    authname = kwargs.get('authname', None)
    authpassword = kwargs.get('authpassword', None)
    authtoken = kwargs.get('authtoken', None)

    if remote_instance is None:
        if 'remote_instance' in sys.modules:
            remote_instance = sys.modules['remote_instance']
        elif 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']

    if remote_instance:
        server = remote_instance.server
        authname = remote_instance.authname
        authpassword = remote_instance.authpassword
        authtoken = remote_instance.authtoken
    elif not remote_instance and None in (server, authname, authpassword, authtoken):
        logger.error('Unable to initialize. Missing credentials: %s' % ''.join(
            [n for n in ['server', 'authname', 'authpassword', 'authtoken'] if n not in kwargs]))
        return None

    # Import R Catmaid
    try:
        catmaid = importr('catmaid')
    except BaseException:
        logger.error(
            'RCatmaid not found. Please install before proceeding.')
        return None

    # Use remote_instance's credentials
    catmaid.server = server
    catmaid.authname = authname
    catmaid.authpassword = authpassword
    catmaid.token = authtoken

    # Create the connection
    con = catmaid.catmaid_connection(server=catmaid.server,
                                     authname=catmaid.authname,
                                     authpassword=catmaid.authpassword,
                                     token=catmaid.token)

    # Login
    catmaid.catmaid_login(con)

    logger.info('Rcatmaid successfully initiated.')

    return catmaid


def data2py(data, **kwargs):
    """ Takes data object from rcatmaid (e.g. ``catmaidneuron`` from
    ``read.neuron.catmaid``) and converts into Python Data.

    Notes
    -----
    (1) Most R data comes as list (even if only 1 entry). This is preserved.
    (2) R lists with headers are converted to dictionaries
    (3) R DataFrames are converted to Pandas DataFrames
    (4) R nblast results are converted to Pandas DataFrames but only the top
        100 hits for which we have reverse scores!

    Parameters
    ----------
    data
        Any kind of R data. Can be nested (e.g. list of lists)!

    Returns
    -------
    Converted data or 'Not converted' if conversion failed.
    """

    if 'neuronlistfh' in cl(data):
        logger.error('On-demand neuronlist found. Conversion cancelled to '
                     'prevent loading large datasets in memory. Please use '
                     'rmaid.dotprops2py() and its "subset" parameter.')
        return None
    elif 'neuronlist' in cl(data):
        if 'neuron' in cl(data):
            return neuron2py(data)
        elif 'dotprops' in cl(data):
            return dotprops2py(data)
        else:
            logger.debug(f'Unable to convert R data of type "{cl(data)}"')
            return 'Not converted'
    elif 'neuron' in cl(data):
        return neuron2py(data)
    elif 'dotprops' in cl(data):
        return dotprops2py(data)
    elif cl(data)[0] == 'integer':
        if not names(data):
            return [int(n) for n in data]
        else:
            return {n: int(data[i]) for i, n in enumerate(names(data))}
    elif cl(data)[0] == 'character':
        if not names(data):
            return [str(n) for n in data]
        else:
            return {n: str(data[i]) for i, n in enumerate(names(data))}
    elif cl(data)[0] == 'numeric':
        if not names(data):
            return [float(n) for n in data]
        else:
            return {n: float(data[i]) for i, n in enumerate(names(data))}
    elif cl(data)[0] == 'data.frame':
        return pandas2ri.ri2py(data)
    elif cl(data)[0] == 'matrix':
        mat = np.array(data)
        df = pd.DataFrame(data=mat)
        if data.names:
            if data.names[1] != robjects.r('NULL'):
                df.columns = data.names[1]
            if data.names[0] != robjects.r('NULL'):
                df.index = data.names[0]
        return df
    elif 'list' in cl(data):
        # If this is just a list, return a list
        if not names(data):
            return [data2py(n) for n in data]
        # If this list has headers, return as dictionary
        else:
            return {n: data2py(data[i]) for i, n in enumerate(names(data))}
    elif cl(data)[0] == 'NULL':
        return None
    elif 'nblastfafb' in cl(data):
        fw_scores = {n: data[0][i] for i, n in enumerate(names(data[0]))}
        rev_scores = {n: data[1][i] for i, n in enumerate(names(data[1]))}
        mu_scores = {n: (fw_scores[n] + rev_scores[n]) / 2 for n in rev_scores}

        df = pd.DataFrame([[n, fw_scores[n], rev_scores[n], mu_scores[n]]
                           for n in rev_scores],
                          columns=['gene_name', 'forward_score',
                                   'reverse_score', 'mu_score']
                          )

        logger.info('Returning only nblast results. Neuron object is stored '
                    'in your original_data[2].')
        return df
    else:
        logger.debug(f'Unable to convert R data of type "{cl(data)}"')
        return 'Not converted'


def neuron2py(x,
              unit_conversion: Union[bool, int, float] = False,
              add_attributes: bool = None
              ) -> 'core.NeuronObject':
    """ Converts an rcatmaid ``neuron`` or ``neuronlist`` object to
    :class:`~navs.TreeNeuron` or :class:`~navis.NeuronList`, respectively.

    Parameters
    ----------
    x :                 R neuron | R neuronlist
                        Neuron to convert to Python
    unit_conversion :   bool | int | float, optional
                        If True will convert units by given factor.
    add_attributes :    None | str | iterable
                        Give additional attributes that should be carried over
                        to TreeNeuron. If ``None``, will only use minimal
                        information to create TreeNeuron. Must match R
                        ``names`` exactly - see examples.

    Returns
    -------
    TreeNeuron/NeuronList
    """

    if 'rpy2' not in str(type(x)):
        raise TypeError(f'This does not look like R object: "{type(x)}"')

    if cl(x)[0] == 'neuronlist':
        nl = pd.DataFrame(data=[[data2py(e) for e in n] for n in x],
                          columns=list(x[0].names))
    elif cl(x)[0] == 'neuron':
        nl = pd.DataFrame(data=[[e for e in x]],
                          columns=x.names)
        nl = nl.applymap(data2py)
    else:
        raise TypeError(f'Must be neuron or neuronlist, got "{cl(x)[0]}"')

    # Raise Error if this is a dotprops:
    if all([c in nl.columns for c in ['points', 'alpha', 'vect']]):
        raise TypeError('This looks to be dotprops. Try using dotprops2py instead.')

    add_attributes = utils.make_iterable(add_attributes)

    # Now that we (should) have all data in Python, we need to extract
    # information relevant to create a `navis.TreeNeuron`
    neurons = []
    for r in nl.itertuples():
        add_data = {}
        if any(add_attributes):
            add_data.update({getattr(r, at, None) for at in add_attributes})

        # Need to convert column titles for R neuron
        new_cols = [c.lower() for c in r.d.columns]
        for k, v in {'pointno': 'node_id', 'w': 'radius', 'parent': 'parent_id'}.items():
            new_cols = [c if c != k else v for c in new_cols]
        r.d.columns = new_cols

        # Set parent to -1 if negative
        r.d.loc[r.d.parent_id < 0, 'parent_id'] = -1

        neurons.append(core.TreeNeuron(r.d, **add_data))

    if isinstance(unit_conversion, (float, int)):
        for n in neurons:
            n *= unit_conversion

    if len(neurons) == 1:
        return neurons[0]

    return core.NeuronList(neurons)


def neuron2r(x: 'core.NeuronObject',
             unit_conversion: Union[bool, int, float] = None,
             add_metadata: bool = False):
    """ Converts a neuron or neuronlist to the corresponding
    neuron/neuronlist object in R.

    Parameters
    ----------
    x :                 TreeNeuron | NeuronList
    unit_conversion :   bool | int | float, optional
                        If not ``False`` will convert units by given factor.
    add_metadata :      bool, optional
                        If False, will use minimal data necessary to construct
                        the R neuron. If True, will add additional data
                        associated with TreeNeuron - this could impact
                        R functions. **Currently not functional!**

    Returns
    -------
    R neuron
        Either R neuron or neuronlist depending on input.
    """

    if isinstance(x, core.NeuronList):
        """
        The way neuronlist are constructed is a bit more complicated:
        They are essentially named lists {'neuronA': neuronobject, ...} BUT
        they also contain a dataframe that as attribute ( attr('df') = df )
        This dataframe looks like this

                pid   skid     name
        skid1
        skid2

        In rpy2, attributes are assigned using the .slots['df'] function.
        """

        nlist = {str(n.id): neuron2r(n, unit_conversion=unit_conversion) for
                 n in config.tqdm(x, desc='Conv.',
                                  leave=config.pbar_leave,
                                  disable=config.pbar_hide)}

        nlist = robjects.ListVector(nlist)
        nlist.rownames = [str(n.id) for n in x]

        nlist.rclass = robjects.r('c("neuronlist","list")')

        return nlist

    elif isinstance(x, core.TreeNeuron):
        # Convert units if applicable
        if isinstance(unit_conversion, (float, int)) and \
           not isinstance(unit_conversion, bool):
            x = x * unit_conversion

        # Prepare list of parents -> root node's parent "None" has to be
        # replaced with -1
        parents = np.array(x.nodes.parent_id.values)
        # should technically be robjects.r('-1L')
        parents[parents == None] = -1 # DO NOT turn this into "parents is None"!

        swc = robjects.DataFrame({'PointNo': robjects.IntVector(x.nodes.node_id.tolist()),
                                  'Label': robjects.IntVector([0] * x.nodes.shape[0]),
                                  'X': robjects.IntVector(x.nodes.x.tolist()),
                                  'Y': robjects.IntVector(x.nodes.y.tolist()),
                                  'Z': robjects.IntVector(x.nodes.z.tolist()),
                                  'W': robjects.FloatVector([w * 2 for w in x.nodes.radius.tolist()]),
                                  'Parent': robjects.IntVector(parents)
                                  })

        soma = utils.make_non_iterable(x.soma)
        soma_id = int(soma) if not isinstance(soma, type(None)) else robjects.r('NULL')

        meta = {}
        if x.has_connectors:
            meta['connectors'] = robjects.DataFrame({'node_id': robjects.IntVector(x.connectors.node_id.tolist()),
                                                     'connector_id': robjects.IntVector(x.connectors.connector_id.tolist()),
                                                     'prepost': robjects.IntVector(x.connectors.relation.tolist()),
                                                     'x': robjects.IntVector(x.connectors.x.tolist()),
                                                     'y': robjects.IntVector(x.connectors.y.tolist()),
                                                     'z': robjects.IntVector(x.connectors.z.tolist())
                                                     })

        # Generate nat neuron - will reroot to soma (I think)
        return nat.as_neuron(swc, origin=soma_id, **meta)
    else:
        raise TypeError(f'Unable to convert data of type "{type(x)}" into R neuron.')


def dotprops2py(dp,
                subset: Optional[Union[List[str], List[int]]] = None
                ) -> 'core.Dotprops':
    """ Converts dotprops into pandas DataFrame.

    Parameters
    ----------
    dp :        dotprops neuronlist | neuronlistfh
                Dotprops object to convert.
    subset :    list of str | list of indices, optional
                Neuron names or indices.

    Returns
    -------
    core.Dotprops
        Subclass of pandas DataFrame. Contains dotprops.
        Can be passed to `plotting.plot3d(dotprops)`
    """

    # Check if list is on demand
    if 'neuronlistfh' in cl(dp) and not subset:
        dp = dp.rx(robjects.IntVector([i + 1 for i in range(len(dp))]))
    elif subset:
        indices = [i for i in subset if isinstance(
            i, int)] + [dp.names.index(n) + 1 for n in subset if isinstance(n, str)]
        dp = dp.rx(robjects.IntVector(indices))

    # This only works if inputs is a neuronlist of dotprops
    if 'neuronlist' in cl(dp):
        df = data2py(dp.slots['df'])
        df.reset_index(inplace=True, drop=True)
    else:
        # If single DataFrame, we don't collect any meta information
        df = pd.DataFrame([])
        dp = [dp]

    points = []
    for i in range(len(dp)):
        this_points = pd.concat([data2py(dp[i][0]), data2py(dp[i][2])], axis=1)
        this_points['alpha'] = dp[i][1]
        this_points.columns = ['x', 'y', 'z', 'x_vec', 'y_vec', 'z_vec', 'alpha']
        points.append(this_points)

    df['points'] = points

    return core.Dotprops(df)


def nblast_allbyall(x: 'core.NeuronList',  # type: ignore  # doesn't like n_cores defau
                    micron_conversion: float,
                    normalize: bool = True,
                    resample: int = 1,
                    n_cores: int = os.cpu_count(),
                    use_alpha: bool = False) -> 'pyclust.ClustResults':
    """ Wrapper to use R's ``nat:nblast_allbyall``
    (https://github.com/jefferislab/nat.nblast/).

    Parameters
    ----------
    x :                 NeuronList | RCatmaid neurons
                        Neurons to blast.
    micron_conversion : int | float
                        Conversion factor to microns. Units in microns is
                        not strictly necessary but highly recommended.
    resample :          int | bool, optional
                        Resampling factor. This makes sure that the neurons
                        are evenly sampled. Applied after micron conversion!
    normalize :         bool, optional
                        If True, matrix is normalized using z-score.
    n_cores :           int, optional
                        Number of cores to use for nblasting. Default is
                        ``os.cpu_count()``.
    use_alpha :         bool, optional
                        Emphasises neurons' straight parts (backbone) over
                        parts that have lots of branches.

    Returns
    -------
    nblast_results
        Instance of :class:`navis.ClustResults` that holds distance
        matrix and contains wrappers to cluster and plot data. Please use
        ``help(nblast_results)`` to learn more and see example below.

    Examples
    --------
    >>> import navis
    >>> import matplotlib.pyplot as plt
    >>> nl = navis.example_neurons()
    >>> # Blast against each other
    >>> res = navis.nblast_allbyall( nl )
    >>> # Cluster and create simple dendrogram
    >>> res.cluster(method='ward')
    >>> res.plot_matrix()
    >>> plt.show()
    """

    domc = importr('doMC')
    cores = robjects.r(f'registerDoMC({n_cores})')

    doParallel = importr('doParallel')
    doParallel.registerDoParallel(cores=n_cores)

    if 'rpy2' in str(type(x)):
        rn = x
    elif isinstance(x, core.NeuronList):
        if x.shape[0] < 2:
            raise ValueError('You have to provide more than a single neuron.')
        rn = neuron2r(x, unit_conversion=micron_conversion)
    elif isinstance(x, core.TreeNeuron):
        raise ValueError('You have to provide more than a single neuron.')
    else:
        raise ValueError(f'Must provide NeuronList, not "{type(x)}"')

    # Make dotprops and resample
    xdp = nat.dotprops(rn, k=5, resample=resample)

    # Calculate scores
    scores = r_nblast.nblast(xdp, xdp, **{'normalised': False,
                                          '.parallel': True,
                                          'UseAlpha': use_alpha})

    # The order is the same as in original NeuronList!
    matrix = pd.DataFrame(np.array(scores))

    if normalize:
        # Perform z-score normalization
        matrix = (matrix - matrix.mean()) / matrix.std()

    if isinstance(x, core.NeuronList):
        res = pyclust.ClustResults(matrix, mat_type='similarity',
                                   labels=x.name if isinstance(x, core.NeuronList) else None,
                                   )
        res.neurons = x
        return res
    else:
        return pyclust.ClustResults(matrix, mat_type='similarity')


def nblast(neuron: 'core.TreeNeuron',  # type: ignore  # doesn't like n_cores default
           db: Optional[str] = None,
           resample: int = 1,
           xform: Optional[str] = None,
           mirror: Optional[str] = None,
           n_cores: int = os.cpu_count(),
           reverse: bool = False,
           normalised: bool = True,
           UseAlpha: bool = False) -> 'NBLASTresults':
    """ Wrapper to use R's nblast (https://github.com/jefferis/nat).


    Parameters
    ----------
    x :             TreeNeuron | nat.neuron
                    Neurons to nblast. This can be either.
    db :            database, optional
                    File containing dotproducts to blast against. This can be
                    either:

                    1. the name of a file in ``'flycircuit.datadir'``,
                    2. a path (e.g. ``'.../gmrdps.rds'``),
                    3. an R file object (e.g. ``robjects.r("load('.../gmrdps.rds')")``)
                    4. a URL to load the list from (e.g. ``'http://.../gmrdps.rds'``)

                    If not provided, will search for a 'dpscanon.rds' file in
                    'flycircuit.datadir'.
    resample :      int, optional
                    Resample to given resolution. For nblasts against light
                    level databases, 1 micron is recommended. Note that this
                    is done AFTER any xform-ing.
    xform :         str, optional
                    If provided, will convert neuron before nblasting. Must
                    be string defining source and target brain space. For
                    example::

                        'FAFB14->JFRC2' converts from FAFB to JFRC2 space
                        'JRC2018F->JRC2013' converts from JRC2018F to JRC2013 space

    mirror :        str, optional
                    If reference space (e.g. "FAFB14" or "JFRC2") is provided
                    the neuron will be mirrored before nblasting. This is
                    relevant for some database as e.g. FlyCircuit neurons are
                    always on the fly's right.
    n_cores :       int, optional
                    Number of cores to use for nblasting. Default is
                    ``os.cpu_count()``.
    reverse :       bool, optional
                    If True, treats the neuron as NBLAST target rather than
                    neurons of database. Makes sense for partial
                    reconstructions.
    UseAlpha :      bool, optional
                    Emphasises neurons' straight parts (backbone) over parts
                    that have lots of branches.
    normalised :    bool, optional
                    Whether to return normalised NBLAST scores.

    Returns
    -------
    nblast_results
        Instance of :class:`navis.rmaid.NBLASTresults` that holds nblast
        results and contains wrappers to plot/extract data. Please use
        help(NBLASTresults) to learn more and see example below.


    Examples
    --------
    # NBLAST example neurons against flycircuit DB

    >>> import navis
    >>> from navis.interfaces import r
    >>> n = navis.example_neurons(1)
    >>> # Example neurons are in FAFB space -> we need to convert and mirror
    >>> # (this also takes care of conversion to microns)
    >>> nbl = r.nblast(n, xform='FAFB14->FCWB', mirror='FCWB')
    >>> # Plot top 5 results
    >>> nbl.plot3d(hits=5)

    """

    start_time = time.time()

    domc = importr('doMC')
    cores = robjects.r(f'registerDoMC({int(n_cores)})')

    doParallel = importr('doParallel')
    doParallel.registerDoParallel(cores=n_cores)

    try:
        flycircuit = importr('flycircuit')
        datadir = robjects.r('getOption("flycircuit.datadir")')[0]
    except BaseException:
        logger.error('R Flycircuit not found.')

    if db is None:
        if not os.path.isfile(datadir + '/dpscanon.rds'):
            raise ValueError('Unable to find default DPS database dpscanon.rds '
                             'in flycircuit.datadir. Please provide database '
                             'using db parameter.')
        logger.info('DPS database not explicitly provided. Loading local '
                    'FlyCircuit DB from dpscanon.rds')
        dps = robjects.r(f'read.neuronlistfh("{datadir}/dpscanon.rds")')
    elif isinstance(db, str):
        if db.startswith('http') or '/' in db:
            dps = robjects.r(f'read.neuronlistfh("{db}")')
        else:
            dps = robjects.r(f'read.neuronlistfh("{datadir}/{db}")')
    elif 'rpy2' in str(type(db)):
        dps = db
    else:
        raise ValueError('Unable to process the DPS database you have provided. '
                         'See help(rmaid.nblast) for details.')

    if 'rpy2' in str(type(neuron)):
        rn = neuron
    elif isinstance(neuron, core.NeuronList):
        if len(neuron) > 1:
            logger.warning('You provided more than a single neuron. Blasting '
                           f'only against the first: {neuron[0].neuron_name}')
        rn = neuron2r(neuron[0])
    elif isinstance(neuron, pd.Series) or isinstance(neuron, core.TreeNeuron):
        rn = neuron2r(neuron)
    else:
        raise TypeError(f'Expected navis or R neuron, got {type(neuron)}')

    # Xform neuron if necessary
    if xform:
        source, target = xform.split('->')
        sample = robjects.r(source)
        reference = robjects.r(target)
        rn = nat_templatebrains.xform_brain(nat.neuronlist(rn),
                                            sample=sample,
                                            reference=reference)
        # Make sure reference checks out if we are going to mirror
        if mirror:
            mirror = target

    # Mirror neuron
    if mirror:
        if not isinstance(mirror, str):
            raise ValueError('"mirror" must be string describing reference, '
                             f'object but got {type(mirror)}')
        reference = robjects.r(mirror)
        rn = nat_templatebrains.mirror_brain(rn, reference)

    # Save template brain for later
    tb = nat_templatebrains.regtemplate(rn)

    # Get neuron object out of the neuronlist
    rn = rn.rx2(1)

    # Reassign template brain
    rn.slots['regtemplate'] = tb

    # The following step are from nat.dotprops_neuron()
    # xdp = nat.dotprops( nat.xyzmatrix(rn) )
    xdp = nat.dotprops(rn, resample=resample, k=5)

    # number of reverse scores to calculate (max 100)
    nrev = min(100, len(dps))

    logger.info('Blasting neuron...')
    if reverse:
        sc = r_nblast.nblast(dps, nat.neuronlist(xdp),
                             **{'normalised': normalised,
                                '.parallel': True,
                                'UseAlpha': UseAlpha})

        # Have to convert to dataframe to sort them -> using
        # 'robjects.r("sort")' looses the names for some reason
        sc_df = pd.DataFrame([[sc.names[0][i], sc[i]] for i in range(len(sc))],
                             columns=['name', 'score'])
        sc_df.sort_values('score', ascending=False, inplace=True)

        # Use ".rx()" like "[]" and "rx2()" like "[[]]" to extract subsets of R
        # objects
        scr = r_nblast.nblast(nat.neuronlist(xdp),
                              dps.rx(robjects.StrVector(sc_df.name.tolist()[:nrev])),
                              **{'normalised': normalised,
                                 '.parallel': True,
                                 'UseAlpha': UseAlpha})
    else:
        sc = r_nblast.nblast(nat.neuronlist(xdp), dps,
                             **{'normalised': normalised,
                                '.parallel': True,
                                'UseAlpha': UseAlpha})

        # Have to convert to dataframe to sort them -> using
        # 'robjects.r("sort")' looses the names for some reason
        sc_df = pd.DataFrame([[sc.names[0][i], sc[i]] for i in range(len(sc))],
                             columns=['name', 'score'])
        sc_df.sort_values('score', ascending=False, inplace=True)

        # Use ".rx()" like "[]" and "rx2()" like "[[]]" to extract subsets of R
        # objects
        scr = r_nblast.nblast(dps.rx(robjects.StrVector(sc_df.name.tolist()[:nrev])),
                              nat.neuronlist(xdp),
                              **{'normalised': normalised,
                                 '.parallel': True,
                                 'UseAlpha': UseAlpha})

    sc_df.set_index('name', inplace=True, drop=True)

    df = pd.DataFrame([[scr.names[i], sc_df.loc[scr.names[i]].score, scr[i], (sc_df.loc[scr.names[i]].score + scr[i]) / 2]
                       for i in range(len(scr))],
                      columns=['gene_name', 'forward_score',
                               'reverse_score', 'mu_score']
                      )

    logger.info(f'Blasting done in {time.time() - start_time:.1f} seconds')

    return NBLASTresults(results=df,
                         sc=sc,
                         scr=scr,
                         neuron=rn,
                         xdp=xdp,
                         dps_db=dps,
                         nblast_param={'mirror': mirror,
                                       'reference': reference,
                                       'UseAlpha': UseAlpha,
                                       'normalised': normalised,
                                       'reverse': reverse})


class NBLASTresults:
    """ Class that holds nblast results and contains wrappers that allow easy
    plotting.

    Attributes
    ----------
    results :   pandas.Dataframe
                Contains top N results.
    sc :        Robject
                Contains original RNblast forward scores.
    scr :       Robject
                Original R Nblast reverse scores (Top N only).
    neuron :    R ``catmaidneuron``
                The neuron that was nblasted transformed into reference space.
    xdp :       robject
                Dotproduct of the transformed neuron.
    param :     dict
                Contains parameters used for nblasting.
    db :        file robject
                Dotproduct database as R object "neuronlistfh".
    date :      datetime object
                Time of nblasting.

    Examples
    --------
    >>> import navis
    >>> # Blast neuron by skeleton ID
    >>> nbl = navis.nblast( skid, remote_instance = rm )
    >>> # Sort results by mu_score
    >>> nbl.sort( 'mu_score' )
    >>> # Show table
    >>> nbl.results
    >>> # 3D plot top 5 hits using vispy
    >>> canvas, view = nbl.plot(hits=5)
    >>> # Show distribution of results
    >>> import matplotlib.pyplot as plt
    >>> nbl.results.hist( layout=(3,1), sharex=True)
    >>> plt.show()
    """

    def __init__(self, results, sc, scr, neuron, xdp, dps_db, nblast_param):
        """ Init function."""
        self.results = results  # this is pandas Dataframe holding top N results
        self.sc = sc  # original Nblast forward scores
        self.scr = scr  # original Nblast reverse scores (Top N only)
        self.neuron = neuron  # the transformed neuron that was nblasted
        self.xdp = xdp  # dotproduct of the transformed neuron
        self.db = dps_db  # dotproduct database as R object "neuronlistfh"
        self.param = nblast_param  # parameters used for nblasting
        self.date = datetime.now()  # time of nblasting

    def sort(self, columns: Union[str, List[str]]):
        """ Sort results by given column(s)."""
        self.results.sort_values(columns, inplace=True, ascending=False)
        self.results.reset_index(inplace=True, drop=True)

    def plot3d(self,
               hits: int = 5,
               plot_neuron: bool = True,
               plot_brain: bool = True,
               **kwargs):
        """ Wrapper to plot nblast hits using ``navis.plot3d()``

        Parameters
        ----------
        hits :  int | str | list thereof, optional
                Nblast hits to plot (default = 5). Can be:

                1. int: e.g. ``hits=5`` for top 5 hits
                2 .list of ints: e.g. ``hits=[2,5]`` to plot hits 2 and 5
                3. string: e.g. ``hits='THMARCM-198F_seg1'`` to plot this neuron
                4. list of strings: e.g. ``['THMARCM-198F_seg1', npfMARCM-'F000003_seg002']`` to plot multiple neurons by their gene name

        plot_neuron :   bool
                        If ``True``, the nblast query neuron will be plotted.
        plot_brain :    bool
                        If ``True``, the reference brain will be plotted.
        **kwargs
                        Parameters passed to :func:`~navis.plot3d`.
                        See ``help(navis.plot3d)`` for details.

        Returns
        -------
        Depending on the backends used by ``navis.plot3d()``:

        vispy (default) : canvas, view
        plotly : plotly figure dictionary

        You can specify the backend by using e.g. ``backend='plotly'`` in
        **kwargs. See ``help(navis.plot3d)`` for details.
        """

        nl = self.get_dps(hits)

        n_py = neuron2py(self.neuron)

        # Create colormap with the query neuron being black
        cmap = {n_py.id: (0, 0, 0)}

        colors = np.linspace(0, 1, len(nl) + 1)
        colors = np.array([hsv_to_rgb(c, 1, 1) for c in colors])
        colors *= 255
        cmap.update({e: colors[i] for i, e in enumerate(nl.names)})

        # Prepare brain
        if plot_brain:
            ref_brain = robjects.r(self.param['reference'][8][0] + '.surf')

            verts = data2py(ref_brain[0])[['X', 'Y', 'Z']].values.tolist()
            faces = data2py(ref_brain[1][0]).values
            faces -= 1  # reduce indices by 1
            faces = faces.tolist()

            volumes = {self.param['reference'][8][0]: {'verts': verts,
                                                       'faces': faces}}
        else:
            volumes = []

        kwargs.update({'color': cmap})

        if nl:
            if plot_neuron is True:
                return plotting.plot3d([n_py, dotprops2py(nl), volumes], **kwargs)
            else:
                return plotting.plot3d([dotprops2py(nl), volumes], **kwargs)

    def get_dps(self, entries: Union[int, str, List[str], List[int]]):
        """ Wrapper to retrieve dotproducts from DPS database (neuronlistfh)
        as neuronslist.

        Parameters
        ----------
        entries :   int | str | list thereof, optional
                    Neurons to extract from DPS database. Can be:

                    1. int: e.g. ``hits=5`` for top 5 hits
                    2 .list of ints: e.g. ``hits=[2,5]`` to plot hits 2 and 5
                    3. string: e.g. ``hits = 'THMARCM-198F_seg1'`` to plot this neuron
                    4. list of strings:
                       e.g. ``['THMARCM-198F_seg1', npfMARCM-'F000003_seg002']``
                       to plot multiple neurons by their gene name

        Returns
        -------
        neuronlist of dotproduct neurons
        """

        if isinstance(entries, int):
            return self.db.rx(robjects.StrVector(self.results.ix[:entries - 1].gene_name.tolist()))
        elif isinstance(entries, str):
            return self.db.rx(entries)
        elif isinstance(entries, (list, np.ndarray)):
            if isinstance(entries[0], int):
                return self.db.rx(robjects.StrVector(self.results.ix[entries].gene_name.tolist()))
            elif isinstance(entries[0], str):
                return self.db.rx(robjects.StrVector(entries))
        else:
            logger.error('Unable to interpret entries provided. See '
                         'help(NBLASTresults.plot3d) for details.')
            return None


def xform_brain(x: Union['core.NeuronObject', 'pd.DataFrame', 'np.ndarray'],
                source: str,
                target: str,
                fallback: Optional[Literal['AFFINE']] = None,
                **kwargs) -> Union['core.NeuronObject',
                                   'pd.DataFrame',
                                   'np.ndarray']:
    """Transform 3D data between template brains. This is just a wrapper for
    ``nat.templatebrains:xform_brain``.

    Parameters
    ----------
    x :         Neuron/List | numpy.ndarray | pandas.DataFrame
                Data to transform. Dataframe must contain ``['x', 'y', 'z']``
                columns. Numpy array must be shape ``(N, 3)``.
    source :    str
                Source template brain that the data currently is in.
    target :    str
                Target template brain that the data should be transformed into.
    fallback :  None | "AFFINE",
                If "AFFINE", will fall back to affine transformation if CMTK
                transformation fails. Else coordinates of points for which the
                transformation failed (e.g. b/c they are out of bounds), will
                be returned as ``None``.
    **kwargs
                Keyword arguments passed to ``nat.templatebrains:xform_brain``

    Returns
    -------
    same type as ``x``
                Copy of input with transformed coordinates.

    """
    if isinstance(x, core.NeuronList):
        if len(x) == 1:
            x = x[0]
        else:
            xf = []
            for n in config.tqdm(x, desc='Xforming',
                                 disable=config.pbar_hide,
                                 leave=config.pbar_leave):
                xf.append(xform_brain(n,
                                      source=source,
                                      target=target,
                                      fallback=fallback,
                                      **kwargs))
            return core.NeuronList(xf)

    if not isinstance(x, (core.TreeNeuron, np.ndarray, pd.DataFrame)):
        raise TypeError(f'Unable to transform data of type "{type(x)}"')

    if isinstance(x, core.TreeNeuron):
        x = x.copy()
        x.nodes = xform_brain(x.nodes,
                              source=source,
                              target=target,
                              fallback=fallback,
                              **kwargs)
        if x.has_connectors:
            x.connectors = xform_brain(x.connectors,
                                       source=source,
                                       target=target,
                                       fallback=fallback,
                                       **kwargs)
        return x
    elif isinstance(x, pd.DataFrame):
        if any([c not in x.columns for c in ['x', 'y', 'z']]):
            raise ValueError('DataFrame must have x, y and z columns.')
        x = x.copy()
        x.loc[:, ['x', 'y', 'z']] = xform_brain(x[['x', 'y', 'z']].values.astype(float),
                                                source=source,
                                                target=target,
                                                fallback=fallback,
                                                **kwargs)
        return x
    elif x.shape[1] != 3:
        raise ValueError('Array must be of shape (N, 3).')

    if isinstance(source, str):
        source = robjects.r(source)
    else:
        TypeError(f'Expected source of type str, got "{type(source)}"')

    if isinstance(target, str):
        target = robjects.r(target)
    else:
        TypeError(f'Expected target of type str, got "{type(target)}"')

    # We need to convert numpy arrays explicitly
    if isinstance(x, np.ndarray):
        x = numpy2ri.py2ro(x)

    xf = nat_templatebrains.xform_brain(x,
                                        sample=source,
                                        reference=target,
                                        FallBackToAffine=fallback == 'AFFINE',
                                        **kwargs)

    return np.array(xf)


def mirror_brain(x: Union['core.NeuronObject', 'pd.DataFrame', 'np.ndarray'],
                 template: str,
                 mirror_axis: Union['X', 'Y', 'Z'] = 'X',
                 transform: Union['warp', 'affine', 'flip'] = 'warp',
                 **kwargs) -> Union['core.NeuronObject',
                                   'pd.DataFrame',
                                   'np.ndarray']:
    """Mirror 3D object along given axixs. This is just a wrapper for
    ``nat.templatebrains:mirror_brain``.

    Parameters
    ----------
    x :             Neuron/List | numpy.ndarray | pandas.DataFrame
                    Data to transform. Dataframe must contain ``['x', 'y', 'z']``
                    columns. Numpy array must be shape ``(N, 3)``.
    template :      str
                    Source template brain space that the data is in.
    mirror_axis :   'X' | 'Y' | 'Z', optional
                    Axis to mirror.
    transform :     'warp' | 'affine' | 'flip', optional
                    Which kind of transform to use.
    **kwargs
                    Keyword arguments passed to
                    ``nat.templatebrains:mirror_brain``.

    Returns
    -------
    same type as ``x``
                Copy of input with transformed coordinates.

    """
    assert transform in ['warp', 'affine', 'flip']
    assert mirror_axis in ['X', 'Y', 'Z']

    if isinstance(x, core.NeuronList):
        if len(x) == 1:
            x = x[0]
        else:
            xf = []
            for n in config.tqdm(x, desc='Mirroring',
                                 disable=config.pbar_hide,
                                 leave=config.pbar_leave):
                xf.append(mirror_brain(n,
                                       template=template,
                                       mirror_axis=mirror_axis,
                                       transform=transform,
                                       **kwargs))
            return core.NeuronList(xf)

    if not isinstance(x, (core.TreeNeuron, np.ndarray, pd.DataFrame)):
        raise TypeError(f'Unable to transform data of type "{type(x)}"')

    if isinstance(x, core.TreeNeuron):
        x = x.copy()
        x.nodes = mirror_brain(x.nodes,
                               template=template,
                               mirror_axis=mirror_axis,
                               transform=transform,
                               **kwargs)
        if x.has_connectors:
            x.connectors = mirror_brain(x.connectors,
                                        template=template,
                                        mirror_axis=mirror_axis,
                                        transform=transform,
                                        **kwargs)
        return x
    elif isinstance(x, pd.DataFrame):
        if any([c not in x.columns for c in ['x', 'y', 'z']]):
            raise ValueError('DataFrame must have x, y and z columns.')
        x = x.copy()
        x.loc[:, ['x', 'y', 'z']] = mirror_brain(x[['x', 'y', 'z']].values.astype(float),
                                                 template=template,
                                                 mirror_axis=mirror_axis,
                                                 transform=transform,
                                                 **kwargs)
        return x
    elif x.shape[1] != 3:
        raise ValueError('Array must be of shape (N, 3).')

    if isinstance(template, str):
        template = robjects.r(template)
    else:
        TypeError(f'Expected template of type str, got "{type(template)}"')

    # We need to convert numpy arrays explicitly
    if isinstance(x, np.ndarray):
        x = numpy2ri.py2ro(x)

    xf = nat_templatebrains.mirror_brain(x,
                                         brain=template,
                                         mirrorAxis=mirror_axis,
                                         transform=transform,
                                         **kwargs)

    return np.array(xf)


def get_brain_template_mesh(x: str) -> core.Volume:
    """Fetch brain surface model from ``nat.flybrains``, ``flycircuit`` or
    ``elmr`` and converts to :class:`navis.Volume`.

    Parameters
    ----------
    x :             str
                    Name of template brain. For example: 'FCWB', 'FAFB14' or
                    'JFRC2'.

    Returns
    -------
    navis.Volume
    """

    if not x.endswith('.surf'):
        x += '.surf'

    # Get the brain volume (this is a named list)
    brain = robjects.r(f'{x}')

    # Vertices are simply called vertices
    vertices = data2py(brain[brain.names.index('Vertices')])

    # For some reason faces are called "Regions" -> we're using the "first"
    # region although some meshes might include more than one
    faces = data2py(brain[brain.names.index('Regions')][0])

    # Offset to account for difference in indexing between R (1, 2, 3, ...)
    # and  Python (0, 1, 2, ....)
    faces -= 1

    return core.Volume(vertices=vertices[['X', 'Y', 'Z']].values,
                       faces=faces[['V1', 'V2', 'V3']].values,
                       name=x)


def get_neuropil(x: str, template: str = 'FCWB') -> core.Volume:
    """ Fetches given neuropil from ``nat.flybrains``, ``flycircuit`` or
    ``elmr`` and converts to :class:`navis.Volume`.

    Parameters
    ----------
    x :             str
                    Name of neuropil.
    template :      'FCWB' | 'FAFB14' | 'JFRC', optional
                    Name of the template brain.

    Returns
    -------
    navis.Volume
    """

    if not template.endswith('NP.surf'):
        template += 'NP.surf'

    # Get the brain volume (this is a named list)
    template = robjects.r(f'{template}')

    reglist = template[template.names.index('RegionList')]  # type: ignore  # confused by R objects

    if x not in reglist:
        raise ValueError('Neuropil not found in this brain. Available '
                         f'regions: {", ".join(reglist)}')

    # Get list of faces for desired region
    regions = template[template.names.index('Regions')]  # type: ignore  # confused by R objects
    faces = data2py(regions[regions.names.index(x)])  # type: ignore  # confused by R objects

    # Offset to account for difference in indexing between R (1, 2, 3, ...)
    # and  Python (0, 1, 2, ....)
    faces -= 1

    # Get pre-defined color for this neuropil
    color = template[template.names.index('RegionColourList')][regions.names.index(x)]  # type: ignore  # confused by R objects

    # Get vertices
    all_vertices = data2py(template[template.names.index('Vertices')])  # type: ignore  # confused by R objects

    # Remove superfluous vertices
    verts_required = np.unique(faces.values)
    this_verts = all_vertices.loc[verts_required]

    # Reorder and remap - DO NOT use ".index" here!
    new_map = {old: new for old, new in zip(this_verts.index,
                                            np.arange(0, this_verts.shape[0]).astype(int))}
    faces['V1'] = faces.V1.map(new_map)
    faces['V2'] = faces.V2.map(new_map)
    faces['V3'] = faces.V3.map(new_map)

    return core.Volume(vertices=this_verts[['X', 'Y', 'Z']].values,
                       faces=faces.values,
                       name=x,
                       color=color)
