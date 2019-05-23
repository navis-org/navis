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

import pandas as pd
import numpy as np

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, numpy2ri

from .. import cluster as pyclust
from .. import core, plotting, config, utils

cl = robjects.r('class')
names = robjects.r('names')
attributes = names = robjects.r('attributes')

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
                  'get_neuropil', 'xform_brain'])

# Activate automatic conversion
numpy2ri.activate()
pandas2ri.activate()

def init_rcatmaid(**kwargs):
    """ Initialize the R catmaid package.

    R package by Greg Jefferis: https://github.com/jefferis/rcatmaid

    Parameters
    ----------
    remote_instance :   CATMAID instance
                        From navis.CatmaidInstance(). This is used to
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
        try:
            df = pandas2ri.ri2py_dataframe(data)
            return df
        except BaseException:
            logger.debug(f'Unable to convert R data of type "{cl(data)}"')
            return 'Not converted'
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


def neuron2py(x, unit_conversion=False, add_attributes=None):
    """ Converts an rcatmaid ``neuron`` or ``neuronlist`` object to
    :class:`~navs.TreeNeuron` or :class:`~navis.NeuronList`, respectively.

    Parameters
    ----------
    x :                 R neuron | R neuronlist
                        Neuron to convert to Python
    unit_conversion :   bool | int | float, optional
                        If True will convert units.
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

    add_attributes = utils._make_iterable(add_attributes)

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

        # Set parent to None if negative
        r.d.loc[r.d.parent_id < 0, 'parent_id'] = None

        neurons.append(core.TreeNeuron(r.d, **add_data))

    if len(neurons) == 1:
        return neurons[0]
    return core.NeuronList(neurons)


def neuron2r(x, unit_conversion=False, add_metadata=False):
    """ Converts a neuron or neuronlist to the corresponding
    neuron/neuronlist object in R.

    Parameters
    ----------
    x :                 TreeNeuron | NeuronList
    unit_conversion :   bool | int | float, optional
                        If not ``False`` will convert coordinates.
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

        nlist = {str(n.uuid): neuron2r(n, unit_conversion=unit_conversion) for
                 n in config.tqdm(x, desc='Conv.',
                                  leave=config.pbar_leave,
                                  disable=config.pbar_hide)}

        nlist = robjects.ListVector(nlist)
        nlist.rownames = [str(n.uuid) for n in x]

        """
        df = robjects.DataFrame({'pid': robjects.IntVector([1] * x.shape[0]),
                                 'name': robjects.StrVector(x.neuron_name.tolist())
                                 })
        df.rownames = x.skeleton_id.tolist()
        nlist.slots['df'] = df
        """
        nlist.rclass = robjects.r('c("neuronlist","list")')

        return nlist

    elif isinstance(x, core.TreeNeuron):
        # Convert units if applicable
        if unit_conversion:
            x = x * unit_conversion

        # First convert into format that rcatmaid expects as server response

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

        soma_id = x.soma if not isinstance(x.soma, type(None)) else robjects.r('NULL')

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


def dotprops2py(dp, subset=None):
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
        Can be passed to `plotting.plot3d( dotprops )`
    """

    # Check if list is on demand
    if 'neuronlistfh' in cl(dp) and not subset:
        dp = dp.rx(robjects.IntVector([i + 1 for i in range(len(dp))]))
    elif subset:
        indices = [i for i in subset if isinstance(
            i, int)] + [dp.names.index(n) + 1 for n in subset if isinstance(n, str)]
        dp = dp.rx(robjects.IntVector(indices))

    df = data2py(dp.slots['df'])
    df.reset_index(inplace=True, drop=True)

    points = []
    for i in range(len(dp)):
        this_points = pd.concat([data2py(dp[i][0]), data2py(dp[i][2])], axis=1)
        this_points['alpha'] = dp[i][1]
        this_points.columns = ['x', 'y', 'z',
                               'x_vec', 'y_vec', 'z_vec', 'alpha']
        points.append(this_points)

    df['points'] = points

    return core.Dotprops(df)


def nblast_allbyall(x, micron_conversion, normalize=True, resample=1,
                    n_cores=os.cpu_count(), use_alpha=False):
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
                        are evenly sampled.
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
    cores = robjects.r('registerDoMC(%i)' % n_cores)

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


def nblast(neuron, remote_instance=None, db=None, n_cores=os.cpu_count(),
           reverse=False, normalised=True, UseAlpha=False, mirror=True,
           reference='nat.flybrains::FCWB'):
    """ Wrapper to use R's nblast (https://github.com/jefferis/nat).

    Provide neuron to nblast either as skeleton ID or neuron object. This
    essentially recapitulates what `elmr's <https://github.com/jefferis/elmr>`_
    ``nblast_fafb`` does.

    Notes
    -----
    Neurons are automatically resampled to 1 micron.

    Parameters
    ----------
    x
                    Neuron to nblast. This can be either:
                    1. A single skeleton ID
                    2. navis neuron from e.g. navis.get_neuron()
                    3. RCatmaid neuron object
    remote_instance :   Catmaid Instance, optional
                        Only neccessary if only a SKID is provided
    db :            database, optional
                    File containing dotproducts to blast against. This can be
                    either:

                    1. the name of a file in ``'flycircuit.datadir'``,
                    2. a path (e.g. ``'.../gmrdps.rds'``),
                    3. an R file object (e.g. ``robjects.r("load('.../gmrdps.rds')")``)
                    4. a URL to load the list from (e.g. ``'http://.../gmrdps.rds'``)

                    If not provided, will search for a 'dpscanon.rds' file in
                    'flycircuit.datadir'.
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
    mirror :        bool, optional
                    Whether to mirror the neuron or not b/c FlyCircuit neurons
                    are on fly's right.
    normalised :    bool, optional
                    Whether to return normalised NBLAST scores.
    reference :     string | R file object, optional
                    Default = 'nat.flybrains::FCWB'

    Returns
    -------
    nblast_results
        Instance of :class:`navis.rmaid.NBLASTresults` that holds nblast
        results and contains wrappers to plot/extract data. Please use
        help(NBLASTresults) to learn more and see example below.

    """

    start_time = time.time()

    domc = importr('doMC')
    cores = robjects.r('registerDoMC(%i)' % n_cores)

    doParallel = importr('doParallel')
    doParallel.registerDoParallel(cores=n_cores)

    if remote_instance is None:
        if 'remote_instance' in sys.modules:
            remote_instance = sys.modules['remote_instance']
        elif 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']

    try:
        flycircuit = importr('flycircuit')
        datadir = robjects.r('getOption("flycircuit.datadir")')[0]
    except BaseException:
        logger.error('R Flycircuit not found.')

    if db is None:
        if not os.path.isfile(datadir + '/dpscanon.rds'):
            logger.error('Unable to find default DPS database dpscanon.rds in '
                         'flycircuit.datadir. Please provide database using '
                         'db parameter.')
            return
        logger.info('DPS database not explicitly provided. Loading local '
                    'FlyCircuit DB from dpscanon.rds')
        dps = robjects.r('read.neuronlistfh("%s")' %(datadir + '/dpscanon.rds'))
    elif isinstance(db, str):
        if db.startswith('http') or '/' in db:
            dps = robjects.r('read.neuronlistfh("%s")' % db)
        else:
            dps = robjects.r('read.neuronlistfh("%s")' % datadir + '/' + db)
    elif 'rpy2' in str(type(db)):
        dps = db
    else:
        logger.error('Unable to process the DPS database you have provided. '
                     'See help(rmaid.nblast) for details.')
        return

    if 'rpy2' in str(type(neuron)):
        rn = neuron
    elif isinstance(neuron, pd.DataFrame) or isinstance(neuron, core.CatmaidNeuronList):
        if neuron.shape[0] > 1:
            logger.warning('You provided more than a single neuron. Blasting '
                           'only against the first: %s' % neuron.ix[0].neuron_name)
        rn = neuron2r(neuron.ix[0], convert_to_um=False)
    elif isinstance(neuron, pd.Series) or isinstance(neuron, core.CatmaidNeuron):
        rn = neuron2r(neuron, convert_to_um=False)
    elif isinstance(neuron, str) or isinstance(neuron, int):
        if not remote_instance:
            logger.error('You have to provide a CATMAID instance using the '
                         '<remote_instance> parameter. See help(rmaid.nblast) '
                         'for details.')
            return
        rn = neuron2r(fetch.get_neuron(
            neuron, remote_instance), convert_to_um=False)
    else:
        logger.error('Unable to intepret <neuron> parameter provided. See '
                     'help(rmaid.nblast) for details.')
        return

    # Bring catmaid neuron into reference brain space -> this also converts to
    # um
    if isinstance(reference, str):
        reference = robjects.r(reference)
    rn = nat_templatebrains.xform_brain(
        nat.neuronlist(rn), sample='FAFB14', reference=reference)

    # Mirror neuron
    if mirror:
        rn = nat_templatebrains.mirror_brain(rn, reference)

    # Save template brain for later
    tb = nat_templatebrains.regtemplate(rn)

    # Get neuron object out of the neuronlist
    rn = rn.rx2(1)

    # Reassign template brain
    rn.slots['regtemplate'] = tb

    # The following step are from nat.dotprops_neuron()
    # xdp = nat.dotprops( nat.xyzmatrix(rn) )
    xdp = nat.dotprops(rn, resample=1, k=5)

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

    logger.info('Blasting done in %s seconds' % round(time.time() - start_time))

    return NBLASTresults(df, sc, scr, rn, xdp, dps, {'mirror': mirror,
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

    def sort(self, columns):
        """ Sort results by given column."""
        self.results.sort_values(columns, inplace=True, ascending=False)
        self.results.reset_index(inplace=True, drop=True)

    def plot3d(self, hits=5, plot_neuron=True, plot_brain=True, **kwargs):
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
        plotly : matplotlib figure

        You can specify the backend by using e.g. ``backend='plotly'`` in
        **kwargs. See ``help(navis.plot3d)`` for details.
        """

        nl = self.get_dps(hits)

        n_py = neuron2py(self.neuron)
        # We have to bring the soma radius down to um -> this may mess
        # up soma detection elsewhere, so be carefull!
        n_py.ix[0].nodes.radius /= 1000

        # Create colormap with the query neuron being black
        cmap = {n_py.ix[0].skeleton_id: (0, 0, 0)}

        colors = np.linspace(0, 1, len(nl) + 1)
        colors = np.array([hsv_to_rgb(c, 1, 1) for c in colors])
        colors *= 255
        cmap.update({e: colors[i] for i, e in enumerate(nl.names)})

        # Prepare brain
        if plot_brain:
            ref_brain = robjects.r(self.param['reference'][8][0] + '.surf')

            verts = data2py(ref_brain[0])[['X', 'Y', 'Z']].as_matrix().tolist()
            faces = data2py(ref_brain[1][0]).as_matrix()
            faces -= 1  # reduce indices by 1
            faces = faces.tolist()
            # [ [i,i+1,i+2] for i in range( int( len(verts)/3 ) ) ]

            volumes = {self.param['reference'][8][0]: {'verts': verts, 'faces': faces}}
        else:
            volumes = []

        kwargs.update({'colors': cmap,
                       'downsampling': 1})

        logger.info('Colormap:' + str(cmap))

        if nl:
            if plot_neuron is True:
                return plotting.plot3d([n_py, dotprops2py(nl), volumes], **kwargs)
            else:
                return plotting.plot3d([dotprops2py(nl), volumes], **kwargs)

    def get_dps(self, entries):
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
            logger.error('Unable to intepret entries provided. See '
                         'help(NBLASTresults.plot3d) for details.')
            return None


def xform_brain(x, source, target, fallback=None, **kwargs):
    """ Transform 3D data between template brains. This is just a wrapper for
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

    if not isinstance(x, (core.TreeNeuron, np.ndarray, pd.DataFrame)):
        raise TypeError(f'Unable to transform data of type "{type(x)}"')

    if isinstance(x, core.TreeNeuron):
        x = x.copy()
        x.nodes = xform_brain(x.nodes, source, target)
        if x.has_connectors:
            x.connectors = xform_brain(x.connectors, source, target)
        return x
    elif isinstance(x, pd.DataFrame):
        if any([c not in x.columns for c in ['x', 'y', 'z']]):
            raise ValueError('DataFrame must have x, y and z columns.')
        x = x.copy()
        x.loc[:, ['x', 'y', 'z']] = xform_brain(x[['x', 'y', 'z']].values.astype(float),
                                                source, target)
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

    xf = nat_templatebrains.xform_brain(x,
                                        sample=source,
                                        reference=target,
                                        FallBackToAffine=fallback == 'AFFINE',
                                        **kwargs)

    return np.array(xf)


def get_neuropil(x, template='FCWB', convert_nm=True):
    """ Fetches given neuropil from ``nat.flybrains``, ``flycircuit`` or
    ``elmr`` and converts to :class:`navis.Volume`.

    Parameters
    ----------
    x :             str
                    Name of neuropil.
    template :      'FCWB' | 'FAFB14' | 'JFRC', optional
                    Name of the template brain.
    convert_nm :    bool, optional
                    If True, will convert from um to nm.
    """

    if not template.endswith('NP.surf'):
        template += 'NP.surf'

    np = data2py(robjects.r(f'subset({x}, "{template}")'))

    n_verts = np['Vertices'].shape[0]
    verts = np['Vertices'][['X', 'Y', 'Z']].values
    faces = np.array([range(0, n_verts, 3),
                      range(1, n_verts, 3),
                      range(2, n_verts, 3)]).T

    if convert_nm:
        verts += 1000

    return core.Volume(verts, faces)
