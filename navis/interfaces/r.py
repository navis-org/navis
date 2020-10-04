
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
import math
import numbers
import os
import sys

from datetime import datetime
from colorsys import hsv_to_rgb
from typing import Union, Optional, List
from typing_extensions import Literal
from scipy.spatial.distance import pdist

import pandas as pd
import numpy as np

import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

from rpy2.robjects import pandas2ri, numpy2ri

# Inconventiently, rpy2's version vector differs in the way it's constructed
# between 2.X ``((2, 9, 4), '')`` and 3.X (3, 3, 2)
rpy2_major_version = int(rpy2.__version__.split('.')[0])
if rpy2_major_version >= 3:
    from rpy2.robjects.conversion import localconverter

from .. import cluster as pyclust
from .. import core, plotting, config, utils

# Set up logging
logger = config.logger


def try_importr(x):
    """Try importing R library. Log error on exception."""
    try:
        return importr(x)
    except BaseException:
        logger.warning(f'Failed to import R library "{x}"! Some functions might'
                        ' not work as expected. Please install from within R.')
        # Return dummy class
        return FailedImport(x)


def try_loadr(x):
    """Try finding robject. Will postpone exception until use time."""
    try:
        return robjects.r(x)
    except BaseException:
        # Return dummy class
        return FailedObject(x)


class FailedImport:
    """Dummy class for failed imports from R. Throws meaningful exceptions."""

    def __init__(self, name):
        self.name = name

    def raise_error(self):
        raise Exception(f'R library "{self.name}" could not be imported. '
                        'Please make sure it is properly installed.')

    def __call__(self):
        self.raise_error()

    def __getattr__(self, name):
        self.raise_error()


class FailedObject(FailedImport):
    def raise_error(self):
        raise Exception(f'R object "{self.name}" was not found.')


cl = try_loadr('class')
names = try_loadr('names')
attributes = try_loadr('attributes')

# Load the entire natverse
nat = try_importr("nat")
r_nblast = try_importr('nat.nblast')
nat_templatebrains = try_importr('nat.templatebrains')
flycircuit = try_importr('flycircuit')

# These are mainly relevant for exposing transforms
nat_flybrains = try_importr('nat.flybrains')
elmr = try_importr('elmr')
nat_jrcbrains = try_importr('nat.jrcbrains')


__all__ = sorted(['neuron2r', 'neuron2py', 'init_rcatmaid', 'dotprops2py',
                  'data2py', 'NBLASTresults', 'nblast', 'nblast_allbyall',
                  'get_neuropil', 'xform_brain', 'mirror_brain'])

# Do not use this! It will convert stuff in unexpected ways. E.g.
# dps = robjects.r(f'read.neuronlistfh("{datadir}/dpscanon.rds")')
# will turn out as just an array

# numpy2ri.activate()
# pandas2ri.activate()


def init_rcatmaid(**kwargs):
    """Initialize the R Catmaid package.

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
    """Convert data from rcatmaid (e.g. ``catmaidneuron`` from
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
        if rpy2.__version_vector__[0] < 3:
            return pandas2ri.ri2py(data)
        else:
            with localconverter(robjects.default_converter + pandas2ri.converter):
                return robjects.conversion.rpy2py(data)
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
    """Convert an rcatmaid ``neuron`` or ``neuronlist`` object to
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
    """Convert TreeNeuron/List to corresponding R neuron/neuronlist object.

    Parameters
    ----------
    x :                 TreeNeuron | NeuronList
    unit_conversion :   bool | int | float, optional
                        If not ``False`` will multiply units by given factor.
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
        they also contain a dataframe as attribute ( attr('df') = df )
        This dataframe looks like this

                pid   skid     name
        skid1
        skid2

        In rpy2, attributes are assigned using the .slots['df'] function.
        """

        nlist = {str(n.id): neuron2r(n, unit_conversion=unit_conversion) for
                 n in config.tqdm(x, desc='Neurons to R',
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
        parents = x.nodes.parent_id.values
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
            try:
                prepost = robjects.IntVector(x.connectors['type'].astype(int).tolist())
            except ValueError:
                prepost = robjects.StrVector(x.connectors['type'].tolist())
            except BaseException:
                raise

            meta['connectors'] = robjects.DataFrame({'node_id': robjects.IntVector(x.connectors.node_id.tolist()),
                                                     'connector_id': robjects.IntVector(x.connectors.connector_id.tolist()),
                                                     'prepost': prepost,
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
    """Convert dotprops into pandas DataFrame.

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


def nblast_allbyall(x: 'core.NeuronList',  # type: ignore  # doesn't like n_cores defaults
                    normalized: bool = True,
                    resample: int = None,
                    n_cores: int = os.cpu_count() - 2,
                    use_alpha: bool = False) -> 'pyclust.ClustResults':
    """Wrapper to use R's ``nat:nblast_allbyall``
    (https://github.com/jefferislab/nat.nblast/).

    NBLAST is optimized for data in microns.

    Parameters
    ----------
    x :                 NeuronList | nat.neurons
                        (Tree)Neurons to blast. While not strictly necessary,
                        data should be in microns.
    resample :          int | bool, optional
                        Resampling during dotprops generation. A good value
                        is ``1`` which means if data is in microns (which it
                        should!) it will be resampled to 1 tangent vector per
                        micron.
    normalized :        bool, optional
                        If True, matrix is normalized using z-score.
    n_cores :           int, optional
                        Number of cores to use for nblasting. Default is
                        ``os.cpu_count() - 2``.
    use_alpha :         bool, optional
                        Emphasizes neurons' straight parts (backbone) over
                        parts that have lots of branches.

    Returns
    -------
    pandas.DataFrame
                        DataFrame containing the results.

    Examples
    --------
    >>> import navis
    >>> from navis.interfaces import r
    >>> nl = navis.example_neurons()
    >>> # Blast against each other (note the division to get to microns)
    >>> scores = r.nblast_allbyall(nl / 1000)

    """
    if n_cores > 1:
        doMC = importr('doMC')
        doMC.registerDoMC(int(n_cores))

    if 'rpy2' in str(type(x)):
        rn = x
    elif isinstance(x, core.NeuronList):
        if x.shape[0] < 2:
            raise ValueError('You have to provide more than a single neuron.')
        # Check if query or targets are not in microns
        # Note this test can return `None` if it can't be determined
        if not _check_microns(x):
            logger.warning('NBLAST is optimized for data in microns and it looks'
                           ' like these neurons are not in microns.')
        rn = neuron2r(x)
    elif isinstance(x, core.TreeNeuron):
        raise ValueError('You have to provide more than a single neuron.')
    else:
        raise ValueError(f'Must provide NeuronList, not "{type(x)}"')

    # Generate dotprops
    logger.info('Generating dotprops for neurons.')
    xdp = _make_dotprops(rn,
                         resample=resample,
                         parallel=n_cores > 1)

    # Calculate scores
    logger.info('Running all-by-all nblast.')
    pbar = 'text' if n_cores <= 1 else "none"
    scores = r_nblast.nblast(xdp, xdp, **{'normalised': normalized,
                                          '.parallel': n_cores > 1,
                                          '.progress': pbar,
                                          'UseAlpha': use_alpha})

    # Generate DataFrame from scores
    res = data2py(scores).T

    return res


def nblast(query: Union['core.TreeNeuron', 'core.NeuronList', 'core.Dotprops'],  # type: ignore  # doesn't like n_cores default
           target: Optional[str] = None,
           resample: Optional[int] = None,
           mean_scores: bool = False,
           n_cores: int = os.cpu_count() - 2,
           normalized: bool = True,
           use_alpha: bool = False) -> 'NBLASTresults':
    """Run nat's NBLAST (https://github.com/jefferis/nat).

    Please note that NBLAST is optimized for units in microns.

    Parameters
    ----------
    query :         TreeNeuron | NeuronList | Dotprops | nat.neuron
                    Query neuron(s) to NBLAST against the targets.
    target :        TreeNeuron | NeuronList | Dotprops | nat.neuron | str
                    Neuron(s) to run the query against. If a str, must be
                    either:

                        1. the name of a file in ``'flycircuit.datadir'``
                        2. a path (e.g. ``'.../gmrdps.rds'``)
                        3. an R file object (e.g. ``robjects.r("load('.../gmrdps.rds')")``)
                        4. a URL to load the list from (e.g. ``'http://.../gmrdps.rds'``)
                        5. "flycircuit"

    resample :      int | bool, optional
                    Resampling during dotprops generation. A good value
                    is ``1`` which means if data is in microns (which it
                    should!) it will be resampled to 1 tangent vector per
                    micron.
    mean_scores :   bool
                    If True, will run query->target NBLAST followed by a
                    target->query NBLAST and return the mean scores.
    n_cores :       int, optional
                    Number of cores to use for nblasting. Default is
                    ``os.cpu_count() - 2``.
    use_alpha :     bool, optional
                    Emphasizes neurons' straight parts (backbone) over parts
                    that have lots of branches.
    normalized :    bool, optional
                    Whether to return normalized NBLAST scores.

    Returns
    -------
    scores :        pandas.DataFrame
                    Matrix with NBLAST scores. Rows are query neurons, columns
                    are targets.

    Examples
    --------
    # NBLAST example neurons against flycircuit DB

    >>> import navis
    >>> from navis.interfaces import r
    >>> nl = navis.example_neurons(n=5)
    >>> nl.units
    array([8, 8, 8, 8, 8]) <Unit('nanometer')>
    >>> # Convert to microns
    >>> nl_um = nl / 1000
    >>> # Run the nblast
    >>> scores = r.nblast(nl_um)

    """
    if n_cores > 1:
        doMC = importr('doMC')
        doMC.registerDoMC(int(n_cores))
    #_ = robjects.r(f'registerDoMC({int(n_cores)})')

    # Check if query or targets are not in microns
    # Note this test can return `None` if it can't be determined
    if _check_microns(query) is False:
        logger.warning('NBLAST is optimized for data in microns and it looks '
                       'like your queries are not in microns.')
    if _check_microns(target) is False:
        logger.warning('NBLAST is optimized for data in microns and it looks '
                       'like your targets are not in microns.')

    # Turn query into dotprops
    try:
        query_dps = _make_dotprops(query,
                                   resample=resample,
                                   parallel=n_cores > 1)
    except NotImplementedError:
        raise NotImplementedError('Unable to produce R dotprops for query of '
                                  f'type {type(query)}')

    # Turn target into dotprops
    try:
        target_dps = _make_dotprops(target,
                                    resample=resample,
                                    parallel=n_cores > 1)
    except NotImplementedError:
        raise NotImplementedError('Unable to produce R dotprops for target of '
                                  f'type {type(target)}')

    # Make sure dotprops are in neuronlists
    if 'neuronlist' not in cl(query_dps):
        logger.info('Generating dotprops for query neurons.')
        target_dps = nat.neuronlist(query_dps)

    if 'neuronlist' not in cl(target_dps):
        logger.info('Generating dotprops for target neurons.')
        target_dps = nat.neuronlist(target_dps)

    logger.info('Running nblast.')
    pbar = 'text' if n_cores <= 1 else "none"
    sc = r_nblast.nblast(query_dps, target_dps,
                         **{'normalised': normalized,
                            '.parallel': n_cores > 1,
                            '.progress': pbar,
                            'UseAlpha': use_alpha})

    # Generate DataFrame from scores
    res = data2py(sc).T

    # Calculate reverse scores
    if mean_scores:
        logger.info('Running reverse nblast.')
        scr = r_nblast.nblast(target_dps, query_dps,
                              **{'normalised': normalized,
                                 '.parallel': n_cores > 1,
                                 '.progress': pbar,
                                 'UseAlpha': use_alpha})

        res = (res + data2py(scr)) / 2

    return res


def _check_microns(x, warn=True):
    """Check if neuron data is in microns.

    Returns either [True, None (=unclear), False]
    """
    if isinstance(x, core.NeuronList):
        check = np.array([_check_microns(n) for n in x])
        if all(check):
            return True
        elif any(check is False):
            return False
        return None

    if isinstance(x, core.TreeNeuron):
        u = getattr(x, 'units', None)
        if isinstance(u, (config.ureg.Quantity, config.ureg.Unit)):
            if not u.unitless:
                return u.to_compact().units == config.ureg.Unit('um')

    return None


def _make_dotprops(x, resample=None, parallel=False):
    """Try to make dotprops from input data."""
    if isinstance(resample, type(None)) or not resample:
        resample = robjects.r('NA')

    # If x is string it might be a flycircuit data type
    if isinstance(x, str) or x is None:
        try:
            flycircuit = importr('flycircuit')
            datadir = robjects.r('getOption("flycircuit.datadir")')[0]
        except BaseException:
            logger.error('R Flycircuit not found.')

    # Fallback are the flycircuit dotprops
    if x is None:
        if not os.path.isfile(datadir + '/dpscanon.rds'):
            raise ValueError('Unable to find default DPS database dpscanon.rds '
                             'in flycircuit.datadir. Please provide database '
                             'using target parameter.')
        logger.info('DPS database not explicitly provided. Loading local '
                    'FlyCircuit DB from dpscanon.rds')
        x = robjects.r(f'read.neuronlistfh("{datadir}/dpscanon.rds")')

    # If string, try to load the R object
    if isinstance(x, str):
        if x.startswith('http') or '/' in x:
            x = robjects.r(f'read.neuronlistfh("{x}")')
        else:
            x = robjects.r(f'read.neuronlistfh("{datadir}/{x}")')

    # If navis neuron convert to R neuron/list
    if isinstance(x, (core.TreeNeuron, core.NeuronList)):
        x = neuron2r(x)

    # At this point we are expecting an R object
    if 'rpy2' not in str(type(x)):
        raise NotImplementedError(f'Unable to convert target of type {type(x)} into '
                                  'R dotprops.')

    # Make sure we have dotprops
    if 'dotprops' in cl(x):
        dps = x
    else:
        dotprops = robjects.r('dotprops')
        if not parallel:
            dps = dotprops(x, resample=resample, k=5)
        else:
            nlapply = robjects.r('nlapply')
            dps = nlapply(x, dotprops,
                          resample=resample, k=5,
                          **{'.parallel': True, '.progress': 'none'})

    return dps


class NBLASTresults:
    """Class holding NBLAST results and wrappers that allow easy plotting.

    Attributes
    ----------
    results :   pandas.Dataframe
                Contains top N results.
    sc :        Robject
                Contains original RNblast forward scores.
    scr :       Robject
                Original R Nblast reverse scores (Top N only).
    query :     R object
                The original query neurons.
    param :     dict
                Contains parameters used for nblasting.
    target :    R robject
                The original target neurons.
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

    def __init__(self, results, sc, scr, query, target, nblast_param):
        """ Init function."""
        self.results = results  # this is pandas Dataframe holding top N results
        self.sc = sc  # original Nblast forward scores
        self.scr = scr  # original Nblast reverse scores (Top N only)
        self.query = query  # the original query neuron(s)
        self.target = target  # the original target neurons
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
        """Plot nblast hits using ``navis.plot3d()``.

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
        """Retrieve dotproducts from DPS database (neuronlistfh) as neuronslist.

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
                bulk: bool = False,
                verbose: bool = False,
                **kwargs) -> Union['core.NeuronObject',
                                   'pd.DataFrame',
                                   'np.ndarray']:
    """Transform 3D data between template brains.

    This is a simple wrapper for ``nat.templatebrains:xform_brain``.

    Notes
    -----
    For Neurons only: whether there is a change in units during transformation
    (e.g. nm -> um) is inferred by comparing distances between x/y/z coordinates
    before and after transform. This guesstimate is then used to convert
    ``.units`` and node radii (for TreeNeurons).

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
    bulk :      bool | int
                If True or number and input is NeuronList, will xform all
                coordinates in chunks (default=100k) instead of neuron-by-neuron.
                This can be ~2x faster (due to reduced overhead) is very memory
                intensive! If ``bulk`` is a number will process chunks of
                given size.
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
            if bulk:
                xf = x.copy()

                # Collect all spatial data
                xyz = []
                for n in xf:
                    if isinstance(n, core.TreeNeuron):
                        xyz.append(n.nodes[['x', 'y', 'z']].values)
                    elif isinstance(n, core.MeshNeuron):
                        xyz.append(n.vertices)
                    else:
                        raise TypeError("Don't know how to transform neuron of "
                                        f"type '{type(n)}'")

                    if n.has_connectors:
                        xyz.append(n.connectors[['x', 'y', 'z']].values)

                # Combine into big matrix
                xyz = np.vstack(xyz)

                if isinstance(bulk, bool):
                    bulk = 100e3

                # Split into chunks of max 100k coordinates
                n_chunks = math.ceil(xyz.shape[0] / bulk)
                chunks = np.array_split(xyz, n_chunks, axis=0)

                # Xform
                xyz_xf = []
                with config.tqdm(desc='Xforming coordinates',
                                 disable=config.pbar_hide,
                                 leave=config.pbar_leave,
                                 total=xyz.shape[0],
                                 ) as pbar:
                    for c in chunks:
                        c_xf = xform_brain(c,
                                           source=source,
                                           target=target,
                                           fallback=fallback,
                                           verbose=verbose,
                                           **kwargs)
                        xyz_xf.append(c_xf)
                        pbar.update(c_xf.shape[0])
                xyz_xf = np.vstack(xyz_xf)

                # Guess change in spatial units
                sample = min(10e3 / xyz.shape[0], .1)
                change, magnitude = _guess_change(xyz, xyz_xf, sample=sample)

                # Some clean-up
                del xyz, chunks

                # Map xformed coordinates back
                offset = 0
                for n in xf:
                    if isinstance(n, core.TreeNeuron):
                        n.nodes.loc[:, ['x', 'y', 'z']] = xyz_xf[offset:offset + n.n_nodes]
                        offset += n.n_nodes
                        # Fix radius based on our best estimate
                        if 'radius' in n.nodes.columns:
                            n.nodes['radius'] *= 10**magnitude
                    elif isinstance(n, core.MeshNeuron):
                        n.vertices = xyz_xf[offset:offset + n.vertices.shape[0]]
                        offset += n.vertices.shape[0]

                    if n.has_connectors:
                        n.connectors.loc[:, ['x', 'y', 'z']] = xyz_xf[offset:offset + n.n_connectors]
                        offset += n.n_connectors

                    # Make an educated guess as to whether the units have changed
                    if hasattr(n, 'units') and magnitude != 0:
                        if isinstance(n.units, (config.ureg.Unit, config.ureg.Quantity)):
                            n.units = (n.units / 10**magnitude).to_compact()

                return xf

            xf = []
            for n in config.tqdm(x, desc='Xforming',
                                 disable=config.pbar_hide,
                                 leave=config.pbar_leave):
                xf.append(xform_brain(n,
                                      source=source,
                                      target=target,
                                      fallback=fallback,
                                      verbose=verbose,
                                      **kwargs))
            return x.__class__(xf)

    if not isinstance(x, (core.BaseNeuron, np.ndarray, pd.DataFrame, core.Volume)):
        raise TypeError(f'Unable to transform data of type "{type(x)}"')

    if isinstance(x, core.BaseNeuron):
        xf = x.copy()
        # We will collate spatial data to reduce overhead from calling
        # R's xform_brain
        if isinstance(xf, core.TreeNeuron):
            xyz = xf.nodes[['x', 'y', 'z']].values
        elif isinstance(xf, core.MeshNeuron):
            xyz = xf.vertices
        else:
            raise TypeError(f"Don't know how to transform neuron of type '{type(xf)}'")

        # Add connectors
        if xf.has_connnectors:
            xyz = np.vstack([xyz, xf.connectors[['x', 'y', 'z']].values])

        # Do the xform of all spatial data
        xyz_xf = xform_brain(xyz,
                             source=source,
                             target=target,
                             fallback=fallback,
                             verbose=verbose,
                             **kwargs)

        # Guess change in spatial units
        change, magnitude = _guess_change(xyz, xyz_xf)

        # Map xformed coordinates back
        if isinstance(xf, core.TreeNeuron):
            xf.nodes.loc[:, ['x', 'y', 'z']] = xyz_xf[:xf.n_nodes]
            # Fix radius based on our best estimate
            if 'radius' in xf.nodes.columns:
                xf.nodes['radius'] *= 10**magnitude
        elif isinstance(xf, core.MeshNeuron):
            xf.vertices = xyz_xf[:xf.vertices.shape[0]]

        if xf.has_connectors:
            xf.connectors.loc[:, ['x', 'y', 'z']] = xyz_xf[-xf.connectors.shape[0]:]

        # Make an educated guess as to whether the units have changed
        if hasattr(xf, 'units') and magnitude != 0:
            if isinstance(xf.units, (config.ureg.Unit, config.ureg.Quantity)):
                xf.units = (xf.units / 10**magnitude).to_compact()

        # Fix soma radius if applicable
        if hasattr(xf, 'soma_radius') and isinstance(xf.soma_radius, numbers.Number):
            xf.soma_radius *= 10**magnitude

        return xf
    elif isinstance(x, pd.DataFrame):
        if any([c not in x.columns for c in ['x', 'y', 'z']]):
            raise ValueError('DataFrame must have x, y and z columns.')
        x = x.copy()
        x.loc[:, ['x', 'y', 'z']] = xform_brain(x[['x', 'y', 'z']].values.astype(float),
                                                source=source,
                                                target=target,
                                                fallback=fallback,
                                                verbose=verbose,
                                                **kwargs)
        return x
    elif isinstance(x, core.Volume):
        x = x.copy()
        x.vertices = xform_brain(x.vertices,
                                 source=source,
                                 target=target,
                                 fallback=fallback,
                                 verbose=verbose,
                                 **kwargs)
        return x
    elif x.shape[1] != 3:
        raise ValueError('Array must be of shape (N, 3).')

    if not isinstance(source, str):
        TypeError(f'Expected source of type str, got "{type(source)}"')

    if not isinstance(target, str):
        TypeError(f'Expected target of type str, got "{type(target)}"')

    # We need to convert numpy arrays explicitly
    if isinstance(x, np.ndarray):
        if rpy2.__version_vector__[0] < 3:
            x = numpy2ri.py2ro(x)
        else:
            x = numpy2ri.py2rpy(x)

    xf = nat_templatebrains.xform_brain(x,
                                        sample=source,
                                        reference=target,
                                        FallBackToAffine=fallback == 'AFFINE',
                                        Verbose=verbose,
                                        **kwargs)

    return np.array(xf)


def _guess_change(xyz_before, xyz_after, sample=.1):
    """Guess change in units during xforming."""
    if isinstance(xyz_before, pd.DataFrame):
        xyz_before = xyz_before[['x', 'y', 'z']].values
    if isinstance(xyz_after, pd.DataFrame):
        xyz_after = xyz_after[['x', 'y', 'z']].values

    # Select the same random sample of points in both spaces
    if sample <= 1:
        sample = int(xyz_before.shape[0] * sample)
    rnd_ix = np.random.choice(xyz_before.shape[0], sample, replace=False)
    sample_bef = xyz_before[rnd_ix, :]
    sample_aft = xyz_after[rnd_ix, :]

    # Get pairwise distance between those points
    dist_pre = pdist(sample_bef)
    dist_post = pdist(sample_aft)

    # Calculate how the distance between nodes changed and get the average
    # Note we are ignoring nans - happens e.g. when points did not transform.
    with np.errstate(divide='ignore', invalid='ignore'):
        change = dist_post / dist_pre
    # Drop infinite values in rare cases where nodes end up on top of another
    mean_change = np.nanmean(change[change < np.inf])

    # Find the order of magnitude
    magnitude = round(math.log10(mean_change))

    return mean_change, magnitude


def mirror_brain(x: Union['core.NeuronObject', 'pd.DataFrame', 'np.ndarray'],
                 template: str,
                 mirror_axis: Union[Literal['X'],
                                    Literal['Y'],
                                    Literal['Z']] = 'X',
                 transform: Union[Literal['warp'],
                                  Literal['affine'],
                                  Literal['flip']] = 'warp',
                 via: Optional[str] = None,
                 **kwargs) -> Union['core.NeuronObject',
                                    'pd.DataFrame',
                                    'np.ndarray']:
    """Mirror 3D object along given axixs.

    This is a simple wrapper for ``nat.templatebrains:mirror_brain``.

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
    via :           str | None
                    If a template brain (e.g. "FCWB") is given will first
                    transform coordinates into that space, then mirror and
                    transform back. Use this if there is no mirror registration
                    for the original template.
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

    # If we go via another brain space
    if via:
        # Xform to "via" space
        xf = xform_brain(x, source=template, target=via)
        # Mirror
        xfm = mirror_brain(xf,
                           template=via,
                           mirror_axis=mirror_axis,
                           transform=transform,
                           via=None,
                           **kwargs)
        # Xform back to original template space
        xfm_inv = xform_brain(xfm, source=via, target=template)
        return xfm_inv

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

    if not isinstance(x, (core.BaseNeuron, np.ndarray, pd.DataFrame)):
        raise TypeError(f'Unable to transform data of type "{type(x)}"')

    if isinstance(x, core.BaseNeuron):
        x = x.copy()
        if isinstance(x, core.TreeNeuron):
            x.nodes = mirror_brain(x.nodes,
                                   template=template,
                                   mirror_axis=mirror_axis,
                                   transform=transform,
                                   **kwargs)
        elif isinstance(x, core.MeshNeuron):
            x.vertices = mirror_brain(x.vertices,
                                      template=template,
                                      mirror_axis=mirror_axis,
                                      transform=transform,
                                      **kwargs)
        else:
            raise TypeError(f"Don't know how to transform neuron of type '{type(x)}'")

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

    if not isinstance(template, str):
        TypeError(f'Expected template of type str, got "{type(template)}"')

    # We need to convert numpy arrays explicitly
    if isinstance(x, np.ndarray):
        if rpy2.__version_vector__[0] < 3:
            x = numpy2ri.py2ro(x)
        else:
            x = numpy2ri.py2rpy(x)

    # It appears we need to fetch the brain template first -> passing the string
    # causes an error
    brain = robjects.r(template)

    xf = nat_templatebrains.mirror_brain(x,
                                         brain=brain,
                                         mirrorAxis=mirror_axis,
                                         transform=transform,
                                         **kwargs)

    return np.array(xf)


def get_brain_template_mesh(x: str) -> core.Volume:
    """Fetch brain surface model from ``nat.flybrains``, ``flycircuit`` or ``elmr``.

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
    """Fetch given neuropil from ``nat.flybrains``, ``flycircuit`` or ``elmr``.

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


def load_rda(fp: str, convert: bool = True):
    """Load and convert R data file (.rda).

    This function should be able to deal with the common data types used in the
    natverse.

    Parameters
    ----------
    fp :        str
                Filepath to rda file.
    convert :   bool | function
                If True, will attempt to convert data from R to Python. If
                ``convert`` is a function, we expect it to accept the raw R data
                and return the converted.

    Returns
    -------
    data

    """
    if not isinstance(fp, str):
        raise TypeError(f'Expected filepath as string, got "{type(fp)}"')

    if not os.path.isfile(fp):
        raise ValueError(f'File not found: {fp}')

    load = robjects.r('load')
    object_names = load(fp)
    data = robjects.r(object_names[0])

    if convert is True:
        data = data2py(data)
    elif callable(convert):
        data = convert(data)

    return data
