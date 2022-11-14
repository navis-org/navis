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
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.

"""Interface with VirtualFlyBrain.org using the vfb_connect library."""

from textwrap import dedent

try:
    from vfb_connect.cross_server_tools import VfbConnect
    vc = VfbConnect(neo_endpoint='http://pdb.v4.virtualflybrain.org', neo_credentials=('neo4j', 'vfb'))
except ImportError:
    msg = dedent("""
          vfb_connect library not found. Please install using pip:

                pip install vfb_connect

          """)
    raise ImportError(msg)
except BaseException:
    raise

import requests

import numpy as np
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO

from .. import config, utils, core

logger = config.get_logger(__name__)


# For convenience some shorthands for datasets
DATABASES = {'hemibrain:1.0.1': 'neuprint_JRC_Hemibrain_1point0point1',
             'hemibrain:v1.1': 'neuprint_JRC_Hemibrain_1point1',
             'hemibrain:v1.1.1': 'neuprint_JRC_Hemibrain_1point1',  # these are effectively the same
             'hemibrain': 'neuprint_JRC_Hemibrain_1point1',         # these are effectively the same
             'FAFB': 'catmaid_FAFB'}


def get_vfb_ids(x, database=None, attach=True):
    """Fetch VFB ID for given neuron(s).

    Parameters
    ----------
    x :         Neuron/List | int | str | iterable
                Neurons, IDs or a list thereof (see Examples). If neuron(s) will
                assume that their IDs correspond to the `accession` (=external)
                ID in VFB.
    database :  str
                Database `x` belongs to. This can be, for example:
                    - 'FAFB'
                    - 'neuronbridge'
                    - 'hemibrain:v1.0.1'
                    - 'hemibrain:v1.1.0'
                    - 'FlyCircuit'
                Use `vfb.vc.neo_query_wrapper.get_dbs()` to get a full list.

    Returns
    -------
    pandas.DataFrame

    Examples
    --------

    >>> import navis
    >>> from navis.interfaces import vfb
    >>> # Example neurons are from the hemibrain dataset
    >>> nl = navis.example_neurons(2)
    >>> # Let's see if VFB knows their IDs ("body IDs" in neuPrint speak)
    >>> vfb_ids = vfb.get_vfb_ids(nl)
    >>> vfb_ids
                                            1734350788    1734350908
    neuronbridge                          VFB_jrchjtdf  VFB_jrchjtdb
    neuprint_JRC_Hemibrain_1point0point1  VFB_jrch078z  VFB_jrch075v
    neuprint_JRC_Hemibrain_1point1        VFB_jrchjtdf  VFB_jrchjtdb

    """
    database = DATABASES.get(database, database)

    if isinstance(x, (core.NeuronList, core.BaseNeuron)):
        x = x.id

    x = utils.make_iterable(x).astype(str)

    res = vc.neo_query_wrapper.xref_2_vfb_id(acc=x, db=database)

    res = {keys: {v['db']: v['vfb_id'] for v in values} for keys, values in res.items()}

    return pd.DataFrame.from_records(res)


def get_vfb_meta(x, database, raw=False):
    """Fetch VFB's meta data for given neuron(s).

    Parameters
    ----------
    x :         Neuron/List | int | str | iterable
                Neurons, IDs or a list thereof (see Examples). If neuron(s) will
                assume that their IDs correspond to the `accession` (=external)
                ID in VFB.
    database :  str
                Database `x` belongs to. This can be, for example:
                    - 'FAFB'
                    - 'neuronbridge'
                    - 'hemibrain:v1.0.1'
                    - 'hemibrain:v1.1.0'
                    - 'FlyCircuit'
                Use `vfb.vc.neo_query_wrapper.get_dbs()` to get a full list.
    raw :       bool
                If True, will return the raw JSON (list of nested dicts).
                If False (default) will return a filtered, flattened DataFrame
                with the most relevant info.

    Returns
    -------
    meta :      pandas.DataFrame

    Examples
    --------


    """
    vfb_ids = get_vfb_ids(x, database=database)

    if isinstance(x, (core.NeuronList, core.BaseNeuron)):
        ids = x.id
    else:
        ids = x

    ids = utils.make_iterable(ids).astype(str)

    miss = ~np.isin(ids, vfb_ids.columns)
    if any(miss):
        logger.warning(f'Did not find VFB meta data for {ids[miss]}')

    to_fetch = vfb_ids.iloc[0][ids[~miss]].values.astype(str)

    json = vc.neo_query_wrapper.get_TermInfo(to_fetch.tolist())

    if raw:
        return json

    meta = []

    for n in json:
        # Some core info
        label = n['term']['core']['label']
        short = n['term']['core']['short_form']
        desc = n['term']['description']
        comment = n['term']['comment']

        # Xref info
        acc = list(set([ref['accession'] for ref in n['xrefs']]))
        sources = [ref['site']['label'] for ref in n['xrefs']]

        # Get available templates
        temps = [im['image']['template_anatomy']['label'] for im in n['channel_image']]
        temps = list(set(temps))

        # Find neurotransmitter (not sure this covers all the terms)
        for trans in ('Cholinergic', 'GABAergic', 'Glutamatergic',
                      'Serotonergic', 'Octopaminergic', 'NA'):
            if trans in n['term']['core']['types']:
                break

        meta.append([label, short, trans, acc, sources, desc, comment, temps])

    meta = pd.DataFrame(meta, columns=['label', 'vfb_id', 'transmitter',
                                       'accession_id', 'sources', 'description',
                                       'comments', 'templates'])

    return meta


def get_skeletons(x, template=None, max_threads=5, verbose=True):
    """Fetch skeletons for given VFB IDs.

    Parameters
    ----------
    x :             str | list thereof
                    VFB IDs to fetch skeletons for.
    template :      str, optional
                    The template space the skeletons should be in. Use
                    `vfb.vc.neo_query_wrapper.get_templates()` to get a fulll
                    list of available templates. Not specifying a template will
                    return all available images.
    max_threads :   int, optional
                    Max number of parallel requests.

    Returns
    -------
    NeuronList

    Examples
    --------
    Fetch skeletons based on their VFB ID (e.g. copy-pasted from website)

    >>> from navis.interfaces import vfb
    >>> n = vfb.get_skeletons('VFB_jrchjtdf', template='JRC2018Unisex')

    Fetch all available skeletons for the "adult antennal lobe projection neuron DA1 lPN"
    class:

    >>> from navis.interfaces import vfb
    >>> nl = vfb.get_skeletons('FBbt_00067363')

    """
    x = utils.make_iterable(x).astype(str)

    # Grab the meta data
    images = _query_images(x, template=template, verbose=verbose)

    if not images:
        return

    with ThreadPoolExecutor(max_workers=1 if not max_threads else max_threads) as executor:
        futures = {}
        for img in images:
            f = executor.submit(_fetch_single_skeleton, img)
            futures[f] = img['template_channel']['short_form']

        nl = []
        with config.tqdm(desc='Downloading',
                         total=len(futures),
                         leave=config.pbar_leave,
                         disable=len(futures) == 1 or config.pbar_hide) as pbar:
            for f in as_completed(futures):
                pbar.update(1)
                try:
                    nl.append(f.result())
                except Exception as exc:
                    print(f'{futures[f]} generated an exception:', exc)

    return core.NeuronList(nl)


def _fetch_single_skeleton(img, **kwargs):
    """Fetch a single skeleton. Intended to be wrapped by ThreadPoolExecutor."""
    # Fetch the SWC table (oddly this seems to stall occasionally)
    r = requests.get(img['image_folder'] + '/volume.swc')
    swc = pd.read_csv(StringIO(r.content.decode()),
                      sep=' ', comment='#', header=None)
    swc.columns = ['node_id', 'label',
                   'x', 'y', 'z', 'radius',
                   'parent_id']
    # Turn into a TreeNeuron
    n = core.TreeNeuron(swc,
                        name=img['label'],
                        id=img['short_form'],
                        vfb_meta=img,
                        **kwargs)

    n._register_attr(name='template', value=img['template_anatomy']['label'])

    return n


def _query_images(short_forms, template, verbose=True):
    """Internal function to search for images."""
    short_forms = utils.make_iterable(short_forms).astype(str)

    # Split short forms into individuals, classes and others
    individuals = [s for s in short_forms if s.startswith('VFB')]
    classes = [s for s in short_forms if s.startswith('FBbt')]
    other = [s for s in short_forms if s not in individuals and s not in classes]

    # Get anything that matches the term(s)
    inds = []
    if individuals or other:
        inds += vc.neo_query_wrapper.get_anatomical_individual_TermInfo(short_forms=individuals + other)
    if classes or other:
        for hit in vc.neo_query_wrapper.get_TermInfo(short_forms=classes + other):
            inds += hit.get('anatomy_channel_image', [])

    # First get all images
    images = []
    for ind in inds:
        if 'term' in ind:
            data = ind['term']['core']
        elif 'anatomy' in ind:
            data = ind['anatomy']

        if not 'has_image' in data['types']:
            continue

        this_images = ind.get('channel_image', [])
        if isinstance(this_images, dict):
            this_images = [this_images]

        for im in this_images:
            images.append(im['image'])
            # Also carry some of the other info over
            images[-1]['label'] = data['label']
            images[-1]['short_form'] = data['short_form']

    # Gather all available templates
    templates = list(set([i['template_anatomy']['label'] for i in images]))

    if template:
        matches = [i for i in images if i['template_anatomy']['label'] == template]
    else:
        matches = images

    # Try giving some useful feedback why the search failed
    if verbose:
        if not matches:
            if not inds:
                logger.info(f'Searching for {short_forms} did not return any results')
            elif not images:
                logger.info(f'Matches for {short_forms} do not appear to have any associated '
                            'images')
            elif template and template not in templates:
                logger.info(f'Matches for {short_forms} do not have any images in template '
                            f'{template}. Try one of the following instead: '
                            f'{", ".join(templates)}')
        elif template:
            logger.info(f'Searching for {short_forms} returned {len(matches)} '
                        f'images in template {template}')
        else:
            logger.info(f'Searching for {short_forms} returned {len(matches)} '
                        f'images in {len(templates)} templates: '
                        f'{", ".join(templates)}')

    return matches
