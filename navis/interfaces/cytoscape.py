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

""" Set of functions to interface with Cytoscape using its CyREST API. This
module requires py2cytoscape (https://github.com/cytoscape/py2cytoscape)
"""

import logging

import networkx as nx
import numpy as np
import pandas as pd
from py2cytoscape.data.cyrest_client import CyRestClient

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if len(logger.handlers) == 0:
    # Generate stream handler
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    # Create formatter and add it to the handlers
    formatter = logging.Formatter(
        '%(levelname)-5s : %(message)s (%(name)s)')
    sh.setFormatter(formatter)
    logger.addHandler(sh)


def get_client():
    """Initialises connection with Cytoscape and returns client."""
    return CyRestClient()


def get_navis_style():
    """Returns our default network style."""

    cy = get_client()

    all_styles = cy.style.get_all()
    s = cy.style.create('navis')

    # If the style already existed, return unchanged
    if 'navis' in all_styles:
        return s

    # If we created the style from scratch, apply some basic settings
    basic_settings = dict(
        # You can set default values as key-value pairs.

        NODE_FILL_COLOR='#FEC44F',
        NODE_SIZE=18,
        NODE_BORDER_WIDTH=7,
        NODE_BORDER_COLOR='#999999',
        NODE_LABEL_COLOR='#555555',
        NODE_LABEL_FONT_SIZE=14,
        NODE_LABEL_POSITION='S,NW,c,0.00,3.00',

        EDGE_WIDTH=2,
        EDGE_TRANSPARENCY=100,
        EDGE_CURVED=True,
        EDGE_BEND='0.728545744495502,-0.684997151948455,0.6456513365424503',
        EDGE_UNSELECTED_PAINT='#CCCCCC',
        EDGE_STROKE_UNSELECTED_PAINT='#333333',
        EDGE_TARGET_ARROW_SHAPE='DELTA',

        NETWORK_BACKGROUND_PAINT='#FFFFFF',
    )

    s.update_defaults(basic_settings)

    return s


def generate_network(x, layout='fruchterman-rheingold', apply_style=True,
                     clear_session=True):
    """Load network into Cytoscape.

    Parameters
    ----------
    x :             networkX Graph | pandas.DataFrame
                    Network to export to Cytoscape. Can be:
                      1. NetworkX Graph e.g. from navis.networkx (preferred!)
                      2. Pandas DataFrame. Mandatory columns:
                         'source','target','interaction'
    layout :        str | None, optional
                    Layout to apply. Set to ``None`` to not apply any.
    apply_style :   bool, optional
                    If True will apply a "navis" style to the network.
    clear_session : bool, optional
                    If True, will clear session before adding network.

    Returns
    -------
    cytoscape Network

    """
    # Initialise connection with Cytoscape
    cy = get_client()

    if layout not in cy.layout.get_all() + [None]:
        raise ValueError('Unknown layout. Available options: '
                         ', '.join(cy.layout.get_all()))

    # Clear session
    if clear_session:
        cy.session.delete()

    if isinstance(x, nx.Graph):
        n = cy.network.create_from_networkx(x)
    elif isinstance(x, np.ndarray):
        n = cy.network.create_from_ndarray(x)
    elif isinstance(x, pd.DataFrame):
        n = cy.network.create_from_dataframe(x)
    else:
        raise TypeError(f'Unable to generate network from data of type "{type(x)}"')

    if layout:
        # Apply basic layout
        cy.layout.apply(name=layout, network=n)

    if apply_style:
        # Get our default style
        s = get_navis_style()

        # Add some passthough mappings to the style
        s.create_passthrough_mapping(
            column='neuron_name', vp='NODE_LABEL', col_type='String')
        max_edge_weight = n.get_edge_column('weight').max()
        s.create_continuous_mapping(column='weight',
                                    vp='EDGE_WIDTH',
                                    col_type='Double',
                                    points=[{'equal': '1.0', 'greater': '1.0', 'lesser': '1.0', 'value': 1.0},
                                            {'equal': max_edge_weight / 3, 'greater': 1.0, 'lesser': max_edge_weight / 3, 'value': max_edge_weight}]
                                    )

        # Apply style
        cy.style.apply(s, n)

    return n
