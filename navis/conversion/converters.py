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
from .. import core, config
from ..plotting.plot_utils import make_tube

logger = config.logger


def tree2meshneuron(x: 'core.TreeNeuron',
                    tube_points: int = 8,
                    use_normals: bool = True) -> 'core.MeshNeuron':
    """Convert TreeNeuron to MeshNeuron.

    Uses the ``radius`` to convert skeleton to 3D tube mesh. Missing radii  are
    treated as zeros.

    Parameters
    ----------
    x :             TreeNeuron | NeuronList
                    Neuron to convert.
    tube_points :   int
                    Number of points making up the circle of the cross-section
                    of the tube.
    use_normals :   bool
                    If True will rotate tube along its curvature.

    Returns
    -------
    TreeNeuron

    """
    if not isinstance(x, core.TreeNeuron):
        raise TypeError(f'Expected TreeNeuron, got "{type(x)}"')

    # Note that we are treating missing radii as "0"
    radii_map = x.nodes.set_index('node_id').radius.fillna(0)

    if (radii_map <= 0).any():
        logger.warning('At least some radii are missing or <= 0. Mesh will '
                       'look funny.')

    # Map radii onto segments
    radii = [radii_map.loc[seg].values for seg in x.segments]
    co_map = x.nodes.set_index('node_id')[['x', 'y', 'z']]
    seg_points = [co_map.loc[seg].values for seg in x.segments]

    vertices, faces = make_tube(seg_points,
                                radii=radii,
                                tube_points=tube_points,
                                use_normals=use_normals)

    return core.MeshNeuron({'vertices': vertices, 'faces': faces},
                           connectors=x.connectors,
                           units=x.units, name=x.name, id=x.id)
