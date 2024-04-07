#    This script is part of navis (http://www.github.com/navis-org/navis).
#    Copyright (C) 2017 Philipp Schlegel
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
#
#    You should have received a copy of the GNU General Public License
#    along

"""Experimental plotting module using pygfx."""

import uuid
import warnings

import pandas as pd
import numpy as np

import matplotlib.colors as mcl

from wgpu.gui.auto import WgpuCanvas, run

from ... import core, config, utils, conversion
from ..colors import eval_color, prepare_colormap, vertex_colors
from ..plot_utils import segments_to_coords

try:
    import pygfx as gfx
except ImportError:
    raise ImportError("Unable to import pygfx. Please install it first.")

__all__ = ["volume2gfx", "neuron2gfx", "dotprop2gfx", "voxel2gfx", "points2gfx"]

logger = config.get_logger(__name__)


class Plotter:
    ALLOWED_KWARGS = []

    def __init__(self, **kwargs):
        self.parse_kwargs(**kwargs)
        self.neurons = []
        self.volumes = []
        self.points = []
        self.visuals = []

    def __call__(self):
        """Plot objects."""
        return self.plot()

    def parse_kwargs(self, **kwargs):
        """Parse kwargs."""
        # Check for invalid kwargs
        invalid_kwargs = list(kwargs)
        for k in self.ALLOWED_KWARGS:
            # If this is a tuple of possible kwargs (e.g. "lw" or "linewidth")
            if isinstance(k, (tuple, list, set)):
                # Check if we have multiple kwargs for the same thing
                if len([kk for kk in k if kk in invalid_kwargs]) > 1:
                    raise ValueError(f'Please use only one of "{k}"')

                for kk in k:
                    if kk in invalid_kwargs:
                        # Make sure we always use the first kwarg
                        kwargs[k[0]] = kwargs.pop(kk)
                        invalid_kwargs.remove(kk)
            else:
                if k in invalid_kwargs:
                    invalid_kwargs.remove(k)

        if len(invalid_kwargs):
            warnings.warn(
                f"Unknown kwargs for {self.BACKEND} backend: {', '.join([f'{k}' for k in invalid_kwargs])}"
            )

        self.kwargs = kwargs

    def add_objects(self, x):
        """Add objects to the plot."""
        (neurons, volumes, points, visuals) = utils.parse_objects(x)
        self.neurons += neurons.neurons   # this is NeuronList
        self.volumes += volumes
        self.points += points
        self.visuals += visuals

        return self

    def plot(self):
        """Plot objects."""
        raise NotImplementedError


class GfxPlotter(Plotter):
    ALLOWED_KWARGS = {
        ("color", "c", "colors"),
        "cn_colors",
        ("linewidth", "lw"),
        "scatter_kws",
        "synapse_layout",
        "dps_scale_vec",
        "width",
        "height",
        "alpha",
        "radius",
        "soma",
        "connectors",
        "connectors_only",
        "palette",
        "color_by",
        "shade_by",
        "vmin",
        "vmax",
        "smin",
        "smax",
        "volume_legend",
    }
    BACKEND = "pygfx"

    def plot(self):
        """Generate the plot."""
        colors = self.kwargs.get('color', None)
        palette = self.kwargs.get("palette", None)

        self.neuron_cmap, volumes_cmap = prepare_colormap(
            colors,
            neurons=self.neurons,
            volumes=self.volumes,
            palette=palette,
            clusters=self.kwargs.get("clusters", None),
            alpha=self.kwargs.get("alpha", None),
            color_range=255,
        )


        # Generate scene
        scene = gfx.Group()
        scene.add(*neuron2gfx(core.NeuronList(self.neurons), **self.kwargs))
        scene.add(*volume2gfx(self.volumes, **self.kwargs))
        scene.add(*points2gfx(self.points, **self.kwargs))

        # Add background
        scene.add(
            gfx.Background(material=gfx.BackgroundMaterial([0, 0, 0]))
        )

        # Add light
        scene.add(gfx.AmbientLight())
        scene.add(gfx.DirectionalLight())

        camera = gfx.OrthographicCamera()
        camera.show_object(scene, scale=1.4)

        #renderer_svg = gfx.SvgRenderer(640, 480, "~/line.svg")
        #renderer_svg.render(scene, camera)

        canvas = WgpuCanvas(size=(1000, 800))
        renderer = gfx.WgpuRenderer(canvas, show_fps=True)
        controller = gfx.TrackballController(camera, register_events=renderer)
        gfx.show(scene,
                 canvas=canvas,
                 controller=controller,
                 camera=camera,
                 renderer=renderer)
                 #draw_function=lambda : poll_for_input(renderer, scene, camera))
        #disp = gfx.Display() # (camera=camera)
        #disp.show(scene)


#def poll_for_input(renderer, scene, camera):
#    renderer.render(scene, camera)


def volume2gfx(x, **kwargs):
    """Convert Volume(s) to pygfx visuals."""
    # Must not use make_iterable here as this will turn into list of keys!
    if not isinstance(x, (list, np.ndarray)):
        x = [x]

    # List to fill with vispy visuals
    visuals = []
    for i, v in enumerate(x):
        if not isinstance(v, core.Volume):
            raise TypeError(f'Expected navis.Volume, got "{type(v)}"')

        object_id = uuid.uuid4()

        if "color" in kwargs or "c" in kwargs:
            color = kwargs.get("color", kwargs.get("c", (0.95, 0.95, 0.95, 0.1)))
        else:
            color = getattr(v, "color", (0.95, 0.95, 0.95, 0.1))

        # Colors might be list, need to pick the correct color for this volume
        if isinstance(color, list):
            if all([isinstance(c, (tuple, list, np.ndarray)) for c in color]):
                color = color[i]

        if isinstance(color, str):
            color = mcl.to_rgb(color)

        color = np.array(color, dtype=float)

        # Add alpha
        if len(color) < 4:
            color = np.append(color, [0.1])

        if max(color) > 1:
            color[:3] = color[:3] / 255

        s = gfx.Mesh(
            gfx.Geometry(
                indices=v.faces.astype(np.int32, copy=False),
                positions=v.vertices.astype(np.float32, copy=False),
            ),
            gfx.MeshStandardMaterial(  # MeshBasicMaterial
                color=color, flat_shading=kwargs.get("shading", "smooth") == "flat"
            ),
        )

        # Add custom attributes
        s._object_type = "volume"
        s._volume_name = getattr(v, "name", None)
        s._object = v
        s._object_id = object_id

        visuals.append(s)

    return visuals


def neuron2gfx(x, color=None, **kwargs):
    """Convert a Neuron/List to pygfx visuals.

    Parameters
    ----------
    x :               TreeNeuron | MeshNeuron | Dotprops | VoxelNeuron | NeuronList
                      Neuron(s) to plot.
    color :           list | tuple | array | str
                      Color to use for plotting.
    colormap :        tuple | dict | array
                      Color to use for plotting. Dictionaries should be mapped
                      by ID. Overrides ``color``.
    connectors :      bool, optional
                      If True, plot connectors.
    connectors_only : bool, optional
                      If True, only connectors are plotted.
    by_strahler :     bool, optional
                      If True, shade neurites by strahler order.
    by_confidence :   bool, optional
                      If True, shade neurites by confidence.
    linewidth :       int, optional
                      Set linewidth. Might not work depending on your backend.
    cn_mesh_colors :  bool, optional
                      If True, connectors will have same color as the neuron.
    synapse_layout :  dict, optional
                      Sets synapse layout. For example::

                        {
                            0: {
                                'name': 'Presynapses',
                                'color': (255, 0, 0)
                                },
                            1: {
                                'name': 'Postsynapses',
                                'color': (0, 0, 255)
                                },
                            2: {
                                'name': 'Gap junctions',
                                'color': (0, 255, 0)
                                },
                            'display': 'lines'  # can also be 'circles'
                        }

    Returns
    -------
    list
                    Contains pygfx objects for each neuron.

    """
    if isinstance(x, core.BaseNeuron):
        x = core.NeuronList(x)
    elif isinstance(x, core.NeuronList):
        pass
    else:
        raise TypeError(f'Unable to process data of type "{type(x)}"')

    colors = color if color is not None else kwargs.get("c", kwargs.get("colors", None))
    palette = kwargs.get("palette", None)
    color_by = kwargs.get("color_by", None)
    shade_by = kwargs.get("shade_by", None)

    if not isinstance(color_by, type(None)):
        if not palette:
            raise ValueError(
                'Must provide `palette` (e.g. "viridis") argument '
                "if using `color_by`"
            )

        colormap = vertex_colors(
            x,
            by=color_by,
            alpha=kwargs.get("alpha", 1),
            palette=palette,
            vmin=kwargs.get("vmin", None),
            vmax=kwargs.get("vmax", None),
            na=kwargs.get("na", "raise"),
            color_range=1,
        )
    else:
        colormap, _ = prepare_colormap(
            colors,
            neurons=x,
            palette=palette,
            alpha=kwargs.get("alpha", None),
            clusters=kwargs.get("clusters", None),
            color_range=1,
        )

    if not isinstance(shade_by, type(None)):
        alphamap = vertex_colors(
            x,
            by=shade_by,
            use_alpha=True,
            palette="viridis",  # palette is irrelevant here
            vmin=kwargs.get("smin", None),
            vmax=kwargs.get("smax", None),
            na=kwargs.get("na", "raise"),
            color_range=1,
        )

        new_colormap = []
        for c, a in zip(colormap, alphamap):
            if not (isinstance(c, np.ndarray) and c.ndim == 2):
                c = np.tile(c, (a.shape[0], 1))

            if c.shape[1] == 4:
                c[:, 3] = a[:, 3]
            else:
                c = np.insert(c, 3, a[:, 3], axis=1)

            new_colormap.append(c)
        colormap = new_colormap

    # List to fill with vispy visuals
    visuals = []
    for i, neuron in enumerate(x):
        # Generate random ID -> we need this in case we have duplicate IDs
        object_id = uuid.uuid4()

        if kwargs.get("radius", False):
            # Convert and carry connectors with us
            if isinstance(neuron, core.TreeNeuron):
                _neuron = conversion.tree2meshneuron(neuron)
                _neuron.connectors = neuron.connectors
                neuron = _neuron

        neuron_color = colormap[i]
        if not kwargs.get("connectors_only", False):
            if isinstance(neuron, core.TreeNeuron):
                visuals += skeleton2gfx(neuron, neuron_color, object_id, **kwargs)
            elif isinstance(neuron, core.MeshNeuron):
                visuals += mesh2gfx(neuron, neuron_color, object_id, **kwargs)
            elif isinstance(neuron, core.Dotprops):
                visuals += dotprop2gfx(neuron, neuron_color, object_id, **kwargs)
            elif isinstance(neuron, core.VoxelNeuron):
                visuals += voxel2gfx(neuron, neuron_color, object_id, **kwargs)
            else:
                logger.warning(
                    f"Don't know how to plot neuron of type '{type(neuron)}'"
                )

        if (
            kwargs.get("connectors", False) or kwargs.get("connectors_only", False)
        ) and neuron.has_connectors:
            visuals += connectors2gfx(neuron, neuron_color, object_id, **kwargs)

    return visuals


def connectors2gfx(neuron, neuron_color, object_id, **kwargs):
    """Convert connectors to pygfx visuals."""
    cn_lay = config.default_connector_colors.copy()
    cn_lay.update(kwargs.get("synapse_layout", {}))

    visuals = []
    cn_colors = kwargs.get("cn_colors", None)
    for j in neuron.connectors.type.unique():
        if isinstance(cn_colors, dict):
            color = cn_colors.get(j, cn_lay.get(j, {}).get("color", (0.1, 0.1, 0.1)))
        elif cn_colors == "neuron":
            color = neuron_color
        elif cn_colors:
            color = cn_colors
        else:
            color = cn_lay.get(j, {}).get("color", (0.1, 0.1, 0.1))

        color = eval_color(color, color_range=1)

        this_cn = neuron.connectors[neuron.connectors.type == j]

        pos = (
            this_cn[["x", "y", "z"]]
            .apply(pd.to_numeric)
            .values.astype(np.float32, copy=False)
        )

        mode = cn_lay["display"]
        if mode == "circles" or isinstance(neuron, core.MeshNeuron):
            con = points2gfx(pos, color=color, size=cn_lay.get("size", 100))[0]
        elif mode == "lines":
            tn_coords = (
                neuron.nodes.set_index("node_id")
                .loc[this_cn.node_id.values][["x", "y", "z"]]
                .apply(pd.to_numeric)
                .values
            )

            # Zip coordinates and add a row of NaNs to indicate breaks in the
            # segments
            coords = np.hstack(
                (pos, tn_coords, np.full(pos.shape, fill_value=np.nan))
            ).reshape((len(pos) * 3, 3))
            coords = coords.astype(np.float32, copy=False)

            # Create line plot from segments
            con = gfx.Line(
                gfx.Geometry(positions=coords),
                gfx.LineMaterial(thickness=kwargs.get("linewidth", 1), color=color),
            )
            # method can also be 'agg' -> has to use connect='strip'
        else:
            raise ValueError(f'Unknown connector display mode "{mode}"')

        # Add custom attributes
        con._object_type = "neuron"
        con._neuron_part = "connectors"
        con._id = neuron.id
        con._name = str(getattr(neuron, "name", neuron.id))
        con._object_id = object_id
        con._object = neuron

        visuals.append(con)
    return visuals


def mesh2gfx(neuron, neuron_color, object_id, **kwargs):
    """Convert mesh (i.e. MeshNeuron) to pygfx visuals."""
    # Skip empty neurons
    if not len(neuron.faces):
        return []

    mat_color_kwargs = dict()
    obj_color_kwargs = dict()
    if isinstance(neuron_color, np.ndarray) and neuron_color.ndim == 2:
        if len(neuron_color) == len(neuron.vertices):
            obj_color_kwargs = dict(colors=neuron_color)
            mat_color_kwargs = dict(color_mode="vertex")
        elif len(neuron_color) == len(neuron.faces):
            obj_color_kwargs = dict(colors=neuron_color)
            mat_color_kwargs = dict(color_mode="face")
        else:
            mat_color_kwargs["color"] = neuron_color
    else:
        mat_color_kwargs["color"] = neuron_color

    m = gfx.Mesh(
        gfx.Geometry(
            indices=neuron.faces.astype(np.int32, copy=False),
            positions=neuron.vertices.astype(np.float32, copy=False),
            **obj_color_kwargs,
        ),
        gfx.MeshPhongMaterial(**mat_color_kwargs),
    )

    # Add custom attributes
    m._object_type = "neuron"
    m._neuron_part = "neurites"
    m._id = neuron.id
    m._name = str(getattr(neuron, "name", neuron.id))
    m._object_id = object_id
    m._object = neuron
    return [m]


def to_pygfx_cmap(color, N=256, gamma=1.0, fade=True):
    """Convert a given colour to a pygfx colormap."""
    # First force RGB
    stop = mcl.to_rgba(color, alpha=1)
    start = mcl.to_rgba(color if not fade else "k", alpha=0)

    # Use matplotlib to interpolate colors
    mpl_cmap = mcl.LinearSegmentedColormap.from_list(
        "", [start, stop], N=N, gamma=gamma
    )

    # Get the colors
    colormap_data = mpl_cmap(np.linspace(0, 1, N)).astype(np.float32, copy=False)

    # Important note:
    # It looks that as things stand now, pygfx expects the colormap to be only
    # rgb, not rgba. So we need to remove the alpha channel.
    colormap_data = colormap_data[:, :3]

    # Convert to vispy cmap
    return gfx.Texture(colormap_data, dim=1)


def voxel2gfx(neuron, neuron_color, object_id, **kwargs):
    """Convert voxels (i.e. VoxelNeuron) to pygfx visuals."""
    # TODOs:
    # - add support for custom color maps
    # - add support for other Volume materials (e.g. gfx.VolumeMipMaterial)

    # Similar to vispy, pygfx seems to expect zyx coordinate space
    grid = neuron.grid.T

    # Avoid boolean matrices here
    if grid.dtype == bool:
        grid = grid.astype(int)

    # Find the potential max value of the volume
    cmax = np.iinfo(grid.dtype).max

    # Initialize texture
    tex = gfx.Texture(grid, dim=3)

    # Initialize the volume
    vol = gfx.Volume(
        gfx.Geometry(grid=tex),
        gfx.VolumeRayMaterial(clim=(0, cmax), map=gfx.cm.cividis),
    )

    # Set scales and offset
    (
        vol.local.scale_x,
        vol.local.scale_y,
        vol.local.scale_z,
    ) = neuron.units_xyz.magnitude
    (vol.local.x, vol.local.y, vol.local.z) = neuron.offset

    # Add custom attributes
    vol._object_type = "neuron"
    vol._neuron_part = "neurites"
    vol._id = neuron.id
    vol._name = str(getattr(neuron, "name", neuron.id))
    vol._object_id = object_id
    vol._object = neuron

    return [vol]


def skeleton2gfx(neuron, neuron_color, object_id, **kwargs):
    """Convert skeleton (i.e. TreeNeuron) into pygfx visuals."""
    if neuron.nodes.empty:
        logger.warning(f"Skipping TreeNeuron w/o nodes: {neuron.id}")
        return []
    elif neuron.nodes.shape[0] == 1:
        logger.warning(f"Skipping single-node TreeNeuron: {neuron.label}")
        return []

    visuals = []
    if not kwargs.get("connectors_only", False):
        # Make sure we have one color for each node
        neuron_color = np.asarray(neuron_color).astype(np.float32, copy=False)

        # if neuron_color.ndim == 1:
        #    neuron_color = np.tile(neuron_color, (neuron.nodes.shape[0], 1))

        # Generate coordinates, breaks in segments are indicated by NaNs
        if neuron_color.ndim == 1:
            coords = segments_to_coords(neuron, neuron.segments)
        else:
            coords, vertex_colors = segments_to_coords(neuron, neuron.segments, node_colors=neuron_color)
            # `neuron_color` is now a list of colors for each segment; we have to flatten it
            # and add `None` to match the breaks
            vertex_colors = np.vstack([np.append(t, [[None] * t.shape[1]], axis=0) for t in vertex_colors]).astype(np.float32, copy=False)

        coords = np.vstack([np.append(t, [[None] * 3], axis=0) for t in coords])
        coords = coords.astype(np.float32, copy=False)

        # Create line plot from segments
        linewidth = kwargs.get("linewidth", kwargs.get("lw", 2))

        if neuron_color.ndim == 1:
            line = gfx.Line(
                gfx.Geometry(positions=coords),
                gfx.LineMaterial(thickness=linewidth, color=neuron_color),
            )
        else:
            line = gfx.Line(
                gfx.Geometry(positions=coords, colors=vertex_colors),
                gfx.LineMaterial(thickness=linewidth, color_mode='vertex'),
            )

        # Add custom attributes
        line._object_type = "neuron"
        line._neuron_part = "neurites"
        line._id = neuron.id
        line._name = str(getattr(neuron, "name", neuron.id))
        line._object = neuron
        line._object_id = object_id

        visuals.append(line)

        # Extract and plot soma
        soma = utils.make_iterable(neuron.soma)
        if kwargs.get("soma", True):
            # If soma detection is messed up we might end up producing
            # hundrets of soma which will freeze the session
            if len(soma) >= 10:
                logger.warning(
                    f"Neuron {neuron.id} appears to have {len(soma)}"
                    " somas. That does not look right - will ignore "
                    "them for plotting."
                )
            else:
                for s in soma:
                    # Skip `None` somas
                    if isinstance(s, type(None)):
                        continue

                    # If we have colors for every vertex, we need to find the
                    # color that corresponds to this root (or it's parent to be
                    # precise)
                    if isinstance(neuron_color, np.ndarray) and neuron_color.ndim > 1:
                        s_ix = np.where(neuron.nodes.node_id == s)[0][0]
                        soma_color = neuron_color[s_ix]
                    else:
                        soma_color = neuron_color

                    n = neuron.nodes.set_index("node_id").loc[s]
                    r = (
                        getattr(n, neuron.soma_radius)
                        if isinstance(neuron.soma_radius, str)
                        else neuron.soma_radius
                    )
                    s = gfx.Mesh(
                        gfx.sphere_geometry(
                            radius=r * 2, width_segments=16, height_segments=8
                        ),
                        gfx.MeshPhongMaterial(color=soma_color),
                    )
                    s.local.y = n.y
                    s.local.x = n.x
                    s.local.z = n.z

                    # Add custom attributes
                    s._object_type = "neuron"
                    s._neuron_part = "soma"
                    s._id = neuron.id
                    s._name = str(getattr(neuron, "name", neuron.id))
                    s._object = neuron
                    s._object_id = object_id

                    visuals.append(s)

    return visuals


def dotprop2gfx(x, neuron_color, object_id, **kwargs):
    """Convert dotprops(s) to vispy visuals.

    Parameters
    ----------
    x :             navis.Dotprops | pd.DataFrame
                    Dotprop(s) to plot.

    Returns
    -------
    list
                    Contains vispy visuals for each dotprop.

    """
    # Skip empty neurons
    if not len(x.points):
        return []

    # Generate TreeNeuron
    scale_vec = kwargs.get("dps_scale_vec", "auto")
    tn = x.to_skeleton(scale_vec=scale_vec)
    return skeleton2gfx(tn, neuron_color, object_id, **kwargs)


def points2gfx(x, **kwargs):
    """Convert points to vispy visuals.

    Parameters
    ----------
    x :             list of arrays
                    Points to plot.
    color :         tuple | array
                    Color to use for plotting.
    size :          int, optional
                    Marker size.

    Returns
    -------
    list
                    Contains pygfx visuals for points.

    """
    # TODOs:
    # - add support for per-vertex sizes and colors
    colors = kwargs.get(
        "color",
        kwargs.get("c", kwargs.get("colors", config.default_color)),
    )
    colors = np.asarray(eval_color(colors, force_alpha=True)).astype(
        np.float32, copy=False
    )

    visuals = []
    for p in x:
        object_id = uuid.uuid4()
        p = np.asarray(p).astype(np.float32, copy=False)

        # Make sure coordinates are c-contiguous
        if not p.flags["C_CONTIGUOUS"]:
            p = np.ascontiguousarray(p)

        con = gfx.Points(
            gfx.Geometry(positions=p),
            gfx.PointsMaterial(
                color=colors, size=kwargs.get("size", kwargs.get("s", 2))
            ),
        )

        # Add custom attributes
        con._object_type = "points"
        con._object_id = object_id

        visuals.append(con)

    return visuals
