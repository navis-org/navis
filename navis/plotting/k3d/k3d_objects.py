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

import k3d
import uuid
import warnings

import numpy as np
import pandas as pd
import trimesh as tm

from .._common import (
    connector_colors,
    resolve_cn_color,
    resolve_cn_layout,
    resolve_connectors,
    resolve_somata,
    use_radius,
)
from ..colors import vertex_colors, eval_color, color_to_int
from ..plot_utils import segments_to_coords
from ... import core, config, conversion

logger = config.get_logger(__name__)

__all__ = ["neuron2k3d", "scatter2k3d", "dotprops2k3d", "voxel2k3d", "volume2k3d"]


def neuron2k3d(x, colormap, settings):
    """Convert neurons to k3d objects."""
    if isinstance(x, core.BaseNeuron):
        x = core.NeuronList(x)
    elif not isinstance(x, core.NeuronList):
        raise TypeError('Unable to process data of type "{}"'.format(type(x)))

    # palette = kwargs.get("palette", None)
    # color_by = kwargs.get("color_by", None)
    # shade_by = kwargs.get("shade_by", None)
    # lg = kwargs.pop("legend_group", None)

    if settings.color_by is not None:
        if not settings.palette:
            raise ValueError(
                'Must provide `palette` (e.g. "viridis") argument '
                "if using `color_by`"
            )

        colormap = vertex_colors(
            x,
            by=settings.color_by,
            alpha=settings.alpha,
            use_alpha=False,
            palette=settings.palette,
            vmin=settings.vmin,
            vmax=settings.vmax,
            na="raise",
            color_range=255,
        )

    if not isinstance(settings.shade_by, type(None)):
        logger.warning("`shade_by` does not work with the k3d backend")

    cn_lay = resolve_cn_layout(settings)

    trace_data = []
    _radius_warned = False
    for i, neuron in enumerate(x):
        name = str(getattr(neuron, "name", neuron.id))
        color = colormap[i]

        try:
            # Try converting this neuron's ID
            neuron_id = str(neuron.id)
        except BaseException:
            # If that doesn't work generate a new ID
            neuron_id = str(str(uuid.uuid1()))

        showlegend = True
        label = neuron.label
        if (
            isinstance(settings.legend_group, dict)
            and neuron.id in settings.legend_group
        ):
            # Check if this the first entry for this legendgroup
            label = legendgroup = settings.legend_group[neuron.id]
            for d in trace_data:
                # If it is not the first entry, hide it
                if getattr(d, "legendgroup", None) == legendgroup:
                    showlegend = False
                    break
        elif isinstance(settings.legend_group, str):
            legendgroup = settings.legend_group
        else:
            legendgroup = neuron_id

        if use_radius(neuron, settings):
            # Warn once if more than 5% of nodes have missing radii
            if not _radius_warned:
                if (
                    (neuron.nodes.radius.fillna(0).values <= 0).sum() / neuron.n_nodes
                ) > 0.05:
                    logger.warning(
                        "Some skeleton nodes have radius <= 0. This may lead to "
                        "rendering artifacts. Set `radius=False` to plot skeletons "
                        "as single-width lines instead."
                    )
                    _radius_warned = True

            _neuron = conversion.tree2meshneuron(
                neuron,
                warn_missing_radii=False,
                radius_scale_factor=settings.get("linewidth", 1),
            )
            _neuron.connectors = neuron.connectors
            neuron = _neuron

            # See if we need to map colors to vertices
            if isinstance(color, np.ndarray) and color.ndim == 2:
                color = color[neuron.vertex_map]

        if not settings.connectors_only:
            if isinstance(neuron, core.Skeleton):
                trace_data += skeleton2k3d(
                    neuron,
                    label=label,
                    legendgroup=legendgroup,
                    showlegend=showlegend,
                    color=color,
                    settings=settings,
                )
            elif isinstance(neuron, core.Mesh):
                trace_data += mesh2k3d(
                    neuron,
                    label=label,
                    legendgroup=legendgroup,
                    showlegend=showlegend,
                    color=color,
                    settings=settings,
                )
            elif isinstance(neuron, core.Dotprops):
                trace_data += dotprops2k3d(
                    neuron,
                    label=label,
                    legendgroup=legendgroup,
                    showlegend=showlegend,
                    color=color,
                    settings=settings,
                )
            elif isinstance(neuron, core.Voxels):
                trace_data += voxel2k3d(
                    neuron,
                    label=label,
                    legendgroup=legendgroup,
                    showlegend=showlegend,
                    color=color,
                    settings=settings,
                )
            else:
                raise TypeError(f'Unable to plot neurons of type "{type(neuron)}"')

        # Add connectors (empty frame when they aren't wanted)
        connectors = resolve_connectors(neuron, settings)

        if settings.get("cn_color_by", None) is not None and not connectors.empty:
            # A colour per connector rather than per type, which leaves no groups
            # to loop over - and no way to draw stalks, since a k3d line carries
            # a single colour
            rgba, _ = connector_colors(connectors, cn_lay, color, settings)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                trace_data.append(
                    k3d.points(
                        positions=connectors[["x", "y", "z"]].values,
                        name=f"Connectors of {name}",
                        shader="flat",
                        point_size=settings.cn_size
                        if settings.cn_size
                        else cn_lay["size"] * 50,
                        colors=[color_to_int(c) for c in (rgba[:, :3] * 255).astype(int)],
                        opacity=settings.get("cn_alpha", 1),
                    )
                )
            connectors = connectors.iloc[:0]

        for j, this_cn in connectors.groupby("type"):
                c = resolve_cn_color(j, cn_lay, color, settings)
                c = color_to_int(eval_color(c, color_range=255))

                cn_label = f'{cn_lay.get(j, {"name": "connector"})["name"]} of {name}'

                if cn_lay["display"] == "circles" or not isinstance(
                    neuron, core.Skeleton
                ):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        trace_data.append(
                            k3d.points(
                                positions=this_cn[["x", "y", "z"]].values,
                                name=cn_label,
                                shader="flat",
                                point_size=settings.cn_size
                                if settings.cn_size
                                else cn_lay["size"] * 50,
                                color=c,
                                opacity=settings.get('cn_alpha', 1),
                            )
                        )
                elif cn_lay["display"] == "lines":
                    # Find associated treenodes
                    co1 = this_cn[["x", "y", "z"]].values
                    co2 = (
                        neuron.nodes.set_index("node_id")
                        .loc[this_cn.node_id.values, ["x", "y", "z"]]
                        .values
                    )

                    coords = np.array(
                        [
                            co
                            for seg in zip(
                                co1, co1, co2, co2, [[np.nan] * 3] * len(co1)
                            )
                            for co in seg
                        ]
                    )

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        trace_data.append(
                            k3d.line(
                                coords,
                                color=c,
                                name=cn_label,
                                width=settings.linewidth,
                                shader="thick",
                            )
                        )
                else:
                    raise ValueError(
                        f'Unknown display type for connectors "{cn_lay["display"]}"'
                    )

    return trace_data


def mesh2k3d(neuron, legendgroup, showlegend, label, color, settings):
    """Convert Mesh to k3d object."""
    # Skip empty neurons
    if neuron.n_vertices == 0:
        return []

    opacity = 1
    if isinstance(color, np.ndarray) and color.ndim == 2:
        if len(color) == len(neuron.vertices):
            color = [color_to_int(c) for c in color]
            color_kwargs = dict(colors=color)
        else:
            raise ValueError("Colors must match number of vertices for K3D meshes.")
    else:
        c = color_to_int(color)
        color_kwargs = dict(color=c)

        if len(color) == 4:
            opacity = color[3]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace_data = [
            k3d.mesh(
                vertices=neuron.vertices.astype("float32"),
                indices=neuron.faces.astype("uint32"),
                name=label,
                flat_shading=False,
                opacity=opacity,
                **color_kwargs,
            )
        ]

    return trace_data


def voxel2k3d(neuron, legendgroup, showlegend, label, color, settings):
    """Convert Voxels to k3d object."""
    # Skip empty neurons
    if min(neuron.shape) == 0:
        return []

    img = neuron.grid
    if img.dtype not in (np.float32, np.float64):
        img = img.astype(np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace_data = [
            k3d.volume(
                img.T,
                bounds=neuron.bbox.flatten(),
                interpolation=False,
            )
        ]

    return trace_data


def skeleton2k3d(neuron, legendgroup, showlegend, label, color, settings):
    """Convert skeleton (i.e. Skeleton) to plotly line plot."""
    if neuron.nodes.empty:
        logger.warning(f"Skipping Skeleton w/o nodes: {neuron.label}")
        return []
    elif neuron.nodes.shape[0] == 1:
        logger.warning(f"Skipping single-node skeleton: {neuron.label}")
        return []

    # `flat=True` ends every segment in a row of NaNs, which is the break k3d
    # needs. Per-node colors ride along on the same call so that they come back
    # in segment order too - lining them up afterwards would mean walking the
    # segments a second time and mapping every vertex through a dict.
    per_node_color = isinstance(color, np.ndarray) and color.ndim == 2
    if per_node_color:
        # k3d wants colors int-packed, which is what `color_to_int` does one at
        # a time. RGB is 0-255 here and the alpha channel (if any) is ignored.
        rgb = np.asarray(color).astype(np.int64)
        packed = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]
        coords, seg_colors = segments_to_coords(neuron, node_colors=packed, flat=True)
    else:
        coords, seg_colors = segments_to_coords(neuron, flat=True), None

    # What `flat=True` does not give us is the duplicated first and last
    # coordinate per segment - for reasons I don't quite understand k3d wants
    # those (possibly a bug). Duplicating a row is just repeating it, so this is
    # one pass with the two end rows bumped. The `+=` matters: a single-node
    # segment is its own first *and* last row, and needs three copies.
    ends = np.flatnonzero(np.isnan(coords[:, 0]))  # the break rows
    starts = np.concatenate(([0], ends[:-1] + 1))
    reps = np.ones(len(coords), dtype=np.intp)
    reps[starts] += 1
    reps[ends - 1] += 1
    coords = np.repeat(coords, reps, axis=0)

    color_kwargs = {}
    if seg_colors is not None:
        color_kwargs["colors"] = np.repeat(seg_colors, reps, axis=0)
    else:
        color_kwargs["color"] = color_to_int(color)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace_data = [
            k3d.line(
                coords,
                width=settings.linewidth,
                shader="thick",
                name=label,
                **color_kwargs,
            )
        ]

    # Add soma(s):
    for soma in resolve_somata(neuron, color, settings):
        sp = tm.primitives.Sphere(radius=soma.radius, subdivisions=2)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trace_data.append(
                k3d.mesh(
                    vertices=sp.vertices + soma.center.astype("float32"),
                    indices=sp.faces.astype("uint32"),
                    color=color_to_int(soma.color),
                    flat_shading=False,
                    name=f"soma of {label}",
                )
            )

    return trace_data


def scatter2k3d(x, **kwargs):
    """Convert DataFrame with x,y,z columns to plotly scatter plot."""
    c = eval_color(
        kwargs.get("color", kwargs.get("c", (100, 100, 100))), color_range=255
    )
    c = color_to_int(c)
    s = kwargs.get("size", kwargs.get("s", 1))
    name = kwargs.get("name", None)

    trace_data = []
    for scatter in x:
        if isinstance(scatter, pd.DataFrame):
            if not all([c in scatter.columns for c in ["x", "y", "z"]]):
                raise ValueError("DataFrame must have x, y and z columns")
            scatter = scatter[["x", "y", "z"]].values

        if not isinstance(scatter, np.ndarray):
            scatter = np.array(scatter)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trace_data.append(
                k3d.points(
                    positions=scatter, name=name, color=c, point_size=s, shader="dot"
                )
            )
    return trace_data


def dotprops2k3d(x, legendgroup, showlegend, label, color, settings):
    """Convert Dotprops to plotly graph object."""
    tn = x.to_skeleton(scale_vec=settings.dps_scale_vec)

    return skeleton2k3d(tn, legendgroup, showlegend, label, color, settings=settings)


def volume2k3d(x, colormap, settings):
    """Convert Volumes to plotly objects."""
    trace_data = []
    for i, v in enumerate(x):
        # Skip empty data
        if isinstance(v.vertices, np.ndarray):
            if not v.vertices.any():
                continue
        elif not v.vertices:
            continue

        name = getattr(v, "name", None)

        c = colormap[i]
        if len(c) == 4:
            opacity = c[3]
        else:
            opacity = 0.5

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trace_data.append(
                k3d.mesh(
                    vertices=v.vertices.astype("float32"),
                    indices=v.faces.astype("uint32"),
                    name=name,
                    color=color_to_int(c[:3]),
                    flat_shading=False,
                    opacity=opacity,
                )
            )

    return trace_data
