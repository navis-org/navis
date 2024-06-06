import numpy as np

from scipy import ndimage
from tqdm.auto import trange

from .. import core, config
from .base import TransformSequence, BaseTransform
from .h5reg import H5transform
from . import registry

logger = config.get_logger(__name__)


class ImageXformer:
    """Class to render images from a source to a target space.

    You should be able to gather most parameters from a TemplateBrain object.

    Parameters
    ----------
    transform :     Transform | Sequence
                    Transform or sequence of transforms to go
                    *from target to source* template space.
    target_dims :   (x, y, z) tuple
                    Dimensions of the target space, i.e. how many voxels in each
                    axis.
    target_spacing : (x, y z) tuple
                    Size of voxels in the target space.
    source_spacing : (x, y, z) tuple
                    Size of voxels in the source space.
    target_offset : (x, y, z) tuple, optional
                    Offset in physical units of the target space. This will
                    typically be (0, 0, 0).
    source_offset : (x, y, z) tuple, optional
                    Offset in physical units of the source space. This will
                    typically be (0, 0, 0).
    progress :      bool
                    Whether to show a progress bar for the rendering.

    """
    def __init__(
        self,
        transform,
        target_dims,
        target_spacing,
        source_spacing,
        target_offset=None,
        source_offset=None,
        progress: bool = True,
    ):
        if isinstance(transform, BaseTransform):
            transform = TransformSequence(transform)

        if isinstance(transform, list):
            transform = TransformSequence(*transform)

        self.transform = transform
        self.target_dims = target_dims
        self.target_spacing = target_spacing
        self.target_offset = target_offset
        self.source_spacing = source_spacing
        self.source_offset = source_offset
        self.progress = progress
        self.stepsize = int(1e6)
        self.interpolation_order = 1

        # See if we can pre-cache the transform(s)
        if self.progress:
            logger.info('Caching transform(s).')
        if isinstance(self.transform, H5transform):
            self.transform.full_ingest()
        elif isinstance(transform, TransformSequence):
            for t in transform.transforms:
                if isinstance(t, H5transform):
                    t.full_ingest()

    def render(self, image):
        """Render an image into the target space.

        Parameters
        ----------
        image : VoxelNeuron | (M, N, K) numpy array
                Image in source space to transform.

        Returns
        -------
        VoxelNeuron
                Image in target space.

        """
        if isinstance(image, core.VoxelNeuron):
            image = image.grid

        # Generate the coordinates for each pixel in the target space
        XX, YY, ZZ = np.meshgrid(
            range(self.target_dims[0]),
            range(self.target_dims[1]),
            range(self.target_dims[2]),
            indexing="ij",
        )
        # Generate a (3, M, N, K) grid
        ix_grid = np.array([XX, YY, ZZ])

        # Convert grid into (N * N * K, 3) voxel array (this is slow)
        ix_array_target = ix_grid.T.reshape(-1, 3)
        del ix_grid  # we don't need it anymore

        # Prepare the image to render onto
        img_xf = np.zeros(self.target_dims, dtype=image.dtype)

        # Render in batches to keep memory consumption down
        bar_format = "{desc}{percentage:3.0f}%|{bar}|[{elapsed}<{remaining}]"
        for i in trange(
            0,
            ix_array_target.shape[0],
            self.stepsize,
            desc="Rendering",
            disable=not self.progress,
            bar_format=bar_format,
        ):
            # Convert indices from voxel to physical coordinates we can transform
            coo_array_target = ix_array_target[i : i + self.stepsize] * self.target_spacing

            # Add physical offset if applicable
            if self.target_offset is not None:
                coo_array_target += self.target_offset

            # Project these coordinates from target to source space
            # This step is the the bottleneck since we are (potentially) xforming
            # millions of coordinates
            current_level = int(logger.level)
            try:
                logger.setLevel("ERROR")
                # Convert this batch of physical coordinates into physical source space
                coo_array_source = self.transform.xform(
                    coo_array_target, affine_fallback=True
                )
            except BaseException:
                raise
            finally:
                logger.setLevel(current_level)

            # Convert physical coordinates into voxels (note that we are NOT rounding here)
            ix_array_source = coo_array_source / self.source_spacing

            if self.source_offset is not None:
                ix_array_source += self.source_offset

            # Use target->source index mapping to interpolate the image
            # order=1 means linear interpolation (much faster)
            img_xf[
                ix_array_target[i : i + self.stepsize, 0],
                ix_array_target[i : i + self.stepsize, 1],
                ix_array_target[i : i + self.stepsize, 2],
            ] = ndimage.map_coordinates(image, ix_array_source.T, order=self.interpolation_order)

        return core.VoxelNeuron(img_xf, offset=self.target_offset, units=self.target_spacing)


def xform_image(img, source, target, progress=True):
    """Experimental function to render image into a target space.

    Parameters
    ----------
    img :       VoxelNeuron | (M, N, K) numpy array
                Image to transform.
    source :    str
                The source template space.
    target :    str
                The target template space.
    progress :  bool
                Whether to show a progress bar.

    Returns
    -------
    VoxelNeuron
                Image in target space.

    """
    if isinstance(img, core.VoxelNeuron):
        img = img.grid
    elif not isinstance(img, np.ndarray):
        raise TypeError(f'Expected VoxelNeuron or numpy array, got "{type(img)}"')
    elif img.ndim != 3:
        raise ValueError('Image must be 3D.')

    # Get the transform from target to source
    transform = registry.find_bridging_path(target, source)

    # We need info on the source and target spaces
    source = registry.find_template(source)
    target = registry.find_template(target)

    # Generate the ImageXformer
    xformer = ImageXformer(
        transform,
        target.dims,
        target.voxdims,
        source.voxdims,
        target_offset=np.asarray(target.boundingbox).reshape(3, 2)[:, 0],
        source_offset=np.asarray(source.boundingbox).reshape(3, 2)[:, 0],
        progress=progress
    )

    return xformer.xform(img)
