import nibabel as nib
import numpy as np

from nicoord import AffineTransform
from nicoord import CoordinateSystem
from nicoord import CoordinateSystemSpace
from nicoord import CoordinateSystemAxes
from nicoord import inverse

import streamlines as sl


# Streamlines in .trk format are always saved in native RAS space.
_ras_mm = CoordinateSystem(
    CoordinateSystemSpace.NATIVE, CoordinateSystemAxes.RAS)


def load(filename: str):
    """Loads the streamlines contained in a file

    Loads the streamlines contained in a .trk file. The streamlines are
    always loaded in a native RAS coordinate system. If the voxel_to_rasmm
    affine transform is present in the header, it is also loaded with
    the streamlines. This allows the transformation to voxel space using the
    transform_to method.

    Args:
        filename: The file name from which to load the streamlines. Only .trk
            files are supported.
    """

    # Load the input streamlines.
    tractogram_file = nib.streamlines.load(filename)
    affine_to_rasmm = tractogram_file.header['voxel_to_rasmm']

    # If there is a transform to RAS, invert it to get the transform to
    # voxel space.
    if not np.allclose(affine_to_rasmm, np.eye(4)):
        affine_to_voxel = np.linalg.inv(affine_to_rasmm)
        target = CoordinateSystem(
            CoordinateSystemSpace.VOXEL, CoordinateSystemAxes.RAS)
        transforms = [AffineTransform(_ras_mm, target, affine_to_voxel)]

    else:
        transforms = None

    tractogram = tractogram_file.tractogram
    streamlines = sl.Streamlines(tractogram.streamlines, _ras_mm, transforms)

    # Add the streamline point data to each streamline.
    for key, values in tractogram.data_per_point.items():
        for streamline, value in zip(streamlines, values):
            streamline.data[key] = value.T

    return streamlines


def save(streamlines, filename, reference_volume_shape=(1, 1, 1),
         voxel_size=(1, 1, 1)):
    """Saves streamlines to a trk file

    Saves the streamlines and their metadata to a trk file.

    Args:
        streamlines (streamlines.Streamlines): The streamlines to save.
        filename (str): The filename of the output file. If the file
            exists, it will be overwritten.
        reference_volume_shape (sequence, optional): A sequence with a
            shape of (3,) which contains the shape of the reference volume
            of the streamlines.
        voxel_size (sequence, optional): A sequence with a shape of (3,)
            which contains the voxel size of the reference volume.

    Examples:
        >>> import numpy as np
        >>> import streamlines as sl

        >>> streamlines = sl.Streamlines(np.random.randn(10, 100, 3))
        >>> sl.io.save(streamlines, 'test.trk')

    """

    # Concatenate all metadata into 2 dicts, one for streamline data and
    # the other for point data.
    data_per_point = {}
    data_per_streamline = {}

    # There might be no streamlines.
    if len(streamlines) > 0:
        for key in streamlines[0].data.keys():
            if streamlines[0].data[key].ndim == 2:
                data_per_point[key] = [s.data[key].T for s in streamlines]
            else:
                data_per_streamline[key] = [s.data[key] for s in streamlines]

    transforms = streamlines.transforms
    if streamlines.coordinate_system != _ras_mm:

        # If we are not in RAS, find the affine to native RAS. If it does not
        # exist, we have to stop because nibabel always wants to save in native
        # RAS.
        valid_transforms = [t for t in transforms if t.target == _ras_mm]

        if len(valid_transforms) == 0:
            raise ValueError(
                f'The streamlines are not in native RAS space and no '
                f'transforms to RAS are available. Cannot save to .trk '
                f'format.')

        # Note that we don't change the coordinate system. Nibabel does it on
        # save.
        affine = valid_transforms[0].affine
        affine_to_rasmm = affine

    else:

        # The points are already in the right coordinate system.
        affine_to_rasmm = np.eye(4)

        # If we are in RAS, we can still find the transform to native RAS as
        # the inverse of the inverse. It is ok if there is none.
        valid_transforms = [t for t in transforms if t.target == _ras_mm]

        if len(valid_transforms) == 0:
            affine = np.eye(4)
        else:
            affine = inverse(valid_transforms[0]).affine

    new_tractogram = nib.streamlines.Tractogram(
        [s.points for s in streamlines],
        affine_to_rasmm=affine_to_rasmm,
        data_per_point=data_per_point,
        data_per_streamline=data_per_streamline)

    hdr_dict = {'dimensions': reference_volume_shape,
                'voxel_sizes': voxel_size,
                'voxel_to_rasmm': affine,
                'voxel_order': "".join(nib.aff2axcodes(affine))}
    trk_file = nib.streamlines.TrkFile(new_tractogram, hdr_dict)
    trk_file.save(filename)
