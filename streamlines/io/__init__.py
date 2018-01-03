import nibabel as nib
import numpy as np

import streamlines as sl


def load(filename):

    # Load the input streamlines.
    tractogram_file = nib.streamlines.load(filename)
    affine_to_rasmm = tractogram_file.header['voxel_to_rasmm']

    tractogram = tractogram_file.tractogram
    if not np.allclose(affine_to_rasmm, np.eye(4)):
        inv_affine = np.linalg.inv(affine_to_rasmm)
        tractogram = tractogram.apply_affine(inv_affine, False)

    streamlines = sl.Streamlines(tractogram.streamlines,
                                 tractogram.affine_to_rasmm)

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

    new_tractogram = nib.streamlines.Tractogram(
        [s.points for s in streamlines],
        affine_to_rasmm=streamlines.affine,
        data_per_point=data_per_point,
        data_per_streamline=data_per_streamline)

    hdr_dict = {'dimensions': reference_volume_shape,
                'voxel_sizes': voxel_size,
                'voxel_to_rasmm': streamlines.affine,
                'voxel_order': "".join(nib.aff2axcodes(streamlines.affine))}
    trk_file = nib.streamlines.TrkFile(new_tractogram, hdr_dict)
    trk_file.save(filename)
