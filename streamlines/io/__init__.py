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

    return streamlines


def save(streamlines, filename, reference_volume_shape=(1, 1, 1),
         voxel_size=(1, 1, 1)):

    new_tractogram = nib.streamlines.Tractogram(
        [s.points for s in streamlines],
         affine_to_rasmm=streamlines.affine)

    hdr_dict = {'dimensions': reference_volume_shape,
                'voxel_sizes': voxel_size,
                'voxel_to_rasmm': streamlines.affine,
                'voxel_order': "".join(nib.aff2axcodes(streamlines.affine))}
    trk_file = nib.streamlines.TrkFile(new_tractogram, hdr_dict)
    trk_file.save(filename)
