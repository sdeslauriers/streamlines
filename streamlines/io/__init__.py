import nibabel as nib

import streamlines as sl

def load(filename):

    # Load the input streamlines.
    tractogram = nib.streamlines.load(filename).tractogram
    streamlines = sl.Streamlines(tractogram.streamlines,
                                 tractogram.affine_to_rasmm)

    return streamlines


def save(streamlines, filename):

    new_tractogram = nib.streamlines.Tractogram(
        [s.points for s in streamlines],
        affine_to_rasmm=streamlines.affine)
    nib.streamlines.save(new_tractogram, filename)
