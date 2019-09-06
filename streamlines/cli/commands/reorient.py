import argparse

from streamlines.io import load
from streamlines.io import save


def add_parser(subparsers):

    # The reorient subparser.
    reorient_subparser = subparsers.add_parser(
        'reorient',
        description='Reorients the streamlines so they all have the same '
                    'orientation (similar start/finish ROI). Reorient only '
                    'makes sense if the file contains a single bundle.',
        help='Reorients streamlines of a bundle.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    reorient_subparser.add_argument(
        'input_filename', metavar='input_file', type=str,
        help='STR The file that contains the streamlines to reorient. Can be '
             'of any file format supported by nibabel.')
    reorient_subparser.add_argument(
        'output_filename', metavar='output_file', type=str,
        help='STR The file where the reoriented streamlines will be saved. '
             'Can be of any file format supported by nibabel.')


def reorient(input_filename, output_filename, **kwargs):
    """Reorients streamlines in a file

    Reorients the streamlines so they all have the same orientation (similar
    start/finish ROI). Reorient only makes sense if the file contains a
    single bundle.

    Args:
        input_filename: The file that contains the streamlines to smooth.
        output_filename: The file where the smoothed streamlines will be saved.

    """

    # Load the input streamlines using the requested parameters.
    streamlines = load(input_filename)
    streamlines.reorient(**kwargs)

    # Save the streamlines to the output file.
    save(streamlines, output_filename)
