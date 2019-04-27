import argparse

from streamlines.io import load
from streamlines.io import save


def add_parser(subparsers):

    # The smooth subparser.
    smooth_subparser = subparsers.add_parser(
        'smooth',
        description='Smooths streamlines using a least square b-spline. The '
                    'distance between knots controls the smoothness of the '
                    'output streamline with larger distances being smoother.',
        help='Smooths streamlines using a least square b-spline.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    smooth_subparser.add_argument(
        'input', metavar='input_file', type=str,
        help='STR The file that contains the streamlines to smooth. Can be of '
             'any file format supported by nibabel.')
    smooth_subparser.add_argument(
        'output', metavar='output_file', type=str,
        help='STR The file where the smoothed streamlines will be saved. Can '
             'be of any file format supported by nibabel.')
    smooth_subparser.add_argument(
        '--knot-distance', metavar='FLOAT', type=float, default=10.0,
        help='The distance between knots. Larger distance yield smoother '
             'streamlines.')
    smooth_subparser.set_defaults(func=smooth)


def smooth(input_filename, output_filename, **kwargs):
    """Smooths streamlines in a file

    Smooths streamlines using a least square b-spline. The distance between
    knots controls the smoothness of the output streamline with larger
    distances being smoother

    Args:
        input_filename: The file that contains the streamlines to smooth.
        output_filename: The file where the smoothed streamlines will be saved.

    """

    # Load the input streamlines using the requested parameters.
    streamlines = load(input_filename)
    streamlines.smooth(**kwargs)

    # Save the streamlines to the output file.
    save(streamlines, output_filename)
