import numpy as np

from streamlines.io import load


def add_parser(subparsers):

    # The information subparser.
    info_subparser = subparsers.add_parser(
        'info',
        description='Prints information about streamlines in a file.')
    info_subparser.add_argument(
        'input', metavar='input_file', type=str,
        help='STR The file that contains the streamlines. Can be of '
             'any file format supported by nibabel.')
    info_subparser.set_defaults(func=info)


def info(streamlines_filename):
    """Print information about a streamlines file

    Prints the number of streamlines in the file and the mean length of the
    streamlines.

    Args:
        streamlines_filename: The file whose info is printed.
    """

    # Load the input streamlines using the requested parameters.
    streamlines = load(streamlines_filename)

    # Print info about the streamlines.
    out = ''
    out += '\nNumber of streamlines: {}'.format(len(streamlines))
    out += '\nMean length: {:.2f}'.format(np.mean(streamlines.lengths))

    print(out)
