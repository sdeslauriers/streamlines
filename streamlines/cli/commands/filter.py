from streamlines.io import load
from streamlines.io import save


def add_parser(subparsers):

    # The filter subparser.
    filter_subparser = subparsers.add_parser(
        'filter',
        description='Removes streamlines from a file based on their features. '
                    'For example, remove all streamlines with a length below '
                    '50mm using --min-length 50.',
        help='Filters streamlines based on their features.')
    filter_subparser.add_argument(
        'input_filename', metavar='input_file', type=str,
        help='STR The file that contains the streamlines to filter. Can be of '
             'any file format supported by nibabel.')
    filter_subparser.add_argument(
        'output_filename', metavar='output_file', type=str,
        help='STR The file where the filtered streamlines will be saved. Can '
             'be of any file format supported by nibabel.')
    filter_subparser.add_argument(
        '--min-length', metavar='FLOAT', type=float,
        help='The minimum length of streamlines included in the output.')
    filter_subparser.set_defaults(func=filter)


def filter(input_filename, output_filename, **kwargs):
    """Removes streamlines from a file based on features

    Removes streamlines from a file based on their features. For example,
    remove all streamlines with a length below 50mm using --min-length 50.

    Args:
        input_filename: The file that contains the streamlines to filter.
        output_filename: The file where the remaining streamlines will be
            saved.

    """

    # Load the input_streamlines using the requested parameters.
    streamlines = load(input_filename)
    streamlines.filter(**kwargs)

    # Save the streamlines to the output file.
    save(streamlines, output_filename)
