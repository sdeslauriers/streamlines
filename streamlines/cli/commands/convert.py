from streamlines.io import load
from streamlines.io import save


def add_parser(subparsers):

    convert_subparser = subparsers.add_parser(
        'convert',
        description='Converts a streamline file from one format to another.')
    convert_subparser.add_argument(
        'input_filename', metavar='input_file', type=str,
        help='STR The file that contains the streamlines. Can be of '
             'any file format supported by nibabel.')
    convert_subparser.add_argument(
        'output_filename', metavar='output_file', type=str,
        help='STR The name of the output file. Can be of any file format '
             'supported by nibabel.')
    convert_subparser.set_defaults(func=convert)


def convert(input_filename, output_filename):
    """Converts a streamlines file from one format to another

    Args:
        input_filename: The file to convert.
        output_filename: The output filename.
    """

    # Let load and save do all the work.
    streamlines = load(input_filename)
    save(streamlines, output_filename)
