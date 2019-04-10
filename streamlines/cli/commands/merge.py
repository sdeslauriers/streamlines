from functools import reduce
from operator import iadd

from streamlines.io import load
from streamlines.io import save


def add_parser(subparsers):

    merge_subparser = subparsers.add_parser(
        'merge',
        description='Merges several streamline files into one. The merging '
                    'operation does not verify if duplicate streamlines '
                    'exist.',
        help='Merges several streamline files into one.')
    merge_subparser.add_argument(
        'inputs', metavar='input_files', nargs='+',
        help='STR STR ... The files to be merged. Can be of any file format '
             'supported by nibabel.')
    merge_subparser.add_argument(
        'output', metavar='output_file', type=str,
        help='STR The file where the merged streamlines will be saved. Can '
             'be of any file format supported by nibabel.')
    merge_subparser.set_defaults(func=merge)


def merge(inputs, output):

    # Load all the input streamlines and merge them.
    streamlines_list = [load(i) for i in inputs]
    streamlines = reduce(iadd, streamlines_list)

    # Save the streamlines to the output file.
    save(streamlines, output)
