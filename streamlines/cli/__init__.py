from functools import reduce
import operator

import nibabel as nib

from streamlines import Streamlines 
from streamlines.io import load, save


def filter(input, output, **kwargs):

    # Load the input streamlines using the requested parameters.
    streamlines = load(input)
    streamlines.filter(**kwargs)

    # Save the streamlines to the output file.
    save(streamlines, output)

def merge(inputs, output):

    # Load all the input streamlines and merge them.
    streamlines_list = [load(i) for i in inputs]
    streamlines = reduce(operator.iadd, streamlines_list)    

    # Save the streamlines to the output file.
    save(streamlines, output)

def reorient(input, output, **kwargs):

    # Load the input streamlines using the requested parameters.
    streamlines = load(input)
    streamlines.reorient(**kwargs)

    # Save the streamlines to the output file.
    save(streamlines, output)

def smooth(input, output, **kwargs):

    # Load the input streamlines using the requested parameters.
    streamlines = load(input)
    streamlines.smooth(**kwargs)

    # Save the streamlines to the output file.
    save(streamlines, output)
