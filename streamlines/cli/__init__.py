from functools import reduce
import operator

import nibabel as nib
import numpy as np

from streamlines import Streamlines 
from streamlines.io import load, save


def filter(input, output, **kwargs):

    # Load the input streamlines using the requested parameters.
    streamlines = load(input)
    streamlines.filter(**kwargs)

    # Save the streamlines to the output file.
    save(streamlines, output)

def info(input):

    # Load the input streamlines using the requested parameters.
    streamlines = load(input)

    # Print info about the streamlines.
    out = ''
    out += '\nNumber of streamlines: {}'.format(len(streamlines))
    out += '\nMean length: {:.2f}'.format(np.mean(streamlines.lengths))

    print(out)

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
