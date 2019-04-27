from streamlines.io import load, save


def filter(input, output, **kwargs):

    # Load the input streamlines using the requested parameters.
    streamlines = load(input)
    streamlines.filter(**kwargs)

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
