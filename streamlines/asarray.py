import builtins

import numpy as np
import scipy


MIN_NB_POINTS = 10
KEY_INDEX = np.concatenate((range(5), range(-1, -6, -1)))


def hash(array):
    """Hashes an array that represents a streamline"""

    # Use just a few data points as hash key. I could use all the data of
    # the streamlines, but then the complexity grows with the number of
    # points.
    if len(array) < MIN_NB_POINTS:
        key = array
    else:
        key = array[KEY_INDEX]

    return builtins.hash(tuple(key.ravel()))


def length(streamline):
    """Measures the length of a streamline"""

    # The length of streamlines with fewer that 2 points is 0.
    if len(streamline) < 2:
        return 0.0

    diff = streamline[1:, :] - streamline[:-1, :]
    return np.sum(np.sqrt(np.sum((diff) ** 2, 1)))


def distance(left, right, nb_points=20):
    """Measures the distance between two streamlines"""

    # Resample the streamlines to the same number of points.
    left_resampled = resample(left, nb_points)
    right_resampled = resample(right, nb_points)

    # The distance between the streamlines is the distance between each
    # point.
    distance = 0.0
    for left_point, right_point in zip(left_resampled, right_resampled):
        distance += np.sqrt(np.sum((left_point - right_point) ** 2))

    return distance / nb_points


def reorient(streamline, template):
    """Reorients a streamline to follow a template"""

    # Find the orientation that is closest to the template.
    if distance(streamline, template) <= distance(streamline[::-1], template):
        return np.array(streamline)
    else:
        return np.array(streamline[::-1])


def resample(streamline, nb_points):
    """Resamples a streamline

    Resamples the streamline to a new number of points which may
    be greater (interpolation) or lower (subsampling) than the
    original number of points.

    """

    # If the streamline has no points, it is interpolated as all zeros.
    if len(streamline) == 0:
        return np.zeros((nb_points, 3))
    elif len(streamline) == 1:
        return np.tile(streamline[0], (nb_points, 1))

    # Cubic interpolation is preferred, but requires a minimum number of
    # points.
    if len(streamline) == 2:
        method = 'slinear'
    elif len(streamline) == 3:
        mehod = 'quadratic'
    else:
        method = 'cubic'
    
    # The x, y, and z coordinates are interpolated independently.
    t = np.linspace(0, 1, len(streamline))

    fx = scipy.interpolate.interp1d(t, streamline[:, 0], method) 
    fy = scipy.interpolate.interp1d(t, streamline[:, 1], method) 
    fz = scipy.interpolate.interp1d(t, streamline[:, 2], method) 

    nt = np.linspace(0, 1, nb_points)
    new_streamline = np.array([fx(nt), fy(nt), fz(nt)]).T

    return new_streamline
