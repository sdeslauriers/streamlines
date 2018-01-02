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

def smooth(array, knot_distance=10):
    """Smoothes the streamline using a b-spline"""

    # If the streamline has 0 or 1 point, it cannot be interpolated.
    nb_points = len(array)
    if nb_points <= 1:
        return array

    # Cubic splines are preferred, but require at least 4 points.
    degree = min(nb_points - 1, 3)

    # The segment length will be used to reparametrize the streamline
    # and also provides the length.
    segment_length = np.sqrt(np.sum((array[1:, :] - array[:-1, :]) ** 2, 1))
    cumulative_length = np.cumsum(segment_length)
    streamline_length = cumulative_length[-1]

    # Evenly placed knots.
    nb_knots = min(streamline_length // knot_distance, nb_points - degree + 1)
    nb_knots = int(max(nb_knots, 2))
    knots = np.concatenate((
        np.zeros((degree,)),
        np.arange(nb_knots),
        np.ones((degree,)) * (nb_knots - 1)))

    # The streamline is parametrized to a line with a length of nb_knots
    # points with the knots evenly spaced along this line.
    x = np.zeros((nb_points,))
    x[1:] = cumulative_length / streamline_length * (nb_knots - 1)

    # Smooth the streamline and return the new points.
    bspline = scipy.interpolate.make_lsq_spline(x, array, knots, degree)

    return bspline(x)

def transform(array, affine):
    """Applies an affine transform to a streamline"""

    padded = np.hstack((array, np.ones((len(array), 1))))
    return np.dot(padded, affine.T)[:, 0:3]
