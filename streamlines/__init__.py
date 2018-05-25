import builtins
import collections

import numpy as np
import scipy.interpolate

from .asarray import distance, hash, length, reorient, resample, smooth
from .asarray import transform
import streamlines.io


class Streamline(object):
    """A diffusion MRI streamline"""

    def __init__(self, points=None, data=None):
        """Diffusion MRI streamline

        An instance of the Streamline class represents a single diffusion MRI
        streamline. A streamline is formed by a sequence of points in 3D
        space (e.g. a (N, 3) numpy array).

        Args:
            points (optional): The points of the streamlines. Any structure
                which can be converted to a numpy array of floats with a shape
                of (N, 3) is accepted. If not supplied, an empty streamline (a
                streamline with 0 points) is returned.

            data (optional): The metadata associated with the streamline.
                Should be a dict whose values are arrays with a shape of (N, M)
                or (N,) where M is the number of points of the streamline. For
                (N, M), the data is associated with every point of the
                streamline whereas for (N,) the data is associated with the
                streamline itself.

        Examples:
            >>> import numpy as np
            >>> import streamlines as sl

            >>> empty_streamline = sl.Streamline()
            >>> empty_streamline.length
            0.0

            >>> streamline = sl.Streamline([[0, 0, 0], [100, 0, 0]])
            >>> streamline.length
            100.0

            >>> other_streamline = sl.Streamline(np.random.randn(100, 3))

        Raises:
            TypeError: If the ``points`` cannot be converted to a numpy array
                of floats.
            ValueError: If the numpy array resulting from ``points`` does not
                have a shape of (N, 3).

        """

        if points is None:
            points = np.empty((0, 3))
        else:
            try:
                points = np.array(points, dtype=float)
            except:
                raise TypeError(
                    'points must be convertible to a numpy array of floats.')

        if points.ndim != 2:
            raise ValueError(
                'points must be a two dimensional array, not {} dimensional.'
                .format(points.ndim))

        if points.shape[1] != 3:
            raise ValueError(
                'points must have a shape of (N, 3), not {}.'
                .format(points.shape))

        if data is None:
            data = {}

        self._data = data
        self._points = points

    def __contains__(self, point):
        """Verifies if a point is part of a streamline"""
        return next((True for p in self._points if np.all(p == point)), False)

    def __eq__(self, other):
        return hash(self._points) == hash(other._points)

    def __getitem__(self, key):
        return self._points[key]

    def __hash__(self):
        return hash(self._points)

    def __iter__(self):
        return iter(self._points)

    def __len__(self):
        return len(self._points)

    def __reversed__(self):
        return reversed(self._points)

    def __str__(self):
        return 'streamline: {} points'.format(len(self))

    @property
    def data(self):
        return self._data

    @property
    def points(self):
        return self._points.copy()

    @property
    def length(self):
        return length(self._points)

    def distance(left, right, nb_points=20):
        return distance(left._points, right._points, nb_points)

    def reorient(self, template):
        """Reorients a streamline using a template streamline"""
        self._points = reorient(self._points, template._points)
        return self

    def resample(self, nb_points):
        self._points = resample(self._points, nb_points)

    def reverse(self):
        """Reverses the order of the points of the streamline"""
        self._points = self._points[::-1]

    def smooth(self, knot_distance=10):
        """Smooths a streamline in place"""
        self._points = smooth(self._points, knot_distance)
        return self

    def transform(self, affine):
        """Applies an affine transformation to a streamline"""
        self._points = transform(self._points, affine)
        return self


class Streamlines(object):
    """A sequence of diffusion MRI streamlines"""

    def __init__(self, iterable=None, affine=np.eye(4)):
        """Sequence of diffusion MRI streamlines

           An instance of the Streamlines class represents a group of diffusion
           MRI streamlines.

           Args:
                iterable (optional): An iterable that contains the individual
                    streamlines. Each item in the iterable must be convertible
                    to a Streamline instance, i.e. it must be convertible to a
                    (N, 3) array of float. See the Streamline class for more
                    details.

        """

        self.affine = affine

        self._items = []
        if iterable is not None:
            self._items = [Streamline(i) for i in iterable]

    def __iadd__(self, other):
        self._items += other._items
        return self

    def __contains__(self, streamline):
        return streamline in self._items

    def __getitem__(self, key):
        """Get a single streamline or a subset of streamlines"""

        # For numpy arrays, we support only bool for now.
        if isinstance(key, np.ndarray):

            if key.dtype is np.dtype(bool):

                if key.ndim != 1 or len(key) != len(self):
                    raise ValueError('When using a numpy of bool to get '
                                     'streamlines, the length of the array '
                                     'must math the number of streamlines '
                                     '({} != {}).'.format(len(key), len(self)))

                streamlines = [s for keep, s in zip(key, self) if keep]
                return Streamlines(streamlines, self.affine)

            else:
                raise TypeError('Only numpy arrays of bool can be '
                                'used as indices to Streamlines, not arrays '
                                'of {}.'.format(key.dtype))

        return self._items[key]

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def __reversed__(self):
        return reversed(self._items)

    def __str__(self):
        return str(self._items)

    @property
    def lengths(self):
        """Returns the length of all streamlines"""
        return [s.length for s in self._items]

    def append(self, streamline):
        """Append a streamline to the sequence"""
        self._items.append(streamline)

    def filter(self, min_length=None):

        if min_length is not None:
            self._items = [i for i in self._items if i.length >= min_length]

        return self

    def reorient(self, template=None):

        if template is None:
            template = self._items[0]

        for streamline in self:
            streamline.reorient(template)

    def reverse(self):
        """Reverses the order of points of the streamlines"""
        for streamline in self:
            streamline.reverse()

    def smooth(self, knot_distance=10):
        """Smooth streamlines in place"""

        for streamline in self._items:
            streamline.smooth(knot_distance)

        return self

    def transform(self, affine):
        """Applies an affine transform to the streamlines"""
        for streamline in self:
            streamline.transform(affine)
