import builtins
import collections

import numpy as np
import scipy.interpolate

from .asarray import distance, hash, length, reorient, resample, smooth
import streamlines.io


class Streamline(object):
    """A diffusion MRI streamline"""

    def __init__(self, points=None):

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
                'points must be a two dimensionnal array, not {} dimensionnal.'
                .format(points.ndim))

        if points.shape[1] != 3:
            raise ValueError(
                'points must have a shape of (N, 3), not {}.'
                .format(points.shape))

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


class Streamlines(object):
    """A sequence of dMRI streamlines"""

    def __init__(self, iterable=None, affine=np.eye(4)):

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
