import unittest

import numpy as np

import streamlines as sl


class TestStreamline(unittest.TestCase):
    """Tests the Streamline class"""

    def test_init(self):
        """Test the __init__ method"""

        # The default streamline has one point at the origin.
        streamline = sl.Streamline()
        self.assertTrue(isinstance(streamline, sl.Streamline))

        # A streamline can also be created using points.
        points = np.random.randn(10, 3)
        streamline = sl.Streamline(points)

        # A streamline cannot be created with a type that is not
        # convertible to a numpy array.
        points = ['abc', 'def']
        self.assertRaises(TypeError, sl.Streamline, points)

        # The points of a streamline must be represented by a 2D array.
        points = [1, 2, 3]
        self.assertRaises(ValueError, sl.Streamline, points)

        # The second dimension of the points must be 3 (the streamline
        # is in 3D space).
        points = [[1, 2], [3, 4]]
        self.assertRaises(ValueError, sl.Streamline, points)

    def test_contains(self):
        """Test the __contains__ magic method"""

        points = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        streamline = sl.Streamline(points)

        # All points of the streamline should be contained in the
        # streamline.
        for point in points:
            self.assertTrue(point in streamline)

        # Other points should not.
        self.assertFalse([1, 2, 4] in streamline)

    def test_eq(self):
        """Test the __eq__ magic method"""

        # If two streamlines have the same points, they are equal.
        points = [[1, 2, 3], [2, 3, 4]]
        streamline1 = sl.Streamline(points)
        points = [[1, 2, 3], [2, 3, 4]]
        streamline2 = sl.Streamline(points)
        self.assertEqual(streamline1, streamline2)

        # If two streamlines have even a small difference in points,
        # they are not equal.
        points = [[1, 2, 3], [2, 3, 4]]
        streamline1 = sl.Streamline(points)
        points = [[1 + 1e-6, 2, 3], [2, 3, 4]]
        streamline2 = sl.Streamline(points)
        self.assertNotEqual(streamline1, streamline2)

    def test_hash(self):
        """Test the __hash__ magic method"""
    
        # If two streamlines have the same points, they have the same hash.
        points = [[1, 2, 3], [2, 3, 4]]
        streamline1 = sl.Streamline(points)
        points = [[1, 2, 3], [2, 3, 4]]
        streamline2 = sl.Streamline(points)
        self.assertEqual(hash(streamline1), hash(streamline2))

        # Changing the points of a streamline slightly should yield different
        # hashes.
        points = [[1, 2, 3], [2, 3, 4]]
        streamline1 = sl.Streamline(points)
        points = [[1 + 1e-6, 2, 3], [2, 3, 4]]
        streamline2 = sl.Streamline(points)
        self.assertNotEqual(hash(streamline1), hash(streamline2))

    def test_iter(self):
        """Test the __iter__ magic method"""

        # Iterating over a streamline and iterating over its points
        # should yield the same result.
        points = [[1,2,3], [4, 5, 6], [7, 8, 9]]
        streamline = sl.Streamline(points)
        for streamline_point, point in zip(streamline, points):
            np.testing.assert_array_almost_equal(streamline_point, point)

        # Iterating over a streamline with no points should do nothing.
        streamline = sl.Streamline()
        for point in streamline:
            self.assertTrue(False)

    def test_len(self):
        """Test the __len__ magic method"""

        # The length of a streamline is its number of points.
        for n in range(10):
            streamline = sl.Streamline(np.random.rand(n, 3))
            self.assertEqual(len(streamline), n)

    def test_reversed(self):
        """Test the __reversed__ magic method"""

        # Reversed iterating over a streamline and iterating over its points
        # should yield the same result.
        points = [[1,2,3], [4, 5, 6], [7, 8, 9]]
        streamline = reversed(sl.Streamline(points))
        for streamline_point, point in zip(streamline, reversed(points)):
            np.testing.assert_array_almost_equal(streamline_point, point)

        # Reversed iterating over a streamline with no points should do nothing.
        streamline = sl.Streamline()
        for point in reversed(streamline):
            self.assertTrue(False)
