import unittest

import numpy as np

from streamlines.asarray import distance, length, reorient, resample, smooth


class TestAsArray(unittest.TestCase):

    def test_distance(self):
        """Tests the distance function"""

        left = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ])
        right = np.array([
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0]
        ])

        self.assertAlmostEqual(distance(left, right), 1.0)

    def test_length(self):
        """Tests the length function"""

        # For streamlines that have less than 2 points, the length is 0.
        self.assertAlmostEqual(length(np.empty((0, 3))), 0.0)
        self.assertAlmostEqual(length(np.array([[1.0, 0.0, 1.0]])), 0.0)

        # Euclidean length for other cases.
        streamline = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ])
        self.assertEqual(length(streamline), 1)

        streamline = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0]
        ])
        self.assertAlmostEqual(length(streamline), 3.0)

    def test_reorient(self):
        """Test the reorient function"""

        streamline = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0]
        ])
        template = np.array([
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
        ])

        reoriented_streamline = reorient(streamline, template)
        np.testing.assert_array_almost_equal(reoriented_streamline,
                                             streamline[::-1])

    def test_resample(self):
        """Test the resample function"""

        streamline = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0]
        ])

        resampled_streamline = resample(streamline, 1)
        self.assertEqual(len(resampled_streamline), 1)
        np.testing.assert_array_almost_equal(
            resampled_streamline[0],
            streamline[0])
        
        # Resampling to 2 points should yield the first and last 
        # points.
        resampled_streamline = resample(streamline, 2)
        self.assertEqual(len(resampled_streamline), 2)
        np.testing.assert_array_almost_equal(
            resampled_streamline[0],
            streamline[0])
        np.testing.assert_array_almost_equal(
            resampled_streamline[-1],
            streamline[-1])

        streamline = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        resampled_streamline = resample(streamline, 3)
        self.assertEqual(len(resampled_streamline), 3)
        np.testing.assert_array_almost_equal(
            resampled_streamline[1],
            [0.5, 0.0, 0.0])

    def test_smooth(self):
        """Test the smooth function"""

        # Streamlines with 0 and 1 point do not change.
        streamline = np.empty((0,3))
        smoothed_streamline = smooth(streamline)
        np.testing.assert_array_almost_equal(smoothed_streamline, streamline)

        streamline = np.array([[0, 0, 0]])
        smoothed_streamline = smooth(streamline)
        np.testing.assert_array_almost_equal(smoothed_streamline, streamline)

        # A straight line should remain straight.
        x = np.linspace(0, 100, 1000)
        yz = np.zeros((1000,))
        streamline = np.array([x, yz, yz]).T

        smoothed_streamline = smooth(streamline)
        np.testing.assert_array_almost_equal(smoothed_streamline, streamline)
