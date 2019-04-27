import os
import tempfile
import unittest

import numpy as np

from streamlines import Streamlines
from streamlines.cli import filter, reorient
from streamlines.cli.commands.info import info
from streamlines.cli.commands.merge import merge
from streamlines.io import load, save


class TestCLI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Generates streamlines files used to test the CLI"""

        cls.test_dir = tempfile.TemporaryDirectory()

        # A file with all short streamlines.
        streamlines = Streamlines([
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        ])
        filename = os.path.join(cls.test_dir.name, 'short.trk')
        save(streamlines, filename)

        # A file with a single random streamline.
        streamlines = Streamlines([
            np.random.randn(100,3)
        ])
        filename = os.path.join(cls.test_dir.name, 'random.trk')
        save(streamlines, filename)

        # A file with no streamlines.
        streamlines = Streamlines()
        filename = os.path.join(cls.test_dir.name, 'empty.trk')
        save(streamlines, filename)

        # A file with a bundle.
        points = np.array([
            np.linspace(0, 100, 1000),
            np.zeros((1000,)),
            np.zeros((1000,))]).T
        points_list = [points + np.random.randn(*points.shape)
                       for _ in range(100)]

        streamlines = Streamlines(points_list)
        filename = os.path.join(cls.test_dir.name, 'bundle.trk')
        save(streamlines, filename)

        # Also save the bundle with flipped orientations.
        flip = [np.random.rand() < 0.5 for _ in points_list]
        flip[0] = False
        points_list = [p[::-1] if f else p for f, p in zip(flip, points_list)]
        streamlines = Streamlines(points_list)
        filename = os.path.join(cls.test_dir.name, 'bundle-flipped.trk')
        save(streamlines, filename)

    @classmethod
    def tearDownClass(cls):
        """Removes the streamlines file used to test the CLI"""

        cls.test_dir.cleanup()

    def test_filter(self):
        """Test the filter command of the CLI"""

        # Filter the short streamlines. The result should be an empty tractogram.
        output = os.path.join(self.test_dir.name, 'test-filter-1.trk')
        filter(
            os.path.join(self.test_dir.name, 'short.trk'),
            output,
            min_length=1.1)
        streamlines = load(output)
        self.assertEqual(len(streamlines), 0)

    def test_info(self):
        """Test the info command of the CLI"""

        # Show the info of the bundle.
        info(os.path.join(self.test_dir.name, 'bundle.trk'))

    def test_merge(self):
        """Test the merge command of the CLI"""

        # Merging the short and random streamlines should yield a tractogram
        # with 4 streamlines.
        inputs = [os.path.join(self.test_dir.name, i)
                  for i in ['short.trk', 'random.trk']]
        output = os.path.join(self.test_dir.name, 'test-merge-1.trk')
        merge(inputs, output)
        streamlines = load(output)
        self.assertEqual(len(streamlines), 4)

        # Merging the shord and empty files should yield 3 streamlines.
        inputs = [os.path.join(self.test_dir.name, i)
                  for i in ['short.trk', 'empty.trk']]
        output = os.path.join(self.test_dir.name, 'test-merge-2.trk')
        merge(inputs, output)
        streamlines = load(output)
        self.assertEqual(len(streamlines), 3)

    def test_reorient(self):
        """Test the reorient command of the CLI"""

        # Reorienting the streamlines should do nothing (they are already)
        # correctly oriented.
        output = os.path.join(self.test_dir.name, 'test-reorient-1.trk')
        reorient(
            os.path.join(self.test_dir.name, 'bundle.trk'),
            output)
        streamlines = load(os.path.join(self.test_dir.name, 'bundle.trk'))
        new_streamlines = load(output)
        for streamline, new_streamline in zip(streamlines, new_streamlines):
            np.testing.assert_array_almost_equal(new_streamline._points,
                                                 streamline._points)

        # Reorienting the bundle with the reversed first streamline
        # should reverse every streamline.
        output = os.path.join(self.test_dir.name, 'test-reorient-2.trk')
        reorient(
            os.path.join(self.test_dir.name, 'bundle-flipped.trk'),
            output)
        streamlines = load(os.path.join(self.test_dir.name, 'bundle.trk'))
        new_streamlines = load(output)
        for streamline, new_streamline in zip(streamlines, new_streamlines):
            np.testing.assert_array_almost_equal(new_streamline._points,
                                                 streamline._points)
