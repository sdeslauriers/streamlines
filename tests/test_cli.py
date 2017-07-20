import os
import tempfile
import unittest

import numpy as np

from streamlines import Streamlines
from streamlines.cli import filter, merge
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
