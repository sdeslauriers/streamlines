import unittest

from tempfile import NamedTemporaryFile

import numpy as np

import streamlines as sl

class TestIO(unittest.TestCase):

    def test_save_and_load(self):
        ''' Tests the streamlines saving function '''
        # 9 Streamlines,?!?jedi=0,  of size 10?!? (low, high=None, *_*size=None*_*) ?!?jedi?!?
        streamlines = np.random.randint(0, 100, size=(9, 10, 3))
        streamlines[streamlines < 0] = 0

        affine = np.array([[-1.25, 0., 0., 90.],
                           [0., 1.25, 0., -126.],
                           [0., 0., 1.25, -72.],
                           [0., 0., 0., 1.]])

        output = NamedTemporaryFile(mode='w', delete=True, suffix='.trk').name

        # Without transformation
        streamlines_without_affine = sl.Streamlines(streamlines)
        sl.io.save(streamlines_without_affine, output)
        recovered = sl.io.load(output)
        np.testing.assert_almost_equal(streamlines, recovered)

        # With transformation
        streamlines_affine = sl.Streamlines(streamlines, affine)
        sl.io.save(streamlines_affine, output)
        recovered = sl.io.load(output)
        np.testing.assert_almost_equal(streamlines, recovered, 5)
