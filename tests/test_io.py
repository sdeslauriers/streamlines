import unittest
from tempfile import NamedTemporaryFile

import numpy as np
from nicoord import AffineTransform
from nicoord import CoordinateSystem
from nicoord import CoordinateSystemSpace
from nicoord import CoordinateSystemAxes

import streamlines as sl


class TestIO(unittest.TestCase):
    """Test the saving and loading streamlines"""

    def test_simple_save_and_load(self):
        """Test the saving and loading for simple streamline data"""

        streamlines = np.random.randint(0, 100, size=(9, 10, 3))
        streamlines[streamlines < 0] = 0

        output = NamedTemporaryFile(mode='w', delete=True, suffix='.trk').name

        # Saving and reloading should give the same points because the
        # streamlines are already in RAS by default.
        streamlines_without_affine = sl.Streamlines(streamlines)
        sl.io.save(streamlines_without_affine, output)
        recovered_streamlines = sl.io.load(output)

        for streamline, recovered in zip(streamlines, recovered_streamlines):
            np.testing.assert_array_almost_equal(streamline, recovered.points)

    def test_with_affine(self):
        """Test saving and loading with an affine transformation"""

        streamlines = np.random.randn(9, 10, 3)
        affine = np.array([[-1.25, 0., 0., 90.],
                           [0., 1.25, 0., -126.],
                           [0., 0., 1.25, -72.],
                           [0., 0., 0., 1.]])

        source = CoordinateSystem(
            CoordinateSystemSpace.NATIVE, CoordinateSystemAxes.RAS)
        target = CoordinateSystem(
            CoordinateSystemSpace.VOXEL, CoordinateSystemAxes.RAS)
        transform = AffineTransform(source, target, np.linalg.inv(affine))

        output = NamedTemporaryFile(mode='w', delete=True, suffix='.trk').name

        streamlines = sl.Streamlines(streamlines, transforms=[transform])
        sl.io.save(streamlines, output)
        recovered_streamlines = sl.io.load(output)

        for streamline, recovered in zip(streamlines, recovered_streamlines):
            np.testing.assert_almost_equal(streamline, recovered.points, 5)

    def test_with_coordinate_change(self):
        """Test saving and loading with a coordinate system change"""

        streamlines = np.random.randint(0, 100, size=(9, 10, 3))
        streamlines[streamlines < 0] = 0
        affine = np.array([[-1.25, 0., 0., 90.],
                           [0., 1.25, 0., -126.],
                           [0., 0., 1.25, -72.],
                           [0., 0., 0., 1.]])

        source = CoordinateSystem(
            CoordinateSystemSpace.VOXEL, CoordinateSystemAxes.RAS)
        target = CoordinateSystem(
            CoordinateSystemSpace.NATIVE, CoordinateSystemAxes.RAS)
        transform = AffineTransform(source, target, np.linalg.inv(affine))

        output = NamedTemporaryFile(mode='w', delete=True, suffix='.trk').name

        streamlines = sl.Streamlines(
            streamlines, coordinate_system=source, transforms=[transform])
        sl.io.save(streamlines, output)
        recovered_streamlines = sl.io.load(output)

        # The loaded streamlines are not in native RAS space and we have to
        # ask to get them in the original voxel space.
        self.assertEqual(recovered_streamlines.coordinate_system, target)

        recovered_streamlines.transform_to(source)
        for streamline, recovered in zip(streamlines, recovered_streamlines):
            np.testing.assert_almost_equal(streamline, recovered.points, 4)


