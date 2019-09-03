import unittest
from tempfile import NamedTemporaryFile

import numpy as np
from nicoord import AffineTransform
from nicoord import CoordinateSystem
from nicoord import CoordinateSystemSpace
from nicoord import CoordinateSystemAxes
from nicoord import VoxelSpace
from nicoord import inverse

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

    def test_simple_save_and_load_tck(self):
        """Test the saving and loading for simple streamline data in tck"""

        streamlines = np.random.randint(0, 100, size=(9, 10, 3))
        streamlines[streamlines < 0] = 0

        output = NamedTemporaryFile(mode='w', delete=True, suffix='.tck').name

        # Saving and reloading should give the same points because the
        # streamlines are already in RAS by default.
        streamlines_without_affine = sl.Streamlines(streamlines)
        sl.io.save(streamlines_without_affine, output)
        recovered_streamlines = sl.io.load(output)

        for streamline, recovered in zip(streamlines, recovered_streamlines):
            np.testing.assert_array_almost_equal(streamline, recovered.points)

    def test_save_with_data(self):
        """Test saving a streamline with attached data"""

        def random_streamline():
            nb_points = np.random.randint(1, 100)
            points = np.random.randint(0, 100, size=(nb_points, 3))
            point_data = np.random.randn(1, nb_points)
            streamline_data = np.random.randn(5)
            data = {
                'point-data': point_data,
                'streamline-data': streamline_data
            }
            return sl.Streamline(points, data)

        streamlines = sl.Streamlines()
        for _ in range(3):
            streamlines += random_streamline()

        output = NamedTemporaryFile(mode='w', delete=True, suffix='.trk').name
        sl.io.save(streamlines, output)
        recovered_streamlines = sl.io.load(output)

        for s, r in zip(streamlines, recovered_streamlines):
            np.testing.assert_array_almost_equal(s.points, r.points)
            np.testing.assert_array_almost_equal(
                s.data['point-data'], r.data['point-data'])

        output = NamedTemporaryFile(mode='w', delete=True, suffix='.tck').name
        sl.io.save(streamlines, output)
        recovered_streamlines = sl.io.load(output)

        # The data is missing because it cannot be saved in .tck.
        for s, r in zip(streamlines, recovered_streamlines):
            np.testing.assert_array_almost_equal(s.points, r.points)

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

    def test_preserve_voxel_sizes(self):
        """Test if the voxel sizes are preserved on save and load"""

        streamlines = np.random.randint(0, 100, size=(9, 10, 3))
        streamlines[streamlines < 0] = 0
        affine = np.array([[-1.25, 0., 0., 90.],
                           [0., 1.25, 0., -126.],
                           [0., 0., 1.25, -72.],
                           [0., 0., 0., 1.]])

        voxel_sizes = (1.25, 1.25, 1.25)
        shape = (256, 256, 256)
        source = VoxelSpace(voxel_sizes, shape, CoordinateSystemAxes.RAS)
        target = CoordinateSystem(
            CoordinateSystemSpace.NATIVE, CoordinateSystemAxes.RAS)
        transform = AffineTransform(source, target, np.linalg.inv(affine))

        output = NamedTemporaryFile(mode='w', delete=True, suffix='.trk').name

        # The streamlines are not in native RAS with a transform to native RAS.
        streamlines = sl.Streamlines(
            streamlines, coordinate_system=source, transforms=[transform])
        sl.io.save(streamlines, output)

        recovered_streamlines = sl.io.load(output)
        recovered_streamlines.transform_to(source)

        new_voxel_sizes = recovered_streamlines.coordinate_system.voxel_sizes
        np.testing.assert_array_almost_equal(voxel_sizes, new_voxel_sizes)

        new_shape = recovered_streamlines.coordinate_system.shape
        np.testing.assert_array_almost_equal(shape, new_shape)

        # The streamlines are in native RAS with a transform to voxel.
        transforms = [inverse(transform)]
        streamlines = sl.Streamlines(
            streamlines, coordinate_system=target, transforms=transforms)
        sl.io.save(streamlines, output)

        recovered_streamlines = sl.io.load(output)
        recovered_streamlines.transform_to(source)

        new_voxel_sizes = recovered_streamlines.coordinate_system.voxel_sizes
        np.testing.assert_array_almost_equal(voxel_sizes, new_voxel_sizes)
