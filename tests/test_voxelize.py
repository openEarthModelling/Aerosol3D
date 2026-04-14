import numpy as np
import pyvista as pv
import pytest


class TestVoxelizeWithMaterials:
    def test_returns_imagedata(self):
        from aerosol3d.geometry.voxelize import voxelize_with_materials
        pytest.skip("Requires AerosolParticle — tested in Task 7 integration")

    def test_select_enclosed_points_basic(self):
        """Verify PyVista's select_enclosed_points works for our use case."""
        sphere = pv.Sphere(radius=50.0, center=(0, 0, 0))
        grid = pv.ImageData(dimensions=(21, 21, 21),
                            spacing=(10, 10, 10),
                            origin=(-100, -100, -100))
        enclosed = grid.select_interior_points(sphere)
        selected = enclosed.point_data["selected_points"]
        n_inside = np.sum(selected)
        assert n_inside > 0
        assert n_inside < grid.n_points