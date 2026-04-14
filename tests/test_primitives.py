import numpy as np
import pyvista as pv
import pytest


class TestCreateSphere:
    def test_returns_polydata(self):
        from aerosol3d.geometry.primitives import create_sphere
        mesh = create_sphere(center=(0, 0, 0), radius=50.0)
        assert isinstance(mesh, pv.PolyData)
        assert mesh.n_points > 0

    def test_field_data_stored(self):
        from aerosol3d.geometry.primitives import create_sphere
        mesh = create_sphere(center=(1, 2, 3), radius=50.0)
        assert mesh.field_data["geometry_type"][0] == "sphere"
        assert mesh.field_data["analytic_radius"][0] == 50.0
        assert np.array_equal(mesh.field_data["analytic_center"], [1.0, 2.0, 3.0])

    def test_volume_approximately_correct(self):
        from aerosol3d.geometry.primitives import create_sphere
        mesh = create_sphere(center=(0, 0, 0), radius=50.0)
        expected = 4/3 * np.pi * 50**3
        assert abs(mesh.volume - expected) / expected < 0.05  # 5% tolerance


class TestCreateEllipsoid:
    def test_returns_polydata(self):
        from aerosol3d.geometry.primitives import create_ellipsoid
        mesh = create_ellipsoid(center=(0, 0, 0), axes=(60, 40, 20))
        assert isinstance(mesh, pv.PolyData)

    def test_field_data_stored(self):
        from aerosol3d.geometry.primitives import create_ellipsoid
        mesh = create_ellipsoid(center=(1, 2, 3), axes=(60, 40, 20))
        assert mesh.field_data["geometry_type"][0] == "ellipsoid"
        assert np.array_equal(mesh.field_data["analytic_axes"], [60.0, 40.0, 20.0])


class TestCreateCube:
    def test_returns_polydata(self):
        from aerosol3d.geometry.primitives import create_cube
        mesh = create_cube(center=(0, 0, 0), side_lengths=(100, 80, 60))
        assert isinstance(mesh, pv.PolyData)

    def test_field_data_stored(self):
        from aerosol3d.geometry.primitives import create_cube
        mesh = create_cube(center=(1, 2, 3), side_lengths=(100, 80, 60))
        assert mesh.field_data["geometry_type"][0] == "cube"
        assert np.array_equal(mesh.field_data["analytic_side_lengths"], [100.0, 80.0, 60.0])