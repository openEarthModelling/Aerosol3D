import pyvista as pv


class TestMeshImporter:
    def test_import_sphere_from_stl(self, tmp_path):
        from Aerosol3D.factory.mesh_importer import from_file

        sphere = pv.Sphere(radius=50.0)
        filepath = str(tmp_path / "test_sphere.stl")
        sphere.save(filepath)

        mesh = from_file(filepath, unit="nm")
        assert isinstance(mesh, pv.PolyData)
        assert mesh.n_points > 0

    def test_unit_scaling(self, tmp_path):
        from Aerosol3D.factory.mesh_importer import from_file

        sphere = pv.Sphere(radius=50.0)
        filepath = str(tmp_path / "test_sphere.stl")
        sphere.save(filepath)

        mesh = from_file(filepath, unit="nm", source_unit="um")
        # Should scale by 1000
        assert mesh.points.max() > 50.0
