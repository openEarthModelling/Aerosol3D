import os
import numpy as np
import pyvista as pv
import pytest


class TestExporter:
    def test_save_vtp(self, soot_material, tmp_path):
        from Aerosol3D.core.particle import AerosolParticle
        from Aerosol3D.geometry.primitives import create_sphere
        from Aerosol3D.io.exporter import save_vtp

        p = AerosolParticle(name="test", unit="nm")
        core = create_sphere(center=(0, 0, 0), radius=50.0)
        p.add_mesh("bc_core", core, soot_material)

        filepath = str(tmp_path / "test.vtp")
        save_vtp(p, filepath)
        assert os.path.exists(filepath)
        loaded = pv.read(filepath)
        assert loaded.n_points > 0

    def test_save_voxel(self, soot_material, tmp_path):
        from Aerosol3D.core.particle import AerosolParticle
        from Aerosol3D.geometry.primitives import create_sphere
        from Aerosol3D.io.exporter import save_voxel

        p = AerosolParticle(name="test", unit="nm")
        core = create_sphere(center=(0, 0, 0), radius=50.0)
        p.add_mesh("bc_core", core, soot_material)

        filepath = str(tmp_path / "test.vti")
        save_voxel(p, filepath, voxel_size=10.0)
        assert os.path.exists(filepath)
        loaded = pv.read(filepath)
        assert "material_id" in loaded.cell_data