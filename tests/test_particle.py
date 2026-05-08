import numpy as np
import pyvista as pv


class TestAerosolParticle:
    def test_create_empty(self):
        from Aerosol3D.core.particle import AerosolParticle, MixingState

        p = AerosolParticle(name="test", unit="nm")
        assert p.name == "test"
        assert p.unit == "nm"
        assert p.mixing_state == MixingState.INTERNAL

    def test_add_mesh(self, soot_material):
        from Aerosol3D.core.particle import AerosolParticle
        from Aerosol3D.geometry.primitives import create_sphere

        p = AerosolParticle(name="test", unit="nm")
        mesh = create_sphere(center=(0, 0, 0), radius=50.0)
        p.add_mesh("core", mesh, soot_material)
        assert len(p.blocks) == 1
        assert "core" in p.blocks

    def test_add_mesh_tags_cell_data(self, soot_material):
        from Aerosol3D.core.particle import AerosolParticle
        from Aerosol3D.geometry.primitives import create_sphere

        p = AerosolParticle(name="test", unit="nm")
        mesh = create_sphere(center=(0, 0, 0), radius=50.0)
        p.add_mesh("core", mesh, soot_material)
        block = p.blocks["core"]
        assert "material_id" in block.cell_data
        assert "ri_n" in block.cell_data
        assert "ri_k" in block.cell_data

    def test_add_mesh_tags_field_data(self, soot_material):
        from Aerosol3D.core.particle import AerosolParticle
        from Aerosol3D.geometry.primitives import create_sphere

        p = AerosolParticle(name="test", unit="nm")
        mesh = create_sphere(center=(0, 0, 0), radius=50.0)
        p.add_mesh("bc_core", mesh, soot_material)
        block = p.blocks["bc_core"]
        assert block.field_data["role"] == ["bc_core"]
        assert block.field_data["material_name"] == ["soot"]

    def test_combined(self, soot_material, sulfate_material):
        from Aerosol3D.core.particle import AerosolParticle
        from Aerosol3D.geometry.primitives import create_sphere

        p = AerosolParticle(name="test", unit="nm")
        core = create_sphere(center=(0, 0, 0), radius=50.0)
        coating = create_sphere(center=(0, 0, 0), radius=60.0)
        p.add_mesh("core", core, soot_material)
        p.add_mesh("coating", coating, sulfate_material)
        combined = p.combined
        assert isinstance(combined, pv.PolyData)
        assert combined.n_cells > 0

    def test_save_vtp(self, soot_material, tmp_path):
        from Aerosol3D.core.particle import AerosolParticle
        from Aerosol3D.geometry.primitives import create_sphere

        p = AerosolParticle(name="test", unit="nm")
        mesh = create_sphere(center=(0, 0, 0), radius=50.0)
        p.add_mesh("core", mesh, soot_material)
        filepath = str(tmp_path / "test.vtp")
        p.save(filepath)
        assert __import__("os").path.exists(filepath)

    def test_from_aggregate(self, soot_material):
        from Aerosol3D.core.aggregate import FractalAggregate
        from Aerosol3D.core.particle import AerosolParticle

        centers = np.random.default_rng(42).random((5, 3)) * 100
        radii = np.full(5, 10.0)
        agg = FractalAggregate(centers=centers, radii=radii, material=soot_material)
        p = AerosolParticle(name="fractal", unit="nm")
        p.add_mesh("bc_core", agg.to_mesh(theta_res=8, phi_res=8), soot_material)
        assert p.blocks["bc_core"].n_points > 0
