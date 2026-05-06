import numpy as np
import pytest


class TestCAMCoating:
    def test_coating_added(self, soot_material, sulfate_material):
        from aerosol3d.core.particle import AerosolParticle
        from aerosol3d.geometry.primitives import create_sphere
        from aerosol3d.modeling.cam import apply_cam_coating

        p = AerosolParticle(name="test", unit="nm")
        core = create_sphere(center=(0, 0, 0), radius=50.0)
        p.add_mesh("bc_core", core, soot_material)

        result = apply_cam_coating(p, f_bc=0.5, material=sulfate_material)
        assert "coating" in result.blocks
        assert result.mixing_state.name == "COATED"

    def test_envelope_radius(self, soot_material, sulfate_material):
        from aerosol3d.core.particle import AerosolParticle
        from aerosol3d.geometry.primitives import create_sphere
        from aerosol3d.modeling.cam import apply_cam_coating

        p = AerosolParticle(name="test", unit="nm")
        core = create_sphere(center=(0, 0, 0), radius=50.0)
        p.add_mesh("bc_core", core, soot_material)

        result = apply_cam_coating(p, f_bc=0.5, material=sulfate_material)
        coating = result.blocks["coating"]
        expected_R = 50.0 * (2.0 ** (1.0/3.0))
        bounds = coating.bounds
        max_dim = max(bounds[1::2][i] - bounds[::2][i] for i in range(3))
        assert abs(max_dim / 2 - expected_R) / expected_R < 0.1

    def test_f_bc_validation(self, soot_material, sulfate_material):
        from aerosol3d.core.particle import AerosolParticle
        from aerosol3d.geometry.primitives import create_sphere
        from aerosol3d.modeling.cam import apply_cam_coating

        p = AerosolParticle(name="test", unit="nm")
        core = create_sphere(center=(0, 0, 0), radius=50.0)
        p.add_mesh("bc_core", core, soot_material)

        with pytest.raises(ValueError):
            apply_cam_coating(p, f_bc=0.0, material=sulfate_material)