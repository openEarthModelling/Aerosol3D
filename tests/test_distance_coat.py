import numpy as np
import pyvista as pv
import pytest


class TestDistanceCoating:
    def test_coating_added_as_new_block(self, soot_material, sulfate_material):
        from Aerosol3D.core.particle import AerosolParticle
        from Aerosol3D.geometry.primitives import create_sphere
        from Aerosol3D.modeling.distance_coat import apply_distance_coating

        p = AerosolParticle(name="test", unit="nm")
        core = create_sphere(center=(0, 0, 0), radius=50.0)
        p.add_mesh("bc_core", core, soot_material)

        result = apply_distance_coating(p, thickness=10.0, material=sulfate_material)
        assert "coating" in result.blocks
        assert result.mixing_state.name == "COATED"

    def test_coating_volume(self, soot_material, sulfate_material):
        from Aerosol3D.core.particle import AerosolParticle
        from Aerosol3D.geometry.primitives import create_sphere
        from Aerosol3D.modeling.distance_coat import apply_distance_coating

        p = AerosolParticle(name="test", unit="nm")
        core = create_sphere(center=(0, 0, 0), radius=50.0)
        p.add_mesh("bc_core", core, soot_material)

        result = apply_distance_coating(p, thickness=10.0, material=sulfate_material)
        coating = result.blocks["coating"]
        expected = 4/3 * np.pi * (60**3 - 50**3)
        assert coating.volume > 0
        assert abs(coating.volume - expected) / expected < 0.15  # 15% tolerance

    def test_thickness_must_be_positive(self, soot_material, sulfate_material):
        from Aerosol3D.core.particle import AerosolParticle
        from Aerosol3D.geometry.primitives import create_sphere
        from Aerosol3D.modeling.distance_coat import apply_distance_coating

        p = AerosolParticle(name="test", unit="nm")
        core = create_sphere(center=(0, 0, 0), radius=50.0)
        p.add_mesh("bc_core", core, soot_material)

        with pytest.raises(ValueError):
            apply_distance_coating(p, thickness=-5.0, material=sulfate_material)
