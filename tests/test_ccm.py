import numpy as np
import pytest


class TestCCMCoating:
    def test_coating_added(self, soot_material, sulfate_material):
        from Aerosol3D.core.particle import AerosolParticle
        from Aerosol3D.geometry.primitives import create_sphere
        from Aerosol3D.modeling.ccm import apply_ccm_coating

        p = AerosolParticle(name="test", unit="nm")
        core = create_sphere(center=(0, 0, 0), radius=50.0)
        p.add_mesh("bc_core", core, soot_material)

        result = apply_ccm_coating(p, f_bc=0.5, material=sulfate_material)
        assert "coating" in result.blocks
        assert result.mixing_state.name == "COATED"

    def test_f_bc_validation(self, soot_material, sulfate_material):
        from Aerosol3D.core.particle import AerosolParticle
        from Aerosol3D.geometry.primitives import create_sphere
        from Aerosol3D.modeling.ccm import apply_ccm_coating

        p = AerosolParticle(name="test", unit="nm")
        core = create_sphere(center=(0, 0, 0), radius=50.0)
        p.add_mesh("bc_core", core, soot_material)

        with pytest.raises(ValueError):
            apply_ccm_coating(p, f_bc=1.5, material=sulfate_material)
        with pytest.raises(ValueError):
            apply_ccm_coating(p, f_bc=0.0, material=sulfate_material)