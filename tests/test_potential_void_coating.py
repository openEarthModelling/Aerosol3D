import numpy as np
import pytest


class TestPotentialVoidCoating:
    def test_coating_added(self, soot_material, sulfate_material):
        from Aerosol3D.core.particle import AerosolParticle
        from Aerosol3D.geometry.primitives import create_sphere
        from Aerosol3D.modeling.potential_void_coating import apply_potential_void_coating

        p = AerosolParticle(name="test", unit="nm")
        core = create_sphere(center=(0, 0, 0), radius=50.0)
        p.add_mesh("bc_core", core, soot_material)

        result = apply_potential_void_coating(
            p, coated_area_fraction=0.8, dp_dc_ratio=1.5,
            material=sulfate_material, resolution=32
        )
        assert "coating" in result.blocks
        assert result.mixing_state.name == "COATED"

    def test_coating_has_cells(self, soot_material, sulfate_material):
        from Aerosol3D.core.particle import AerosolParticle
        from Aerosol3D.geometry.primitives import create_sphere
        from Aerosol3D.modeling.potential_void_coating import apply_potential_void_coating

        p = AerosolParticle(name="test", unit="nm")
        core = create_sphere(center=(0, 0, 0), radius=50.0)
        p.add_mesh("bc_core", core, soot_material)

        result = apply_potential_void_coating(
            p, coated_area_fraction=0.8, dp_dc_ratio=1.5,
            material=sulfate_material, resolution=32
        )

        # On simple spheres coating may be patchy; verify mesh exists
        coating = result.blocks["coating"]
        assert coating.n_cells > 0

    def test_area_fraction_validation(self, soot_material, sulfate_material):
        from Aerosol3D.core.particle import AerosolParticle
        from Aerosol3D.geometry.primitives import create_sphere
        from Aerosol3D.modeling.potential_void_coating import apply_potential_void_coating

        p = AerosolParticle(name="test", unit="nm")
        core = create_sphere(center=(0, 0, 0), radius=50.0)
        p.add_mesh("bc_core", core, soot_material)

        with pytest.raises(ValueError):
            apply_potential_void_coating(
                p, coated_area_fraction=0.0, dp_dc_ratio=1.5,
                material=sulfate_material, resolution=32
            )
        with pytest.raises(ValueError):
            apply_potential_void_coating(
                p, coated_area_fraction=1.5, dp_dc_ratio=1.5,
                material=sulfate_material, resolution=32
            )

    def test_dp_dc_validation(self, soot_material, sulfate_material):
        from Aerosol3D.core.particle import AerosolParticle
        from Aerosol3D.geometry.primitives import create_sphere
        from Aerosol3D.modeling.potential_void_coating import apply_potential_void_coating

        p = AerosolParticle(name="test", unit="nm")
        core = create_sphere(center=(0, 0, 0), radius=50.0)
        p.add_mesh("bc_core", core, soot_material)

        with pytest.raises(ValueError):
            apply_potential_void_coating(
                p, coated_area_fraction=0.5, dp_dc_ratio=1.0,
                material=sulfate_material, resolution=32
            )
        with pytest.raises(ValueError):
            apply_potential_void_coating(
                p, coated_area_fraction=0.5, dp_dc_ratio=0.5,
                material=sulfate_material, resolution=32
            )

    def test_infeasible_constraint(self, soot_material, sulfate_material):
        from Aerosol3D.core.particle import AerosolParticle
        from Aerosol3D.geometry.primitives import create_sphere
        from Aerosol3D.modeling.potential_void_coating import apply_potential_void_coating

        p = AerosolParticle(name="test", unit="nm")
        core = create_sphere(center=(0, 0, 0), radius=50.0)
        p.add_mesh("bc_core", core, soot_material)

        with pytest.raises(ValueError):
            apply_potential_void_coating(
                p, coated_area_fraction=1.0, dp_dc_ratio=1.01,
                material=sulfate_material, resolution=32
            )
