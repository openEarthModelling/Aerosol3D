import numpy as np
import pytest


class TestEquivalentDiameter:
    def test_single_sphere(self, soot_material):
        from Aerosol3D.core.particle import AerosolParticle
        from Aerosol3D.geometry.primitives import create_sphere

        p = AerosolParticle(name="test")
        mesh = create_sphere(center=(0, 0, 0), radius=50.0)
        p.add_mesh("core", mesh, soot_material)

        d_ve = p.equivalent_diameter
        # Volume of sphere: 4/3 * pi * 50^3 = 523598.7756
        # d_ve = (6V/pi)^(1/3) = (6 * 523598.7756 / pi)^(1/3) = 100.0
        assert d_ve == pytest.approx(100.0, rel=0.05)

    def test_two_identical_spheres(self, soot_material):
        from Aerosol3D.core.particle import AerosolParticle
        from Aerosol3D.geometry.primitives import create_sphere

        p = AerosolParticle(name="test")
        mesh = create_sphere(center=(0, 0, 0), radius=50.0)
        p.add_mesh("core1", mesh, soot_material)
        p.add_mesh("core2", mesh.copy(), soot_material)

        d_ve = p.equivalent_diameter
        # Total volume = 2 * 523598.7756
        # d_ve = (6 * 2V / pi)^(1/3) = 100 * 2^(1/3) ≈ 125.99
        assert d_ve == pytest.approx(125.99, rel=0.05)

    def test_empty_particle_raises(self):
        from Aerosol3D.core.particle import AerosolParticle

        p = AerosolParticle(name="empty")
        with pytest.raises(ValueError, match="no volume"):
            _ = p.equivalent_diameter


class TestEffectiveRefractiveIndex:
    def test_single_material(self, soot_material):
        from Aerosol3D.core.particle import AerosolParticle
        from Aerosol3D.geometry.primitives import create_sphere

        p = AerosolParticle(name="test")
        mesh = create_sphere(center=(0, 0, 0), radius=50.0)
        p.add_mesh("core", mesh, soot_material)

        m_eff = p.effective_refractive_index
        assert m_eff == pytest.approx(complex(1.8, 0.7), rel=0.01)

    def test_two_materials_volume_weighted(self, soot_material, sulfate_material):
        from Aerosol3D.core.particle import AerosolParticle
        from Aerosol3D.geometry.primitives import create_sphere

        p = AerosolParticle(name="test")
        core = create_sphere(center=(0, 0, 0), radius=50.0)
        coating = create_sphere(center=(0, 0, 0), radius=60.0)
        p.add_mesh("core", core, soot_material)
        p.add_mesh("coating", coating, sulfate_material)

        m_eff = p.effective_refractive_index
        # V_core = 4/3*pi*50^3, V_coating = 4/3*pi*60^3
        # m_eff = (V_core * 1.8 + V_coating * 1.4) / (V_core + V_coating)
        v_core = (4.0 / 3.0) * np.pi * 50.0**3
        v_coat = (4.0 / 3.0) * np.pi * 60.0**3
        expected = (v_core * complex(1.8, 0.7) + v_coat * complex(1.4, 0.0)) / (v_core + v_coat)
        assert m_eff == pytest.approx(expected, rel=0.01)

    def test_empty_particle_raises(self):
        from Aerosol3D.core.particle import AerosolParticle

        p = AerosolParticle(name="empty")
        with pytest.raises(ValueError, match="no volume"):
            _ = p.effective_refractive_index
