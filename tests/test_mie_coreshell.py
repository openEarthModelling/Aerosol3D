# tests/test_mie_coreshell.py
import numpy as np
import pytest


class TestCoreShellGeometry:
    def test_extracts_correct_geometry(self, soot_material, sulfate_material):
        from Aerosol3D.core.particle import AerosolParticle
        from Aerosol3D.geometry.primitives import create_sphere

        p = AerosolParticle(name="test")
        p.add_mesh("core", create_sphere((0, 0, 0), 30.0), soot_material)
        p.add_mesh("shell", create_sphere((0, 0, 0), 50.0), sulfate_material)

        d_core, d_outer, m_core, m_shell = p.coreshell_geometry

        # d_core from core block volume
        V_core = p._block_volume(p.blocks["core"])
        expected_d_core = (6 * V_core / np.pi) ** (1 / 3)
        assert d_core == pytest.approx(expected_d_core, rel=0.01)

        # d_outer from total volume
        V_total = V_core + p._block_volume(p.blocks["shell"])
        expected_d_outer = (6 * V_total / np.pi) ** (1 / 3)
        assert d_outer == pytest.approx(expected_d_outer, rel=0.01)

        # Refractive indices
        assert abs(m_core - soot_material.refractive_index) < 1e-10
        assert abs(m_shell - sulfate_material.refractive_index) < 1e-10

    def test_single_block_raises(self, soot_material):
        from Aerosol3D.core.particle import AerosolParticle
        from Aerosol3D.geometry.primitives import create_sphere

        p = AerosolParticle(name="test")
        p.add_mesh("core", create_sphere((0, 0, 0), 50.0), soot_material)

        with pytest.raises(ValueError, match="exactly 2"):
            p.coreshell_geometry

    def test_three_blocks_raises(self, soot_material):
        from Aerosol3D.core.particle import AerosolParticle
        from Aerosol3D.geometry.primitives import create_sphere

        p = AerosolParticle(name="test")
        p.add_mesh("a", create_sphere((0, 0, 0), 30.0), soot_material)
        p.add_mesh("b", create_sphere((0, 0, 0), 40.0), soot_material)
        p.add_mesh("c", create_sphere((0, 0, 0), 50.0), soot_material)

        with pytest.raises(ValueError, match="exactly 2"):
            p.coreshell_geometry

    def test_smaller_volume_is_core(self, sulfate_material, soot_material):
        from Aerosol3D.core.particle import AerosolParticle
        from Aerosol3D.geometry.primitives import create_sphere

        p = AerosolParticle(name="test")
        # Add in reverse order — sulfate (larger) first, soot (smaller) second
        p.add_mesh("big", create_sphere((0, 0, 0), 50.0), sulfate_material)
        p.add_mesh("small", create_sphere((0, 0, 0), 30.0), soot_material)

        d_core, d_outer, m_core, m_shell = p.coreshell_geometry

        # Core should be the smaller block (soot)
        assert abs(m_core - soot_material.refractive_index) < 1e-10
        assert abs(m_shell - sulfate_material.refractive_index) < 1e-10
        assert d_core < d_outer
