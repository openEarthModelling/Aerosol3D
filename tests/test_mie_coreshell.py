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


class TestSolveMieCoreShell:
    def test_returns_optical_result(self, soot_material, sulfate_material):
        # Import mie_solver first to apply scipy.integrate.trapz patch
        from Aerosol3D.optics.mie_solver import solve_mie_coreshell

        pytest.importorskip("PyMieScatt")
        from Aerosol3D.core.particle import AerosolParticle
        from Aerosol3D.geometry.primitives import create_sphere
        from Aerosol3D.optics.datastructs import SimulationConfig

        p = AerosolParticle(name="test")
        p.add_mesh("core", create_sphere((0, 0, 0), 30.0), soot_material)
        p.add_mesh("shell", create_sphere((0, 0, 0), 50.0), sulfate_material)
        config = SimulationConfig(wavelength=550.0)

        result = solve_mie_coreshell(p, config, verbose=False)
        assert result.solver == "MIE_CORESHELL"
        assert result.cross_sections.Q_ext > 0
        assert result.cross_sections.Q_sca > 0
        assert result.cross_sections.g >= -1.0
        assert result.cross_sections.g <= 1.0

    def test_with_phase_function(self, soot_material, sulfate_material):
        from Aerosol3D.optics.mie_solver import solve_mie_coreshell

        pytest.importorskip("PyMieScatt")
        from Aerosol3D.core.particle import AerosolParticle
        from Aerosol3D.geometry.primitives import create_sphere
        from Aerosol3D.optics.datastructs import SimulationConfig

        p = AerosolParticle(name="test")
        p.add_mesh("core", create_sphere((0, 0, 0), 30.0), soot_material)
        p.add_mesh("shell", create_sphere((0, 0, 0), 50.0), sulfate_material)
        config = SimulationConfig(wavelength=550.0)

        result = solve_mie_coreshell(p, config, compute_phase_func=True, verbose=False)
        assert result.phase_function is not None
        assert result.phase_function.P11.shape[0] == 181

    def test_degeneracy_with_homogeneous(self, soot_material):
        """When core and shell have the same RI, core-shell ≈ homogeneous Mie."""
        from Aerosol3D.optics.mie_solver import solve_mie, solve_mie_coreshell

        pytest.importorskip("PyMieScatt")
        from Aerosol3D.core.material import Material
        from Aerosol3D.core.particle import AerosolParticle
        from Aerosol3D.geometry.primitives import create_sphere
        from Aerosol3D.optics.datastructs import SimulationConfig

        m = complex(1.5, 0.0)
        mat = Material(name="uniform", refractive_index=m, density=1.8)
        mat2 = Material(name="uniform2", refractive_index=m, density=1.8)

        p_cs = AerosolParticle(name="test")
        p_cs.add_mesh("core", create_sphere((0, 0, 0), 30.0), mat)
        p_cs.add_mesh("shell", create_sphere((0, 0, 0), 50.0), mat2)

        # The homogeneous sphere must have the same outer diameter as the
        # core-shell particle (d_outer comes from total volume via
        # coreshell_geometry, so we pass it explicitly to solve_mie).
        d_core, d_outer, _, _ = p_cs.coreshell_geometry
        r_outer = d_outer / 2.0

        p_homo = AerosolParticle(name="test_homo")
        p_homo.add_mesh("core", create_sphere((0, 0, 0), r_outer), mat)

        config = SimulationConfig(wavelength=550.0)
        r_cs = solve_mie_coreshell(p_cs, config, verbose=False)
        r_homo = solve_mie(p_homo, config, verbose=False)

        # Outer diameter of core-shell ≈ diameter of homogeneous sphere
        # (small tolerance due to voxelization-induced diameter differences)
        assert r_cs.cross_sections.Q_ext == pytest.approx(r_homo.cross_sections.Q_ext, rel=0.02)


class TestIntegration:
    def test_solve_optics_mie_coreshell(self, soot_material, sulfate_material):
        """Full pipeline: particle -> solve_optics(solver='MIE_CORESHELL')."""
        # Import mie_solver first to apply scipy.integrate.trapz patch
        from Aerosol3D.optics.mie_solver import solve_mie_coreshell  # noqa: F401

        pytest.importorskip("PyMieScatt")
        from Aerosol3D.core.particle import AerosolParticle
        from Aerosol3D.geometry.primitives import create_sphere
        from Aerosol3D.optics.datastructs import SimulationConfig
        from Aerosol3D.optics.dda_solver import solve_optics

        p = AerosolParticle(name="bc_sulfate")
        p.add_mesh("core", create_sphere((0, 0, 0), 30.0), soot_material)
        p.add_mesh("shell", create_sphere((0, 0, 0), 50.0), sulfate_material)
        config = SimulationConfig(wavelength=550.0)

        result = solve_optics(p, config, solver="MIE_CORESHELL", verbose=False)
        assert result.solver == "MIE_CORESHELL"
        assert result.cross_sections.Q_ext > 0

    def test_solve_optics_mie_with_ema_methods(self, soot_material, sulfate_material):
        """Homogeneous Mie with different EMA methods."""
        # Import mie_solver first to apply scipy.integrate.trapz patch
        from Aerosol3D.optics.mie_solver import solve_mie

        pytest.importorskip("PyMieScatt")
        from Aerosol3D.core.particle import AerosolParticle
        from Aerosol3D.geometry.primitives import create_sphere
        from Aerosol3D.optics.datastructs import SimulationConfig

        p = AerosolParticle(name="bc_sulfate")
        p.add_mesh("core", create_sphere((0, 0, 0), 30.0), soot_material)
        p.add_mesh("shell", create_sphere((0, 0, 0), 50.0), sulfate_material)
        config = SimulationConfig(wavelength=550.0)

        r_vw = solve_mie(p, config, ema_method="volume_weighted", verbose=False)
        r_mg = solve_mie(p, config, ema_method="maxwell_garnett", verbose=False)
        r_bg = solve_mie(p, config, ema_method="bruggeman", verbose=False)

        # All should produce valid results
        for r in [r_vw, r_mg, r_bg]:
            assert r.cross_sections.Q_ext > 0
            assert r.solver == "MIE"

        # Different EMA methods should give different results
        # (because soot and sulfate have very different RIs)
        assert r_vw.cross_sections.Q_ext != r_mg.cross_sections.Q_ext

    def test_coreshell_vs_homogeneous_different(self, soot_material, sulfate_material):
        """Core-shell should differ from homogeneous Mie for absorbing core."""
        # Import mie_solver first to apply scipy.integrate.trapz patch
        from Aerosol3D.optics.mie_solver import solve_mie_coreshell  # noqa: F401

        pytest.importorskip("PyMieScatt")
        from Aerosol3D.core.particle import AerosolParticle
        from Aerosol3D.geometry.primitives import create_sphere
        from Aerosol3D.optics.datastructs import SimulationConfig
        from Aerosol3D.optics.dda_solver import solve_optics

        p = AerosolParticle(name="test")
        p.add_mesh("core", create_sphere((0, 0, 0), 30.0), soot_material)
        p.add_mesh("shell", create_sphere((0, 0, 0), 50.0), sulfate_material)
        config = SimulationConfig(wavelength=550.0)

        r_cs = solve_optics(p, config, solver="MIE_CORESHELL", verbose=False)
        r_homo = solve_optics(p, config, solver="MIE", verbose=False)

        # Results should be physically different (core-shell ≠ homogeneous EMA)
        assert r_cs.cross_sections.Q_abs != r_homo.cross_sections.Q_abs
        assert r_cs.cross_sections.Q_ext != r_homo.cross_sections.Q_ext
