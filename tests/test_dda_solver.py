# tests/test_dda_solver.py
import os
import numpy as np
import pytest

JULIA_AVAILABLE = os.environ.get("SKIP_JULIA_TESTS") != "1"


@pytest.fixture(scope="module")
def julia_available():
    if not JULIA_AVAILABLE:
        pytest.skip("Julia runtime not available")


class TestSolveOptics:
    def test_sphere_basic(self, julia_available, soot_material):
        """Solve a soot sphere and verify optical result structure."""
        from aerosol3d import AerosolParticle, create_sphere
        from aerosol3d.optics.datastructs import SimulationConfig
        from aerosol3d.optics.dda_solver import solve_optics

        p = AerosolParticle(name="soot_sphere", unit="nm")
        p.add_mesh("core", create_sphere((0, 0, 0), 50.0), soot_material)

        config = SimulationConfig(wavelength=550.0, dipole_spacing=10.0)
        result = solve_optics(p, config, voxel_size=10.0)

        assert result.n_dipoles > 0
        assert result.cross_sections.C_ext > 0
        assert result.cross_sections.C_sca > 0
        assert result.cross_sections.C_abs >= 0
        assert result.cross_sections.SSA >= 0
        assert result.cross_sections.SSA <= 1.0
        assert -1 <= result.cross_sections.g <= 1
        assert result.cross_sections.r_eff > 0
        # Optical theorem: C_ext = C_abs + C_sca
        assert result.cross_sections.C_ext == pytest.approx(
            result.cross_sections.C_abs + result.cross_sections.C_sca, abs=1e-10
        )

    def test_sphere_validity_warning(self, julia_available, soot_material):
        """Large dipole spacing should trigger validity failure."""
        from aerosol3d import AerosolParticle, create_sphere
        from aerosol3d.optics.datastructs import SimulationConfig
        from aerosol3d.optics.dda_solver import solve_optics

        p = AerosolParticle(name="soot_sphere", unit="nm")
        p.add_mesh("core", create_sphere((0, 0, 0), 50.0), soot_material)

        # Very large spacing -> invalid DDA
        config = SimulationConfig(wavelength=550.0, dipole_spacing=100.0)
        result = solve_optics(p, config, voxel_size=100.0)
        assert result.validity is not None
        assert result.validity["valid"] is False

    def test_coated_sphere(self, julia_available, soot_material, sulfate_material):
        """Solve a coated sphere (two materials) and verify."""
        from aerosol3d import (
            AerosolParticle, create_sphere, apply_distance_coating
        )
        from aerosol3d.optics.datastructs import SimulationConfig
        from aerosol3d.optics.dda_solver import solve_optics

        p = AerosolParticle(name="coated", unit="nm")
        p.add_mesh("core", create_sphere((0, 0, 0), 50.0), soot_material)
        apply_distance_coating(p, thickness=10.0, material=sulfate_material)

        config = SimulationConfig(wavelength=550.0, dipole_spacing=10.0)
        result = solve_optics(p, config, voxel_size=10.0)
        assert result.n_dipoles > 0
        assert result.cross_sections.C_ext > 0

    def test_result_has_voxel_grid(self, julia_available, soot_material):
        """Verify voxel_grid is attached to result."""
        from aerosol3d import AerosolParticle, create_sphere
        from aerosol3d.optics.datastructs import SimulationConfig
        from aerosol3d.optics.dda_solver import solve_optics

        p = AerosolParticle(name="soot_sphere", unit="nm")
        p.add_mesh("core", create_sphere((0, 0, 0), 50.0), soot_material)

        config = SimulationConfig(wavelength=550.0, dipole_spacing=10.0)
        result = solve_optics(p, config, voxel_size=10.0)

        assert result.voxel_grid is not None
        assert "E_intensity" in result.voxel_grid.cell_data


class TestPhaseFunction:
    def test_phase_function_structure(self, julia_available, soot_material):
        from aerosol3d import AerosolParticle, create_sphere
        from aerosol3d.optics.datastructs import SimulationConfig
        from aerosol3d.optics.dda_solver import solve_optics

        p = AerosolParticle(name="soot_sphere", unit="nm")
        p.add_mesh("core", create_sphere((0, 0, 0), 50.0), soot_material)

        config = SimulationConfig(wavelength=550.0, dipole_spacing=10.0)
        result = solve_optics(p, config, voxel_size=10.0, compute_phase_func=True)

        assert result.phase_function is not None
        assert result.phase_function.P11.ndim == 2
        assert result.phase_function.P11.shape[0] == result.phase_function.theta.shape[0]
        assert np.all(result.phase_function.P11 >= 0)

    def test_forward_peak(self, julia_available, soot_material):
        """Forward scattering should generally be >= backward for Mie-sized particles."""
        from aerosol3d import AerosolParticle, create_sphere
        from aerosol3d.optics.datastructs import SimulationConfig
        from aerosol3d.optics.dda_solver import solve_optics

        p = AerosolParticle(name="soot_sphere", unit="nm")
        p.add_mesh("core", create_sphere((0, 0, 0), 50.0), soot_material)

        config = SimulationConfig(wavelength=550.0, dipole_spacing=10.0)
        result = solve_optics(p, config, voxel_size=10.0, compute_phase_func=True)

        # Forward (theta ~ 0) should be strong
        forward_idx = np.argmin(result.phase_function.theta)
        backward_idx = np.argmax(result.phase_function.theta)
        assert result.phase_function.P11[forward_idx, 0] >= result.phase_function.P11[backward_idx, 0]


class TestAutoVoxelSize:
    def test_auto_voxel_size_produces_valid_mkd(self, julia_available, soot_material):
        """Auto-computed voxel_size should satisfy |m|*k*d <= target."""
        from aerosol3d import AerosolParticle, create_sphere
        from aerosol3d.optics.datastructs import SimulationConfig
        from aerosol3d.optics.dda_solver import solve_optics

        p = AerosolParticle(name="soot_sphere", unit="nm")
        p.add_mesh("core", create_sphere((0, 0, 0), 50.0), soot_material)

        config = SimulationConfig(wavelength=550.0, precision="high")
        result = solve_optics(p, config, voxel_size=None, verbose=False)

        assert result.n_dipoles > 0
        assert result.validity is not None
        assert result.validity["m_k_d"] < 1.0

    def test_explicit_voxel_size_overrides_precision(self, julia_available, soot_material):
        """Explicit voxel_size should be used regardless of precision setting."""
        from aerosol3d import AerosolParticle, create_sphere
        from aerosol3d.optics.datastructs import SimulationConfig
        from aerosol3d.optics.dda_solver import solve_optics

        p = AerosolParticle(name="soot_sphere", unit="nm")
        p.add_mesh("core", create_sphere((0, 0, 0), 50.0), soot_material)

        config = SimulationConfig(wavelength=550.0, precision="high")
        result = solve_optics(p, config, voxel_size=44.0, verbose=False)

        assert result.n_dipoles > 0
        # With large voxel_size, m_k_d should exceed high precision target (0.95)
        assert result.validity["m_k_d"] > 0.95


class TestPrepareDDA:
    def test_prepare_dda_returns_expected(self, julia_available, soot_material):
        from aerosol3d import AerosolParticle, create_sphere
        from aerosol3d.optics.datastructs import SimulationConfig
        from aerosol3d.optics.dda_solver import _prepare_dda

        p = AerosolParticle(name="soot_sphere", unit="nm")
        p.add_mesh("core", create_sphere((0, 0, 0), 50.0), soot_material)
        config = SimulationConfig(wavelength=550.0, dipole_spacing=10.0)

        positions, alpha_e, grid, material_map, voxel_size, m_max, material_names = _prepare_dda(
            p, config, voxel_size=10.0
        )

        assert positions.ndim == 2 and positions.shape[1] == 3
        assert alpha_e.ndim == 1
        assert positions.shape[0] == alpha_e.shape[0]
        assert positions.shape[0] > 0
        assert voxel_size == 10.0
        assert m_max > 0


class TestSolveSingleWL:
    def test_solve_single_wl_matches_solve_optics(self, julia_available, soot_material):
        """_solve_single_wl should produce same result as original solve_optics for single wavelength."""
        from aerosol3d import AerosolParticle, create_sphere
        from aerosol3d.optics.datastructs import SimulationConfig
        from aerosol3d.optics.dda_solver import solve_optics, _prepare_dda, _solve_single_wl

        p = AerosolParticle(name="soot_sphere", unit="nm")
        p.add_mesh("core", create_sphere((0, 0, 0), 50.0), soot_material)
        config_direct = SimulationConfig(wavelength=550.0, dipole_spacing=10.0)
        config_extracted = SimulationConfig(wavelength=550.0, dipole_spacing=10.0)

        # Direct solve_optics
        result_direct = solve_optics(p, config_direct, voxel_size=10.0, verbose=False)

        # Via _prepare_dda + _solve_single_wl
        positions, alpha_e, grid, material_map, voxel_size, m_max, material_names = _prepare_dda(
            p, config_extracted, voxel_size=10.0
        )
        result_extracted = _solve_single_wl(
            positions, alpha_e, grid, material_map, config_extracted, m_max, voxel_size, material_names,
            compute_near_field=True, compute_phase_func=True, verbose=False,
        )

        assert result_direct.n_dipoles == result_extracted.n_dipoles
        assert result_direct.cross_sections.C_ext == pytest.approx(result_extracted.cross_sections.C_ext, abs=1e-10)
        assert result_direct.cross_sections.C_sca == pytest.approx(result_extracted.cross_sections.C_sca, abs=1e-10)
        assert result_direct.cross_sections.C_abs == pytest.approx(result_extracted.cross_sections.C_abs, abs=1e-10)
        assert result_direct.cross_sections.g == pytest.approx(result_extracted.cross_sections.g, abs=1e-10)


class TestVerbose:
    def test_verbose_prints_output(self, julia_available, soot_material, capsys):
        """verbose=True should print configuration table to stdout."""
        from aerosol3d import AerosolParticle, create_sphere
        from aerosol3d.optics.datastructs import SimulationConfig
        from aerosol3d.optics.dda_solver import solve_optics

        p = AerosolParticle(name="soot_sphere", unit="nm")
        p.add_mesh("core", create_sphere((0, 0, 0), 50.0), soot_material)

        config = SimulationConfig(wavelength=550.0, dipole_spacing=10.0)
        solve_optics(p, config, voxel_size=10.0, verbose=True)

        captured = capsys.readouterr()
        assert "DDA Simulation Configuration" in captured.out
        assert "Solve time" in captured.out

    def test_verbose_false_suppresses_output(self, julia_available, soot_material, capsys):
        """verbose=False should not print configuration table."""
        from aerosol3d import AerosolParticle, create_sphere
        from aerosol3d.optics.datastructs import SimulationConfig
        from aerosol3d.optics.dda_solver import solve_optics

        p = AerosolParticle(name="soot_sphere", unit="nm")
        p.add_mesh("core", create_sphere((0, 0, 0), 50.0), soot_material)

        config = SimulationConfig(wavelength=550.0, dipole_spacing=10.0)
        solve_optics(p, config, voxel_size=10.0, verbose=False)

        captured = capsys.readouterr()
        assert "DDA Simulation Configuration" not in captured.out
