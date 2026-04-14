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
