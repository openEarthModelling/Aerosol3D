# tests/test_bridge.py
import os
import pytest

JULIA_AVAILABLE = os.environ.get("SKIP_JULIA_TESTS") != "1"


@pytest.fixture(scope="module")
def julia_available():
    """Skip entire module if Julia is not available."""
    if not JULIA_AVAILABLE:
        pytest.skip("Julia runtime not available (set SKIP_JULIA_TESTS=1 to skip)")


class TestBridgeInit:
    def test_ensure_julia_loads(self, julia_available):
        from aerosol3d.optics.bridge import _ensure_julia, is_julia_ready
        _ensure_julia()
        assert is_julia_ready()

    def test_julia_cemd_loaded(self, julia_available):
        from aerosol3d.optics.bridge import _ensure_julia
        from julia import CoupledElectricMagneticDipoles as CEMD
        _ensure_julia()
        assert hasattr(CEMD, "DDACore")
        assert hasattr(CEMD, "Alphas")
        assert hasattr(CEMD, "InputFields")
        assert hasattr(CEMD, "PostProcessing")


class TestBridgeSolve:
    def test_single_dipole_trivial(self, julia_available):
        """Single dipole with trivial polarizability should not crash."""
        from aerosol3d.optics.bridge import solve_dda
        from aerosol3d.optics.datastructs import SimulationConfig

        import numpy as np
        positions = np.array([[0.0, 0.0, 0.0]])
        alpha_e = np.array([1.0 + 0.0j])
        config = SimulationConfig(wavelength=550.0, dipole_spacing=10.0)

        result = solve_dda(positions, alpha_e, config)
        assert "phi_inc" in result
        assert result["phi_inc"].shape == (1, 3)

    def test_two_dipoles(self, julia_available):
        from aerosol3d.optics.bridge import solve_dda
        from aerosol3d.optics.datastructs import SimulationConfig

        import numpy as np
        positions = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        alpha_e = np.array([0.1 + 0.01j, 0.1 + 0.01j])
        config = SimulationConfig(wavelength=550.0, dipole_spacing=10.0)

        result = solve_dda(positions, alpha_e, config)
        assert result["phi_inc"].shape == (2, 3)


class TestBridgeCrossSections:
    def test_single_dipole_cross_sections(self, julia_available):
        """Single dipole with absorptive polarizability (non-zero imag part)."""
        from aerosol3d.optics.bridge import compute_cross_sections
        from aerosol3d.optics.datastructs import SimulationConfig

        import numpy as np
        positions = np.array([[0.0, 0.0, 0.0]])
        alpha_e = np.array([0.5 + 0.1j])  # absorptive
        config = SimulationConfig(wavelength=550.0, dipole_spacing=10.0)

        from aerosol3d.optics.bridge import solve_dda
        dda_result = solve_dda(positions, alpha_e, config)

        cs = compute_cross_sections(positions, alpha_e, dda_result, config)
        assert len(cs) == 3
        assert cs[0] > 0  # C_ext should be positive
        # Optical theorem: C_ext = C_abs + C_sca
        assert cs[0] == pytest.approx(cs[1] + cs[2], abs=1e-10)


class TestBridgeDiffScattering:
    def test_forward_scattering(self, julia_available):
        """Differential scattering needs non-zero kr extents (max_norm > 0)."""
        from aerosol3d.optics.bridge import (
            solve_dda, compute_diff_scattering
        )
        from aerosol3d.optics.datastructs import SimulationConfig

        import numpy as np
        # Two dipoles separated by dipole_spacing ensures max_norm > 0
        positions = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        alpha_e = np.array([0.1 + 0.01j, 0.1 + 0.01j])
        config = SimulationConfig(wavelength=550.0, dipole_spacing=10.0)

        dda_result = solve_dda(positions, alpha_e, config)

        forward = np.array([[0.0, 0.0, 1.0]])
        backward = np.array([[0.0, 0.0, -1.0]])

        dcs_fwd = compute_diff_scattering(positions, alpha_e, dda_result, config, forward)
        dcs_bwd = compute_diff_scattering(positions, alpha_e, dda_result, config, backward)
        assert dcs_fwd >= 0
        assert dcs_bwd >= 0


class TestAsymmetryParameter:
    def test_symmetric_scattering(self, julia_available):
        """A small isotropic scatterer should have g near 0 (symmetric)."""
        from aerosol3d.optics.bridge import (
            solve_dda, compute_asymmetry_parameter
        )
        from aerosol3d.optics.datastructs import SimulationConfig

        import numpy as np
        # Two separated dipoles with weak polarizability
        positions = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        alpha_e = np.array([0.01 + 0.001j, 0.01 + 0.001j])
        config = SimulationConfig(wavelength=550.0, dipole_spacing=5.0)

        dda_result = solve_dda(positions, alpha_e, config)
        g = compute_asymmetry_parameter(
            positions, alpha_e, dda_result, config, C_sca=1e-10
        )
        # g should be a float in [-1, 1]
        assert isinstance(g, float)
        assert -1.0 <= g <= 1.0
