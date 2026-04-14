# tests/test_datastructs.py
import numpy as np
import pytest


class TestSimulationConfig:
    def test_defaults(self):
        from aerosol3d.optics.datastructs import SimulationConfig
        c = SimulationConfig(wavelength=550.0)
        assert c.wavelength == 550.0
        assert c.n_host == 1.0
        assert c.solver == "CPU"
        assert c.polarization == (1.0, 0.0, 0.0)
        assert c.propagation == (0.0, 0.0, 1.0)

    def test_validity_check_passes(self):
        from aerosol3d.optics.datastructs import SimulationConfig
        c = SimulationConfig(wavelength=550.0, dipole_spacing=5.0)
        # |m|*k*d for m=2, k=2pi/550, d=5 -> 2 * 0.0114 * 5 = 0.114 < 1
        result = c.validity_check(m_max=2.0)
        assert result["valid"] is True
        assert result["m_k_d"] < 1.0

    def test_validity_check_fails(self):
        from aerosol3d.optics.datastructs import SimulationConfig
        c = SimulationConfig(wavelength=550.0, dipole_spacing=200.0)
        result = c.validity_check(m_max=2.0)
        assert result["valid"] is False


class TestCrossSections:
    def test_fields(self):
        from aerosol3d.optics.datastructs import CrossSections
        cs = CrossSections(
            wavelength=550.0,
            C_ext=1000.0, C_sca=600.0, C_abs=400.0,
            Q_ext=2.0, Q_sca=1.2, Q_abs=0.8,
            SSA=0.6, g=0.7, r_eff=12.6
        )
        assert cs.wavelength == 550.0
        assert cs.SSA == pytest.approx(600.0 / 1000.0)


class TestPhaseFunction:
    def test_optional_fields(self):
        from aerosol3d.optics.datastructs import PhaseFunction
        pf = PhaseFunction(
            theta=np.array([0.0, 0.5, 1.0]),
            phi=np.array([0.0, 1.0, 2.0, 3.0]),
            P11=np.ones((3, 4)),
        )
        assert pf.P12 is None
        assert pf.mueller_matrix is None

    def test_with_mueller(self):
        from aerosol3d.optics.datastructs import PhaseFunction
        pf = PhaseFunction(
            theta=np.array([0.0, 0.5]),
            phi=np.array([0.0, 1.0]),
            P11=np.ones((2, 2)),
            mueller_matrix=np.zeros((2, 2, 4, 4)),
        )
        assert pf.mueller_matrix.shape == (2, 2, 4, 4)


class TestOpticalResult:
    def test_minimal(self):
        from aerosol3d.optics.datastructs import (
            OpticalResult, SimulationConfig, CrossSections
        )
        config = SimulationConfig(wavelength=550.0)
        cs = CrossSections(
            wavelength=550.0,
            C_ext=1000.0, C_sca=600.0, C_abs=400.0,
            Q_ext=2.0, Q_sca=1.2, Q_abs=0.8,
            SSA=0.6, g=0.7, r_eff=12.6,
        )
        result = OpticalResult(config=config, cross_sections=cs, n_dipoles=100)
        assert result.n_dipoles == 100
        assert result.phase_function is None
        assert result.voxel_grid is None

    def test_no_pyvista_import_at_load(self):
        """Verify importing datastructs does NOT trigger pyvista import."""
        import importlib
        import sys
        # Ensure pyvista is not already loaded
        pv_was_loaded = "pyvista" in sys.modules
        # Force reimport datastructs
        if "aerosol3d.optics.datastructs" in sys.modules:
            del sys.modules["aerosol3d.optics.datastructs"]
        # If pyvista wasn't loaded before, datastructs shouldn't load it
        if not pv_was_loaded:
            from aerosol3d.optics.datastructs import SimulationConfig
            assert "pyvista" not in sys.modules