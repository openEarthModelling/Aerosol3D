import numpy as np
import pytest


@pytest.fixture
def default_config():
    from Aerosol3D.optics.datastructs import SimulationConfig

    return SimulationConfig(wavelength=550.0, dipole_spacing=10.0)


class TestSimulationConfig:
    def test_defaults(self):
        from Aerosol3D.optics.datastructs import SimulationConfig

        config = SimulationConfig()
        assert config.wavelength == 550.0

    def test_validity_check_passes(self, default_config):
        m_max = 1.5
        result = default_config.validity_check(m_max, default_config.dipole_spacing)
        assert result["valid"] is True
        assert "m_k_d" in result

    def test_validity_check_fails(self, default_config):
        # Override spacing to be very large
        default_config.dipole_spacing = 1000.0
        result = default_config.validity_check(
            m_max=1.5, dipole_spacing=default_config.dipole_spacing
        )
        assert result["valid"] is False


class TestCrossSections:
    def test_fields(self, default_config):
        from Aerosol3D.optics.datastructs import CrossSections

        cs = CrossSections(
            wavelength=default_config.wavelength,
            C_ext=100.0,
            C_sca=60.0,
            C_abs=40.0,
            Q_ext=2.0,
            Q_sca=1.2,
            Q_abs=0.8,
            SSA=0.6,
            g=0.5,
            r_eff=50.0,
        )
        assert cs.wavelength == 550.0
        assert cs.SSA == 0.6


class TestPhaseFunction:
    def test_optional_fields(self):
        from Aerosol3D.optics.datastructs import PhaseFunction

        pf = PhaseFunction(
            theta=np.array([0.0, 1.0]), phi=np.array([0.0, 1.0]), P11=np.array([1.0, 2.0])
        )
        assert pf.P12 is None
        assert pf.mueller_matrix is None

    def test_with_mueller(self):
        from Aerosol3D.optics.datastructs import PhaseFunction

        pf = PhaseFunction(
            theta=np.array([0.0, 1.0]),
            phi=np.array([0.0, 1.0]),
            P11=np.array([1.0, 2.0]),
            mueller_matrix=np.zeros((2, 2, 4, 4)),
        )
        assert pf.mueller_matrix is not None


class TestOpticalResult:
    def test_minimal(self, default_config):
        from Aerosol3D.optics.datastructs import CrossSections, OpticalResult

        cs = CrossSections(
            wavelength=default_config.wavelength,
            C_ext=100.0,
            C_sca=60.0,
            C_abs=40.0,
            Q_ext=2.0,
            Q_sca=1.2,
            Q_abs=0.8,
            SSA=0.6,
            g=0.5,
            r_eff=50.0,
        )
        result = OpticalResult(config=default_config, cross_sections=cs)
        assert result.n_dipoles == 0
        assert result.voxel_grid is None

    def test_no_pyvista_import_at_load(self):
        """SimulationConfig should be importable without pyvista installed."""
        import sys

        # Remove pyvista from sys.modules if present
        pv_modules = [m for m in sys.modules if m and "pyvista" in m]
        for m in pv_modules:
            del sys.modules[m]

        # Should not raise
        from Aerosol3D.optics.datastructs import SimulationConfig

        config = SimulationConfig()
        assert config.wavelength == 550.0


class TestSimulationConfigWavelengthList:
    def test_wavelength_accepts_list(self):
        from Aerosol3D.optics.datastructs import SimulationConfig

        config = SimulationConfig(wavelength=[400.0, 550.0, 700.0])
        assert config.wavelength == [400.0, 550.0, 700.0]

    def test_wavelength_accepts_float(self):
        from Aerosol3D.optics.datastructs import SimulationConfig

        config = SimulationConfig(wavelength=550.0)
        assert config.wavelength == 550.0


class TestPrecisionLevel:
    def test_auto_voxel_size_low(self):
        """Low precision: |m|*k*d should be < 0.5."""
        from Aerosol3D.optics.datastructs import auto_voxel_size

        d = auto_voxel_size(550.0, 2.08, "low")
        k = 2.0 * np.pi / 550.0
        mkd = 2.08 * k * d
        assert mkd <= 0.55

    def test_auto_voxel_size_high(self):
        """High precision: |m|*k*d should be < 0.95."""
        from Aerosol3D.optics.datastructs import auto_voxel_size

        d = auto_voxel_size(550.0, 2.08, "high")
        k = 2.0 * np.pi / 550.0
        mkd = 2.08 * k * d
        assert mkd <= 1.0
        assert mkd > 0.7

    def test_auto_voxel_size_medium(self):
        from Aerosol3D.optics.datastructs import auto_voxel_size

        d = auto_voxel_size(550.0, 2.08, "medium")
        k = 2.0 * np.pi / 550.0
        mkd = 2.08 * k * d
        assert mkd <= 0.8
        assert mkd > 0.3

    def test_invalid_precision_raises(self):
        from Aerosol3D.optics.datastructs import auto_voxel_size

        with pytest.raises(ValueError, match="precision"):
            auto_voxel_size(550.0, 1.5, "ultra")

    def test_default_config_new_fields(self):
        from Aerosol3D.optics.datastructs import SimulationConfig

        config = SimulationConfig()
        assert config.wavelength == 550.0
        assert config.precision == "medium"
        assert config.source == "solar"
        assert config.polarization is None
        assert config.solver == "CPU"
