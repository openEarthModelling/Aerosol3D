import numpy as np
import pytest

from Aerosol3D.optics.datastructs import (
    CrossSections,
    OpticalResult,
    PhaseFunction,
    SimulationConfig,
)


class TestOpticalResultsToPyradtranData:
    def test_shape_and_keys(self):
        from Aerosol3D.optics.pyradtran_export import optical_results_to_pyradtran_data

        # Build 3 mock OpticalResults
        theta = np.linspace(0, np.pi, 90)
        phi = np.linspace(0, 2 * np.pi, 180, endpoint=False)
        P11 = np.ones((90, 180))

        results = []
        for wl in [400.0, 550.0, 700.0]:
            cs = CrossSections(
                wavelength=wl,
                C_ext=100.0,
                C_sca=80.0,
                C_abs=20.0,
                Q_ext=2.0,
                Q_sca=1.6,
                Q_abs=0.4,
                SSA=0.8,
                g=0.7,
                r_eff=50.0,
            )
            pf = PhaseFunction(theta=theta, phi=phi, P11=P11)
            cfg = SimulationConfig(wavelength=wl)
            results.append(
                OpticalResult(
                    config=cfg,
                    cross_sections=cs,
                    phase_function=pf,
                    n_dipoles=1000,
                    validity={"m_k_d": 0.5, "valid": True},
                )
            )

        data = optical_results_to_pyradtran_data(results, n_legendre=32)

        assert data["wavelength_um"] == [0.4, 0.55, 0.7]
        assert data["radius_um"] == [50.0]
        assert data["Qext"].shape == (3, 1)
        assert data["Qsca"].shape == (3, 1)
        assert data["g"].shape == (3, 1)
        assert data["legendre_moments"].shape == (3, 1, 32)

    def test_values_reasonable(self):
        from Aerosol3D.optics.pyradtran_export import optical_results_to_pyradtran_data

        theta = np.linspace(0, np.pi, 90)
        phi = np.linspace(0, 2 * np.pi, 180, endpoint=False)
        P11 = np.ones((90, 180))

        results = []
        for wl in [400.0, 550.0]:
            cs = CrossSections(
                wavelength=wl,
                C_ext=100.0,
                C_sca=80.0,
                C_abs=20.0,
                Q_ext=2.0,
                Q_sca=1.6,
                Q_abs=0.4,
                SSA=0.8,
                g=0.7,
                r_eff=50.0,
            )
            pf = PhaseFunction(theta=theta, phi=phi, P11=P11)
            cfg = SimulationConfig(wavelength=wl)
            results.append(
                OpticalResult(
                    config=cfg,
                    cross_sections=cs,
                    phase_function=pf,
                    n_dipoles=1000,
                )
            )

        data = optical_results_to_pyradtran_data(results, n_legendre=16)

        assert np.all(data["Qsca"] <= data["Qext"] + 1e-10)
        assert np.all(data["Qext"] >= 0)
        assert np.all(data["Qsca"] >= 0)
        assert np.all((data["g"] >= -1) & (data["g"] <= 1))
        assert np.allclose(data["legendre_moments"][:, :, 0], 1.0)

    def test_requires_phase_function(self):
        from Aerosol3D.optics.pyradtran_export import optical_results_to_pyradtran_data

        cs = CrossSections(
            wavelength=550.0,
            C_ext=100.0,
            C_sca=80.0,
            C_abs=20.0,
            Q_ext=2.0,
            Q_sca=1.6,
            Q_abs=0.4,
            SSA=0.8,
            g=0.7,
            r_eff=50.0,
        )
        cfg = SimulationConfig(wavelength=550.0)
        result = OpticalResult(config=cfg, cross_sections=cs, phase_function=None)

        with pytest.raises(ValueError, match="phase_function"):
            optical_results_to_pyradtran_data([result])
