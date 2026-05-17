import numpy as np
import pytest

from Aerosol3D.optics.datastructs import (
    CrossSections,
    OpticalResult,
    PhaseFunction,
    SimulationConfig,
)


def _make_result(wl=550.0, g=0.7, r_eff=200.0, has_pf=True):
    cs = CrossSections(
        wavelength=wl,
        C_ext=100.0,
        C_sca=80.0,
        C_abs=20.0,
        Q_ext=2.0,
        Q_sca=1.6,
        Q_abs=0.4,
        SSA=0.8,
        g=g,
        r_eff=r_eff,
    )
    pf = None
    if has_pf:
        n_theta = 91
        n_phi = 36
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        mu = np.cos(theta[:, None])
        P11 = (1 - g ** 2) / (1 + g ** 2 - 2 * g * mu) ** 1.5
        pf = PhaseFunction(theta=theta, phi=phi, P11=P11)
    cfg = SimulationConfig(wavelength=wl)
    return OpticalResult(config=cfg, cross_sections=cs, phase_function=pf, solver="MIE")


class TestFromOpticalResults:
    def test_extracts_scalar_fields(self):
        from Aerosol3D.optics.optics_export import from_optical_results

        results = [_make_result(400.0), _make_result(550.0), _make_result(700.0)]
        data = from_optical_results(results, n_legendre=16)

        assert data.wavelength_nm.shape == (3,)
        assert np.allclose(data.wavelength_nm, [400.0, 550.0, 700.0])
        assert data.C_ext.shape == (3,)
        assert data.C_sca.shape == (3,)
        assert data.C_abs.shape == (3,)
        assert data.SSA.shape == (3,)
        assert data.g.shape == (3,)
        assert data.r_eff_nm == 200.0
        assert data.n_legendre == 16
        assert data.solver == "MIE"

    def test_auto_computes_legendre_moments(self):
        from Aerosol3D.optics.optics_export import from_optical_results

        results = [_make_result(550.0, g=0.7)]
        data = from_optical_results(results, n_legendre=8)

        assert data.legendre_moments is not None
        assert data.legendre_moments.shape == (1, 8)
        assert data.legendre_moments[0, 0] == pytest.approx(1.0, abs=1e-6)
        for l in range(1, 8):
            expected = (2 * l + 1) * (0.7 ** l)
            assert data.legendre_moments[0, l] == pytest.approx(expected, rel=0.05)

    def test_no_phase_function_gives_no_legendre(self):
        from Aerosol3D.optics.optics_export import from_optical_results

        results = [_make_result(550.0, has_pf=False)]
        data = from_optical_results(results)

        assert data.P11 is None
        assert data.legendre_moments is None
        assert data.theta_rad is None
        assert data.phi_rad is None

    def test_empty_results_raises(self):
        from Aerosol3D.optics.optics_export import from_optical_results

        with pytest.raises(ValueError, match="empty"):
            from_optical_results([])

    def test_p11_extracted(self):
        from Aerosol3D.optics.optics_export import from_optical_results

        results = [_make_result(550.0)]
        data = from_optical_results(results)

        assert data.P11 is not None
        assert data.P11.shape[0] == 1
        assert data.theta_rad is not None
        assert data.phi_rad is not None
