import numpy as np

from Aerosol3D.optics.optics_export import AerosolOpticsData


def _make_data(n_wl=3, g=0.7, n_legendre=16):
    n_theta = 91
    n_phi = 36
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    wl = np.array([400.0, 550.0, 700.0][:n_wl])

    mu = np.cos(theta[:, None])
    P11_base = (1 - g ** 2) / (1 + g ** 2 - 2 * g * mu) ** 1.5
    P11 = np.broadcast_to(P11_base, (n_wl, n_theta, n_phi)).copy()

    legendre_moments = np.zeros((n_wl, n_legendre))
    for l in range(n_legendre):
        legendre_moments[:, l] = (2 * l + 1) * g ** l

    return AerosolOpticsData(
        wavelength_nm=wl,
        C_ext=np.full(n_wl, 100.0),
        C_sca=np.full(n_wl, 80.0),
        C_abs=np.full(n_wl, 20.0),
        SSA=np.full(n_wl, 0.8),
        g=np.full(n_wl, g),
        r_eff_nm=200.0,
        theta_rad=theta,
        phi_rad=phi,
        P11=P11,
        legendre_moments=legendre_moments,
        n_legendre=n_legendre,
    )


class TestPlotSpectralProperties:
    def test_creates_file(self, tmp_path):
        from Aerosol3D.optics.visualization import plot_spectral_properties

        data = _make_data()
        plot_spectral_properties(data, str(tmp_path))
        assert (tmp_path / "spectral_properties.png").exists()


class TestPlotPhaseFunction:
    def test_creates_file(self, tmp_path):
        from Aerosol3D.optics.visualization import plot_phase_function

        data = _make_data()
        plot_phase_function(data, wl_idx=1, output_dir=str(tmp_path))
        assert (tmp_path / "phase_function_550nm.png").exists()


class TestPlotOpticalComparison:
    def test_creates_file(self, tmp_path):
        from Aerosol3D.optics.visualization import plot_optical_comparison

        dda = _make_data(n_wl=3, g=0.72)
        mie = _make_data(n_wl=3, g=0.70)
        plot_optical_comparison([dda, mie], ["DDA", "Mie"], str(tmp_path))
        assert (tmp_path / "optical_comparison.png").exists()


class TestPlotPhaseFunctionComparison:
    def test_creates_file(self, tmp_path):
        from Aerosol3D.optics.visualization import plot_phase_function_comparison

        dda = _make_data(n_wl=2, g=0.72)
        mie = _make_data(n_wl=2, g=0.70)
        plot_phase_function_comparison([dda, mie], ["DDA", "Mie"], str(tmp_path))
        assert len(list(tmp_path.glob("p11_comparison_*.png"))) == 2


class TestPlotLegendreConvergence:
    def test_creates_file(self, tmp_path):
        from Aerosol3D.optics.visualization import plot_legendre_convergence

        data = _make_data(n_wl=1, n_legendre=16)
        plot_legendre_convergence(data, wl_idx=0, output_dir=str(tmp_path))
        assert (tmp_path / "legendre_convergence.png").exists()


class TestPlotLegendreMomentsSpectrum:
    def test_creates_file(self, tmp_path):
        from Aerosol3D.optics.visualization import plot_legendre_moments_spectrum

        data = _make_data(n_wl=3, n_legendre=16)
        plot_legendre_moments_spectrum(data, str(tmp_path))
        assert (tmp_path / "legendre_moments_spectrum.png").exists()


class TestGenerateComparisonSummary:
    def test_returns_string(self):
        from Aerosol3D.optics.visualization import generate_comparison_summary

        dda = _make_data(n_wl=3, g=0.72)
        mie = _make_data(n_wl=3, g=0.70)
        summary = generate_comparison_summary([dda, mie], ["DDA", "Mie"])
        assert isinstance(summary, str)
        assert "DDA" in summary
        assert "Mie" in summary
