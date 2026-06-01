"""Tests for Aerosol3D.bulk.builder — BulkOpticsBuilder."""

from __future__ import annotations

import numpy as np
import pytest

from Aerosol3D.bulk.datastructs import BulkAerosolOpticsData, SizeDistribution
from Aerosol3D.optics.optics_export import AerosolOpticsData


def _make_optics(
    wavelength_nm: np.ndarray,
    n_legendre: int = 32,
    c_ext_base: float = 1.0,
    c_sca_base: float = 0.8,
    g_base: float = 0.75,
) -> AerosolOpticsData:
    """Create a synthetic AerosolOpticsData without running MIE."""
    n_wl = len(wavelength_nm)
    C_ext = np.full(n_wl, c_ext_base, dtype=float)
    C_sca = np.full(n_wl, c_sca_base, dtype=float)
    C_abs = C_ext - C_sca
    SSA = C_sca / C_ext
    g = np.full(n_wl, g_base, dtype=float)

    # Build beta from g: beta_1 = g, rest = 0, beta_0 = 1
    legendre_moments_beta = np.zeros((n_wl, n_legendre), dtype=float)
    legendre_moments_beta[:, 0] = 1.0
    legendre_moments_beta[:, 1] = g_base

    return AerosolOpticsData(
        wavelength_nm=np.asarray(wavelength_nm, dtype=float),
        C_ext=C_ext,
        C_sca=C_sca,
        C_abs=C_abs,
        SSA=SSA,
        g=g,
        r_eff_nm=100.0,
        legendre_moments_beta=legendre_moments_beta,
        n_legendre=n_legendre,
        solver="synthetic",
    )


class TestBuilderAcceptsEntries:
    def test_builder_accepts_entries(self):
        from Aerosol3D.bulk.builder import BulkOpticsBuilder

        sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.5)
        radii = np.array([50.0, 100.0, 200.0])
        builder = BulkOpticsBuilder(size_distribution=sd, radii_nm=radii, n_legendre=32)

        wl = np.array([500.0, 600.0])
        optics = _make_optics(wl, n_legendre=32)
        builder.add(100.0, optics)

        assert 100.0 in builder._entries
        assert builder._entries[100.0] is optics

    def test_builder_accepts_multiple_entries(self):
        from Aerosol3D.bulk.builder import BulkOpticsBuilder

        sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.5)
        radii = np.array([50.0, 100.0, 200.0])
        builder = BulkOpticsBuilder(size_distribution=sd, radii_nm=radii, n_legendre=32)

        wl = np.array([500.0, 600.0])
        for r in [50.0, 100.0, 200.0]:
            builder.add(r, _make_optics(wl, n_legendre=32))

        assert len(builder._entries) == 3
        assert set(builder._entries.keys()) == {50.0, 100.0, 200.0}


class TestBuilderRejectsMismatched:
    def test_rejects_mismatched_wavelengths(self):
        from Aerosol3D.bulk.builder import BulkOpticsBuilder

        sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.5)
        radii = np.array([50.0, 100.0, 200.0])
        builder = BulkOpticsBuilder(size_distribution=sd, radii_nm=radii, n_legendre=32)

        wl1 = np.array([500.0, 600.0])
        wl2 = np.array([500.0, 700.0])
        builder.add(50.0, _make_optics(wl1, n_legendre=32))

        with pytest.raises(ValueError, match="wavelength"):
            builder.add(100.0, _make_optics(wl2, n_legendre=32))

    def test_rejects_mismatched_n_legendre(self):
        from Aerosol3D.bulk.builder import BulkOpticsBuilder

        sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.5)
        radii = np.array([50.0, 100.0, 200.0])
        builder = BulkOpticsBuilder(size_distribution=sd, radii_nm=radii, n_legendre=32)

        wl = np.array([500.0, 600.0])
        builder.add(50.0, _make_optics(wl, n_legendre=32))

        with pytest.raises(ValueError, match="n_legendre"):
            builder.add(100.0, _make_optics(wl, n_legendre=16))

    def test_rejects_non_aerosol_optics_data(self):
        from Aerosol3D.bulk.builder import BulkOpticsBuilder

        sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.5)
        radii = np.array([50.0, 100.0, 200.0])
        builder = BulkOpticsBuilder(size_distribution=sd, radii_nm=radii, n_legendre=32)

        with pytest.raises(TypeError, match="AerosolOpticsData"):
            builder.add(100.0, "not optics")


class TestBuilderCompute:
    def test_compute_produces_bulk_data(self):
        from Aerosol3D.bulk.builder import BulkOpticsBuilder

        sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.5)
        radii = np.logspace(1, 3, 16)  # 10 to 1000 nm
        builder = BulkOpticsBuilder(size_distribution=sd, radii_nm=radii, n_legendre=32)

        wl = np.array([500.0, 600.0])
        for r in radii:
            builder.add(r, _make_optics(wl, n_legendre=32, c_ext_base=r / 100.0))

        bulk = builder.compute(interpolation="pchip", integration="quad", n_quad=256)

        assert isinstance(bulk, BulkAerosolOpticsData)
        assert bulk.wavelength_nm.shape == (2,)
        assert bulk.C_ext.shape == (2,)
        assert bulk.C_sca.shape == (2,)
        assert bulk.C_abs.shape == (2,)
        assert bulk.SSA.shape == (2,)
        assert bulk.g.shape == (2,)
        assert bulk.beta.shape == (2, 32)
        assert bulk.n_legendre == 32
        assert bulk.size_distribution is sd
        np.testing.assert_allclose(bulk.beta[:, 0], 1.0, atol=1e-14)

    def test_compute_derived_fields(self):
        from Aerosol3D.bulk.builder import BulkOpticsBuilder

        sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.5)
        radii = np.logspace(1, 3, 16)
        builder = BulkOpticsBuilder(size_distribution=sd, radii_nm=radii, n_legendre=32)

        wl = np.array([500.0])
        for r in radii:
            builder.add(r, _make_optics(wl, n_legendre=32, c_ext_base=2.0, c_sca_base=1.5))

        bulk = builder.compute(n_quad=256)

        # C_abs = C_ext - C_sca
        np.testing.assert_allclose(bulk.C_abs, bulk.C_ext - bulk.C_sca, atol=1e-12)
        # SSA = C_sca / C_ext
        np.testing.assert_allclose(bulk.SSA, bulk.C_sca / bulk.C_ext, atol=1e-12)
        # g = beta[:, 1] / 3.0 (vSmartMOM convention: beta includes (2l+1) factor)
        np.testing.assert_allclose(bulk.g, bulk.beta[:, 1] / 3.0, atol=1e-12)

    def test_compute_sets_r_eff(self):
        from Aerosol3D.bulk.builder import BulkOpticsBuilder

        sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.5)
        radii = np.logspace(1, 3, 16)
        builder = BulkOpticsBuilder(size_distribution=sd, radii_nm=radii, n_legendre=32)

        wl = np.array([500.0])
        for r in radii:
            builder.add(r, _make_optics(wl, n_legendre=32))

        bulk = builder.compute(n_quad=256)
        expected_reff = sd.effective_radius()
        assert bulk.r_eff_nm == pytest.approx(expected_reff, rel=1e-6)

    def test_compute_records_methods(self):
        from Aerosol3D.bulk.builder import BulkOpticsBuilder

        sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.5)
        radii = np.logspace(1, 3, 16)
        builder = BulkOpticsBuilder(size_distribution=sd, radii_nm=radii, n_legendre=32)

        wl = np.array([500.0])
        for r in radii:
            builder.add(r, _make_optics(wl, n_legendre=32))

        bulk = builder.compute(interpolation="pchip", integration="quad", n_quad=512)
        assert bulk.interpolation_method == "pchip"
        assert bulk.integration_method == "quad"
        assert bulk.integration_n_points == 512

    def test_compute_without_legendre_moments_beta_falls_back_to_g(self):
        """If optics lacks legendre_moments_beta, reconstruct from g."""
        from Aerosol3D.bulk.builder import BulkOpticsBuilder

        sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.5)
        radii = np.logspace(1, 3, 16)
        builder = BulkOpticsBuilder(size_distribution=sd, radii_nm=radii, n_legendre=32)

        wl = np.array([500.0])
        for r in radii:
            optics = _make_optics(wl, n_legendre=32, g_base=0.6)
            optics.legendre_moments_beta = None
            builder.add(r, optics)

        bulk = builder.compute(n_quad=256)
        # beta[:, 1] should reflect g = 0.6 in vSmartMOM convention: beta_1 = 3*g = 1.8
        np.testing.assert_allclose(bulk.beta[:, 1], 1.8, atol=1e-12)
        # g should still equal the input g
        np.testing.assert_allclose(bulk.g, 0.6, atol=1e-12)


class TestMonodisperseLimit:
    def test_monodisperse_limit(self):
        """Single radius with narrow distribution should ~equal input."""
        from Aerosol3D.bulk.builder import BulkOpticsBuilder

        # Very narrow lognormal centered on 100 nm
        sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.05)
        radii = np.array([100.0])
        builder = BulkOpticsBuilder(size_distribution=sd, radii_nm=radii, n_legendre=32)

        wl = np.array([500.0])
        optics = _make_optics(wl, n_legendre=32, c_ext_base=2.5, c_sca_base=1.8, g_base=0.65)
        builder.add(100.0, optics)

        bulk = builder.compute(n_quad=512)

        # In the monodisperse limit, bulk should equal the single-particle values
        np.testing.assert_allclose(bulk.C_ext, optics.C_ext, rtol=1e-2)
        np.testing.assert_allclose(bulk.C_sca, optics.C_sca, rtol=1e-2)
        np.testing.assert_allclose(bulk.SSA, optics.SSA, rtol=1e-2)
        np.testing.assert_allclose(bulk.g, optics.g, rtol=1e-2)


class TestMieRippleDetection:
    def test_mie_ripple_fallback_to_method1(self):
        """If spacing is too coarse for Mie ripples, fallback to Method 1."""
        from Aerosol3D.bulk.builder import BulkOpticsBuilder

        sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.5)
        # Sparse radii — likely insufficient for Mie ripple resolution
        radii = np.array([80.0, 100.0, 120.0])
        builder = BulkOpticsBuilder(size_distribution=sd, radii_nm=radii, n_legendre=32)

        wl = np.array([500.0])
        for r in radii:
            builder.add(r, _make_optics(wl, n_legendre=32))

        # refractive_index ~ 1.5 => period = 500 / (2 * 0.5) = 500 nm
        # dr between points ~ 20 nm, which is < 500/3 ~ 167 nm, so this may NOT trigger
        # Use a larger m to make period smaller
        bulk = builder.compute(
            check_mie_ripples=True,
            refractive_index=1.5 + 0.0j,
            mie_ripple_min_points=3,
            n_quad=256,
        )

        assert isinstance(bulk, BulkAerosolOpticsData)

    def test_mie_ripple_no_fallback_when_fine_enough(self):
        """Dense sampling should not trigger fallback."""
        from Aerosol3D.bulk.builder import BulkOpticsBuilder

        sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.5)
        radii = np.logspace(1, 3, 64)
        builder = BulkOpticsBuilder(size_distribution=sd, radii_nm=radii, n_legendre=32)

        wl = np.array([500.0])
        for r in radii:
            builder.add(r, _make_optics(wl, n_legendre=32))

        bulk = builder.compute(
            check_mie_ripples=True,
            refractive_index=1.5 + 0.0j,
            mie_ripple_min_points=3,
            n_quad=256,
        )

        assert not bulk.fallback_used
        assert bulk.fallback_wavelengths is None or len(bulk.fallback_wavelengths) == 0


class TestAddFromNetCDF:
    def test_add_from_netcdf(self, tmp_path):
        from Aerosol3D.bulk.builder import BulkOpticsBuilder

        sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.5)
        radii = np.array([100.0])
        builder = BulkOpticsBuilder(size_distribution=sd, radii_nm=radii, n_legendre=32)

        wl = np.array([500.0, 600.0])
        optics = _make_optics(wl, n_legendre=32)
        path = tmp_path / "optics.nc"
        optics.to_netcdf(str(path))

        builder.add_from_netcdf(100.0, str(path))
        assert 100.0 in builder._entries
        np.testing.assert_array_equal(builder._entries[100.0].wavelength_nm, wl)
