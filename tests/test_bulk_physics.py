"""Physical validation tests for bulk aerosol optics computation.

Tests verify fundamental physical constraints that must hold regardless
of the specific size distribution or optical properties:
- SSA bounds [0, 1]
- beta[:, 0] == 1 (normalization of Legendre moments)
- Energy conservation: C_ext = C_sca + C_abs
- Monodisperse limit: narrow distribution approximates single-particle result
"""

from __future__ import annotations

import numpy as np

from Aerosol3D.bulk.builder import BulkOpticsBuilder
from Aerosol3D.bulk.datastructs import SizeDistribution
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

    # Build beta from g: beta_1 = 3*g (vSmartMOM convention), rest = 0, beta_0 = 1
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


class TestSSABounds:
    """Verify 0 <= SSA <= 1 for all wavelengths."""

    def test_ssa_between_zero_and_one__scattering_only(self):
        """Non-absorbing particles: SSA == 1."""
        sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.5)
        radii = np.logspace(1, 3, 16)
        builder = BulkOpticsBuilder(size_distribution=sd, radii_nm=radii, n_legendre=32)

        wl = np.array([400.0, 500.0, 600.0, 700.0])
        for r in radii:
            builder.add(r, _make_optics(wl, c_ext_base=2.0, c_sca_base=2.0))

        bulk = builder.compute(n_quad=256)
        assert np.all(bulk.SSA >= 0.0)
        assert np.all(bulk.SSA <= 1.0)
        np.testing.assert_allclose(bulk.SSA, 1.0, atol=1e-12)

    def test_ssa_between_zero_and_one__absorbing(self):
        """Strongly absorbing particles: SSA near 0."""
        sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.5)
        radii = np.logspace(1, 3, 16)
        builder = BulkOpticsBuilder(size_distribution=sd, radii_nm=radii, n_legendre=32)

        wl = np.array([400.0, 500.0, 600.0])
        for r in radii:
            builder.add(r, _make_optics(wl, c_ext_base=2.0, c_sca_base=0.1))

        bulk = builder.compute(n_quad=256)
        assert np.all(bulk.SSA >= 0.0)
        assert np.all(bulk.SSA <= 1.0)
        np.testing.assert_allclose(bulk.SSA, 0.05, atol=1e-12)

    def test_ssa_between_zero_and_one__mixed(self):
        """Varying C_ext and C_sca across radii — some absorbing, some scattering."""
        sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.5)
        radii = np.logspace(1, 3, 16)
        builder = BulkOpticsBuilder(size_distribution=sd, radii_nm=radii, n_legendre=32)

        wl = np.array([400.0, 500.0, 600.0, 700.0, 800.0])
        for i, r in enumerate(radii):
            # Alternate between scattering-dominated and absorption-dominated
            if i % 2 == 0:
                c_sca = 1.5
                c_ext = 1.6
            else:
                c_sca = 0.3
                c_ext = 1.0
            builder.add(r, _make_optics(wl, c_ext_base=c_ext, c_sca_base=c_sca))

        bulk = builder.compute(n_quad=256)
        assert np.all(bulk.SSA >= 0.0)
        assert np.all(bulk.SSA <= 1.0)

    def test_ssa_between_zero_and_one__wavelength_dependent(self):
        """Wavelength-dependent optical properties with varying absorption."""
        sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.5)
        radii = np.logspace(1, 3, 16)
        builder = BulkOpticsBuilder(size_distribution=sd, radii_nm=radii, n_legendre=32)

        wl = np.array([300.0, 400.0, 500.0, 600.0, 700.0, 800.0])
        for r in radii:
            # Make SSA wavelength-dependent: UV absorbing, VIS scattering
            c_ext = np.full_like(wl, 2.0)
            c_sca = np.array([0.2, 0.5, 1.0, 1.5, 1.8, 1.9])
            optics = _make_optics(wl, c_ext_base=1.0, c_sca_base=1.0)
            optics.C_ext = c_ext
            optics.C_sca = c_sca
            optics.C_abs = c_ext - c_sca
            optics.SSA = c_sca / c_ext
            builder.add(r, optics)

        bulk = builder.compute(n_quad=256)
        assert np.all(bulk.SSA >= 0.0)
        assert np.all(bulk.SSA <= 1.0)


class TestBeta0Normalization:
    """Verify beta[:, 0] == 1.0 exactly (within machine epsilon)."""

    def test_beta0_is_exactly_one__single_wavelength(self):
        sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.5)
        radii = np.logspace(1, 3, 16)
        builder = BulkOpticsBuilder(size_distribution=sd, radii_nm=radii, n_legendre=32)

        wl = np.array([500.0])
        for r in radii:
            builder.add(r, _make_optics(wl, c_ext_base=2.0, c_sca_base=1.5))

        bulk = builder.compute(n_quad=256)
        np.testing.assert_allclose(bulk.beta[:, 0], 1.0, atol=np.finfo(float).eps)

    def test_beta0_is_exactly_one__multiple_wavelengths(self):
        sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.5)
        radii = np.logspace(1, 3, 16)
        builder = BulkOpticsBuilder(size_distribution=sd, radii_nm=radii, n_legendre=32)

        wl = np.array([400.0, 500.0, 600.0, 700.0, 800.0])
        for r in radii:
            builder.add(r, _make_optics(wl, c_ext_base=2.0, c_sca_base=1.5))

        bulk = builder.compute(n_quad=256)
        np.testing.assert_allclose(bulk.beta[:, 0], 1.0, atol=np.finfo(float).eps)

    def test_beta0_is_exactly_one__varying_g(self):
        """beta_0 must remain 1 even when asymmetry parameter varies across radii."""
        sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.5)
        radii = np.logspace(1, 3, 16)
        builder = BulkOpticsBuilder(size_distribution=sd, radii_nm=radii, n_legendre=32)

        wl = np.array([500.0])
        for i, r in enumerate(radii):
            g = 0.5 + 0.4 * (i / max(1, len(radii) - 1))  # g from 0.5 to 0.9
            builder.add(r, _make_optics(wl, c_ext_base=2.0, c_sca_base=1.5, g_base=g))

        bulk = builder.compute(n_quad=256)
        np.testing.assert_allclose(bulk.beta[:, 0], 1.0, atol=np.finfo(float).eps)


class TestEnergyConservation:
    """Verify C_ext = C_sca + C_abs (energy conservation)."""

    def test_energy_conservation__single_wavelength(self):
        sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.5)
        radii = np.logspace(1, 3, 16)
        builder = BulkOpticsBuilder(size_distribution=sd, radii_nm=radii, n_legendre=32)

        wl = np.array([500.0])
        for r in radii:
            builder.add(r, _make_optics(wl, c_ext_base=2.0, c_sca_base=1.5))

        bulk = builder.compute(n_quad=256)
        np.testing.assert_allclose(bulk.C_ext, bulk.C_sca + bulk.C_abs, atol=1e-12)

    def test_energy_conservation__multiple_wavelengths(self):
        sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.5)
        radii = np.logspace(1, 3, 16)
        builder = BulkOpticsBuilder(size_distribution=sd, radii_nm=radii, n_legendre=32)

        wl = np.array([400.0, 500.0, 600.0, 700.0, 800.0])
        for r in radii:
            builder.add(r, _make_optics(wl, c_ext_base=2.0, c_sca_base=1.5))

        bulk = builder.compute(n_quad=256)
        np.testing.assert_allclose(bulk.C_ext, bulk.C_sca + bulk.C_abs, atol=1e-12)

    def test_energy_conservation__wavelength_dependent(self):
        """Energy conservation must hold even with wavelength-varying properties."""
        sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.5)
        radii = np.logspace(1, 3, 16)
        builder = BulkOpticsBuilder(size_distribution=sd, radii_nm=radii, n_legendre=32)

        wl = np.array([300.0, 400.0, 500.0, 600.0, 700.0, 800.0])
        for r in radii:
            c_ext = np.array([1.5, 1.8, 2.0, 2.2, 2.5, 2.8])
            c_sca = np.array([0.5, 1.0, 1.5, 1.8, 2.0, 2.2])
            optics = _make_optics(wl, c_ext_base=1.0, c_sca_base=1.0)
            optics.C_ext = c_ext
            optics.C_sca = c_sca
            optics.C_abs = c_ext - c_sca
            optics.SSA = c_sca / c_ext
            builder.add(r, optics)

        bulk = builder.compute(n_quad=256)
        np.testing.assert_allclose(bulk.C_ext, bulk.C_sca + bulk.C_abs, atol=1e-12)

    def test_energy_conservation__derived_C_abs_matches(self):  # noqa: N802
        """Explicitly verify that bulk.C_abs equals the difference."""
        sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.5)
        radii = np.logspace(1, 3, 16)
        builder = BulkOpticsBuilder(size_distribution=sd, radii_nm=radii, n_legendre=32)

        wl = np.array([500.0, 600.0])
        for r in radii:
            builder.add(r, _make_optics(wl, c_ext_base=3.0, c_sca_base=0.7))

        bulk = builder.compute(n_quad=256)
        expected_C_abs = bulk.C_ext - bulk.C_sca
        np.testing.assert_allclose(bulk.C_abs, expected_C_abs, atol=1e-12)


class TestMonodisperseLimit:
    """Single radius with very narrow distribution should approximate the single-particle result."""

    def test_monodisperse_limit_C_ext(self):  # noqa: N802
        sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.01)
        radii = np.array([100.0])
        builder = BulkOpticsBuilder(size_distribution=sd, radii_nm=radii, n_legendre=32)

        wl = np.array([500.0])
        optics = _make_optics(wl, c_ext_base=2.5, c_sca_base=1.8, g_base=0.65)
        builder.add(100.0, optics)

        bulk = builder.compute(n_quad=512)
        np.testing.assert_allclose(bulk.C_ext, optics.C_ext, rtol=1e-2)

    def test_monodisperse_limit_C_sca(self):  # noqa: N802
        sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.01)
        radii = np.array([100.0])
        builder = BulkOpticsBuilder(size_distribution=sd, radii_nm=radii, n_legendre=32)

        wl = np.array([500.0])
        optics = _make_optics(wl, c_ext_base=2.5, c_sca_base=1.8, g_base=0.65)
        builder.add(100.0, optics)

        bulk = builder.compute(n_quad=512)
        np.testing.assert_allclose(bulk.C_sca, optics.C_sca, rtol=1e-2)

    def test_monodisperse_limit_SSA(self):  # noqa: N802
        sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.01)
        radii = np.array([100.0])
        builder = BulkOpticsBuilder(size_distribution=sd, radii_nm=radii, n_legendre=32)

        wl = np.array([500.0])
        optics = _make_optics(wl, c_ext_base=2.5, c_sca_base=1.8, g_base=0.65)
        builder.add(100.0, optics)

        bulk = builder.compute(n_quad=512)
        np.testing.assert_allclose(bulk.SSA, optics.SSA, rtol=1e-2)

    def test_monodisperse_limit_g(self):
        sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.01)
        radii = np.array([100.0])
        builder = BulkOpticsBuilder(size_distribution=sd, radii_nm=radii, n_legendre=32)

        wl = np.array([500.0])
        optics = _make_optics(wl, c_ext_base=2.5, c_sca_base=1.8, g_base=0.65)
        builder.add(100.0, optics)

        bulk = builder.compute(n_quad=512)
        np.testing.assert_allclose(bulk.g, optics.g, rtol=1e-2)

    def test_monodisperse_limit_beta(self):
        sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.01)
        radii = np.array([100.0])
        builder = BulkOpticsBuilder(size_distribution=sd, radii_nm=radii, n_legendre=32)

        wl = np.array([500.0])
        optics = _make_optics(wl, c_ext_base=2.5, c_sca_base=1.8, g_base=0.65)
        builder.add(100.0, optics)

        bulk = builder.compute(n_quad=512)
        # beta in bulk is in vSmartMOM convention (beta_l = (2l+1)*g_l)
        # Input optics has legendre_moments_beta in Aerosol3D convention (g_l)
        # So bulk.beta[:, 1] = 3 * g, while optics.legendre_moments_beta[:, 1] = g
        expected_beta1 = optics.legendre_moments_beta[:, 1] * 3.0
        np.testing.assert_allclose(bulk.beta[:, 1], expected_beta1, rtol=1e-2)

    def test_monodisperse_limit_multiple_wavelengths(self):
        """Narrow distribution should approximate single-particle result at all wavelengths."""
        sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.01)
        radii = np.array([100.0])
        builder = BulkOpticsBuilder(size_distribution=sd, radii_nm=radii, n_legendre=32)

        wl = np.array([400.0, 500.0, 600.0, 700.0])
        optics = _make_optics(wl, c_ext_base=2.5, c_sca_base=1.8, g_base=0.65)
        builder.add(100.0, optics)

        bulk = builder.compute(n_quad=512)
        np.testing.assert_allclose(bulk.C_ext, optics.C_ext, rtol=1e-2)
        np.testing.assert_allclose(bulk.C_sca, optics.C_sca, rtol=1e-2)
        np.testing.assert_allclose(bulk.SSA, optics.SSA, rtol=1e-2)
        np.testing.assert_allclose(bulk.g, optics.g, rtol=1e-2)
