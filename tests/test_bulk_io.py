"""Tests for bulk aerosol data structures."""

import numpy as np
import pytest

from Aerosol3D.bulk.datastructs import BulkAerosolOpticsData, SizeDistribution


def test_create_basic():
    """Create a BulkAerosolOpticsData instance and verify basic field access."""
    n_wavelength = 3
    n_legendre = 8

    wavelength_nm = np.array([400.0, 550.0, 700.0])
    C_ext = np.array([1.0, 2.0, 3.0])
    C_sca = np.array([0.9, 1.8, 2.7])
    C_abs = C_ext - C_sca
    SSA = C_sca / C_ext
    g = np.array([0.65, 0.70, 0.75])

    # beta[..., 0] must be 1 by definition (k_0 / k_0)
    beta = np.ones((n_wavelength, n_legendre))
    beta[:, 1] = 3.0 * g  # g = beta_1 / 3.0

    sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.3)
    radii_nm = np.array([50.0, 100.0, 200.0])
    radii_weights = np.array([0.2, 0.6, 0.2])

    per_radius_C_ext = np.random.rand(len(radii_nm), n_wavelength)
    per_radius_C_sca = np.random.rand(len(radii_nm), n_wavelength)
    per_radius_beta = np.ones((len(radii_nm), n_wavelength, n_legendre))

    data = BulkAerosolOpticsData(
        wavelength_nm=wavelength_nm,
        C_ext=C_ext,
        C_sca=C_sca,
        C_abs=C_abs,
        SSA=SSA,
        g=g,
        beta=beta,
        n_legendre=n_legendre,
        size_distribution=sd,
        radii_nm=radii_nm,
        radii_weights=radii_weights,
        per_radius_C_ext=per_radius_C_ext,
        per_radius_C_sca=per_radius_C_sca,
        per_radius_beta=per_radius_beta,
        r_eff_nm=120.0,
        interpolation_method="linear",
        integration_method="quad",
        integration_n_points=64,
        fallback_used=False,
        fallback_wavelengths=[],
        tau_ref=0.5,
        concentration_method="mass_mixing_ratio",
        concentration_kwargs={"mmr": 1e-6},
    )

    assert data.n_legendre == n_legendre
    assert np.allclose(data.beta[:, 0], 1.0)
    assert np.allclose(data.SSA, C_sca / C_ext)
    assert data.size_distribution.dist_type == "lognormal"
    assert data.r_eff_nm == 120.0
    assert data.integration_n_points == 64
    assert not data.fallback_used
    assert data.tau_ref == 0.5
