"""Tests for bulk aerosol NetCDF I/O."""

import os
import tempfile

import numpy as np
import pytest

from Aerosol3D.bulk.datastructs import BulkAerosolOpticsData, SizeDistribution
from Aerosol3D.bulk.io import (
    bulk_from_netcdf,
    bulk_to_netcdf,
    bulk_to_vsmartmom_netcdf,
)


def _make_bulk():
    """Create a fully-populated BulkAerosolOpticsData for testing."""
    n_wavelength = 3
    n_legendre = 8
    n_radii = 4

    wavelength_nm = np.array([400.0, 550.0, 700.0])
    C_ext = np.array([1.0, 2.0, 3.0])
    C_sca = np.array([0.9, 1.8, 2.7])
    C_abs = C_ext - C_sca
    SSA = C_sca / C_ext
    g = np.array([0.65, 0.70, 0.75])

    beta = np.ones((n_wavelength, n_legendre))
    beta[:, 1] = 3.0 * g

    sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.3)
    radii_nm = np.array([50.0, 100.0, 150.0, 200.0])
    radii_weights = np.array([0.1, 0.4, 0.3, 0.2])

    per_radius_C_ext = np.random.rand(n_radii, n_wavelength)
    per_radius_C_sca = np.random.rand(n_radii, n_wavelength)
    per_radius_beta = np.ones((n_radii, n_wavelength, n_legendre))

    return BulkAerosolOpticsData(
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
        fallback_used=True,
        fallback_wavelengths=[550.0],
        tau_ref=0.5,
        concentration_method="mass_mixing_ratio",
        concentration_kwargs={"mmr": 1e-6},
    )


def test_to_netcdf_and_back():
    """Round-trip a BulkAerosolOpticsData through NetCDF and verify all fields."""
    bulk = _make_bulk()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "bulk.nc")
        bulk_to_netcdf(bulk, path)
        restored = bulk_from_netcdf(path)

    assert np.allclose(restored.wavelength_nm, bulk.wavelength_nm)
    assert np.allclose(restored.C_ext, bulk.C_ext)
    assert np.allclose(restored.C_sca, bulk.C_sca)
    assert np.allclose(restored.C_abs, bulk.C_abs)
    assert np.allclose(restored.SSA, bulk.SSA)
    assert np.allclose(restored.g, bulk.g)
    assert np.allclose(restored.beta, bulk.beta)
    assert restored.n_legendre == bulk.n_legendre

    assert restored.size_distribution is not None
    assert restored.size_distribution.dist_type == bulk.size_distribution.dist_type
    assert restored.size_distribution.params == bulk.size_distribution.params

    assert np.allclose(restored.radii_nm, bulk.radii_nm)
    assert np.allclose(restored.radii_weights, bulk.radii_weights)
    assert np.allclose(restored.per_radius_C_ext, bulk.per_radius_C_ext)
    assert np.allclose(restored.per_radius_C_sca, bulk.per_radius_C_sca)
    assert np.allclose(restored.per_radius_beta, bulk.per_radius_beta)

    assert restored.r_eff_nm == bulk.r_eff_nm
    assert restored.interpolation_method == bulk.interpolation_method
    assert restored.integration_method == bulk.integration_method
    assert restored.integration_n_points == bulk.integration_n_points
    assert restored.fallback_used == bulk.fallback_used
    assert restored.fallback_wavelengths == bulk.fallback_wavelengths
    assert restored.tau_ref == bulk.tau_ref
    assert restored.concentration_method == bulk.concentration_method
    assert restored.concentration_kwargs == bulk.concentration_kwargs


def test_vsmartmom_export():
    """Verify vSmartMOM-compatible NetCDF writes and contains expected data."""
    bulk = _make_bulk()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "vsmartmom.nc")
        bulk_to_vsmartmom_netcdf(bulk, path, tau_ref=0.3)

        import xarray as xr

        ds = xr.open_dataset(path)
        assert "bulk_SSA" in ds
        assert "bulk_C_ext_nm2" in ds
        assert "bulk_C_sca_nm2" in ds
        assert "bulk_beta" in ds
        assert np.allclose(ds["wavelength_nm"].values, bulk.wavelength_nm)
        assert np.allclose(ds["bulk_SSA"].values, bulk.SSA)
        assert np.allclose(ds["bulk_beta"].values, bulk.beta)
        assert ds.attrs["tau_ref"] == pytest.approx(0.3)
        assert ds.attrs["n_legendre"] == bulk.n_legendre
        ds.close()
