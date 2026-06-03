"""Tests for VSmartMOMResult dataclass and NetCDF I/O."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from Aerosol3D.vsmartmom.result import VSmartMOMResult


class TestVSmartMOMResult:
    """Test suite for VSmartMOMResult."""

    def test_result_creation(self) -> None:
        """Verify dataclass can be constructed with all fields."""
        n_stokes = 3
        n_vza = 5
        n_wl = 4
        n_layer = 2

        R = np.ones((n_stokes, n_vza, n_wl))
        T = np.zeros((n_stokes, n_vza, n_wl))
        wavelengths = np.array([500.0, 600.0, 700.0, 800.0])
        wavenumbers = np.array([20000.0, 16666.7, 14285.7, 12500.0])
        vza = np.array([0.0, 15.0, 30.0, 45.0, 60.0])
        vaz = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        sza = 30.0
        tau_per_layer = np.ones((n_wl, n_layer))
        model_info = {"description": "test model", "version": "1.0"}

        result = VSmartMOMResult(
            R=R,
            T=T,
            wavelengths=wavelengths,
            wavenumbers=wavenumbers,
            vza=vza,
            vaz=vaz,
            sza=sza,
            tau_per_layer=tau_per_layer,
            model_info=model_info,
        )

        assert result.R is R
        assert result.T is T
        assert result.wavelengths is wavelengths
        assert result.wavenumbers is wavenumbers
        assert result.vza is vza
        assert result.vaz is vaz
        assert result.sza == sza
        assert result.tau_per_layer is tau_per_layer
        assert result.model_info is model_info

    def test_result_to_netcdf_roundtrip(self) -> None:
        """Save to NetCDF and reload, verify all fields match."""
        n_stokes = 3
        n_vza = 5
        n_wl = 4
        n_layer = 2

        original = VSmartMOMResult(
            R=np.random.rand(n_stokes, n_vza, n_wl),
            T=np.random.rand(n_stokes, n_vza, n_wl),
            wavelengths=np.array([500.0, 600.0, 700.0, 800.0]),
            wavenumbers=np.array([20000.0, 16666.7, 14285.7, 12500.0]),
            vza=np.array([0.0, 15.0, 30.0, 45.0, 60.0]),
            vaz=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            sza=30.0,
            tau_per_layer=np.ones((n_wl, n_layer)) * 0.5,
            model_info={"description": "test model", "version": "1.0"},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_result.nc"
            original.to_netcdf(str(path))

            loaded = VSmartMOMResult.from_netcdf(str(path))

            np.testing.assert_array_equal(loaded.R, original.R)
            np.testing.assert_array_equal(loaded.T, original.T)
            np.testing.assert_array_equal(loaded.wavelengths, original.wavelengths)
            np.testing.assert_array_equal(loaded.wavenumbers, original.wavenumbers)
            np.testing.assert_array_equal(loaded.vza, original.vza)
            np.testing.assert_array_equal(loaded.vaz, original.vaz)
            assert loaded.sza == original.sza
            np.testing.assert_array_equal(loaded.tau_per_layer, original.tau_per_layer)
            assert loaded.model_info == original.model_info
