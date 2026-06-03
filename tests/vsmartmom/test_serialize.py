"""Tests for vSmartMOM input serialization."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from Aerosol3D.vsmartmom.serialize import compute_tau_profile, serialize_input


class TestComputeTauProfile:
    """Test suite for compute_tau_profile."""

    def test_compute_tau_profile_basic(self) -> None:
        """Verify tau formula with known values."""
        # 2 wavelengths, 3 layers
        C_ext = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # shape (2, 3), nm^2
        heights = np.array([0.0, 1000.0, 2000.0, 3000.0])  # 4 iface -> 3 layers, m
        number_conc = np.array([1.0, 2.0, 3.0])  # cm^-3

        tau = compute_tau_profile(C_ext, heights, number_conc)

        assert tau.shape == (2, 3)
        dz = 1000.0  # m per layer
        scale = 1e-6  # conversion factor
        expected_wl0 = np.array(
            [
                1.0 * 1.0 * dz * scale,
                2.0 * 2.0 * dz * scale,
                3.0 * 3.0 * dz * scale,
            ]
        )
        expected_wl1 = np.array(
            [
                1.0 * 4.0 * dz * scale,
                2.0 * 5.0 * dz * scale,
                3.0 * 6.0 * dz * scale,
            ]
        )
        np.testing.assert_array_almost_equal(tau[0], expected_wl0)
        np.testing.assert_array_almost_equal(tau[1], expected_wl1)

    def test_compute_tau_profile_single_wavelength(self) -> None:
        """Single wavelength case: C_ext shape (n_layer,)."""
        C_ext = np.array([1.5, 2.5, 3.5])  # shape (3,), nm^2
        heights = np.array([0.0, 500.0, 1000.0, 1500.0])  # m
        number_conc = np.array([10.0, 20.0, 30.0])  # cm^-3

        tau = compute_tau_profile(C_ext, heights, number_conc)

        assert tau.shape == (1, 3)
        dz = 500.0
        scale = 1e-6
        expected = np.array(
            [
                10.0 * 1.5 * dz * scale,
                20.0 * 2.5 * dz * scale,
                30.0 * 3.5 * dz * scale,
            ]
        )
        np.testing.assert_array_almost_equal(tau[0], expected)

    def test_compute_tau_profile_zero_conc(self) -> None:
        """Zero concentration should yield zero tau."""
        C_ext = np.array([[1.0, 2.0], [3.0, 4.0]])  # shape (2, 2)
        heights = np.array([0.0, 1000.0, 2000.0])  # m
        number_conc = np.array([0.0, 0.0])  # cm^-3

        tau = compute_tau_profile(C_ext, heights, number_conc)

        assert tau.shape == (2, 2)
        np.testing.assert_array_equal(tau, np.zeros((2, 2)))


class TestSerializeInput:
    """Test suite for serialize_input."""

    def test_serialize_input_creates_file(self) -> None:
        """Verify NetCDF file is created with correct structure and variables."""
        n_wl = 3
        n_layer = 2
        n_legendre = 4
        n_vza = 5

        wavelengths_nm = np.array([500.0, 600.0, 700.0])
        C_ext = np.ones((n_wl, n_layer)) * 1.5  # nm^2
        SSA = np.array([0.9, 0.85, 0.8])
        beta = np.ones((n_wl, n_legendre)) * 0.5
        heights = np.array([0.0, 1000.0, 2000.0])  # m, 3 iface -> 2 layers
        number_conc = np.array([100.0, 200.0])  # cm^-3
        sza = 30.0
        vza = np.array([0.0, 15.0, 30.0, 45.0, 60.0])
        vaz = np.array([0.0, 0.0, 90.0, 180.0, 270.0])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "vsmartmom_input.nc"
            serialize_input(
                wavelengths_nm=wavelengths_nm,
                C_ext=C_ext,
                SSA=SSA,
                beta=beta,
                heights=heights,
                number_conc=number_conc,
                sza=sza,
                vza=vza,
                vaz=vaz,
                output_path=str(output_path),
                n_legendre=n_legendre,
                r_eff_nm=150.0,
                size_distribution_type="lognormal",
            )

            assert output_path.exists()

            import xarray as xr

            ds = xr.open_dataset(str(output_path))

            # Check dimensions
            assert "wavelength" in ds.dims
            assert "layer" in ds.dims
            assert "layer_iface" in ds.dims
            assert "legendre_order" in ds.dims
            assert "vza" in ds.dims
            assert ds.sizes["wavelength"] == n_wl
            assert ds.sizes["layer"] == n_layer
            assert ds.sizes["layer_iface"] == n_layer + 1
            assert ds.sizes["legendre_order"] == n_legendre
            assert ds.sizes["vza"] == n_vza

            # Check variables exist
            assert "wavelength_nm" in ds
            assert "wavenumber_cm" in ds
            assert "height_m" in ds
            assert "layer_thickness_m" in ds
            assert "number_conc_cm3" in ds
            assert "tau" in ds
            assert "SSA" in ds
            assert "beta" in ds
            assert "sza" in ds
            assert "vza" in ds
            assert "vaz" in ds

            # Check variable shapes
            assert ds["wavelength_nm"].shape == (n_wl,)
            assert ds["wavenumber_cm"].shape == (n_wl,)
            assert ds["height_m"].shape == (n_layer + 1,)
            assert ds["layer_thickness_m"].shape == (n_layer,)
            assert ds["number_conc_cm3"].shape == (n_layer,)
            assert ds["tau"].shape == (n_wl, n_layer)
            assert ds["SSA"].shape == (n_wl,)
            assert ds["beta"].shape == (n_wl, n_legendre)
            assert ds["sza"].shape == ()
            assert ds["vza"].shape == (n_vza,)
            assert ds["vaz"].shape == (n_vza,)

            # Check coordinate values
            np.testing.assert_array_equal(ds["wavelength_nm"].values, wavelengths_nm)
            np.testing.assert_array_equal(ds["height_m"].values, heights)
            np.testing.assert_array_equal(ds["vza"].values, vza)
            np.testing.assert_array_equal(ds["vaz"].values, vaz)
            assert float(ds["sza"].values) == sza

            # Check attributes
            assert ds.attrs.get("source") == "Aerosol3D"
            assert ds.attrs.get("n_legendre") == n_legendre
            assert ds.attrs.get("r_eff_nm") == 150.0
            assert ds.attrs.get("size_distribution_type") == "lognormal"

            # Check wavenumber conversion: wavenumber = 1e7 / wavelength(nm)
            expected_wavenumbers = 1e7 / wavelengths_nm
            np.testing.assert_array_almost_equal(ds["wavenumber_cm"].values, expected_wavenumbers)

            # Check layer thickness
            expected_dz = np.diff(heights)
            np.testing.assert_array_equal(ds["layer_thickness_m"].values, expected_dz)

            # Check tau computation
            expected_tau = compute_tau_profile(C_ext, heights, number_conc)
            np.testing.assert_array_almost_equal(ds["tau"].values, expected_tau)

            ds.close()
