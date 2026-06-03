"""Input serialization for vSmartMOM radiative transfer.

Provides functions to compute per-layer optical depth profiles and serialize
them along with single-scattering properties to a NetCDF file compatible with
vSmartMOM.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def compute_tau_profile(
    C_ext: np.ndarray,  # noqa: N803
    heights: Sequence[float],
    number_conc: Sequence[float],
) -> np.ndarray:
    """Compute per-layer optical depth profile.

    Formula::

        tau = N * C_ext * dz * 1e-6

    where ``N`` is number concentration [cm^-3], ``C_ext`` is extinction
    cross-section [nm^2], ``dz`` is layer thickness [m], and ``1e-6`` is the
    unit conversion factor (nm^2 * m * cm^-3 -> dimensionless).

    Args:
        C_ext: Extinction cross-section. Shape ``(n_wl, n_layer)`` or
            ``(n_layer,)`` for a single wavelength.
        heights: Layer interface heights in metres. Length ``n_layer + 1``.
        number_conc: Number concentration per layer in cm^-3.
            Length ``n_layer``.

    Returns:
        Optical depth per layer. Shape ``(n_wl, n_layer)``.  If ``C_ext`` is
        one-dimensional, the returned array has shape ``(1, n_layer)``.
    """
    C_ext_arr = np.atleast_2d(np.asarray(C_ext, dtype=float))
    heights_arr = np.asarray(heights, dtype=float)
    number_conc_arr = np.asarray(number_conc, dtype=float)

    dz = np.diff(heights_arr)
    # tau = N * C_ext * dz * 1e-6
    tau = number_conc_arr[np.newaxis, :] * C_ext_arr * dz[np.newaxis, :] * 1e-6
    return tau


def serialize_input(
    wavelengths_nm: np.ndarray,
    C_ext: np.ndarray,  # noqa: N803
    SSA: np.ndarray,  # noqa: N803
    beta: np.ndarray,
    heights: Sequence[float],
    number_conc: Sequence[float],
    sza: float,
    vza: np.ndarray,
    vaz: np.ndarray,
    output_path: str,
    n_legendre: int = 32,
    r_eff_nm: float | None = None,
    size_distribution_type: str = "",
    **kwargs: Any,
) -> None:
    """Write vSmartMOM input to a NetCDF file.

    Args:
        wavelengths_nm: Wavelengths in nm, shape ``(n_wl,)``.
        C_ext: Extinction cross-section per layer in nm^2,
            shape ``(n_wl, n_layer)`` or ``(n_layer,)``.
        SSA: Single-scattering albedo, shape ``(n_wl,)``.
        beta: Legendre expansion coefficients (beta_l = k_l / (2l+1)),
            shape ``(n_wl, n_legendre)``.
        heights: Layer interface heights in metres, length ``n_layer + 1``.
        number_conc: Number concentration per layer in cm^-3,
            length ``n_layer``.
        sza: Solar zenith angle in degrees.
        vza: Viewing zenith angles in degrees, shape ``(n_vza,)``.
        vaz: Viewing azimuth angles in degrees, shape ``(n_vza,)``.
        output_path: Output NetCDF file path.
        n_legendre: Number of Legendre moments.
        r_eff_nm: Effective radius in nm (optional).
        size_distribution_type: Type of size distribution (optional).
        **kwargs: Additional attributes to store in the NetCDF file.
    """
    import xarray as xr

    wavelengths_nm_arr = np.asarray(wavelengths_nm, dtype=float)
    C_ext_arr = np.atleast_2d(np.asarray(C_ext, dtype=float))
    SSA_arr = np.asarray(SSA, dtype=float)
    beta_arr = np.asarray(beta, dtype=float)
    heights_arr = np.asarray(heights, dtype=float)
    number_conc_arr = np.asarray(number_conc, dtype=float)
    vza_arr = np.asarray(vza, dtype=float)
    vaz_arr = np.asarray(vaz, dtype=float)

    n_layer = len(number_conc_arr)
    n_legendre_actual = beta_arr.shape[1] if beta_arr.ndim > 1 else len(beta_arr)

    # Compute tau profile
    tau = compute_tau_profile(C_ext_arr, heights_arr, number_conc_arr)

    # Layer thicknesses
    dz = np.diff(heights_arr)

    # Wavenumbers in cm^-1
    wavenumber_cm = 1e7 / wavelengths_nm_arr

    data_vars: dict[str, Any] = {
        "wavelength_nm": (["wavelength"], wavelengths_nm_arr),
        "wavenumber_cm": (["wavelength"], wavenumber_cm),
        "height_m": (["layer_iface"], heights_arr),
        "layer_thickness_m": (["layer"], dz),
        "number_conc_cm3": (["layer"], number_conc_arr),
        "tau": (["wavelength", "layer"], tau),
        "SSA": (["wavelength"], SSA_arr),
        "beta": (["wavelength", "legendre_order"], beta_arr),
        "sza": ([], float(sza)),
        "vza": (["vza"], vza_arr),
        "vaz": (["vza"], vaz_arr),
    }

    coords: dict[str, Any] = {
        "layer": (["layer"], np.arange(n_layer)),
        "layer_iface": (["layer_iface"], np.arange(n_layer + 1)),
        "legendre_order": (["legendre_order"], np.arange(n_legendre_actual)),
    }

    attrs: dict[str, Any] = {
        "source": "Aerosol3D",
        "n_legendre": n_legendre,
    }

    if r_eff_nm is not None:
        attrs["r_eff_nm"] = float(r_eff_nm)
    if size_distribution_type:
        attrs["size_distribution_type"] = size_distribution_type

    # Add any extra kwargs as attributes
    attrs.update(kwargs)

    ds = xr.Dataset(data_vars, coords=coords, attrs=attrs)
    ds.to_netcdf(output_path)
