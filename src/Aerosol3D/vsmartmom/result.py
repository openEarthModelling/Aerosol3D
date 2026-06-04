"""VSmartMOMResult dataclass with NetCDF I/O for radiative transfer results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class VSmartMOMResult:
    """Result container for vSmartMOM radiative transfer simulations.

    Args:
        R: TOA reflectance, shape [n_stokes, n_vza, n_wl].
        T: BOA transmittance, shape [n_stokes, n_vza, n_wl].
        wavelengths: Wavelengths in nm, shape [n_wl].
        wavenumbers: Wavenumbers in cm^-1, shape [n_wl].
        vza: Viewing zenith angles in degrees, shape [n_vza].
        vaz: Viewing azimuth angles in degrees, shape [n_vza].
        sza: Solar zenith angle in degrees.
        tau_per_layer: Optical depth per layer, shape [n_wl, n_layer].
        model_info: Metadata dictionary.
    """

    R: np.ndarray
    T: np.ndarray
    wavelengths: np.ndarray
    wavenumbers: np.ndarray
    vza: np.ndarray
    vaz: np.ndarray
    sza: float
    tau_per_layer: np.ndarray
    model_info: dict[str, Any]

    def to_netcdf(self, path: str) -> None:
        """Save result to NetCDF file using xarray.

        Args:
            path: Output file path.
        """
        import xarray as xr

        data_vars = {
            "R": (["stokes", "vza", "wavelength"], self.R),
            "T": (["stokes", "vza", "wavelength"], self.T),
            "tau_per_layer": (["wavelength", "layer"], self.tau_per_layer),
        }
        coords = {
            "wavelength": (["wavelength"], self.wavelengths),
            "wavenumber": (["wavelength"], self.wavenumbers),
            "vza": (["vza"], self.vza),
            "vaz": (["vza"], self.vaz),
        }
        attrs = {
            "sza": float(self.sza),
            "model_info_json": json.dumps(self.model_info),
        }

        ds = xr.Dataset(data_vars, coords=coords, attrs=attrs)
        ds.to_netcdf(path)

    @classmethod
    def from_netcdf(cls, path: str) -> VSmartMOMResult:
        """Load result from NetCDF file.

        Args:
            path: Input file path.

        Returns:
            VSmartMOMResult instance.
        """
        import xarray as xr

        ds = xr.open_dataset(path)

        model_info = json.loads(ds.attrs.get("model_info_json", "{}"))

        R = _reorder_rt_array(ds["R"].values, ds["R"].dims)
        T = _reorder_rt_array(ds["T"].values, ds["T"].dims)
        tau = ds["tau_per_layer"].values
        _tau_dims = ds["tau_per_layer"].dims
        if _tau_dims in (("layer", "wavelength"), ("input_layer", "wavelength")):
            tau = tau.transpose(1, 0)

        obj = cls(
            R=R,
            T=T,
            wavelengths=ds["wavelength"].values,
            wavenumbers=ds["wavenumber"].values,
            vza=ds["vza"].values,
            vaz=ds["vaz"].values,
            sza=float(ds.attrs["sza"]),
            tau_per_layer=tau,
            model_info=model_info,
        )
        ds.close()
        return obj


def _reorder_rt_array(arr: np.ndarray, dims: tuple[Any, ...]) -> np.ndarray:
    """Reorder RT array from NetCDF dims to (stokes, vza, wavelength).

    NCDatasets.jl (Julia) writes column-major arrays to NetCDF C-order,
    which can reverse dimension order.  This helper detects the actual
    order and permutes back to our convention.
    """
    if dims == ("stokes", "vza", "wavelength"):
        return arr
    if dims == ("wavelength", "vza", "stokes"):
        return arr.transpose(2, 1, 0)
    if dims == ("vza", "stokes", "wavelength"):
        return arr.transpose(1, 0, 2)
    if dims == ("wavelength", "stokes", "vza"):
        return arr.transpose(1, 2, 0)
    if dims == ("stokes", "wavelength", "vza"):
        return arr.transpose(0, 2, 1)
    if dims == ("vza", "wavelength", "stokes"):
        return arr.transpose(2, 0, 1)
    # Fallback: infer by typical sizes (n_stokes <= 4)
    s = arr.shape
    if s[0] <= 4 and s[1] > 4:
        return arr  # likely (stokes, vza, wavelength)
    if s[2] <= 4 and s[1] > 4:
        return arr.transpose(2, 1, 0)  # (wl, vza, stokes)
    if s[1] <= 4 and s[0] > 4:
        return arr.transpose(1, 0, 2)  # (vza, stokes, wl)
    return arr
