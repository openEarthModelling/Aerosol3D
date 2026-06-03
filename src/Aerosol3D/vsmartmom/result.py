"""VSmartMOMResult dataclass with NetCDF I/O for radiative transfer results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class VSmartMOMResult:
    """Result container for vSmartMOM radiative transfer simulations.

    Attributes:
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

        obj = cls(
            R=ds["R"].values,
            T=ds["T"].values,
            wavelengths=ds["wavelength"].values,
            wavenumbers=ds["wavenumber"].values,
            vza=ds["vza"].values,
            vaz=ds["vaz"].values,
            sza=float(ds.attrs["sza"]),
            tau_per_layer=ds["tau_per_layer"].values,
            model_info=model_info,
        )
        ds.close()
        return obj
