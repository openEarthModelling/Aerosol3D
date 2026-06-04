"""VSmartMOMRunner: orchestrates vSmartMOM radiative transfer via Julia subprocess."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from Aerosol3D.vsmartmom.result import VSmartMOMResult
from Aerosol3D.vsmartmom.serialize import compute_tau_profile, serialize_input

if TYPE_CHECKING:
    from Aerosol3D.bulk.datastructs import BulkAerosolOpticsData


@dataclass
class VSmartMOMRunner:
    """Orchestrate vSmartMOM radiative transfer simulations.

    Args:
        julia_project: Path to the Julia project (environment) containing
            vSmartMOM and its dependencies.
        julia_executable: Name or path of the Julia executable. Defaults to
            ``"julia"``.
        cleanup_temp: Whether to delete the temporary working directory after
            the run. Defaults to ``True``.
    """

    julia_project: str | Path | None
    julia_executable: str = "julia"
    cleanup_temp: bool = True

    def __post_init__(self) -> None:
        """Normalize julia_project to a Path object when provided."""
        if self.julia_project is not None:
            self.julia_project = Path(self.julia_project)

    def _check_julia(self) -> None:
        """Verify the Julia executable exists on the system PATH.

        Raises:
            RuntimeError: If the executable cannot be found.
        """
        if shutil.which(self.julia_executable) is None:
            msg = f"Julia executable '{self.julia_executable}' not found on PATH."
            raise RuntimeError(msg)

    def _validate_inputs(
        self,
        heights: np.ndarray,
        number_conc: np.ndarray,
    ) -> None:
        """Validate geometry and concentration inputs.

        Checks:
            * ``len(heights) == len(number_conc) + 1``
            * All concentrations are non-negative.
            * Heights are strictly increasing.

        Args:
            heights: Layer interface heights in metres.
            number_conc: Number concentration per layer in cm^-3.

        Raises:
            ValueError: If any validation check fails.
        """
        heights_arr = np.asarray(heights, dtype=float)
        number_conc_arr = np.asarray(number_conc, dtype=float)

        if len(heights_arr) != len(number_conc_arr) + 1:
            msg = (
                f"heights length ({len(heights_arr)}) must equal "
                f"number_conc length ({len(number_conc_arr)}) + 1"
            )
            raise ValueError(msg)

        if np.any(number_conc_arr < 0):
            msg = "All number_conc values must be non-negative."
            raise ValueError(msg)

        if not np.all(np.diff(heights_arr) > 0):
            msg = "heights must be strictly increasing."
            raise ValueError(msg)

    def run_rt(
        self,
        bulk: BulkAerosolOpticsData,
        heights: np.ndarray,
        number_conc: np.ndarray,
        sza: float,
        vza: np.ndarray,
        vaz: np.ndarray | None = None,
        tau_ref: float | None = None,
    ) -> VSmartMOMResult:
        """Run the vSmartMOM radiative transfer model.

        Workflow:
            1. Validate inputs.
            2. Check that the Julia executable exists.
            3. Extract bulk optical properties (wavelengths, ``C_ext``, ``SSA``,
               ``beta``).
            4. If ``tau_ref`` is provided, scale ``number_conc`` so that the
               total optical depth at the first wavelength matches
               ``tau_ref``.
            5. Create a temporary directory and serialize the inputs to a
               NetCDF file.
            6. Run the Julia script ``scripts/run_rt.jl`` as a subprocess.
            7. Deserialize the output NetCDF into a
               :class:`VSmartMOMResult`.
            8. Return the result.

        Args:
            bulk: Bulk aerosol optical properties.
            heights: Layer interface heights in metres.
            number_conc: Number concentration per layer in cm^-3.
            sza: Solar zenith angle in degrees.
            vza: Viewing zenith angles in degrees.
            vaz: Viewing azimuth angles in degrees. If ``None``, defaults to
                zeros.
            tau_ref: Optional total optical depth override. If given,
                concentrations are scaled so that the column optical depth at
                the first wavelength equals this value.

        Returns:
            The vSmartMOM radiative transfer result.

        Raises:
            ValueError: If input validation fails.
            RuntimeError: If the Julia executable is missing or the subprocess
                returns a non-zero exit code.
        """
        # 1. Validate inputs
        self._validate_inputs(heights, number_conc)

        # 2. Check Julia exists
        self._check_julia()

        # 3. Extract bulk data
        wavelengths = np.asarray(bulk.wavelength_nm, dtype=float)
        C_ext = np.asarray(bulk.C_ext, dtype=float)
        SSA = np.asarray(bulk.SSA, dtype=float)
        beta = np.asarray(bulk.beta, dtype=float)
        n_legendre = bulk.n_legendre

        # 4. tau_ref override: scale concentrations
        number_conc_arr = np.asarray(number_conc, dtype=float)
        if tau_ref is not None:
            # Compute tau at the first wavelength
            tau_first = compute_tau_profile(C_ext[0], heights, number_conc_arr)
            total_tau = float(np.sum(tau_first))
            if total_tau > 0:
                scale = tau_ref / total_tau
                number_conc_arr = number_conc_arr * scale
            elif tau_ref > 0:
                msg = (
                    "Cannot scale concentrations to tau_ref: original total optical depth is zero."
                )
                raise RuntimeError(msg)

        # Default vaz to zeros if not provided
        vza_arr = np.asarray(vza, dtype=float)
        if vaz is None:
            vaz_arr = np.zeros_like(vza_arr)
        else:
            vaz_arr = np.asarray(vaz, dtype=float)

        # 5. Create temp dir and serialize input
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "input.nc"
            output_path = tmp_path / "output.nc"

            serialize_input(
                wavelengths_nm=wavelengths,
                C_ext=C_ext,
                SSA=SSA,
                beta=beta,
                heights=heights,
                number_conc=number_conc_arr,
                sza=sza,
                vza=vza_arr,
                vaz=vaz_arr,
                output_path=str(input_path),
                n_legendre=n_legendre,
            )

            # 6. Run Julia subprocess
            script_path = Path(__file__).parent / "scripts" / "run_rt.jl"
            cmd = [self.julia_executable]
            if self.julia_project is not None:
                cmd.append(f"--project={self.julia_project}")
            cmd.extend([str(script_path), str(input_path), str(output_path)])

            result = subprocess.run(cmd, capture_output=True)  # noqa: S603

            if result.returncode != 0:
                stderr = result.stderr.decode("utf-8", errors="replace")
                msg = f"vSmartMOM Julia script failed (exit {result.returncode}): {stderr}"
                raise RuntimeError(msg)

            # 7. Deserialize output NetCDF
            vsmartmom_result = VSmartMOMResult.from_netcdf(str(output_path))

            # 8. Return result
            return vsmartmom_result
