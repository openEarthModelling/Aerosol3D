"""Generic multi-wavelength aerosol optical property export."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .datastructs import OpticalResult

from .legendre import compute_legendre_moments


@dataclass
class AerosolOpticsData:
    """Multi-wavelength aerosol optical properties."""

    wavelength_nm: np.ndarray
    C_ext: np.ndarray
    C_sca: np.ndarray
    C_abs: np.ndarray
    SSA: np.ndarray
    g: np.ndarray
    r_eff_nm: float

    theta_rad: np.ndarray | None = None
    phi_rad: np.ndarray | None = None
    P11: np.ndarray | None = None

    legendre_moments: np.ndarray | None = None
    n_legendre: int = 32

    solver: str = ""
    material: str = ""
    refractive_index_real: np.ndarray | None = None
    refractive_index_imag: np.ndarray | None = None


def from_optical_results(
    results: list[OpticalResult],
    n_legendre: int = 32,
    material_name: str = "",
) -> AerosolOpticsData:
    """Build AerosolOpticsData from a list of OpticalResult.

    If phase functions are available, Legendre moments are auto-computed.

    Args:
        results: List of OpticalResult, one per wavelength.
        n_legendre: Number of Legendre moments to compute.
        material_name: Optional material name for metadata.

    Returns:
        AerosolOpticsData with all extracted optical properties.

    Raises:
        ValueError: If results list is empty.
    """
    if not results:
        raise ValueError("results list cannot be empty")

    n_wl = len(results)
    wavelength_nm = np.array([r.cross_sections.wavelength for r in results])
    C_ext = np.array([r.cross_sections.C_ext for r in results])
    C_sca = np.array([r.cross_sections.C_sca for r in results])
    C_abs = np.array([r.cross_sections.C_abs for r in results])
    SSA = np.array([r.cross_sections.SSA for r in results])
    g = np.array([r.cross_sections.g for r in results])
    r_eff_nm = results[0].cross_sections.r_eff

    has_pf = results[0].phase_function is not None
    theta_rad = None
    phi_rad = None
    P11 = None
    legendre_moments = None

    if has_pf:
        pf0 = results[0].phase_function
        assert pf0 is not None  # guaranteed by has_pf
        theta_rad = pf0.theta
        phi_rad = pf0.phi
        n_theta = len(theta_rad)
        n_phi = len(phi_rad)
        P11 = np.zeros((n_wl, n_theta, n_phi))
        legendre_moments = np.zeros((n_wl, n_legendre))

        for i, r in enumerate(results):
            pf = r.phase_function
            assert pf is not None
            P11[i, :, :] = pf.P11
            moments = compute_legendre_moments(pf, n_legendre=n_legendre)
            legendre_moments[i, :] = moments

    return AerosolOpticsData(
        wavelength_nm=wavelength_nm,
        C_ext=C_ext,
        C_sca=C_sca,
        C_abs=C_abs,
        SSA=SSA,
        g=g,
        r_eff_nm=r_eff_nm,
        theta_rad=theta_rad,
        phi_rad=phi_rad,
        P11=P11,
        legendre_moments=legendre_moments,
        n_legendre=n_legendre,
        solver=results[0].solver,
        material=material_name,
    )
