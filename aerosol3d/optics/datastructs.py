# aerosol3d/optics/datastructs.py
"""Pure data classes for DDA optical computation results and configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    import pyvista as pv


@dataclass
class SimulationConfig:
    """All parameters for a single DDA simulation run."""

    wavelength: float
    polarization: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    propagation: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    n_host: float = 1.0
    solver: str = "CPU"
    dipole_spacing: float = 0.0

    def validity_check(self, m_max: float) -> dict:
        """Check |m|*k*d convergence criterion.

        Returns dict with m_max, k, m_k_d, and valid (bool).
        DDA convergence requires m_k_d < 1.
        """
        k = 2.0 * np.pi / self.wavelength
        mkd = abs(m_max) * k * self.dipole_spacing
        return {"m_max": m_max, "k": k, "m_k_d": mkd, "valid": mkd < 1.0}


@dataclass
class CrossSections:
    """Core aerosol optical quantities for a single wavelength."""

    wavelength: float
    C_ext: float
    C_sca: float
    C_abs: float
    Q_ext: float
    Q_sca: float
    Q_abs: float
    SSA: float
    g: float
    r_eff: float


@dataclass
class PhaseFunction:
    """Angular scattering distribution with polarization information."""

    theta: np.ndarray
    phi: np.ndarray
    P11: np.ndarray
    P12: Optional[np.ndarray] = None
    P22: Optional[np.ndarray] = None
    mueller_matrix: Optional[np.ndarray] = None
    depolarization_ratio: Optional[float] = None


@dataclass
class OpticalResult:
    """Complete result of a DDA simulation."""

    config: SimulationConfig
    cross_sections: CrossSections
    phase_function: Optional[PhaseFunction] = None
    voxel_grid: Optional[pv.ImageData] = None
    n_dipoles: int = 0
    validity: Optional[dict] = None