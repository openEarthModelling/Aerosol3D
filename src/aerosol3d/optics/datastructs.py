# aerosol3d/optics/datastructs.py
"""Pure data classes for DDA optical computation results and configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    import pyvista as pv


# Precision level targets: max allowed |m|*k*d for each level
_PRECISION_TARGETS = {
    "low": 0.5,
    "medium": 0.75,
    "high": 0.95,
}


def auto_voxel_size(wavelength: float, m_max: float, precision: str = "medium") -> float:
    """Compute maximum voxel_size satisfying the |m|*k*d convergence criterion.

    Args:
        wavelength: Wavelength in nm.
        m_max: Maximum |refractive_index| across all materials.
        precision: One of "low", "medium", "high".

    Returns:
        Maximum voxel_size in nm.

    Raises:
        ValueError: If precision is not a valid level.
    """
    if precision not in _PRECISION_TARGETS:
        available = ", ".join(sorted(_PRECISION_TARGETS.keys()))
        raise ValueError(f"Unknown precision '{precision}'. Available: {available}")

    target = _PRECISION_TARGETS[precision]
    k = 2.0 * np.pi / wavelength
    return target / (m_max * k)


@dataclass
class SimulationConfig:
    """All parameters for a single DDA simulation run."""

    wavelength: float | list[float] = 550.0
    polarization: Optional[Tuple[float, float, float]] = None
    propagation: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    n_host: float = 1.0
    solver: str = "CPU"
    dipole_spacing: float = 0.0
    precision: str = "medium"
    source: str = "solar"

    def validity_check(self, m_max: float, dipole_spacing: float) -> dict:
        """Check |m|*k*d convergence criterion.

        Args:
            m_max: Maximum |refractive_index| across all materials.
            dipole_spacing: Actual dipole spacing used.

        Returns:
            dict with m_max, k, m_k_d, and valid (bool).
        """
        k = 2.0 * np.pi / self.wavelength
        mkd = abs(m_max) * k * dipole_spacing
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
    solve_time: Optional[float] = None
