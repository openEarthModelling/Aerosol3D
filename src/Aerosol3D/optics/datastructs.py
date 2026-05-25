# Aerosol3D/optics/datastructs.py
"""Pure data classes for DDA optical computation results and configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    import pyvista as pv


# Precision level targets: max allowed |m|*k*d for each level.
# Lower |m|*k*d -> smaller dipole spacing -> more dipoles -> higher accuracy.
_PRECISION_TARGETS = {
    "low": 0.63,  # Draine "rule of thumb": 10 dipoles/wavelength in medium
    "medium": 0.5,  # |m|<=2: few % accuracy with LDR
    "high": 0.3,  # High accuracy for all |m|
    "ultra": 0.15,  # Very high accuracy, large computational cost
}


def auto_voxel_size(wavelength: float, m_max: float, precision: str = "medium") -> float:
    """Compute maximum voxel_size satisfying the ``|m|*k*d`` convergence criterion.

    Args:
        wavelength: Wavelength in nm.
        m_max: Maximum ``|refractive_index|`` across all materials.
        precision: One of "low", "medium", "high", "ultra".

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
    polarization: tuple[float, float, float] | None = None
    propagation: tuple[float, float, float] = (0.0, 0.0, 1.0)
    n_host: float = 1.0
    solver: str = "CPU"
    dipole_spacing: float = 0.0
    precision: str = "medium"
    source: str = "solar"

    def validity_check(self, m_max: float, dipole_spacing: float) -> dict:
        """Check ``|m|*k*d`` convergence criterion.

        Args:
            m_max: Maximum ``|refractive_index|`` across all materials.
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
    P12: np.ndarray | None = None
    P22: np.ndarray | None = None
    mueller_matrix: np.ndarray | None = None
    depolarization_ratio: float | None = None


@dataclass
class OpticalResult:
    """Complete result of an optical simulation."""

    config: SimulationConfig
    cross_sections: CrossSections
    phase_function: PhaseFunction | None = None
    voxel_grid: pv.ImageData | None = None
    n_dipoles: int = 0
    validity: dict | None = None
    solve_time: float | None = None
    solver: Literal["DDA", "MIE"] = "DDA"
