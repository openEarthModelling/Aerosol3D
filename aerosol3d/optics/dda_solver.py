"""DDA optical solver: geometry -> dipoles -> Julia solve -> postprocess."""

import logging

import numpy as np

from .datastructs import SimulationConfig

logger = logging.getLogger(__name__)


def _apply_radiative_correction(alpha0, k):
    """Apply radiative reaction correction to Clausius-Mossotti polarizability.

    Formula: alpha = alpha0 / (1 - (2/3) * i * k^3 * alpha0)

    Args:
        alpha0: Clausius-Mossotti polarizability (scalar or array, complex).
        k: Wavenumber (scalar, float).

    Returns:
        Corrected polarizability with same shape as alpha0, dtype complex128.
    """
    alpha0 = np.asarray(alpha0, dtype=np.complex128)
    correction = 1.0 - (2.0 / 3.0) * 1j * k ** 3 * alpha0
    return alpha0 / correction


def _extract_dipoles(grid, config, material_map):
    """Extract dipole positions and polarizabilities from voxelized grid.

    V1: Binary discretization (filling_fraction = 1.0 for all filled voxels).

    Args:
        grid: pv.ImageData with cell_data["material_id"].
        config: SimulationConfig with wavelength, dipole_spacing, n_host.
        material_map: dict mapping material_id -> object with .refractive_index (complex).

    Returns:
        positions: (N, 3) float64 C-contiguous array of dipole centers.
        alpha_e: (N,) complex128 C-contiguous array of polarizabilities.
    """
    mask = grid.cell_data["material_id"] > 0
    mat_ids = grid.cell_data["material_id"][mask]
    positions = grid.cell_centers().points[mask].copy()

    k = 2.0 * np.pi / config.wavelength
    d = config.dipole_spacing
    eps_h = config.n_host ** 2

    alpha_e = np.zeros(len(positions), dtype=np.complex128)

    for mat_id, mat in material_map.items():
        sel = mat_ids == mat_id
        if not np.any(sel):
            continue
        eps = mat.refractive_index ** 2
        # Clausius-Mossotti for a cube (V = d^3)
        a_cm = 3.0 * d ** 3 * (eps - eps_h) / (eps + 2.0 * eps_h)
        # Radiative correction
        alpha_e[sel] = _apply_radiative_correction(a_cm, k)

    positions = np.ascontiguousarray(positions, dtype=np.float64)
    alpha_e = np.ascontiguousarray(alpha_e, dtype=np.complex128)
    return positions, alpha_e


def _compute_near_field_intensity(grid, phi_inc):
    """Attach |E|^2 intensity to voxel grid cell_data.

    Args:
        grid: pv.ImageData with cell_data["material_id"].
        phi_inc: (N, 3) complex array of induced E-fields at each dipole.

    Returns:
        Modified grid with cell_data["E_intensity"] = |E|^2.
    """
    mask = grid.cell_data["material_id"] > 0
    intensity = np.zeros(grid.n_cells, dtype=np.float64)
    # |E|^2 = |Ex|^2 + |Ey|^2 + |Ez|^2
    intensity[mask] = np.sum(np.abs(phi_inc) ** 2, axis=1)
    grid.cell_data["E_intensity"] = intensity
    return grid