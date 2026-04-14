"""DDA optical solver: geometry -> dipoles -> Julia solve -> postprocess."""

import copy
import logging

import numpy as np

from .datastructs import SimulationConfig

logger = logging.getLogger(__name__)


def _apply_radiative_correction(alpha0, k):
    """Apply radiative reaction correction and normalize for Julia convention.

    Matches Julia's Alphas.alpha_radiative:
        alpha_dimless = (k^3 / (4*pi)) * alpha0 / (1 - i*k^3/(6*pi) * alpha0)

    The output is dimensionless (units of k^3 * volume), which is what the
    Julia DDA solver expects.

    Note: The textbook Draine formula uses factor (2/3)*i*k^3 in the
    denominator. Our factor 1/(6*pi) equals (2/3)/(4*pi), which accounts
    for the dimensionless normalization being folded into the correction.
    The two forms are mathematically equivalent up to the normalization.

    Args:
        alpha0: Clausius-Mossotti polarizability in volume units (scalar or array, complex).
        k: Wavenumber (scalar, float).

    Returns:
        Dimensionless polarizability with same shape as alpha0, dtype complex128.
    """
    alpha0 = np.asarray(alpha0, dtype=np.complex128)
    k3 = k ** 3
    # Radiative-corrected polarizability in volume units
    alpha_rad = alpha0 / (1.0 - 1j * k3 / (6.0 * np.pi) * alpha0)
    # Normalize to dimensionless (matching Julia convention)
    return (k3 / (4.0 * np.pi)) * alpha_rad


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


def solve_optics(
    particle,
    config: SimulationConfig,
    voxel_size: float,
    compute_near_field: bool = True,
    compute_phase_func: bool = False,
) -> "OpticalResult":
    """Main entry: aerosol particle -> DDA optical result.

    Args:
        particle: AerosolParticle instance with .blocks.
        config: SimulationConfig with wavelength, polarization, etc.
        voxel_size: Side length of each voxel (same unit as particle).
        compute_near_field: If True, attach |E|^2 to voxel grid.
        compute_phase_func: If True, compute P11 phase function on (theta, phi) grid.

    Returns:
        OpticalResult with cross_sections, voxel_grid, and validity.
    """
    from .datastructs import CrossSections, OpticalResult
    from .bridge import (
        solve_dda, compute_cross_sections, compute_asymmetry_parameter
    )

    # Step 1: Build material map from particle blocks
    material_map = {}
    for block_name, block_mesh in particle.blocks.items():
        if block_mesh is None:
            continue
        mat_id = int(block_mesh.field_data["material_id"][0])
        if mat_id in material_map:
            continue
        ri_n = float(np.mean(block_mesh.cell_data["ri_n"]))
        ri_k = float(np.mean(block_mesh.cell_data["ri_k"]))
        material_map[mat_id] = type("Mat", (), {
            "refractive_index": complex(ri_n, ri_k),
        })()

    # Step 2: Voxelize
    from aerosol3d.geometry.voxelize import voxelize_with_materials
    grid = voxelize_with_materials(particle, voxel_size)

    # Step 3: Copy config and set dipole_spacing (avoid mutating caller's object)
    config = copy.copy(config)
    config.dipole_spacing = voxel_size

    # Step 4: Validity check
    m_max = max(abs(mat.refractive_index) for mat in material_map.values())
    validity = config.validity_check(m_max)
    if not validity["valid"]:
        logger.warning(
            "DDA convergence criterion violated: |m|*k*d = %.3f (should be < 1). "
            "Results may be inaccurate.",
            validity["m_k_d"],
        )

    # Step 5: Extract dipoles
    positions, alpha_e = _extract_dipoles(grid, config, material_map)

    if len(positions) == 0:
        # Return a zero-result with validity info rather than crashing
        n_filled = int(np.sum(grid.cell_data["material_id"] > 0))
        volume = n_filled * voxel_size ** 3
        r_eff = (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0) if volume > 0 else 0.0
        geo_cs = np.pi * r_eff ** 2 if r_eff > 0 else 0.0
        cross_sections = CrossSections(
            wavelength=config.wavelength,
            C_ext=0.0, C_sca=0.0, C_abs=0.0,
            Q_ext=0.0, Q_sca=0.0, Q_abs=0.0,
            SSA=0.0, g=0.0, r_eff=r_eff,
        )
        return OpticalResult(
            config=config,
            cross_sections=cross_sections,
            voxel_grid=grid,
            n_dipoles=0,
            validity=validity,
        )

    # Step 6: DDA solve
    dda_result = solve_dda(positions, alpha_e, config)

    # Step 7: Cross sections (Julia returns in knorm^{-2} = nm^2)
    cs_raw = compute_cross_sections(positions, alpha_e, dda_result, config)
    C_ext = float(cs_raw[0])
    C_abs = float(cs_raw[1])
    C_sca = float(cs_raw[2])

    # Step 8: Equivalent volume sphere radius
    n_filled = int(np.sum(grid.cell_data["material_id"] > 0))
    volume = n_filled * voxel_size ** 3
    r_eff = (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0)
    geo_cs = np.pi * r_eff ** 2

    # Step 9: Efficiency factors
    Q_ext = C_ext / geo_cs if geo_cs > 0 else 0.0
    Q_sca = C_sca / geo_cs if geo_cs > 0 else 0.0
    Q_abs = C_abs / geo_cs if geo_cs > 0 else 0.0

    # Step 10: SSA
    SSA = C_sca / C_ext if C_ext > 0 else 0.0

    # Step 11: Asymmetry parameter g
    g = compute_asymmetry_parameter(
        positions, alpha_e, dda_result, config, C_sca=C_sca
    )

    cross_sections = CrossSections(
        wavelength=config.wavelength,
        C_ext=C_ext, C_sca=C_sca, C_abs=C_abs,
        Q_ext=Q_ext, Q_sca=Q_sca, Q_abs=Q_abs,
        SSA=SSA, g=g, r_eff=r_eff,
    )

    # Step 12: Near-field (optional)
    if compute_near_field:
        _compute_near_field_intensity(grid, dda_result["phi_inc"])

    # Step 13: Phase function (optional)
    phase_function = None
    if compute_phase_func:
        phase_function = _compute_phase_function(
            positions, alpha_e, dda_result, config
        )

    return OpticalResult(
        config=config,
        cross_sections=cross_sections,
        phase_function=phase_function,
        voxel_grid=grid,
        n_dipoles=len(positions),
        validity=validity,
    )


def _compute_phase_function(
    positions, alpha_e, dda_result, config,
    n_theta: int = 90, n_phi: int = 180,
) -> "PhaseFunction":
    """Compute P11 phase function on a (theta, phi) grid.

    Args:
        positions: (N, 3) dipole positions.
        alpha_e: (N,) polarizabilities.
        dda_result: dict from solve_dda.
        config: SimulationConfig.
        n_theta: Number of polar angle bins.
        n_phi: Number of azimuthal angle bins.

    Returns:
        PhaseFunction with P11(theta, phi).
    """
    from .datastructs import PhaseFunction
    from .bridge import compute_diff_scattering

    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

    # Convert to Cartesian directions
    directions = np.column_stack([
        np.sin(theta_grid).ravel() * np.cos(phi_grid).ravel(),
        np.sin(theta_grid).ravel() * np.sin(phi_grid).ravel(),
        np.cos(theta_grid).ravel(),
    ])

    # Compute differential scattering cross section
    dcs = compute_diff_scattering(positions, alpha_e, dda_result, config, directions)
    P11 = dcs.reshape(n_theta, n_phi)

    return PhaseFunction(theta=theta, phi=phi, P11=P11)
