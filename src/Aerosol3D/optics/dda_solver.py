"""DDA optical solver: geometry -> dipoles -> Julia solve -> postprocess."""

import copy
import logging
import time

import numpy as np

from .datastructs import OpticalResult, PhaseFunction, SimulationConfig

logger = logging.getLogger(__name__)


def _apply_radiative_correction(alpha0, k):
    """Apply radiative reaction correction and normalize for Julia convention."""
    alpha0 = np.asarray(alpha0, dtype=np.complex128)
    k3 = k**3
    alpha_rad = alpha0 / (1.0 - 1j * k3 / (6.0 * np.pi) * alpha0)
    return (k3 / (4.0 * np.pi)) * alpha_rad


def _extract_dipoles(grid, config, material_map):
    """Extract dipole positions and polarizabilities from voxelized grid."""
    mask = grid.cell_data["material_id"] > 0
    mat_ids = grid.cell_data["material_id"][mask]
    positions = grid.cell_centers().points[mask].copy()
    k = 2.0 * np.pi / config.wavelength
    d = config.dipole_spacing
    eps_h = config.n_host**2
    alpha_e = np.zeros(len(positions), dtype=np.complex128)
    for mat_id, mat in material_map.items():
        sel = mat_ids == mat_id
        if not np.any(sel):
            continue
        eps = mat.refractive_index**2
        a_cm = 3.0 * d**3 * (eps - eps_h) / (eps + 2.0 * eps_h)
        alpha_e[sel] = _apply_radiative_correction(a_cm, k)
    positions = np.ascontiguousarray(positions, dtype=np.float64)
    alpha_e = np.ascontiguousarray(alpha_e, dtype=np.complex128)
    return positions, alpha_e


def _compute_near_field_intensity(grid, phi_inc):
    mask = grid.cell_data["material_id"] > 0
    intensity = np.zeros(grid.n_cells, dtype=np.float64)
    intensity[mask] = np.sum(np.abs(phi_inc) ** 2, axis=1)
    grid.cell_data["E_intensity"] = intensity
    return grid


def _prepare_dda(particle, config, voxel_size=None):
    """Prepare DDA geometry: material map -> voxelize -> extract dipoles.

    This is wavelength-independent (except for auto voxel_size which
    uses config.wavelength).  Call once for multi-wavelength solves.

    Returns
    -------
    positions, alpha_e, grid, material_map, voxel_size, m_max, material_names
    """
    # Step 1: Build material map from particle blocks
    material_map = {}
    material_names = []
    for block_name, block_mesh in particle.blocks.items():
        if block_mesh is None:
            continue
        mat_id = int(block_mesh.field_data["material_id"][0])
        if mat_id in material_map:
            continue
        ri_n = float(np.mean(block_mesh.cell_data["ri_n"]))
        ri_k = float(np.mean(block_mesh.cell_data["ri_k"]))
        material_names.append(str(block_mesh.field_data["material_name"][0]))
        material_map[mat_id] = type(
            "Mat",
            (),
            {
                "refractive_index": complex(ri_n, ri_k),
            },
        )()

    m_max = max(abs(mat.refractive_index) for mat in material_map.values())

    # Step 2: Auto voxel_size when None
    auto_computed = voxel_size is None
    if auto_computed:
        from .datastructs import auto_voxel_size

        voxel_size = auto_voxel_size(config.wavelength, m_max, config.precision)

    # Step 3: Voxelize
    from Aerosol3D.geometry.voxelize import voxelize_with_materials

    grid = voxelize_with_materials(particle, voxel_size)

    # Step 4: Copy config and set dipole_spacing
    config = copy.copy(config)
    config.dipole_spacing = voxel_size

    # Step 7: Extract dipoles
    positions, alpha_e = _extract_dipoles(grid, config, material_map)

    return positions, alpha_e, grid, material_map, voxel_size, m_max, material_names


def _solve_single_wl(
    positions,
    alpha_e,
    grid,
    material_map,
    config,
    m_max,
    voxel_size,
    material_names,
    compute_near_field=True,
    compute_phase_func=False,
    verbose=True,
):
    """Solve DDA for a single wavelength.

    Parameters
    ----------
    positions : np.ndarray
        Dipole positions, shape (N, 3).
    alpha_e : np.ndarray
        Dipole polarizabilities, shape (N,).
    grid : pyvista.UnstructuredGrid
        Voxel grid with material_id cell data.
    material_map : dict
        Mapping material_id -> material object with refractive_index attr.
    config : SimulationConfig
        Simulation configuration. ``config.wavelength`` is guaranteed to be
        a single ``float``.
    m_max : float
        Maximum refractive index magnitude.
    voxel_size : float
        Voxel / dipole spacing in nm.
    material_names : list[str]
        Human-readable material names.
    compute_near_field : bool
        Compute and attach near-field intensity to grid.
    compute_phase_func : bool
        Compute phase function.
    verbose : bool
        Print configuration and timing.

    Returns
    -------
    OpticalResult
    """
    t_start = time.time()

    from .bridge import compute_asymmetry_parameter, compute_cross_sections, solve_dda
    from .datastructs import CrossSections, OpticalResult

    # Step 5: Handle polarization=None
    do_depolarized = False
    if config.polarization is None:
        if config.source == "solar":
            do_depolarized = True
            config.polarization = (1.0, 0.0, 0.0)
        else:
            config.polarization = (1.0, 0.0, 0.0)

    # Step 6: Validity check
    validity = config.validity_check(m_max, voxel_size)
    if not validity["valid"]:
        logger.warning(
            "DDA convergence criterion violated: |m|*k*d = %.3f (should be < 1). "
            "Results may be inaccurate.",
            validity["m_k_d"],
        )

    # Ensure config has dipole_spacing set for downstream use
    config = copy.copy(config)
    config.dipole_spacing = voxel_size

    if verbose:
        print(f"{'=' * 52}")
        print("  DDA Simulation Configuration")
        print(f"{'=' * 52}")
        print(f"  wavelength     = {config.wavelength:.1f} nm")
        print(f"  source         = {config.source}")
        print(f"  polarization   = {config.polarization}")
        if do_depolarized:
            print("  mode           = depolarized (2 solves)")
        print(f"  propagation    = {config.propagation}")
        print(f"  n_host         = {config.n_host}")
        print(f"  solver         = {config.solver}")
        print(f"  precision      = {config.precision}")
        print(f"  dipole_spacing = {voxel_size:.2f} nm" + (" (auto)" if voxel_size is None else ""))
        print(f"  |m|_max        = {m_max:.4f}")
        print(f"  k              = {2.0 * np.pi / config.wavelength:.6f} nm^-1")
        mkd = m_max * (2.0 * np.pi / config.wavelength) * voxel_size
        status = "OK" if mkd < 1.0 else "WARNING"
        threshold = " (< 1.0)" if mkd >= 1.0 else ""
        print(f"  |m|*k*d        = {mkd:.4f}  {status}{threshold}")
        print(f"  N_materials    = {len(material_map)} ({', '.join(material_names)})")
        print(f"{'=' * 52}")

    if len(positions) == 0:
        n_filled = int(np.sum(grid.cell_data["material_id"] > 0))
        volume = n_filled * voxel_size**3
        r_eff = (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0) if volume > 0 else 0.0
        cross_sections = CrossSections(
            wavelength=config.wavelength,
            C_ext=0.0,
            C_sca=0.0,
            C_abs=0.0,
            Q_ext=0.0,
            Q_sca=0.0,
            Q_abs=0.0,
            SSA=0.0,
            g=0.0,
            r_eff=r_eff,
        )
        return OpticalResult(
            config=config,
            cross_sections=cross_sections,
            voxel_grid=grid,
            n_dipoles=0,
            validity=validity,
            solve_time=time.time() - t_start,
        )

    # Step 8: DDA solve (single or depolarized)
    if do_depolarized:
        config_x = copy.copy(config)
        config_x.polarization = (1.0, 0.0, 0.0)
        dda_result_x = solve_dda(positions, alpha_e, config_x)
        cs_x_raw = compute_cross_sections(positions, alpha_e, dda_result_x, config_x)

        config_y = copy.copy(config)
        config_y.polarization = (0.0, 1.0, 0.0)
        dda_result_y = solve_dda(positions, alpha_e, config_y)
        cs_y_raw = compute_cross_sections(positions, alpha_e, dda_result_y, config_y)

        C_ext = (float(cs_x_raw[0]) + float(cs_y_raw[0])) / 2
        C_abs = (float(cs_x_raw[1]) + float(cs_y_raw[1])) / 2
        C_sca = (float(cs_x_raw[2]) + float(cs_y_raw[2])) / 2
        dda_result = dda_result_x  # representative for phase func / near field
    else:
        dda_result = solve_dda(positions, alpha_e, config)
        cs_raw = compute_cross_sections(positions, alpha_e, dda_result, config)
        C_ext = float(cs_raw[0])
        C_abs = float(cs_raw[1])
        C_sca = float(cs_raw[2])

    # Step 9: Equivalent volume sphere radius
    n_filled = int(np.sum(grid.cell_data["material_id"] > 0))
    volume = n_filled * voxel_size**3
    r_eff = (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0)
    geo_cs = np.pi * r_eff**2

    # Step 10: Efficiency factors
    Q_ext = C_ext / geo_cs if geo_cs > 0 else 0.0
    Q_sca = C_sca / geo_cs if geo_cs > 0 else 0.0
    Q_abs = C_abs / geo_cs if geo_cs > 0 else 0.0

    # Step 11: SSA
    SSA = C_sca / C_ext if C_ext > 0 else 0.0

    # Step 12: Asymmetry parameter g
    g = compute_asymmetry_parameter(positions, alpha_e, dda_result, config, c_sca=C_sca)

    cross_sections = CrossSections(
        wavelength=config.wavelength,
        C_ext=C_ext,
        C_sca=C_sca,
        C_abs=C_abs,
        Q_ext=Q_ext,
        Q_sca=Q_sca,
        Q_abs=Q_abs,
        SSA=SSA,
        g=g,
        r_eff=r_eff,
    )

    # Step 13: Near-field (optional)
    if compute_near_field:
        _compute_near_field_intensity(grid, dda_result["phi_inc"])

    # Step 14: Phase function (optional)
    phase_function = None
    if compute_phase_func:
        phase_function = _compute_phase_function(positions, alpha_e, dda_result, config)

    elapsed = time.time() - t_start

    if verbose:
        print(f"\n  Solve time     = {elapsed:.1f} s")
        print(f"  N_dipoles     = {len(positions)}")
        if do_depolarized:
            print("  mode           = depolarized average")

    return OpticalResult(
        config=config,
        cross_sections=cross_sections,
        phase_function=phase_function,
        voxel_grid=grid,
        n_dipoles=len(positions),
        validity=validity,
        solve_time=elapsed,
    )


def _orientational_average(results):
    """Average OpticalResults over multiple incident directions.

    Averages C_ext, C_sca, C_abs arithmetically.
    Averages P11 over all directions, then recomputes g from the averaged P11.
    """
    from .datastructs import CrossSections, OpticalResult, PhaseFunction

    n = len(results)
    if n == 0:
        raise ValueError("Cannot average empty result list")
    if n == 1:
        return results[0]

    # Average cross-sections
    C_ext = sum(r.cross_sections.C_ext for r in results) / n
    C_sca = sum(r.cross_sections.C_sca for r in results) / n
    C_abs = sum(r.cross_sections.C_abs for r in results) / n

    r_eff = results[0].cross_sections.r_eff  # Same for all directions
    geo_cs = np.pi * r_eff**2
    Q_ext = C_ext / geo_cs if geo_cs > 0 else 0.0
    Q_sca = C_sca / geo_cs if geo_cs > 0 else 0.0
    Q_abs = C_abs / geo_cs if geo_cs > 0 else 0.0
    SSA = C_sca / C_ext if C_ext > 0 else 0.0

    # Average phase function P11
    phase_function = None
    if results[0].phase_function is not None:
        theta = results[0].phase_function.theta
        phi = results[0].phase_function.phi
        P11_avg = sum(r.phase_function.P11 for r in results) / n
        phase_function = PhaseFunction(theta=theta, phi=phi, P11=P11_avg)

        # Recompute g from averaged P11
        # Azimuthal average: P11(theta) = mean over phi
        P11_theta = np.mean(P11_avg, axis=1)
        theta_grid = theta
        # Use trapezoidal integration for cos(theta) weighting
        sin_theta = np.sin(theta_grid)
        integrand = P11_theta * sin_theta * np.cos(theta_grid)
        numerator = np.trapz(integrand, theta_grid)
        denominator = np.trapz(P11_theta * sin_theta, theta_grid)
        g = numerator / denominator if denominator > 0 else 0.0
        g = float(np.clip(g, -1.0, 1.0))
    else:
        g = sum(r.cross_sections.g for r in results) / n

    cross_sections = CrossSections(
        wavelength=results[0].cross_sections.wavelength,
        C_ext=C_ext,
        C_sca=C_sca,
        C_abs=C_abs,
        Q_ext=Q_ext,
        Q_sca=Q_sca,
        Q_abs=Q_abs,
        SSA=SSA,
        g=g,
        r_eff=r_eff,
    )

    return OpticalResult(
        config=results[0].config,
        cross_sections=cross_sections,
        phase_function=phase_function,
        voxel_grid=results[0].voxel_grid,
        n_dipoles=results[0].n_dipoles,
        validity=results[0].validity,
        solve_time=sum(r.solve_time for r in results),
    )


def _fibonacci_sphere(n):
    """Generate n approximately uniform points on the unit sphere."""
    if n <= 1:
        return [(0.0, 0.0, 1.0)]
    directions = []
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle
    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        directions.append((x, y, z))
    return directions


def solve_optics(
    particle,
    config: SimulationConfig,
    *,
    solver: str = "DDA",
    voxel_size: float = None,
    compute_near_field: bool = True,
    compute_phase_func: bool = False,
    orientational_average: bool = False,
    n_dirs: int = 100,
    propagations: list | None = None,
    verbose: bool = True,
) -> "OpticalResult | list[OpticalResult]":
    """Main entry: aerosol particle -> optical result(s).

    Supports single wavelength (float) or multi-wavelength batch (list).
    Supports orientational averaging via ``propagations`` or the
    ``orientational_average`` flag (DDA only).

    Parameters
    ----------
    particle : AerosolParticle
        The aerosol particle to simulate.
    config : SimulationConfig
        Simulation configuration, including wavelength(s).
    solver : {"DDA", "MIE"}, optional
        Which solver to use. Default is "DDA".
    voxel_size : float, optional
        Voxel size for DDA discretization. If None, auto-computed.
    compute_near_field : bool, optional
        Whether to compute near-field (DDA only). Default True.
    compute_phase_func : bool, optional
        Whether to compute phase function. Default False.
    orientational_average : bool, optional
        If True, average over ``n_dirs`` random orientations (DDA only).
        Ignored if ``propagations`` is provided. Default False.
    n_dirs : int, optional
        Number of orientations for orientational averaging. Default 100.
    propagations : list, optional
        Explicit list of propagation directions for orientational averaging.
    verbose : bool, optional
        Print progress and timing. Default True.
    """
    if solver not in ("DDA", "MIE"):
        raise ValueError(f"solver must be 'DDA' or 'MIE', got {solver!r}")

    # MIE solver dispatch
    if solver == "MIE":
        from .mie_solver import solve_mie

        if isinstance(config.wavelength, list | tuple | np.ndarray):
            wavelengths = list(config.wavelength)
        else:
            wavelengths = [float(config.wavelength)]

        results = []
        for wl in wavelengths:
            wl_config = copy.copy(config)
            wl_config.wavelength = float(wl)
            result = solve_mie(
                particle,
                wl_config,
                compute_phase_func=compute_phase_func,
                verbose=verbose and len(wavelengths) == 1,
            )
            results.append(result)

        return results[0] if len(results) == 1 else results

    # Determine wavelength(s)
    if isinstance(config.wavelength, list | tuple | np.ndarray):
        wavelengths = list(config.wavelength)
    else:
        wavelengths = [float(config.wavelength)]

    # Determine effective wavelength for voxel size
    if voxel_size is None:
        effective_wl = float(min(wavelengths))
    else:
        effective_wl = float(wavelengths[0])

    prep_config = copy.copy(config)
    prep_config.wavelength = effective_wl

    # Steps 1-4, 7: Prepare DDA geometry once
    positions, alpha_e, grid, material_map, voxel_size, m_max, material_names = _prepare_dda(
        particle, prep_config, voxel_size
    )

    # DDA orientational averaging
    if orientational_average and propagations is None:
        if n_dirs < 1:
            raise ValueError("n_dirs must be >= 1")
        propagations = _fibonacci_sphere(n_dirs)
        if compute_phase_func and n_dirs < 100:
            logger.warning(
                "compute_phase_func=True with n_dirs=%d. "
                "Phase function convergence may require n_dirs >= 100. "
                "Consider increasing n_dirs.",
                n_dirs,
            )

    results = []
    for wl in wavelengths:
        wl_config = copy.copy(config)
        wl_config.wavelength = float(wl)

        if propagations is not None:
            dir_results = []
            for prop in propagations:
                prop_config = copy.copy(wl_config)
                prop_config.propagation = tuple(prop)
                dir_result = _solve_single_wl(
                    positions,
                    alpha_e,
                    grid,
                    material_map,
                    prop_config,
                    m_max,
                    voxel_size,
                    material_names,
                    compute_near_field=compute_near_field,
                    compute_phase_func=compute_phase_func,
                    verbose=verbose and len(wavelengths) == 1 and len(propagations) == 1,
                )
                dir_results.append(dir_result)
            averaged_result = _orientational_average(dir_results)
            results.append(averaged_result)
        else:
            result = _solve_single_wl(
                positions,
                alpha_e,
                grid,
                material_map,
                wl_config,
                m_max,
                voxel_size,
                material_names,
                compute_near_field=compute_near_field,
                compute_phase_func=compute_phase_func,
                verbose=verbose and len(wavelengths) == 1,
            )
            results.append(result)

    if len(wavelengths) == 1:
        return results[0]
    return results


def _compute_phase_function(
    positions,
    alpha_e,
    dda_result,
    config,
    n_theta: int = 90,
    n_phi: int = 180,
) -> "PhaseFunction":
    """Compute P11 phase function on a (theta, phi) grid."""
    from .bridge import compute_diff_scattering
    from .datastructs import PhaseFunction

    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

    directions = np.column_stack(
        [
            np.sin(theta_grid).ravel() * np.cos(phi_grid).ravel(),
            np.sin(theta_grid).ravel() * np.sin(phi_grid).ravel(),
            np.cos(theta_grid).ravel(),
        ]
    )

    dcs = compute_diff_scattering(positions, alpha_e, dda_result, config, directions)
    P11 = dcs.reshape(n_theta, n_phi)

    return PhaseFunction(theta=theta, phi=phi, P11=P11)
