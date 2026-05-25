"""DDA optical solver: geometry -> dipoles -> Julia solve -> postprocess."""

import copy
import logging
import time

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from .datastructs import OpticalResult, PhaseFunction, SimulationConfig

logger = logging.getLogger(__name__)


# Lattice Dispersion Relation (LDR) constants (Draine & Goodman 1993)
_LDR_B1 = 1.8915316
_LDR_B2 = -0.1648469
_LDR_B3 = 0.7700004


def _ldr_s(propagation, polarization):
    """Compute S = sum_mu (e_mu * a_mu)^2 for LDR polarizability."""
    return sum((e * a) ** 2 for e, a in zip(polarization, propagation))


def _compute_polarizability(grid, config, material_map):
    """Compute LDR polarizabilities for all dipoles in the grid.

    Uses the Lattice Dispersion Relation (Draine & Goodman 1993):
        alpha_LDR = alpha_CM / {1 + correction}
    where correction includes (kd)^2 and (kd)^3 order terms.

    Returns alpha_e normalized by k^3/(4*pi) for the Julia convention.
    """
    mask = grid.cell_data["material_id"] > 0
    mat_ids = grid.cell_data["material_id"][mask]

    k = 2.0 * np.pi / config.wavelength
    d = config.dipole_spacing
    eps_h = config.n_host**2
    kd = k * d
    k3 = k**3
    S = _ldr_s(config.propagation, config.polarization)

    alpha_e = np.zeros(int(np.sum(mask)), dtype=np.complex128)
    for mat_id, mat in material_map.items():
        sel = mat_ids == mat_id
        if not np.any(sel):
            continue

        eps = mat.refractive_index**2
        m_rel = mat.refractive_index / config.n_host
        m_rel2 = m_rel**2

        # Clausius-Mossotti polarizability (code convention: 4pi * alpha_CM_paper)
        a_cm = 3.0 * d**3 * (eps - eps_h) / (eps + 2.0 * eps_h)

        # LDR correction: adds (kd)^2 order terms beyond radiative reaction
        correction = (a_cm / (4.0 * np.pi * d**3)) * (
            _LDR_B1 + (_LDR_B2 + _LDR_B3 * S) * m_rel2
        ) * kd**2 - 1j * k3 / (6.0 * np.pi) * a_cm

        alpha_ldr = a_cm / (1.0 + correction)
        alpha_e[sel] = (k3 / (4.0 * np.pi)) * alpha_ldr

    return np.ascontiguousarray(alpha_e, dtype=np.complex128)


def _compute_near_field_intensity(grid, phi_inc):
    mask = grid.cell_data["material_id"] > 0
    intensity = np.zeros(grid.n_cells, dtype=np.float64)
    intensity[mask] = np.sum(np.abs(phi_inc) ** 2, axis=1)
    grid.cell_data["E_intensity"] = intensity
    return grid


def _prepare_dda(particle, config, voxel_size=None):
    """Prepare DDA geometry: material map -> voxelize.

    Polarizabilities are NOT computed here -- they depend on wavelength
    and polarization direction, so they are computed per-solve in
    _solve_single_wl via _compute_polarizability.

    Returns
    -------
    grid, material_map, voxel_size, m_max, material_names
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

    return grid, material_map, voxel_size, m_max, material_names


def _solve_single_wl(
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

    Computes LDR polarizabilities internally for the given wavelength
    and polarization direction(s).

    Parameters
    ----------
    grid : pyvista.UnstructuredGrid
        Voxel grid with material_id cell data.
    material_map : dict
        Mapping material_id -> material object with refractive_index attr.
    config : SimulationConfig
        Simulation configuration.
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

    # Ensure config has dipole_spacing set for polarizability computation
    config = copy.copy(config)
    config.dipole_spacing = voxel_size

    # Extract positions from grid
    mask = grid.cell_data["material_id"] > 0
    positions = grid.cell_centers().points[mask].copy()

    # Handle polarization=None
    do_depolarized = False
    if config.polarization is None:
        if config.source == "solar":
            do_depolarized = True
            config.polarization = (1.0, 0.0, 0.0)
        else:
            config.polarization = (1.0, 0.0, 0.0)

    # Validity check
    validity = config.validity_check(m_max, voxel_size)
    if not validity["valid"]:
        logger.warning(
            "DDA convergence criterion violated: |m|*k*d = %.3f (should be < 1). "
            "Results may be inaccurate.",
            validity["m_k_d"],
        )

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
        print(f"  dipole_spacing = {voxel_size:.2f} nm")
        print(f"  |m|_max        = {m_max:.4f}")
        print(f"  k              = {2.0 * np.pi / config.wavelength:.6f} nm^-1")
        mkd = m_max * (2.0 * np.pi / config.wavelength) * voxel_size
        status = "OK" if mkd < 1.0 else "WARNING"
        threshold = " (< 1.0)" if mkd >= 1.0 else ""
        print(f"  |m|*k*d        = {mkd:.4f}  {status}{threshold}")
        print(f"  N_materials    = {len(material_map)} ({', '.join(material_names)})")
        print(f"{'=' * 52}")

    if len(positions) == 0:
        n_filled = int(np.sum(mask))
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

    positions = np.ascontiguousarray(positions, dtype=np.float64)

    # Initialize variables set in branches below (for static analysis)
    alpha_e = alpha_e_x = alpha_e_y = None
    dda_result_x = dda_result_y = None
    config_x = config_y = None
    cs_x_raw = cs_y_raw = None

    # DDA solve (single or depolarized)
    if do_depolarized:
        config_x = copy.copy(config)
        config_x.polarization = (1.0, 0.0, 0.0)
        alpha_e_x = _compute_polarizability(grid, config_x, material_map)
        dda_result_x = solve_dda(positions, alpha_e_x, config_x)
        cs_x_raw = compute_cross_sections(positions, alpha_e_x, dda_result_x, config_x)

        config_y = copy.copy(config)
        config_y.polarization = (0.0, 1.0, 0.0)
        alpha_e_y = _compute_polarizability(grid, config_y, material_map)
        dda_result_y = solve_dda(positions, alpha_e_y, config_y)
        cs_y_raw = compute_cross_sections(positions, alpha_e_y, dda_result_y, config_y)

        C_ext = (float(cs_x_raw[0]) + float(cs_y_raw[0])) / 2
        C_abs = (float(cs_x_raw[1]) + float(cs_y_raw[1])) / 2
        C_sca = (float(cs_x_raw[2]) + float(cs_y_raw[2])) / 2
        dda_result = dda_result_x
    else:
        alpha_e = _compute_polarizability(grid, config, material_map)
        dda_result = solve_dda(positions, alpha_e, config)
        cs_raw = compute_cross_sections(positions, alpha_e, dda_result, config)
        C_ext = float(cs_raw[0])
        C_abs = float(cs_raw[1])
        C_sca = float(cs_raw[2])

    # Equivalent volume sphere radius
    n_filled = int(np.sum(mask))
    volume = n_filled * voxel_size**3
    r_eff = (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0)
    geo_cs = np.pi * r_eff**2

    # Efficiency factors
    Q_ext = C_ext / geo_cs if geo_cs > 0 else 0.0
    Q_sca = C_sca / geo_cs if geo_cs > 0 else 0.0
    Q_abs = C_abs / geo_cs if geo_cs > 0 else 0.0

    # SSA
    SSA = C_sca / C_ext if C_ext > 0 else 0.0

    # Asymmetry parameter g
    if do_depolarized:
        alpha_e_for_g = alpha_e_x
    else:
        alpha_e_for_g = alpha_e
    g = compute_asymmetry_parameter(positions, alpha_e_for_g, dda_result, config, c_sca=C_sca)

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

    # Near-field (optional)
    if compute_near_field:
        _compute_near_field_intensity(grid, dda_result["phi_inc"])

    # Phase function (optional)
    phase_function = None
    if compute_phase_func:
        if do_depolarized:
            from .datastructs import PhaseFunction

            pf_x = _compute_phase_function(
                positions, alpha_e_x, dda_result_x, config_x, c_sca=float(cs_x_raw[2])
            )
            pf_y = _compute_phase_function(
                positions, alpha_e_y, dda_result_y, config_y, c_sca=float(cs_y_raw[2])
            )
            P11_avg = (pf_x.P11 + pf_y.P11) / 2.0
            phase_function = PhaseFunction(theta=pf_x.theta, phi=pf_x.phi, P11=P11_avg)
        else:
            phase_function = _compute_phase_function(
                positions, alpha_e, dda_result, config, c_sca=C_sca
            )

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

    Averages C_ext, C_sca, C_abs, and g arithmetically.
    Averages P11 over all directions in the lab frame.
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

    # Average asymmetry parameter g arithmetically across orientations.
    # Recomputing g from the lab-frame averaged P11 is incorrect because
    # the lab-frame average of a rotationally invariant particle is isotropic
    # (g -> 0), whereas the average of individual g values is the correct
    # orientational average.
    g = sum(r.cross_sections.g for r in results) / n

    # Average phase function P11 in the lab frame
    phase_function = None
    if results[0].phase_function is not None:
        theta = results[0].phase_function.theta
        phi = results[0].phase_function.phi
        P11_avg = sum(r.phase_function.P11 for r in results) / n
        phase_function = PhaseFunction(theta=theta, phi=phi, P11=P11_avg)

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


def _solve_single_orientation(
    propagation,
    grid,
    material_map,
    wl_config,
    m_max,
    voxel_size,
    material_names,
    *,
    compute_near_field=True,
    compute_phase_func=False,
):
    """Solve DDA for a single propagation direction (one orientation)."""
    from . import bridge

    bridge._ensure_julia()

    prop_config = copy.copy(wl_config)
    prop_config.propagation = tuple(propagation)
    return _solve_single_wl(
        grid,
        material_map,
        prop_config,
        m_max,
        voxel_size,
        material_names,
        compute_near_field=compute_near_field,
        compute_phase_func=compute_phase_func,
        verbose=False,
    )


def _solve_single_orientation_safe(
    propagation,
    grid,
    material_map,
    wl_config,
    m_max,
    voxel_size,
    material_names,
    *,
    compute_near_field=True,
    compute_phase_func=False,
):
    """Error-tolerant wrapper for parallel orientation averaging."""
    try:
        return _solve_single_orientation(
            propagation,
            grid,
            material_map,
            wl_config,
            m_max,
            voxel_size,
            material_names,
            compute_near_field=compute_near_field,
            compute_phase_func=compute_phase_func,
        )
    except Exception:
        logger.exception(
            "DDA solve failed for propagation direction %s at λ=%.0fnm",
            tuple(propagation),
            wl_config.wavelength,
        )
        return None


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
    ema_method: str = "volume_weighted",
    voxel_size: float = None,
    compute_near_field: bool = True,
    compute_phase_func: bool = False,
    orientational_average: bool = False,
    n_dirs: int = 50,
    propagations: list | None = None,
    n_jobs: int = 32,
    show_progress: bool = True,
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
    solver : {"DDA", "MIE", "MIE_CORESHELL"}, optional
        Which solver to use. Default is "DDA".
    ema_method : str, optional
        EMA method for Mie homogeneous-sphere path. One of
        'volume_weighted', 'maxwell_garnett', 'bruggeman'.
        Default is 'volume_weighted'.
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
        Number of orientations for orientational averaging. Default 50.
    propagations : list, optional
        Explicit list of propagation directions for orientational averaging.
    n_jobs : int, optional
        Number of parallel workers for orientation averaging.
        1 = serial. Default 32.
    show_progress : bool, optional
        Display tqdm progress bars. Default True.
    verbose : bool, optional
        Print per-solve configuration and timing. Default True.
    """
    if solver not in ("DDA", "MIE", "MIE_CORESHELL"):
        raise ValueError(f"solver must be 'DDA', 'MIE', or 'MIE_CORESHELL', got {solver!r}")

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
                ema_method=ema_method,
                verbose=verbose and len(wavelengths) == 1,
            )
            results.append(result)

        return results[0] if len(results) == 1 else results

    # MIE_CORESHELL solver dispatch
    if solver == "MIE_CORESHELL":
        from .mie_solver import solve_mie_coreshell

        if isinstance(config.wavelength, list | tuple | np.ndarray):
            wavelengths = list(config.wavelength)
        else:
            wavelengths = [float(config.wavelength)]

        results = []
        for wl in wavelengths:
            wl_config = copy.copy(config)
            wl_config.wavelength = float(wl)
            result = solve_mie_coreshell(
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

    # Steps 1-4: Prepare DDA geometry once
    grid, material_map, voxel_size, m_max, material_names = _prepare_dda(
        particle, prep_config, voxel_size
    )

    # DDA orientational averaging
    if orientational_average and propagations is None:
        if n_dirs < 1:
            raise ValueError("n_dirs must be >= 1")
        if n_dirs < 30:
            logger.warning(
                "n_dirs=%d is low. Consider >= 50 for reliable averaging.",
                n_dirs,
            )
        if compute_phase_func and n_dirs < 100:
            logger.warning(
                "compute_phase_func=True with n_dirs=%d. "
                "Phase function convergence may require n_dirs >= 100.",
                n_dirs,
            )
        propagations = _fibonacci_sphere(n_dirs)

    results = []
    if propagations is not None:
        # Flatten all (wavelength, orientation) pairs into a single task list
        # so joblib can utilize all cores even when n_dirs < n_jobs.
        n_wl = len(wavelengths)
        n_prop = len(propagations)
        wl_configs = []
        for wl in wavelengths:
            wl_config = copy.copy(config)
            wl_config.wavelength = float(wl)
            wl_configs.append(wl_config)

        # Build flat list of (wl_config_index, propagation) tuples
        tasks = [(wl_idx, prop) for wl_idx in range(n_wl) for prop in propagations]

        total = len(tasks)
        task_iter = tqdm(
            tasks,
            desc=f"DDA ({n_wl}λ × {n_prop} dirs, {total} tasks)",
            disable=not show_progress,
        )

        flat_results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_solve_single_orientation_safe)(
                prop,
                grid,
                material_map,
                wl_configs[wl_idx],
                m_max,
                voxel_size,
                material_names,
                compute_near_field=False,  # Near-field discarded in orientation averaging
                compute_phase_func=compute_phase_func,
            )
            for wl_idx, prop in task_iter
        )

        # Group results by wavelength and average orientations
        # Filter out None results from failed tasks
        n_failed = sum(1 for r in flat_results if r is None)
        if n_failed > 0:
            logger.warning("%d of %d orientation tasks failed", n_failed, len(flat_results))
        for wl_idx in range(n_wl):
            dir_results = [
                r for r in flat_results[wl_idx * n_prop : (wl_idx + 1) * n_prop] if r is not None
            ]
            if not dir_results:
                raise RuntimeError(
                    f"All orientation tasks failed for λ={wavelengths[wl_idx]:.0f}nm"
                )
            averaged = _orientational_average(dir_results)
            results.append(averaged)
    else:
        for wl in wavelengths:
            wl_config = copy.copy(config)
            wl_config.wavelength = float(wl)
            result = _solve_single_wl(
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
    c_sca: float,
    n_theta: int = 90,
    n_phi: int = 180,
) -> "PhaseFunction":
    """Compute P11 phase function on a (theta, phi) grid.

    P11 is normalized such that the integral over the full sphere is 1:
        ∫ P11(θ, φ) dΩ = 1
    """
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

    # Normalize: phase function integrates to 1 over 4π
    if c_sca > 0:
        P11 = P11 / c_sca

    return PhaseFunction(theta=theta, phi=phi, P11=P11)
