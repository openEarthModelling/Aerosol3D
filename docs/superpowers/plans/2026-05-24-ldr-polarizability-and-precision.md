# LDR Polarizability and Precision Level Improvement — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace RR polarizability with LDR (Draine & Goodman 1993) and expand precision levels from 3 to 4, based on Yurkin & Hoekstra (2007) convergence benchmarks.

**Architecture:** Delete `_apply_radiative_correction` and `_extract_dipoles`. Add `_compute_polarizability` that computes LDR polarizabilities per-solve (each wavelength/polarization gets correct alpha_e). Move alpha_e computation from `_prepare_dda` into `_solve_single_wl` so depolarized mode can use polarization-dependent LDR.

**Tech Stack:** Python, NumPy, pytest, existing Julia bridge (no Julia changes).

**Spec:** `docs/superpowers/specs/2026-05-24-ldr-polarizability-and-precision-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/Aerosol3D/optics/datastructs.py` | `_PRECISION_TARGETS` dict and `auto_voxel_size()` |
| `src/Aerosol3D/optics/dda_solver.py` | LDR polarizability, prepare, solve, orientational averaging, main entry point |
| `tests/test_dda_solver.py` | All DDA solver tests |

---

### Task 1: Update precision levels in datastructs.py

**Files:**
- Modify: `src/Aerosol3D/optics/datastructs.py:17-19`
- Test: `tests/test_dda_solver.py` (run existing tests)

- [ ] **Step 1: Update `_PRECISION_TARGETS`**

In `src/Aerosol3D/optics/datastructs.py`, replace lines 17-19:

```python
# BEFORE:
_PRECISION_TARGETS = {
    "low": 0.95,
    "medium": 0.75,
    "high": 0.5,
}

# AFTER:
_PRECISION_TARGETS = {
    "low": 0.63,       # Draine "rule of thumb": 10 dipoles/wavelength in medium
    "medium": 0.5,     # |m|<=2: few % accuracy with LDR
    "high": 0.3,       # High accuracy for all |m|
    "ultra": 0.15,     # Very high accuracy, large computational cost
}
```

- [ ] **Step 2: Run existing auto_voxel_size tests to verify the change**

Run: `cd /home/zhangfan/Project/20260319_SPEMBSSBDART/aerosol3d && python -m pytest tests/test_dda_solver.py::TestAutoVoxelSize -v -x 2>&1 | head -30`

The test `test_auto_voxel_size_produces_valid_mkd` uses `precision="high"` which changes from 0.5 to 0.3. This still passes since it checks `m_k_d < 1.0`.

Expected: PASS (the test checks `m_k_d < 1.0`, and high=0.3 < 1.0).

The test `test_explicit_voxel_size_overrides_precision` uses `voxel_size=44.0` and checks `m_k_d > 0.95`. This still passes since 44nm spacing gives even higher m_k_d.

- [ ] **Step 3: Verify "ultra" precision is accepted**

Run: `python -c "from Aerosol3D.optics.datastructs import auto_voxel_size; print(auto_voxel_size(550.0, 2.0, 'ultra'))"`

Expected: prints a small voxel size (approximately 6.5 nm).

- [ ] **Step 4: Commit**

```bash
git add src/Aerosol3D/optics/datastructs.py
git commit -m "feat: expand precision levels to 4 tiers based on Yurkin & Hoekstra (2007)

low=0.63, medium=0.5, high=0.3, ultra=0.15 — aligned with DDA
convergence benchmarks for reliable scattering calculations.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 2: Add LDR polarizability functions and refactor `_prepare_dda`

**Files:**
- Modify: `src/Aerosol3D/optics/dda_solver.py:16-42` (delete `_apply_radiative_correction` and `_extract_dipoles`) and lines 53-104 (`_prepare_dda`)

- [ ] **Step 1: Delete `_apply_radiative_correction` and `_extract_dipoles`, add LDR functions**

In `src/Aerosol3D/optics/dda_solver.py`, replace lines 16-42 (the two deleted functions) with:

```python
# Lattice Dispersion Relation (LDR) constants (Draine & Goodman 1993)
_LDR_B1 = 1.8915316
_LDR_B2 = -0.1648469
_LDR_B3 = 0.7700004


def _ldr_S(propagation, polarization):
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
    eps_h = config.n_host ** 2
    kd = k * d
    k3 = k ** 3
    S = _ldr_S(config.propagation, config.polarization)

    alpha_e = np.zeros(int(np.sum(mask)), dtype=np.complex128)
    for mat_id, mat in material_map.items():
        sel = mat_ids == mat_id
        if not np.any(sel):
            continue

        eps = mat.refractive_index ** 2
        m_rel = mat.refractive_index / config.n_host
        m_rel2 = m_rel ** 2

        # Clausius-Mossotti polarizability (code convention: 4pi * alpha_CM_paper)
        a_cm = 3.0 * d ** 3 * (eps - eps_h) / (eps + 2.0 * eps_h)

        # LDR correction: adds (kd)^2 order terms beyond radiative reaction
        correction = (
            (a_cm / (4.0 * np.pi * d ** 3))
            * (_LDR_B1 + (_LDR_B2 + _LDR_B3 * S) * m_rel2)
            * kd ** 2
            - 1j * k3 / (6.0 * np.pi) * a_cm
        )

        alpha_ldr = a_cm / (1.0 + correction)
        alpha_e[sel] = (k3 / (4.0 * np.pi)) * alpha_ldr

    return np.ascontiguousarray(alpha_e, dtype=np.complex128)
```

- [ ] **Step 2: Update `_prepare_dda` to stop computing alpha_e**

Replace the `_prepare_dda` function (lines 53-104) with:

```python
def _prepare_dda(particle, config, voxel_size=None):
    """Prepare DDA geometry: material map -> voxelize -> extract positions.

    This is wavelength-independent (except for auto voxel_size which
    uses config.wavelength).  Call once for multi-wavelength solves.

    Polarizabilities are NOT computed here — they depend on wavelength
    and polarization direction, so they are computed per-solve in
    _solve_single_wl via _compute_polarizability.

    Returns
    -------
    positions, grid, material_map, voxel_size, m_max, material_names
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

    # Step 4: Extract positions from voxel grid
    mask = grid.cell_data["material_id"] > 0
    positions = np.ascontiguousarray(
        grid.cell_centers().points[mask].copy(), dtype=np.float64
    )

    return positions, grid, material_map, voxel_size, m_max, material_names
```

Key changes from original:
- Return tuple has 6 items (was 7) — `alpha_e` removed
- `_extract_dipoles` call removed — replaced by direct position extraction
- Config dipole_spacing is NOT set here (set in `_solve_single_wl` before computing alpha_e)

- [ ] **Step 3: Commit**

```bash
git add src/Aerosol3D/optics/dda_solver.py
git commit -m "feat: add LDR polarizability, refactor _prepare_dda to separate geometry from alpha_e

Replaces _apply_radiative_correction and _extract_dipoles with
_compute_polarizability using Draine & Goodman (1993) LDR formula.
_prepare_dda now returns positions without alpha_e; polarizabilities
are computed per-solve for correct wavelength/polarization handling.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 3: Update `_solve_single_wl` to compute alpha_e internally

**Files:**
- Modify: `src/Aerosol3D/optics/dda_solver.py` (`_solve_single_wl` function)

- [ ] **Step 1: Update `_solve_single_wl` signature and body**

Replace the entire `_solve_single_wl` function with the updated version below. The key changes are:
- Remove `alpha_e` from parameters
- Remove `positions` from parameters (extract from grid, matching new `_prepare_dda`)
- Compute alpha_e internally via `_compute_polarizability` for each polarization direction

New signature: `_solve_single_wl(grid, material_map, config, m_max, voxel_size, material_names, ...)`

```python
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
        volume = n_filled * voxel_size ** 3
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
    volume = n_filled * voxel_size ** 3
    r_eff = (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0)
    geo_cs = np.pi * r_eff ** 2

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
    g = compute_asymmetry_parameter(
        positions, alpha_e_for_g, dda_result, config, c_sca=C_sca
    )

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
```

- [ ] **Step 2: Commit**

```bash
git add src/Aerosol3D/optics/dda_solver.py
git commit -m "refactor: _solve_single_wl computes LDR alpha_e per-polarization

Each polarization direction gets its own alpha_e array computed from
_compute_polarizability, enabling correct LDR S-parameter handling
in depolarized mode.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 4: Update `_solve_single_orientation` and `solve_optics`

**Files:**
- Modify: `src/Aerosol3D/optics/dda_solver.py` (`_solve_single_orientation`, `_solve_single_orientation_safe`, `solve_optics`)

- [ ] **Step 1: Update `_solve_single_orientation`**

Remove `positions` and `alpha_e` from parameters (they are now extracted/computed inside `_solve_single_wl`). The function becomes:

```python
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
    """Solve DDA for a single propagation direction (one orientation).

    Designed as a top-level function so joblib can pickle it for
    multi-process dispatch.
    """
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
```

- [ ] **Step 2: Update `_solve_single_orientation_safe`**

```python
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
```

- [ ] **Step 3: Update `solve_optics`**

In the `solve_optics` function, update three places:

**3a.** Update the `_prepare_dda` call and unpacking (around the existing `positions, alpha_e, grid, ... = _prepare_dda(...)` line):

Replace:
```python
    positions, alpha_e, grid, material_map, voxel_size, m_max, material_names = _prepare_dda(
        particle, prep_config, voxel_size
    )
```
With:
```python
    positions, grid, material_map, voxel_size, m_max, material_names = _prepare_dda(
        particle, prep_config, voxel_size
    )
```

**3b.** Update the `_solve_single_orientation_safe` calls in the parallel dispatch block. Replace the `delayed(...)` call:

Replace:
```python
        flat_results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_solve_single_orientation_safe)(
                prop,
                positions,
                alpha_e,
                grid,
                material_map,
                wl_configs[wl_idx],
                m_max,
                voxel_size,
                material_names,
                compute_near_field=False,
                compute_phase_func=compute_phase_func,
            )
            for wl_idx, prop in task_iter
        )
```
With:
```python
        flat_results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_solve_single_orientation_safe)(
                prop,
                grid,
                material_map,
                wl_configs[wl_idx],
                m_max,
                voxel_size,
                material_names,
                compute_near_field=False,
                compute_phase_func=compute_phase_func,
            )
            for wl_idx, prop in task_iter
        )
```

**3c.** Update the single-wavelength `_solve_single_wl` call in the `else` branch:

Replace:
```python
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
```
With:
```python
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
```

- [ ] **Step 4: Commit**

```bash
git add src/Aerosol3D/optics/dda_solver.py
git commit -m "refactor: remove alpha_e from orientation solver and solve_optics call chain

_solve_single_orientation and solve_optics no longer pass pre-computed
alpha_e; it is computed inside _solve_single_wl per wavelength/polarization.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 5: Update existing tests for new signatures

**Files:**
- Modify: `tests/test_dda_solver.py`

- [ ] **Step 1: Update `TestPrepareDDA.test_prepare_dda_returns_expected`**

The test unpacks `_prepare_dda` return value. Update to expect 6 items (no `alpha_e`):

Replace:
```python
    def test_prepare_dda_returns_expected(self, julia_available, soot_material):
        from Aerosol3D import AerosolParticle, create_sphere
        from Aerosol3D.optics.datastructs import SimulationConfig
        from Aerosol3D.optics.dda_solver import _prepare_dda

        p = AerosolParticle(name="soot_sphere", unit="nm")
        p.add_mesh("core", create_sphere((0, 0, 0), 50.0), soot_material)
        config = SimulationConfig(wavelength=550.0, dipole_spacing=10.0)

        positions, alpha_e, grid, material_map, voxel_size, m_max, material_names = _prepare_dda(
            p, config, voxel_size=10.0
        )

        assert positions.ndim == 2 and positions.shape[1] == 3
        assert alpha_e.ndim == 1
        assert positions.shape[0] == alpha_e.shape[0]
        assert positions.shape[0] > 0
        assert voxel_size == 10.0
        assert m_max > 0
```
With:
```python
    def test_prepare_dda_returns_expected(self, julia_available, soot_material):
        from Aerosol3D import AerosolParticle, create_sphere
        from Aerosol3D.optics.datastructs import SimulationConfig
        from Aerosol3D.optics.dda_solver import _prepare_dda

        p = AerosolParticle(name="soot_sphere", unit="nm")
        p.add_mesh("core", create_sphere((0, 0, 0), 50.0), soot_material)
        config = SimulationConfig(wavelength=550.0, dipole_spacing=10.0)

        positions, grid, material_map, voxel_size, m_max, material_names = _prepare_dda(
            p, config, voxel_size=10.0
        )

        assert positions.ndim == 2 and positions.shape[1] == 3
        assert positions.shape[0] > 0
        assert voxel_size == 10.0
        assert m_max > 0
```

- [ ] **Step 2: Update `TestSolveSingleWL.test_solve_single_wl_matches_solve_optics`**

The test calls `_prepare_dda` and then `_solve_single_wl` with the old signature. Update both:

Replace:
```python
    def test_solve_single_wl_matches_solve_optics(self, julia_available, soot_material):
        """_solve_single_wl should produce same result as original solve_optics for single wavelength."""
        from Aerosol3D import AerosolParticle, create_sphere
        from Aerosol3D.optics.datastructs import SimulationConfig
        from Aerosol3D.optics.dda_solver import _prepare_dda, _solve_single_wl, solve_optics

        p = AerosolParticle(name="soot_sphere", unit="nm")
        p.add_mesh("core", create_sphere((0, 0, 0), 50.0), soot_material)
        config_direct = SimulationConfig(wavelength=550.0, dipole_spacing=10.0)
        config_extracted = SimulationConfig(wavelength=550.0, dipole_spacing=10.0)

        # Direct solve_optics
        result_direct = solve_optics(p, config_direct, voxel_size=10.0, verbose=False)

        # Via _prepare_dda + _solve_single_wl
        positions, alpha_e, grid, material_map, voxel_size, m_max, material_names = _prepare_dda(
            p, config_extracted, voxel_size=10.0
        )
        result_extracted = _solve_single_wl(
            positions,
            alpha_e,
            grid,
            material_map,
            config_extracted,
            m_max,
            voxel_size,
            material_names,
            compute_near_field=True,
            compute_phase_func=True,
            verbose=False,
        )

        assert result_direct.n_dipoles == result_extracted.n_dipoles
        assert result_direct.cross_sections.C_ext == pytest.approx(
            result_extracted.cross_sections.C_ext, abs=1e-10
        )
        assert result_direct.cross_sections.C_sca == pytest.approx(
            result_extracted.cross_sections.C_sca, abs=1e-10
        )
        assert result_direct.cross_sections.C_abs == pytest.approx(
            result_extracted.cross_sections.C_abs, abs=1e-10
        )
        assert result_direct.cross_sections.g == pytest.approx(
            result_extracted.cross_sections.g, abs=1e-10
        )
```
With:
```python
    def test_solve_single_wl_matches_solve_optics(self, julia_available, soot_material):
        """_solve_single_wl should produce same result as solve_optics for single wavelength."""
        from Aerosol3D import AerosolParticle, create_sphere
        from Aerosol3D.optics.datastructs import SimulationConfig
        from Aerosol3D.optics.dda_solver import _prepare_dda, _solve_single_wl, solve_optics

        p = AerosolParticle(name="soot_sphere", unit="nm")
        p.add_mesh("core", create_sphere((0, 0, 0), 50.0), soot_material)
        config_direct = SimulationConfig(wavelength=550.0, dipole_spacing=10.0)
        config_extracted = SimulationConfig(wavelength=550.0, dipole_spacing=10.0)

        # Direct solve_optics
        result_direct = solve_optics(p, config_direct, voxel_size=10.0, verbose=False)

        # Via _prepare_dda + _solve_single_wl
        positions, grid, material_map, voxel_size, m_max, material_names = _prepare_dda(
            p, config_extracted, voxel_size=10.0
        )
        result_extracted = _solve_single_wl(
            grid,
            material_map,
            config_extracted,
            m_max,
            voxel_size,
            material_names,
            compute_near_field=True,
            compute_phase_func=True,
            verbose=False,
        )

        assert result_direct.n_dipoles == result_extracted.n_dipoles
        assert result_direct.cross_sections.C_ext == pytest.approx(
            result_extracted.cross_sections.C_ext, abs=1e-10
        )
        assert result_direct.cross_sections.C_sca == pytest.approx(
            result_extracted.cross_sections.C_sca, abs=1e-10
        )
        assert result_direct.cross_sections.C_abs == pytest.approx(
            result_extracted.cross_sections.C_abs, abs=1e-10
        )
        assert result_direct.cross_sections.g == pytest.approx(
            result_extracted.cross_sections.g, abs=1e-10
        )
```

- [ ] **Step 3: Run the full test suite**

Run: `cd /home/zhangfan/Project/20260319_SPEMBSSBDART/aerosol3d && python -m pytest tests/test_dda_solver.py -v -x 2>&1 | tail -30`

Expected: All tests PASS. The LDR polarizability produces valid results for all existing test cases.

- [ ] **Step 4: Commit**

```bash
git add tests/test_dda_solver.py
git commit -m "test: update test signatures for LDR refactor

_prepare_dda returns 6 items (no alpha_e), _solve_single_wl
computes alpha_e internally.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 6: Add new LDR-specific tests

**Files:**
- Modify: `tests/test_dda_solver.py`

- [ ] **Step 1: Add `test_ldr_polarizability_reduces_to_rr_for_standard_geometry`**

This test verifies the LDR formula produces reasonable values. For propagation along z with transverse polarization, S=0, so the LDR correction simplifies. Add this test class to the end of `tests/test_dda_solver.py`:

```python
class TestLDRPolarizability:
    def test_ldr_s_parameter(self):
        """S should be 0 for transverse polarization along z, nonzero otherwise."""
        from Aerosol3D.optics.dda_solver import _ldr_S

        # Propagation along z, x-polarization: S = 0
        assert _ldr_S((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)) == 0.0
        assert _ldr_S((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)) == 0.0
        # Propagation along z, z-polarization: S = 1
        assert _ldr_S((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)) == 1.0
        # Propagation along x, x-polarization: S = 1
        assert _ldr_S((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)) == 1.0

    def test_ldr_polarizability_finite_and_complex(self, soot_material):
        """LDR alpha_e should be finite complex array for a real particle."""
        from Aerosol3D import AerosolParticle, create_sphere
        from Aerosol3D.optics.datastructs import SimulationConfig
        from Aerosol3D.optics.dda_solver import _compute_polarizability, _prepare_dda

        p = AerosolParticle(name="soot_sphere", unit="nm")
        p.add_mesh("core", create_sphere((0, 0, 0), 50.0), soot_material)
        config = SimulationConfig(wavelength=550.0, dipole_spacing=10.0)

        _, grid, material_map, _, _, _ = _prepare_dda(p, config, voxel_size=10.0)
        config.dipole_spacing = 10.0
        alpha_e = _compute_polarizability(grid, config, material_map)

        assert alpha_e.ndim == 1
        assert len(alpha_e) > 0
        assert np.all(np.isfinite(alpha_e))
        assert alpha_e.dtype == np.complex128

    def test_precision_ultra_accepted(self):
        """The 'ultra' precision level should be accepted by auto_voxel_size."""
        from Aerosol3D.optics.datastructs import auto_voxel_size

        vs = auto_voxel_size(550.0, 2.0, "ultra")
        assert vs > 0
        # Verify |m|*k*d <= 0.15
        k = 2.0 * np.pi / 550.0
        mkd = 2.0 * k * vs
        assert mkd <= 0.15
```

- [ ] **Step 2: Run new tests**

Run: `cd /home/zhangfan/Project/20260319_SPEMBSSBDART/aerosol3d && python -m pytest tests/test_dda_solver.py::TestLDRPolarizability -v 2>&1 | tail -20`

Expected: All 3 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_dda_solver.py
git commit -m "test: add LDR polarizability and ultra precision tests

Verifies S-parameter computation, finite alpha_e output, and ultra
precision level acceptance.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 7: Lint and full test suite

**Files:**
- No new changes — verification only

- [ ] **Step 1: Run ruff lint**

Run: `cd /home/zhangfan/Project/20260319_SPEMBSSBDART/aerosol3d && python -m ruff check src/Aerosol3D/optics/dda_solver.py src/Aerosol3D/optics/datastructs.py`

Expected: No errors. If any, fix them.

- [ ] **Step 2: Run ruff format check**

Run: `cd /home/zhangfan/Project/20260319_SPEMBSSBDART/aerosol3d && python -m ruff format --check src/Aerosol3D/optics/dda_solver.py src/Aerosol3D/optics/datastructs.py`

Expected: All files formatted. If not, run `python -m ruff format src/Aerosol3D/optics/dda_solver.py src/Aerosol3D/optics/datastructs.py` and commit the fix.

- [ ] **Step 3: Run full DDA test suite**

Run: `cd /home/zhangfan/Project/20260319_SPEMBSSBDART/aerosol3d && python -m pytest tests/test_dda_solver.py -v 2>&1 | tail -40`

Expected: All tests PASS (both old and new).

- [ ] **Step 4: Run other optics tests**

Run: `cd /home/zhangfan/Project/20260319_SPEMBSSBDART/aerosol3d && python -m pytest tests/test_mie_solver.py tests/test_bridge.py tests/test_orientational_average.py tests/test_optics_integration.py -v 2>&1 | tail -30`

Expected: All tests PASS (these tests don't depend on DDA solver internals).

- [ ] **Step 5: Final commit if any formatting fixes were needed**

```bash
git add -A
git commit -m "style: lint and format fixes for LDR refactor

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```
