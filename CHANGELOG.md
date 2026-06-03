# Changelog

## [Unreleased]

## [0.8.0] - 2026-06-03

### Added
- Bulk aerosol optics computation module (`Aerosol3D.bulk`) with two merge methods:
  - Method 1 (bin weights): discrete radius-bin weighted averaging
  - Method 2 (continuous quadrature): adaptive Gauss-Kronrod and fixed-quadrature integration over continuous size distributions
- `BulkOpticsBuilder` — builder pattern for constructing bulk optical properties with validation, Mie ripple detection, and automatic fallback to Method 1
- `SizeDistribution` dataclass with lognormal and gamma distribution factories
- `BulkAerosolOpticsData` frozen dataclass for bulk optical property storage
- Physics-safe interpolation kernels (log-log and linear PCHIP) for cross-wavelength and cross-radius data
- NetCDF I/O for bulk aerosol optics: standard format (full provenance) and vSmartMOM-compatible format
- `BulkAerosolOpticsData.to_netcdf()` / `.from_netcdf()` convenience methods
- Top-level exports: `BulkOpticsBuilder`, `SizeDistribution`, `BulkAerosolOpticsData` from `Aerosol3D`
- Cross-validation example script (`bulk_method_comparison.py`) for dual-path asymmetry factor consistency check
- Bulk cross-validation tutorial demonstrating validation methodology
- Bulk optics workflow tutorial demonstrating step-by-step Method 1 usage
- Bulk aerosol optics API reference and user guide with mathematical derivations

### Changed
- `merge_method2` now supports both `"quad"` (adaptive Gauss-Kronrod, default) and `"fixed_quad"` integration methods
- `BulkOpticsBuilder.compute()` auto-extracts refractive index from `AerosolOpticsData` fields when `check_mie_ripples=True` and no explicit refractive index is provided
- `cspline` interpolation triggers a `UserWarning` with automatic fallback to `pchip`

### Fixed
- Beta convention: correctly convert Aerosol3D `g_l` to vSmartMOM `beta_l = (2l+1)g_l` in `BulkOpticsBuilder`
- `beta_0 = 1.0` normalization enforced exactly in both `merge_method1` and `merge_method2`

### Tests
- Physical validation tests for bulk optics: SSA bounds, `beta_0 = 1`, energy conservation, monodisperse limit
- End-to-end Mie multisize to bulk integration test
- Cross-validation example for dual-path `g` consistency check

## [0.7.0] - 2026-05-26

### Added
- LDR (Lattice Dispersion Relation) polarizability (Draine & Goodman 1993) replacing Radiative Reaction model — reduces DDA-Mie discrepancy from 10-30% to ~1% at medium precision
- `_compute_polarizability()` function with per-solve LDR correction, supporting separate S parameters for x/y polarizations in depolarized mode
- Fourth precision level `"ultra"` (|m|kd ≤ 0.15) for very high accuracy DDA
- `test_extract_dipoles.py` rewritten with LDR unit tests: S parameter, single/multi material, LDR→CM convergence check
- `TestLDRPolarizability` test class in `test_dda_solver.py`
- `test_auto_voxel_size_ultra` precision test

### Changed
- Precision targets updated based on Yurkin & Hoekstra (2007): `low` (0.63), `medium` (0.5), `high` (0.3), `ultra` (0.15)
- `_prepare_dda` no longer returns positions (5-tuple); positions extracted per-solve in `_solve_single_wl`
- Removed `_apply_radiative_correction` and `_extract_dipoles` functions (superseded by `_compute_polarizability`)
- `auto_voxel_size` docstring updated to document `"ultra"` precision
- NumPy dependency upper bound `<2.0` removed

### Fixed
- NumPy 2.x compatibility: `_trapz` compat shim (`hasattr`-based) replaces bare `np.trapz`/`np.trapezoid` in `mie_solver.py` and test files
- `test_datastructs.py` precision tests updated for new target values
- Stale `test_extract_dipoles.py` that referenced deleted functions rewritten

### Documentation
- LDR Polarizability section with Draine & Goodman 1993 formula and 4-tier precision level table added to optical computation guide
- Mie vs DDA validation tutorial rewritten for multi-precision LDR convergence comparison
- `precision` parameter added to quickstart key parameters
- README updated with LDR polarizability, precision levels in features/API/example table
- `validate_mie_vs_dda.py` example rewritten for multi-precision convergence analysis

### Tests
- All Julia-dependent tests marked `@pytest.mark.slow` — fast suite (`-m "not slow"`) runs in ~1s
- Orientational averaging tests: `n_dirs` reduced to 2 for CI speed

## [0.6.0] - 2026-05-24

### Added
- Parallel DDA orientation averaging using `joblib.Parallel` with `loky` backend
- Flattened (wavelength × orientation) task grid into single parallel dispatch for full CPU utilization
- `n_jobs` parameter on `solve_optics()` — number of parallel workers (default 32)
- `show_progress` parameter on `solve_optics()` — tqdm progress bars (default True)
- Error-tolerant parallel execution — failed orientation tasks are logged and skipped, not discarded

### Changed
- `n_dirs` default changed from 100 to 50
- Warning threshold for low direction count changed to `n_dirs < 30`
- Near-field computation disabled in parallel orientation path (results are discarded)
- Updated `validate_mie_vs_dda` example to use parallel orientation averaging

### Fixed
- Restored phase-function-specific warning when `compute_phase_func=True` with `n_dirs < 100`
- Explicitly set `backend="loky"` to prevent unsafe threading with PyJulia

### Documentation
- Expanded orientational averaging section in optical computation guide with parallel execution, parameters, and code example
- Updated Mie vs DDA validation tutorial to reflect `n_dirs=50` and `n_jobs=32`
- Added `orientational_average`, `n_dirs`, `n_jobs`, `show_progress` to quickstart key parameters
- Added Parallel Orientation Averaging to README features and updated `solve_optics` API signature
- Documented `tqdm` and `joblib` as required dependencies in installation guide

## [0.5.0] - 2026-05-23

### Added
- EMA (Effective Medium Approximation) module (`core/ema.py`) with three methods:
  - `volume_weighted` — simple volume-weighted average of refractive indices
  - `maxwell_garnett` — Maxwell-Garnett mixing rule for inclusion-in-host geometries
  - `bruggeman` — Bruggeman symmetric mixing rule with Newton-iteration solver
- `coreshell_geometry` property on `Particle` — computes core-shell geometry using volume-based core identification
- `solve_mie_coreshell()` — Mie solver for coated spheres using PyMieScatt's core-shell API
- `ema_method` parameter on `solve_mie()`, `solve_optics()`, and `MIE_CORESHELL` dispatcher
- Integration tests for all EMA methods and core-shell solver (`test_ema_integration.py`)
- `legendre_moments_beta` field on `AerosolOpticsData` — stores DISORT-ready normalized Legendre moments (β_l = k_l/(2l+1))
- Updated `coated_fractal_aggregate` example — compares four Mie approximations with spectral plots

### Changed
- Refactored `Particle.effective_refractive_index` to delegate to EMA module (backward compatible)
- Simplified pipeline example using `ParticleOptics.from_aerosol3d()`

### Fixed
- Removed unsupported `deltam` flag, reverted to ASCII output for libRadtran 2.0.6 compatibility

## [0.4.0] - 2026-05-18

### Added
- `AerosolOpticsData` dataclass — generic multi-wavelength optical property container with auto-computed Legendre moments (`optics_export.py`)
- `from_optical_results()` factory function builds `AerosolOpticsData` from `list[OpticalResult]`
- `AerosolOpticsData.to_netcdf()` / `.from_netcdf()` for NetCDF persistence
- Visualization functions: `plot_spectral_properties`, `plot_phase_function`, `plot_optical_comparison`, `plot_phase_function_comparison`, `plot_legendre_convergence`, `plot_legendre_moments_spectrum`, `generate_comparison_summary`
- Smoke tests for all new visualization functions (`test_optics_vis_new.py`)
- Unit tests for `AerosolOpticsData` construction, Legendre auto-computation, and NetCDF round-trip (`test_optics_export.py`)

### Changed
- Replaced `pyradtran_export.py` with format-agnostic `optics_export.py` — aerosol3d no longer produces pyRadtran-specific output
- Simplified `compute_optics.py` example (-204 lines) — uses `from_optical_results()` and library visualization
- Simplified `compare_results.py` example (-192 lines) — uses library comparison functions
- `run_radiative_transfer.py` now loads `AerosolOpticsData.from_netcdf()` and passes Legendre moments to pyRadtran

### Fixed
- Legendre moments from phase functions are now auto-computed and passed through the RT pipeline (previously fell back to Henyey-Greenstein approximation)
- Convert k_l → beta_l (divide by 2l+1) before passing to DISORT PMOM — DISORT expects normalized moments, not raw coefficients
- `plot_phase_function_comparison` handles datasets with different theta grid sizes via interpolation

### Documentation
- Updated README.md features, API overview, quick start, and examples table to reflect Mie solver, `AerosolOpticsData`, and optical visualization
- Added `optics_export`, `visualization`, `legendre` modules to Sphinx API reference
- Added optical data export, Legendre moments, and visualization sections to optical-computation user guide
- Added step 4 (export and visualize) to quickstart guide
- Added DDA-Mie-pyRadtran pipeline tutorial and coated fractal aggregate tutorial
- Added optional dependencies section (matplotlib, xarray, netCDF4, pyRadtran) to installation guide
- Expanded mie-vs-dda-validation tutorial with detailed explanations
- Added documentation contribution guidelines to contributing guide
- Synced CONTRIBUTING.md with Sphinx version

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-05-16

### Added
- DDA-Mie-pyRadtran optical computation pipeline with stage 1 (compute optics), stages 2-3, and pipeline config/directory structure
- Sphinx documentation site with user guide, tutorials (with embedded example scripts), API reference (autodoc), and contributing guide
- Sphinx optional-dependencies group (`sphinx`, `sphinx-rtd-theme`, `myst-parser`, `sphinx-copybutton`)
- GitHub Actions workflow for building and deploying Sphinx docs to GitHub Pages

### Changed
- DDA particle radius increased to 200nm with manual 5nm dipole spacing

### Fixed
- DDA multi-wavelength polarizability: alpha_e is now recomputed per wavelength instead of reusing the first wavelength's value (previously caused up to 435% error)
- DDA depolarized P11 phase function: now properly averages both orthogonal polarizations instead of using only one
- DDA precision targets: swapped inverted high/low values so "high" produces finer dipole spacing
- P11 comparison plots: handle different theta grid dimensions between DDA and Mie

## [0.2.0] - 2026-05-11

### Added
- MIE solver bridge using PyMieScatt with `solve_optics` dispatch
- DDA orientational averaging over Fibonacci sphere
- MIE vs DDA validation example script
- Scattering asymmetry parameter (g) and P11 agreement assertions in validation

### Changed
- Improved MIE solver code quality and tests
- Improved particle properties robustness and tests

### Fixed
- Scattering angle and orientational averaging for g parameter
- P11 normalization by C_sca in `_compute_phase_function`
- Spherical quadrature weights in `_spherical_grid`
- Handle `n_dirs=1` edge case in Fibonacci sphere and orientational averaging
- Parameter name `C_sca`→`c_sca` and tightened spherical grid tolerance in tests

## [0.1.0] - 2026-05-07

### Added
- Initial release of aerosol3d
- DDA solver for computing optical properties of non-spherical aerosol particles
- Particle shape generation (sphere, spheroid, cube, coated sphere)
- Core data structures for optical properties and simulation configuration

[0.8.0]: https://github.com/openEarthModelling/aerosol3d/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/openEarthModelling/aerosol3d/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/openEarthModelling/aerosol3d/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/openEarthModelling/aerosol3d/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/openEarthModelling/aerosol3d/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/openEarthModelling/aerosol3d/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/openEarthModelling/aerosol3d/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/openEarthModelling/aerosol3d/releases/tag/v0.1.0
