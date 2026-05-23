# Changelog

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

[0.4.0]: https://github.com/openEarthModelling/aerosol3d/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/openEarthModelling/aerosol3d/compare/v0.2.0...v0.3.0
[0.5.0]: https://github.com/openEarthModelling/aerosol3d/compare/v0.4.0...v0.5.0
[Unreleased]: https://github.com/openEarthModelling/aerosol3d/compare/v0.5.0...HEAD
[0.2.0]: https://github.com/openEarthModelling/aerosol3d/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/openEarthModelling/aerosol3d/releases/tag/v0.1.0
