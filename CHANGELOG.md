# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/openEarthModelling/aerosol3d/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/openEarthModelling/aerosol3d/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/openEarthModelling/aerosol3d/releases/tag/v0.1.0
