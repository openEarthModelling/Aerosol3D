# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2026-05-11

### Added

- **MIE solver integration** — New `solver="MIE"` option in `solve_optics()` using PyMieScatt
  for spherical particles. MIE theory provides exact optical properties (Q_ext, Q_sca, Q_abs, g, P11)
  for comparison and validation against DDA.
- `solver` field added to `OpticalResult` dataclass for result provenance tracking.
- `equivalent_diameter` and `effective_refractive_index` properties added to `AerosolParticle`
  class for solver-agnostic particle characterization.
- DDA **orientational averaging** via Fibonacci sphere sampling. Averages cross-sections and
  phase function over multiple incident directions.
- `compute_legendre_moments()` for expanding phase functions into Legendre polynomial coefficients.
- `validate_mie_vs_dda.py` example script for comparing MIE and DDA results.
- `validate_grid_convergence.py` example script for demonstrating DDA grid convergence.

### Fixed

- **DDA asymmetry parameter g** — Corrected spherical quadrature weights in `_spherical_grid()`.
  Weights were incorrectly proportional to `sin(theta)`, vanishing at the forward scattering
  direction and causing ~95% error in g. Now uses constant solid-angle weights
  `dOmega = d(cos theta) * dphi` for the uniform cos(theta) grid.
- **DDA scattering angle computation** — `compute_asymmetry_parameter()` now uses
  `dot(k_inc, k_scat)` instead of the polar angle from the z-axis, fixing g for non-z-axis
  incident directions.
- **DDA orientational averaging for g** — `_orientational_average()` now averages g
  arithmetically across orientations instead of recomputing from lab-frame P11 (which is
  isotropic by construction, yielding g ~ 0).
- **P11 phase function normalization** — `_compute_phase_function()` now normalizes P11
  by C_sca so that the integral over the full sphere equals 1, matching MIE theory convention.
- `n_dirs=1` no longer crashes Fibonacci sphere generation.

### Changed

- `solve_optics()` now accepts explicit `solver="MIE"` or `solver="DDA"` parameter.

## [0.1.0] - 2026-05-08

### Added

- Initial release with DDA solver via CEMD.jl (Julia) integration.
- Fractal aggregate modeling with PyFrac.
- Geometry primitives: sphere, ellipsoid, cube.
- Coating models: CAM, CCM, distance-based, potential edge, potential void.
- Voxelization and visualization utilities.
- pyRadtran export for radiative transfer applications.
