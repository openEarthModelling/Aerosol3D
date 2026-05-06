# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GitHub Actions CI workflow
- ruff linting configuration
- pre-commit hooks
- py.typed marker for type hint support

## [0.2.0] - 2026-04-15

### Added
- GPU acceleration support for DDA optical solver
- Auto voxel size estimation in `solve_optics`
- Verbose output mode for optical computations
- Solar depolarized mode in `SimulationConfig`
- `preset_material()` helper for common aerosol materials
- Rotation video generation for 3D particle visualization
- Enhanced `print_macroscopic()` to auto-extract solve_time from `OpticalResult`

### Fixed
- Camera position update issue in rotation video generation
- Missing CUDA import for GPU detection in bridge.py
- Various code review issues in optics module

## [0.1.0] - 2026-04-01

### Added
- Initial release with core geometry modeling
- DDA optical computation via Julia bridge
- Coating algorithms: distance, potential, CCM, CAM
- Fractal aggregate import from pyFracAggregate
- 3D visualization via PyVista
- Material database with refractive indices
- VTP and voxel export formats
