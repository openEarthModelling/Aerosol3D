# Aerosol3D

Light scattering simulations for aerosol particles using DDA (Discrete Dipole Approximation) and Mie theory.

## Commands

```bash
python -m pytest tests/ -v                    # Run tests
python -m ruff check src/                     # Lint
python -m ruff format src/ tests/             # Format
pip install -e ".[dev]"                       # Dev install
```

DDA solver requires Julia with CoupledElectricMagneticDipoles.jl installed.

## Architecture

```
src/Aerosol3D/
  core/           # Particle, Material, Aggregate dataclasses
  factory/        # Mesh creation (from_file, from_fractal)
  geometry/       # Primitives (sphere, cube, ellipsoid), boolean ops, voxelization
  io/             # Voxel/VTP export
  modeling/       # Coating models (CAM, CCM, distance, potential-edge, potential-void)
  optics/         # Optical property computation
    datastructs.py    # OpticalResult, CrossSections, PhaseFunction, SimulationConfig
    dda_solver.py     # solve_optics() — DDA + Mie dispatch
    mie_solver.py     # Mie theory solver
    bridge.py         # PyJulia bridge for DDA
    legendre.py       # compute_legendre_moments() — k_l = (2l+1)*integral convention
    optics_export.py   # AerosolOpticsData dataclass + from_optical_results() + NetCDF I/O
    visualization.py  # Plotting: spectral, phase function, comparison, Legendre diagnostics
  physics/        # Unit handling
  utils/          # Plotting utilities
```

## Key Patterns

- `solve_optics(particle, config, solver="DDA"|"MIE")` dispatches to DDA or Mie solver
- `from_optical_results(results, n_legendre=32)` builds `AerosolOpticsData` with auto-computed Legendre moments
- `AerosolOpticsData.to_netcdf()` / `.from_netcdf()` for persistence
- `OpticalResult` = single wavelength; `AerosolOpticsData` = multi-wavelength export container

## Gotchas

- **DISORT PMOM format**: `compute_legendre_moments()` returns k_l = (2l+1)*integral (coefficient form). DISORT/libRadtran expects beta_l = k_l/(2l+1). Divide by (2l+1) before passing to pyRadtran.
- **DDA solver is slow**: Each wavelength takes 1-10 minutes depending on particle size. Mie is near-instant.
- **Julia bridge**: First call initializes Julia runtime (~30s). DDA requires CoupledElectricMagneticDipoles.jl.
- **Phase function angles**: DDA and Mie produce different theta grids. Comparison functions must interpolate.
- **`docs/superpowers/`**: Never git-commit files under this directory. They are internal workflow artifacts.

## Example Pipeline

`examples/dda_mie_pyradtran_pipeline/` — three-stage DDA vs Mie comparison:

1. `compute_optics.py` — Compute optical properties, save to NetCDF via AerosolOpticsData
2. `run_radiative_transfer.py` — Run DISORT RT via pyRadtran (requires libRadtran)
3. `compare_results.py` — Compare optical properties and RT results

Set `PYRADTRAN_DATA_PATH` to libRadtran data directory before running RT.
