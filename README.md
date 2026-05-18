# Aerosol3D

[![CI](https://github.com/openEarthModelling/Aerosol3D/actions/workflows/ci.yml/badge.svg)](https://github.com/openEarthModelling/Aerosol3D/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Aerosol3D is a Python toolkit for modeling the 3D geometry and optical properties of atmospheric aerosol particles. It provides a unified pipeline from geometric construction to Discrete Dipole Approximation (DDA) optical computation.

## Features

- **3D Geometry Modeling** — Build spheres, ellipsoids, cubes, and import fractal aggregates
- **Coating Algorithms** — Apply distance-based, potential-based, CCM (Closed-Cell Model), and CAM (Coated-Aggregate Model) coatings
- **Optical Computation** — Solve optical properties via DDA (Julia backend, optional GPU) or Mie theory (PyMieScatt, near-instant for spheres)
- **Optical Property Export** — Multi-wavelength `AerosolOpticsData` container with auto-computed Legendre moments and NetCDF I/O
- **Optical Visualization** — Spectral properties, phase function comparison, Legendre convergence diagnostics, and comparison summary plots
- **3D Visualization** — Generate screenshots and rotation videos using PyVista
- **Material Database** — Built-in refractive index data for common aerosol materials
- **Flexible I/O** — Export to VTP and voxel formats

## Installation

Requires Python >= 3.10.

```bash
pip install Aerosol3D
```

For development:

```bash
git clone https://github.com/openEarthModelling/Aerosol3D.git
cd Aerosol3D
pip install -e ".[dev]"
```

### Optional Dependencies

- **Julia backend** (required for DDA optical computation): Install Julia and run `pip install pyjulia`
- **GPU acceleration**: Requires CUDA-capable GPU and Julia CUDA packages

## Quick Start

```python
from Aerosol3D import (
    AerosolParticle, create_sphere, MixingState,
    preset_material, save_screenshot, solve_optics, SimulationConfig
)

# Create a black carbon sphere
soot = preset_material("black_carbon")
particle = AerosolParticle(
    name="bc_sphere",
    mixing_state=MixingState.INTERNAL,
    unit="nm",
)
particle.add_mesh("core", create_sphere((0, 0, 0), 50.0), soot)

# 3D visualization
save_screenshot(particle, "sphere.png", colors={"core": "black"})

# Optical computation (Mie or DDA)
config = SimulationConfig(wavelength=550.0, source="solar")
result = solve_optics(particle, config, solver="MIE")
print(f"Extinction efficiency: {result.qext}")
```

Export multi-wavelength results and visualize:

```python
from Aerosol3D.optics import from_optical_results

results = [solve_optics(particle, SimulationConfig(wavelength=w), solver="MIE")
           for w in [450, 550, 650]]
data = from_optical_results(results, n_legendre=32)
data.to_netcdf("optics.nc")
```

See the [`examples/`](examples/) directory for complete workflows including fractal aggregates, coated particles, and DDA-Mie comparison pipelines.

## Examples

| Example | Description |
|---------|-------------|
| [`black_carbon_sphere.py`](examples/black_carbon_sphere.py) | Bare BC sphere with DDA optics |
| [`black_carbon_fractal.py`](examples/black_carbon_fractal.py) | Fractal aggregate via pyFracAggregate with full pipeline |
| [`coated_fractal_aggregate.py`](examples/coated_fractal_aggregate.py) | Coated fractal aggregate with coating models |
| [`validate_mie_vs_dda.py`](examples/validate_mie_vs_dda.py) | Mie vs DDA validation comparison |
| [`dda_mie_pyradtran_pipeline/`](examples/dda_mie_pyradtran_pipeline/) | 3-stage DDA/Mie optics + DISORT radiative transfer pipeline |

## API Overview

### Core Classes

- `AerosolParticle` — Particle container with multiple meshes/materials
- `MixingState` — Internal / external mixing state
- `Material` — Refractive index and density
- `FractalAggregate` — Fractal aggregate geometry

### Geometry

- `create_sphere(center, radius)`
- `create_ellipsoid(center, radii)`
- `create_cube(center, size)`

### Coating

- `apply_distance_coating(particle, thickness, material)`
- `apply_potential_coating(particle, thickness, material)`
- `apply_ccm_coating(particle, thickness, material)`
- `apply_cam_coating(particle, thickness, material)`

### Optics

- `solve_optics(particle, config, solver="DDA"|"MIE")` — Optical solver dispatch
- `SimulationConfig(wavelength, source)` — Simulation parameters
- `AerosolOpticsData` / `from_optical_results(results, n_legendre)` — Multi-wavelength optical property container
- `compute_legendre_moments(phase_function, theta)` — Legendre expansion of scattering phase function

### Optical Visualization

- `plot_spectral_properties(data)` — Extinction/scattering/absorption spectra
- `plot_phase_function(data)` — Phase function P11 vs scattering angle
- `plot_optical_comparison(data1, data2)` — Side-by-side comparison of two datasets
- `plot_phase_function_comparison(data1, data2)` — Phase function comparison with relative difference
- `plot_legendre_convergence(data)` — Legendre moment convergence diagnostics

### I/O & Visualization

- `save_screenshot(particle, path)`
- `save_rotation_video(particle, path)`
- `save_vtp(particle, path)` / `save_voxel(particle, path)`

## Development

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=Aerosol3D --cov-report=term-missing
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

DDA optical computation is powered by [CEMD.jl](https://github.com/krcools/CEMD.jl) via a Python-Julia bridge.
