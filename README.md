# aerosol3d

[![CI](https://github.com/openEarthModelling/aerosol3d/actions/workflows/ci.yml/badge.svg)](https://github.com/openEarthModelling/aerosol3d/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

aerosol3d is a Python toolkit for modeling the 3D geometry and optical properties of atmospheric aerosol particles. It provides a unified pipeline from geometric construction to Discrete Dipole Approximation (DDA) optical computation.

## Features

- **3D Geometry Modeling** — Build spheres, ellipsoids, cubes, and import fractal aggregates
- **Coating Algorithms** — Apply distance-based, potential-based, CCM (Closed-Cell Model), and CAM (Coated-Aggregate Model) coatings
- **Optical Computation** — Solve optical properties via DDA using a Julia-based backend with optional GPU acceleration
- **Visualization** — Generate 3D screenshots and rotation videos using PyVista
- **Material Database** — Built-in refractive index data for common aerosol materials
- **Flexible I/O** — Export to VTP and voxel formats

## Installation

Requires Python >= 3.10.

```bash
pip install aerosol3d
```

For development:

```bash
git clone https://github.com/openEarthModelling/aerosol3d.git
cd aerosol3d
pip install -e ".[dev]"
```

### Optional Dependencies

- **Julia backend** (required for DDA optical computation): Install Julia and run `pip install pyjulia`
- **GPU acceleration**: Requires CUDA-capable GPU and Julia CUDA packages

## Quick Start

```python
from aerosol3d import (
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

# DDA optical computation
config = SimulationConfig(wavelength=550.0, source="solar")
result = solve_optics(particle, config)
print(f"Extinction efficiency: {result.qext}")
```

See the [`examples/`](examples/) directory for complete workflows including fractal aggregates and coated particles.

## Examples

| Example | Description |
|---------|-------------|
| [`black_carbon_sphere.py`](examples/black_carbon_sphere.py) | Bare BC sphere with DDA optics |
| [`black_carbon_fractal.py`](examples/black_carbon_fractal.py) | Fractal aggregate via pyFracAggregate with full pipeline |

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

- `solve_optics(particle, config)` — DDA optical solver
- `SimulationConfig(wavelength, source)` — Simulation parameters

### I/O & Visualization

- `save_screenshot(particle, path)`
- `save_rotation_video(particle, path)`
- `save_vtp(particle, path)` / `save_voxel(particle, path)`

## Development

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=aerosol3d --cov-report=term-missing
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

DDA optical computation is powered by [CEMD.jl](https://github.com/krcools/CEMD.jl) via a Python-Julia bridge.
