# DDA-Mie-pyRadtran Pipeline Example

Complete example: from micro-scale optical computation to macro-scale radiative transfer.

## Overview

This example demonstrates how to integrate Aerosol3D's optical computations (DDA and Mie) with pyRadtran for radiative transfer calculations, and compare the differences between DDA and Mie results.

## Particle Configuration

- **Material**: Black carbon (black_carbon)
- **Shape**: Sphere
- **Radius**: 50 nm
- **Wavelength range**: 400-700 nm (7 points)

## Radiative Transfer Scene

- **Atmosphere**: US Standard, ground level
- **Solar**: Zenith angle 30 degrees
- **Solver**: DISORT, 16 streams
- **Surface**: Albedo 0.1
- **Aerosol vertical profile**: Exponential decay, scale height 1.5 km, total optical depth tau_550 = 0.3

## Dependencies

### System Dependencies

- Julia (required for DDA computation)
- libRadtran (pyRadtran backend)

### Python Dependencies

```bash
pip install -r requirements.txt
```

## Running Steps

### Stage 1: Optical Computation

```bash
cd examples/dda_mie_pyradtran_pipeline
python compute_optics.py
```

This will:
- Compute optical properties using DDA for a black carbon sphere
- Compute optical properties using Mie theory for the same particle
- Save to `output/optics_dda.nc` and `output/optics_mie.nc`
- Generate P11 phase function comparison plots

**Test with Mie only (fast, no Julia needed):**
```bash
python compute_optics.py --solver MIE
```

### Stage 2: Radiative Transfer

```bash
python run_radiative_transfer.py
```

This will:
- Read optical data and build pyRadtran `CompositeAerosol`
- Run DISORT radiative transfer solver
- Save to `output/rt_dda.nc` and `output/rt_mie.nc`

**Required environment variable:**
```bash
export PYRADTRAN_DATA_PATH=/usr/local/share/libRadtran/data
```

### Stage 3: Result Comparison

```bash
python compare_results.py
```

This will:
- Plot irradiance spectral comparison (6 PNG files)
- Generate statistical summary `output/summary.txt`

## Output Files

```
output/
├── optics_dda.nc              # DDA optical properties
├── optics_mie.nc              # Mie optical properties
├── p11_comparison_*.png       # P11 comparison plots per wavelength
├── rt_dda.nc                  # DDA radiative transfer results
├── rt_mie.nc                  # Mie radiative transfer results
├── compare_*.png              # Irradiance comparison plots
└── summary.txt                # Statistical summary
```

## Expected Results

For a 50 nm black carbon sphere, DDA and Mie results should be very close (difference < 1%) because:
- Spherical particles are the exact solution for Mie theory
- DDA should converge to Mie results with sufficient dipole resolution

If differences are significant, possible causes:
- DDA precision insufficient (too few dipoles)
- Numerical integration error (insufficient phase function angular resolution)

## Future Extensions

Replace the particle geometry (non-spherical, coated) to study radiative effects of complex particles:

```python
# Example: Coated fractal aggregate
from Aerosol3D import from_fractal, apply_ccm_coating
fractal = from_fractal(aggregate, soot)
coated = apply_ccm_coating(fractal, sulfate)
particle = coated.to_particle()
```

Then re-run the three-stage pipeline.
