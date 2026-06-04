# Aerosol3D Examples

This directory contains standalone example scripts demonstrating the main
workflows of Aerosol3D.  Each script can be run directly after installing the
package (`pip install -e .`).

## Quick Start

```bash
# Install Aerosol3D
pip install -e ".[dev]"

# Run an example
python examples/black_carbon_sphere.py
```

Some examples require additional system dependencies (Julia, libRadtran).  See
the per-example notes below.

---

## Single-Particle Optics

### `black_carbon_sphere.py`
Uncoated black carbon sphere — 3D visualization and DDA optical computation.

**Dependencies:** Aerosol3D, matplotlib, PyVista  
**Runtime:** ~1 min (mostly PyVista initialization)

### `black_carbon_fractal.py`
Black carbon fractal aggregate — full pipeline from geometry to optical
properties (DDA).

**Dependencies:** Aerosol3D, Julia + CoupledElectricMagneticDipoles.jl  
**Runtime:** ~5–10 min per wavelength

### `coated_fractal_aggregate.py`
Coated black carbon fractal aggregate — compare four Mie scattering
approximations for the coated particle.

**Dependencies:** Aerosol3D, matplotlib  
**Runtime:** ~30 s

### `validate_mie_vs_dda.py`
Validate DDA against Mie theory for a spherical particle.

**Dependencies:** Aerosol3D, Julia + CoupledElectricMagneticDipoles.jl  
**Runtime:** ~2 min

---

## Bulk Aerosol Optics

### `bulk_method_comparison.py`
Compare bulk optical properties computed by **Method 1** (bin weights) versus
**Method 2** (continuous interpolation).  Uses synthetic Henyey–Greenstein data.

**Dependencies:** Aerosol3D, matplotlib  
**Runtime:** ~10 s

### `bulk_cross_validation.py`
Cross-validation of the asymmetry parameter `g` via a dual-path consistency
check: quadrature from `P11(theta)` vs. analytic `g` from Mie/DDA.

**Dependencies:** Aerosol3D  
**Runtime:** ~5 s

---

## Radiative Transfer

### `dda_mie_pyradtran_pipeline/`
Three-stage pipeline: DDA/Mie optical computation → DISORT radiative transfer
(via pyRadtran) → result comparison.

**Dependencies:** Aerosol3D, Julia, libRadtran (with `PYRADTRAN_DATA_PATH`
environment variable), matplotlib  
**Runtime:** ~10 min total

See [`dda_mie_pyradtran_pipeline/README.md`](dda_mie_pyradtran_pipeline/README.md)
for detailed instructions.

### `vsmartmom_rt_demo.py`
Run column radiative transfer with **vSmartMOM** using `BulkAerosolOpticsData`
as input.  Demonstrates:

1. Create synthetic bulk aerosol optics (multi-wavelength)
2. Define a vertical number-concentration profile
3. Run RT via `VSmartMOMRunner` (Julia subprocess)
4. Plot TOA reflectance, BOA transmittance, and optical-depth profile

**Dependencies:** Aerosol3D, Julia + vSmartMOM.jl, matplotlib  
**Runtime:** ~30 s (Julia startup dominates)

If Julia is unavailable, the script falls back to showing the input setup and
expected workflow:

```bash
# Preview without Julia
python examples/vsmartmom_rt_demo.py --skip-rt

# Full run (requires --julia-project)
python examples/vsmartmom_rt_demo.py --julia-project /path/to/vsmartmom-project
```

---

## Summary Table

| Example | What it demonstrates | Needs Julia | Needs libRadtran |
|---------|----------------------|-------------|------------------|
| `black_carbon_sphere.py` | DDA + 3D viz | ✅ | ❌ |
| `black_carbon_fractal.py` | Fractal geometry → DDA | ✅ | ❌ |
| `coated_fractal_aggregate.py` | Coating + Mie approx | ❌ | ❌ |
| `validate_mie_vs_dda.py` | DDA validation against Mie | ✅ | ❌ |
| `bulk_method_comparison.py` | Bulk optics methods 1 vs 2 | ❌ | ❌ |
| `bulk_cross_validation.py` | Asymmetry parameter check | ❌ | ❌ |
| `dda_mie_pyradtran_pipeline/` | Full micro→macro RT pipeline | ✅ | ✅ |
| `vsmartmom_rt_demo.py` | vSmartMOM column RT | ✅ | ❌ |
