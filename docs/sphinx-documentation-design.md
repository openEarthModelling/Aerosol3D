---
orphan: true
---

# Sphinx Documentation Site Design

**Date:** 2026-05-11
**Scope:** Build a complete Sphinx documentation site for Aerosol3D v0.2.0
**Target:** Researchers (tutorials + physics) and Developers (API reference)
**Deployment:** GitHub Pages via GitHub Actions
**URL:** https://openearthmodelling.github.io/Aerosol3D/

---

## 1. Document Architecture

```
docs/                           # Sphinx root
├── conf.py                     # Sphinx config (RTD theme, autodoc, myst)
├── index.rst                   # Home page with toctree
├── user-guide/                 # For researchers
│   ├── index.rst
│   ├── installation.rst
│   ├── quickstart.rst
│   ├── particle-geometry.rst
│   ├── coating-models.rst
│   └── optical-computation.rst
├── tutorials/                  # Step-by-step guides
│   ├── index.rst
│   ├── black-carbon-sphere.rst
│   ├── coated-particle.rst
│   ├── fractal-aggregate.rst
│   └── mie-vs-dda-validation.rst
├── api-reference/              # For developers (autodoc)
│   ├── index.rst
│   ├── core.rst
│   ├── geometry.rst
│   ├── modeling.rst
│   ├── optics.rst
│   └── io.rst
├── examples/                   # Links to example scripts
└── contributing.rst
```

## 2. Tech Stack

| Component | Package | Purpose |
|-----------|---------|---------|
| Builder | Sphinx >= 7.0 | Core documentation build |
| Theme | sphinx-rtd-theme >= 2.0 | ReadTheDocs visual style |
| Markdown | myst-parser >= 2.0 | Markdown support in docs |
| API docs | sphinx-autodoc, sphinx-autosummary | Auto-generate from docstrings |
| Math | sphinx.ext.mathjax | LaTeX equation rendering |
| UX | sphinx-copybutton | Copy-to-clipboard for code blocks |

Added to `pyproject.toml` under `[project.optional-dependencies]`:

```toml
docs = [
    "sphinx>=7.0",
    "sphinx-rtd-theme>=2.0",
    "myst-parser>=2.0",
]
```

## 3. Content Plan

### 3.1 Home Page
- Project intro (condensed from README)
- Feature cards: Geometry / Coating / Optics / Visualization
- Quick navigation: Install → Quick Start → API Reference

### 3.2 User Guide (researcher-facing)

| Page | Content |
|------|---------|
| Installation | pip install, Julia backend setup, GPU notes |
| Quick Start | Extended README quickstart with parameter explanations |
| Particle Geometry | Voxelization concept, internal/external mixing state |
| Coating Models | CCM/CAM/distance/potential edge/potential void — physics and use cases |
| Optical Computation | DDA principle, Mie theory scope, convergence criterion \|m\|kd < 1 |

### 3.3 Tutorials (step-by-step)

Each tutorial uses `literalinclude` to embed example scripts from `examples/`:

| Tutorial | Source Script |
|----------|--------------|
| Black Carbon Sphere | `examples/black_carbon_sphere.py` |
| Coated Particle | `examples/black_carbon_fractal.py` |
| Fractal Aggregate | `examples/coated_fractal_aggregate.py` |
| MIE vs DDA Validation | `examples/validate_mie_vs_dda.py` |

### 3.4 API Reference (developer-facing)

Generated via `automodule` / `autoclass`, grouped by module:
- `core` — AerosolParticle, Material, FractalAggregate, MixingState
- `geometry` — create_sphere, create_ellipsoid, create_cube
- `modeling` — 8 coating functions
- `optics` — solve_optics, SimulationConfig, OpticalResult, CrossSections, PhaseFunction
- `io` — save_voxel, save_vtp

## 4. Deployment

### GitHub Actions Workflow
File: `.github/workflows/docs.yml`

Triggers: push to `main`, PRs to `main`

Steps:
1. Checkout code
2. Setup Python 3.11
3. `pip install -e ".[docs]"`
4. `sphinx-build -b html docs/ docs/_build/html`
5. Upload artifact
6. Deploy to GitHub Pages (only on `main` push)

### URL
`https://openearthmodelling.github.io/Aerosol3D/`

Requires enabling GitHub Pages in repo settings (source: GitHub Actions).

## 5. Out of Scope

- Multi-version docs (readthedocs-style version switcher)
- Full mathematical derivations (brief physical intuition only)
- Julia API documentation (only Python API)
- Interactive notebooks / Binder integration

## 6. Success Criteria

- [ ] `sphinx-build` completes with 0 warnings
- [ ] All public APIs have docstring-generated reference pages
- [ ] All 5 example scripts have corresponding tutorials
- [ ] GitHub Actions deploys successfully on push to main
- [ ] Site is accessible at the GitHub Pages URL
