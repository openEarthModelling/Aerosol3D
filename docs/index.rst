Aerosol3D Documentation
=======================

Aerosol3D is a Python toolkit for modeling the 3D geometry and optical
properties of atmospheric aerosol particles. It provides a unified pipeline
from geometric construction to Discrete Dipole Approximation (DDA) optical
computation, with an optional Mie theory solver for spherical particles.

Features
--------

- **Geometry Modeling** — Build spheres, ellipsoids, cubes, and import fractal aggregates.
- **Coating Algorithms** — Apply distance-based, potential-based, CCM, and CAM coatings.
- **Optical Computation** — Solve optical properties via DDA (Julia backend) or Mie theory.
- **Visualization** — Generate 3D screenshots and rotation videos with PyVista.

Quick Navigation
----------------

- :doc:`user-guide/installation` — Install Aerosol3D and optional dependencies
- :doc:`user-guide/quickstart` — Your first particle model and optical solve
- :doc:`tutorials/index` — Step-by-step tutorials for common workflows
- :doc:`api-reference/index` — Complete API reference for developers

.. toctree::
   :maxdepth: 2
   :hidden:

   user-guide/index
   tutorials/index
   api-reference/index
   contributing
