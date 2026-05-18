Mie vs DDA Validation
=====================

This tutorial validates DDA results against exact Mie theory for a spherical
particle. Mie theory provides an analytical solution for spheres, making it an
ideal reference for verifying the DDA numerical solver.

Full Script
-----------

.. literalinclude:: ../../examples/validate_mie_vs_dda.py
   :language: python
   :linenos:

Step-by-Step Explanation
------------------------

1. **Create a spherical particle**

   A 100 nm radius sphere with refractive index m = 1.5 + 0.01i is created.
   The particle's equivalent diameter and effective refractive index are printed
   for reference.

2. **Run Mie solver** (exact solution for spheres)

   ``solve_optics(particle, config, solver="MIE")`` computes the exact Mie
   solution, including Q_ext, Q_sca, g, and the full phase function P11. This
   serves as the ground truth.

3. **Run DDA solver** (approximate, grid-dependent)

   Two DDA runs are performed: a single-orientation solve and an orientational
   average over 100 directions. DDA approximates the sphere as a dipole grid,
   so accuracy depends on dipole spacing. The orientational average improves
   agreement with Mie for non-symmetric dipole arrangements.

4. **Compare and validate**

   Cross-sections (Q_ext, Q_sca) and asymmetry parameter (g) are compared.
   The g parameter must agree within 15% (single orientation) and 30%
   (orientational average). The phase function P11 is checked via log-space
   correlation (r > 0.90) after interpolating DDA results onto the Mie theta grid.

   If matplotlib is available, a semi-log phase function plot is saved as
   ``mie_vs_dda_phase_function.png``.

Further Analysis
-----------------

For multi-wavelength comparison with the full visualization toolkit, see
the :doc:`dda-mie-pyradtran-pipeline` tutorial which uses
``plot_optical_comparison()``, ``plot_phase_function_comparison()``, and
``generate_comparison_summary()`` for detailed DDA vs Mie analysis.
