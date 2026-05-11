MIE vs DDA Validation
=====================

This tutorial validates DDA results against exact Mie theory for a spherical
particle.

Full Script
-----------

.. literalinclude:: ../../examples/validate_mie_vs_dda.py
   :language: python
   :linenos:

Step-by-Step Explanation
------------------------

1. **Create a spherical particle**

2. **Run Mie solver** (exact solution for spheres)

3. **Run DDA solver** (approximate, grid-dependent)

4. **Compare results** and check convergence with finer grids
