Black Carbon Sphere
===================

This tutorial creates a bare black carbon sphere and computes its optical
properties using DDA.

Full Script
-----------

.. literalinclude:: ../../examples/black_carbon_sphere.py
   :language: python
   :linenos:

Step-by-Step Explanation
------------------------

1. **Import modules**

   We import the core classes and functions needed for geometry creation,
   material definition, and optical computation.

2. **Define material**

   Black carbon has a high refractive index with significant absorption
   (large imaginary part).

3. **Create sphere**

   ``create_sphere(center, radius)`` returns a PyVista mesh representing
   the spherical geometry.

4. **Run DDA**

   ``solve_optics`` voxelizes the mesh, computes polarizabilities, and
   solves the DDA linear system via the Julia backend.
