Optical Computation
===================

Discrete Dipole Approximation (DDA)
-----------------------------------

DDA approximates a continuous particle as an array of interacting point
dipoles. The electromagnetic scattering problem is solved by finding the
induced dipole moments that satisfy the self-consistent field equations.

Convergence Criterion
---------------------

DDA accuracy depends on the product :math:`|m|kd`, where:

- :math:`|m|` is the magnitude of the refractive index
- :math:`k = 2\pi/\lambda` is the wavenumber
- :math:`d` is the dipole spacing

For accurate results, :math:`|m|kd \ll 1`. A common rule of thumb is
:math:`|m|kd < 0.5` for high accuracy.

Mie Theory
----------

For spherical particles, Aerosol3D can use Mie theory (via PyMieScatt) as an
alternative to DDA. Mie theory is exact for spheres and much faster, making
it ideal for validation and spherical particle studies.

Orientational Averaging
-----------------------

Non-spherical particles have orientation-dependent optical properties.
Aerosol3D can average over multiple incident directions using Fibonacci
sphere sampling to obtain orientationally averaged cross-sections.

Phase Function
--------------

The phase function :math:`P_{11}(\theta, \phi)` describes the angular
distribution of scattered light. It is normalized such that::

    \int P_{11}(\theta, \phi) \, d\Omega = 1
