Particle Geometry
=================

Voxelization
------------

Aerosol3D represents 3D particle geometry using a voxel grid. Each voxel is a
small cubic cell that is either filled (part of the particle) or empty. The
voxel size controls the spatial resolution and directly affects the number of
dipoles used in DDA computation.

Internal vs External Mixing
---------------------------

- **Internal mixing**: Different materials occupy the same spatial region
  (homogeneous mixture at the voxel scale).

- **External mixing**: Different materials occupy distinct spatial regions
  (e.g., a coated core-shell particle).

Use ``MixingState.INTERNAL`` or ``MixingState.EXTERNAL`` when creating an
``AerosolParticle``.

Equivalent Diameter
-------------------

The volume-equivalent diameter :math:`d_{ve}` is computed from the total
occupied voxel volume::

    d_{ve} = \left(\frac{6V}{\pi}\right)^{1/3}

This is useful for comparing DDA results with Mie theory, which assumes a
spherical particle of equivalent volume.
