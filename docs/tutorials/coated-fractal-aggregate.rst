Coated Fractal Aggregate
=========================

This tutorial generates a black carbon fractal aggregate and applies two
coating methods: potential void-filling and potential edge-growing.

Full Script
-----------

.. literalinclude:: ../../examples/coated_fractal_aggregate.py
   :language: python
   :linenos:

Step-by-Step Explanation
-------------------------

1. **Generate fractal aggregate**

   ``pyFracAggregate`` generates a 30-monomer fractal aggregate with fractal
   dimension Df=1.8. ``from_fractal()`` converts it to an ``AerosolParticle``.

2. **Visualize bare aggregate**

   ``save_screenshot()`` renders the mesh, and
   ``save_particle_voxel_screenshot()`` shows the voxelized representation.

3. **Apply void-filling coating**

   ``apply_potential_void_coating()`` fills internal voids between monomers
   with coating material (sulfate). The ``coated_area_fraction`` and
   ``dp_dc_ratio`` parameters control coating thickness and coverage.

4. **Apply edge-growing coating**

   Starting from a fresh copy of the bare aggregate,
   ``apply_potential_edge_coating()`` grows coating outward from the particle
   surface. This produces a different morphology than void-filling.

5. **Rotation videos** (optional)

   ``save_rotation_video()`` and ``save_particle_voxel_video()`` generate MP4
   animations. Use ``--no-video`` to skip this step.

Coating Parameters
------------------

Both coating methods share these parameters:

- ``coated_area_fraction`` — Fraction of the particle surface to coat (0–1)
- ``dp_dc_ratio`` — Ratio of particle diameter to coating thickness
- ``material`` — Coating material (e.g., sulfate)
- ``resolution`` — Voxelization resolution for coating computation

Running
-------

::

    python coated_fractal_aggregate.py
    python coated_fractal_aggregate.py --no-video
