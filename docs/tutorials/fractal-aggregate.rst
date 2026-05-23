Fractal Aggregate
=================

This tutorial generates a coated black carbon fractal aggregate and
compares four Mie approximation methods: volume-weighted EMA,
Maxwell-Garnett EMA, Bruggeman EMA, and exact core-shell Mie.

Full Script
-----------

.. literalinclude:: ../../examples/coated_fractal_aggregate.py
   :language: python
   :linenos:

Step-by-Step Explanation
------------------------

1. **Generate fractal aggregate**

   ``pyFracAggregate`` creates a 30-monomer fractal aggregate with
   fractal dimension Df=1.8. ``from_fractal()`` converts it to an
   ``AerosolParticle`` with black carbon material.

2. **Apply coating**

   ``apply_potential_void_coating()`` fills internal voids with sulfate
   coating material, creating a coated fractal aggregate.

3. **Compare Mie approximations**

   The example computes optical properties using four approaches:

   - ``volume_weighted`` — linear average of refractive indices
   - ``maxwell_garnett`` — inclusion-in-host mixing rule (soot in sulfate)
   - ``bruggeman`` — symmetric mixing rule
   - ``MIE_CORESHELL`` — exact layered-sphere solution

   See :doc:`/user-guide/ema-and-coreshell` for the theory behind each method.

4. **Spectral comparison plots**

   Extinction, scattering, and absorption cross-sections are plotted
   across the visible spectrum for all four methods, allowing visual
   comparison of the approximation quality.

Running
-------

::

    python coated_fractal_aggregate.py
