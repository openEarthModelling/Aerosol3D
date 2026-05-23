Effective Medium Approximation and Core-Shell Mie
==================================================

When a particle contains multiple materials (e.g., a black carbon core
with sulfate coating), its optical properties depend on how the
refractive indices are combined. Aerosol3D offers two approaches:

- **Effective Medium Approximation (EMA)** — compute a single effective
  refractive index and solve with homogeneous-sphere Mie theory
- **Core-shell Mie** — solve the exact layered-sphere Mie problem using
  the core and shell refractive indices separately

Effective Medium Approximation
------------------------------

EMA replaces a multi-component particle with an equivalent homogeneous
sphere whose effective refractive index :math:`m_{\text{eff}}` represents
the mixture. The ``core/ema.py`` module provides three methods:

Volume-Weighted
^^^^^^^^^^^^^^^

.. math::

   m_{\text{eff}} = \sum_i f_i \, m_i

where :math:`f_i = V_i / V_{\text{total}}` is the volume fraction.

Simplest method; suitable when components have similar refractive indices
or as a quick estimate.

.. code-block:: python

    from Aerosol3D.core.ema import volume_weighted

    m_eff = volume_weighted([300, 700], [1.95 + 0.66j, 1.53 + 0.0j])

Maxwell-Garnett
^^^^^^^^^^^^^^^

.. math::

   \frac{\varepsilon_{\text{eff}} - \varepsilon_h}{\varepsilon_{\text{eff}} + 2\varepsilon_h}
   = \sum_i f_i \frac{\varepsilon_i - \varepsilon_h}{\varepsilon_i + 2\varepsilon_h}

The largest-volume component is treated as the host (:math:`\varepsilon_h`);
all others are inclusions. Best for inclusion-in-host geometries such as
soot inclusions in a sulfate matrix.

.. code-block:: python

    from Aerosol3D.core.ema import maxwell_garnett

    m_eff = maxwell_garnett([300, 700], [1.95 + 0.66j, 1.53 + 0.0j])

Bruggeman
^^^^^^^^^

.. math::

   \sum_i f_i \frac{\varepsilon_i - \varepsilon_{\text{eff}}}{\varepsilon_i + 2\varepsilon_{\text{eff}}} = 0

A symmetric mixing rule — no component is privileged as the host. Solved
via Newton iteration. Best for randomly intermixed compositions.

.. code-block:: python

    from Aerosol3D.core.ema import bruggeman

    m_eff = bruggeman([300, 700], [1.95 + 0.66j, 1.53 + 0.0j])

Choosing an EMA Method
^^^^^^^^^^^^^^^^^^^^^^

+------------------+--------------------------------------------+
| Method           | When to use                                |
+==================+============================================+
| volume_weighted  | Quick estimate, similar refractive indices |
+------------------+--------------------------------------------+
| maxwell_garnett  | Inclusion-in-host geometry (e.g., soot     |
|                  | inclusions in sulfate host)                |
+------------------+--------------------------------------------+
| bruggeman        | Randomly intermixed materials, no clear    |
|                  | host-inclusion structure                   |
+------------------+--------------------------------------------+

Using EMA with the Mie Solver
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``Particle.effective_refractive_index`` property accepts an
``ema_method`` parameter:

.. code-block:: python

    from Aerosol3D import AerosolParticle, MixingState, solve_optics, SimulationConfig

    particle = AerosolParticle(name="coated", mixing_state=MixingState.INTERNAL)
    # ... add meshes ...

    # Use Maxwell-Garnett EMA with the Mie solver
    config = SimulationConfig(wavelength=550.0)
    result = solve_optics(particle, config, solver="MIE", ema_method="maxwell_garnett")

Core-Shell Mie Solver
---------------------

For coated spheres with a distinct core and shell, the core-shell Mie
solver solves the exact layered-sphere problem using PyMieScatt's
``MieQCoreShell`` function.

The ``Particle.coreshell_geometry`` property automatically determines
the core-shell structure from the particle's volume composition:

.. code-block:: python

    from Aerosol3D import AerosolParticle, MixingState, solve_optics, SimulationConfig

    particle = AerosolParticle(name="coated", mixing_state=MixingState.INTERNAL)
    # ... add core and shell meshes ...

    # Get core-shell geometry
    d_core, d_outer, m_core, m_shell = particle.coreshell_geometry

    # Solve with core-shell Mie
    config = SimulationConfig(wavelength=550.0)
    result = solve_optics(particle, config, solver="MIE_CORESHELL")

EMA vs Core-Shell: Which to Use?
---------------------------------

+-----------------+----------------------------+----------------------------+
|                 | EMA + Homogeneous Mie      | Core-Shell Mie             |
+=================+============================+============================+
| Geometry        | Equivalent homogeneous     | Exact layered sphere       |
|                 | sphere                     |                            |
+-----------------+----------------------------+----------------------------+
| Best for        | Irregular internal mixing  | Concentric core-shell      |
|                 | patterns                   | structure                  |
+-----------------+----------------------------+----------------------------+
| Accuracy        | Approximate (depends on    | Exact for spheres          |
|                 | EMA method)                |                            |
+-----------------+----------------------------+----------------------------+
| Speed           | Fast (Mie solver)          | Fast (Mie solver)          |
+-----------------+----------------------------+----------------------------+

For irregularly shaped particles (e.g., fractal aggregates), both
approaches are approximations. Comparing multiple methods is recommended —
see the ``coated_fractal_aggregate`` example.
