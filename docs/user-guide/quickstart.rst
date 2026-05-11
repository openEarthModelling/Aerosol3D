Quick Start
===========

This guide walks through creating a simple spherical particle and computing
its optical properties.

1. Create a Particle
--------------------

.. code-block:: python

    from Aerosol3D import AerosolParticle, Material, create_sphere

    soot = Material(name="soot", refractive_index=complex(1.8, 0.7), density=1.8)
    particle = AerosolParticle(name="bc_sphere", unit="nm")
    particle.add_mesh("core", create_sphere((0, 0, 0), 50.0), soot)

2. Visualize
------------

.. code-block:: python

    from Aerosol3D import save_screenshot
    save_screenshot(particle, "sphere.png", colors={"core": "black"})

3. Compute Optical Properties
-----------------------------

.. code-block:: python

    from Aerosol3D import solve_optics, SimulationConfig

    config = SimulationConfig(wavelength=550.0, dipole_spacing=10.0)
    result = solve_optics(particle, config, compute_phase_func=True)

    print(f"Q_ext = {result.cross_sections.Q_ext:.4f}")
    print(f"Q_sca = {result.cross_sections.Q_sca:.4f}")
    print(f"g     = {result.cross_sections.g:.4f}")

Key Parameters
--------------

- ``wavelength``: Incident light wavelength in nm
- ``dipole_spacing``: DDA dipole spacing in nm (smaller = more accurate but slower)
- ``compute_phase_func``: Whether to compute the phase function P11
