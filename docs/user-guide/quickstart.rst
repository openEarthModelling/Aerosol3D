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
- ``orientational_average``: Average over ``n_dirs`` random orientations (DDA only)
- ``n_dirs``: Number of orientations for averaging (default 50)
- ``n_jobs``: Number of parallel workers for orientation averaging (default 32)
- ``show_progress``: Display tqdm progress bars during averaging (default True)
- ``precision``: DDA dipole spacing precision — ``"low"``, ``"medium"`` (default), ``"high"``, or ``"ultra"`` — controls the :math:`|m|kd` convergence criterion

4. Export and Visualize Results
-------------------------------

.. code-block:: python

    from Aerosol3D.optics import from_optical_results
    from Aerosol3D.optics.visualization import plot_spectral_properties

    # Collect results at multiple wavelengths
    wavelengths = [450.0, 550.0, 650.0]
    results = [
        solve_optics(particle, SimulationConfig(wavelength=w), solver="MIE")
        for w in wavelengths
    ]

    # Build export container with auto-computed Legendre moments
    data = from_optical_results(results, n_legendre=32)
    data.to_netcdf("optics.nc")

    # Plot spectral properties
    plot_spectral_properties(data)
