Bulk Optics Workflow
====================

This tutorial walks through the complete workflow for computing bulk
aerosol optical properties from single-particle Mie calculations.

Prerequisites
-------------

- Completion of :doc:`mie-vs-dda-validation` (understanding single-particle optics)
- ``Aerosol3D`` installed with bulk module dependencies

Overview
--------

We will:

1. Define a lognormal size distribution.
2. Generate synthetic single-particle optical data at several radii.
3. Use ``BulkOpticsBuilder`` to aggregate into bulk properties.
4. Compare Method 1 (bin weights) and Method 2 (continuous integration).
5. Export results to NetCDF.

For the mathematical background, see :doc:`../user-guide/bulk-aerosol-optics`.


Step 1: Define the Size Distribution
------------------------------------

.. code-block:: python

   import numpy as np
   from Aerosol3D import SizeDistribution

   # Lognormal distribution: rg = 200 nm, sigma_ln = 0.5
   dist = SizeDistribution.lognormal(rg_nm=200.0, sigma_ln=0.5)

   # Check effective radius
   print(f"Effective radius: {dist.effective_radius():.1f} nm")

The effective radius :math:`r_{\text{eff}} = r_g \exp(5\sigma_{\ln}^2/2)`
should be approximately 374 nm for these parameters.


Step 2: Generate Single-Particle Optics
---------------------------------------

For this tutorial we use synthetic data that mimics Mie scattering
behavior. In a real workflow, you would use :func:`solve_optics` with
``solver="MIE"`` for each radius.

.. code-block:: python

   from Aerosol3D.optics.optics_export import AerosolOpticsData

   WAVELENGTHS_NM = np.array([400.0, 550.0, 700.0])
   N_LEGENDRE = 8
   RADII_NM = np.array([50.0, 100.0, 200.0, 400.0, 800.0])

   def make_synthetic_optics(radius_nm: float) -> AerosolOpticsData:
       """Create synthetic optical data for a given radius."""
       n_wl = len(WAVELENGTHS_NM)

       # Geometric cross-section scaling
       C_ext = np.pi * radius_nm**2 * (
           1.0 + 0.3 * (400.0 / WAVELENGTHS_NM)
       )

       # Size parameter x = 2*pi*r/lambda
       x = 2 * np.pi * radius_nm / WAVELENGTHS_NM
       SSA = np.clip(0.4 + 0.5 / (1.0 + np.exp(-(x - 3.0))), 0.3, 0.95)

       C_sca = C_ext * SSA
       g = np.clip(0.1 + 0.7 * (x / (x + 2.0)), 0.0, 0.85)

       # Legendre coefficients: Henyey-Greenstein approximation
       beta = np.zeros((n_wl, N_LEGENDRE))
       for l in range(N_LEGENDRE):
           beta[:, l] = (2 * l + 1) * g**l

       return AerosolOpticsData(
           wavelength_nm=WAVELENGTHS_NM.copy(),
           C_ext=C_ext,
           C_sca=C_sca,
           C_abs=C_ext - C_sca,
           SSA=SSA,
           g=g,
           r_eff_nm=radius_nm,
           legendre_moments_beta=beta / np.arange(1, 2 * N_LEGENDRE + 1, 2),
           n_legendre=N_LEGENDRE,
           solver="SYNTHETIC",
       )

   # Generate optics for all radii
   optics_list = [make_synthetic_optics(r) for r in RADII_NM]


Step 3: Build Bulk Properties with BulkOpticsBuilder
----------------------------------------------------

.. code-block:: python

   from Aerosol3D import BulkOpticsBuilder

   builder = BulkOpticsBuilder(
       size_distribution=dist,
       radii_nm=RADII_NM,
       n_legendre=N_LEGENDRE,
   )

   for r, opt in zip(RADII_NM, optics_list):
       builder.add(radius=float(r), optics=opt)

   # Compute with Method 2 (continuous integration, default)
   bulk = builder.compute(n_quad=512)

   print(f"Bulk C_ext: {bulk.C_ext}")
   print(f"Bulk SSA:   {bulk.SSA}")
   print(f"Bulk g:     {bulk.g}")

The ``bulk`` object is a ``BulkAerosolOpticsData`` dataclass containing:

- ``wavelength_nm`` — wavelengths
- ``C_ext``, ``C_sca``, ``C_abs`` — bulk cross-sections
- ``SSA`` — single-scattering albedo
- ``g`` — asymmetry parameter
- ``beta`` — Legendre coefficients (vSmartMOM convention)
- ``fallback_used`` — whether Method 1 fallback was triggered


Step 4: Compare Method 1 and Method 2
-------------------------------------

For sparse sampling, Method 1 and Method 2 may differ. You can compute
both directly for comparison:

.. code-block:: python

   from Aerosol3D.bulk.merge import compute_bin_weights, merge_method1, merge_method2

   # Prepare arrays: (n_radius, n_wavelength)
   C_ext = np.array([opt.C_ext for opt in optics_list])
   C_sca = np.array([opt.C_sca for opt in optics_list])

   # Prepare beta: convert from g_l to (2l+1)*g_l
   beta = np.zeros((len(RADII_NM), len(WAVELENGTHS_NM), N_LEGENDRE))
   for i, opt in enumerate(optics_list):
       l_vals = np.arange(N_LEGENDRE)
       beta[i, :, :] = opt.legendre_moments_beta * (2 * l_vals + 1)

   # Method 1
   weights = compute_bin_weights(RADII_NM, dist)
   m1_C_ext, m1_C_sca, m1_beta = merge_method1(C_ext, C_sca, beta, weights)

   # Method 2
   m2_C_ext, m2_C_sca, m2_beta = merge_method2(
       RADII_NM, C_ext, C_sca, beta, dist, n_quad=512
   )

   print(f"Method 1 C_ext: {m1_C_ext}")
   print(f"Method 2 C_ext: {m2_C_ext}")

With only 5 sample radii, you may see differences of a few percent.
With 20+ radii, the methods converge.


Step 5: Export to NetCDF
------------------------

.. code-block:: python

   from Aerosol3D.bulk.io import bulk_to_netcdf

   bulk_to_netcdf(bulk, "bulk_aerosol.nc")
   print("Saved to bulk_aerosol.nc")

For vSmartMOM compatibility:

.. code-block:: python

   from Aerosol3D.bulk.io import bulk_to_vsmartmom_netcdf

   bulk_to_vsmartmom_netcdf(bulk, "vsmartmom_input.nc", tau_ref=0.3)


Complete Script
---------------

The full working example is available at
``examples/bulk_method_comparison.py``. Run it with:

.. code-block:: bash

   python examples/bulk_method_comparison.py
