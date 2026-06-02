Bulk Cross-Validation
=====================

This tutorial demonstrates how to validate the internal consistency of
bulk optical property computations by computing the asymmetry parameter
:math:`g` via two independent paths and comparing the results.

Background
----------

The asymmetry parameter :math:`g = \langle\cos\theta\rangle` describes
the average direction of scattered radiation. It can be computed in two
independent ways from the same underlying data:

**Path A — Phase Function Quadrature:**
Integrate the phase function :math:`P_{11}(\theta)` directly:

.. math::

   g(r) = \frac{\int_0^{\pi} P_{11}(\theta) \cos\theta \sin\theta \, d\theta}
                {\int_0^{\pi} P_{11}(\theta) \sin\theta \, d\theta}

Then weight by the size distribution and scattering cross-section:

.. math::

   g_{\text{bulk}}^{(A)} = \frac{\int g(r) \, C_{\text{sca}}(r) \, n(r) \, dr}
                                  {\int C_{\text{sca}}(r) \, n(r) \, dr}

**Path B — Legendre Moment:**
Extract :math:`\beta_1` from the bulk Legendre coefficients (already
computed by ``BulkOpticsBuilder``):

.. math::

   g_{\text{bulk}}^{(B)} = \frac{\beta_1}{3}

(In the vSmartMOM convention, :math:`\beta_l` includes the :math:`(2l+1)`
factor, so :math:`\beta_1 = 3g`.)

If both computations are correct, :math:`|g^{(A)} - g^{(B)}|` should be
very small (typically :math:`< 10^{-5}`).

Prerequisites
-------------

- Bulk aerosol NetCDF file (from ``bulk_to_netcdf``)
- Single-particle NetCDF files (one per radius, from ``AerosolOpticsData.to_netcdf``)

For the workflow to produce these files, see :doc:`bulk-optics-workflow`.


Step 1: Load the Data
---------------------

.. code-block:: python

   from Aerosol3D.bulk.io import bulk_from_netcdf
   from Aerosol3D.optics.optics_export import AerosolOpticsData

   # Load bulk result
   bulk = bulk_from_netcdf("output/bulk_aerosol.nc")

   # Load single-size results
   single_files = ["output/mie_r050.nc", "output/mie_r100.nc", "output/mie_r200.nc"]
   singles = [AerosolOpticsData.from_netcdf(p) for p in single_files]

   # Sort by radius
   singles.sort(key=lambda d: d.r_eff_nm)


Step 2: Compute g via Path A
----------------------------

.. code-block:: python

   import numpy as np
   from Aerosol3D.bulk.interpolation import LinearPCHIPInterpolator
   from Aerosol3D.bulk.integration import integrate_distribution_vectorized

   def g_from_phase_function(theta_rad, P11):
       """Compute g from azimuthally-averaged P11."""
       sin_t = np.sin(theta_rad)
       cos_t = np.cos(theta_rad)
       num = np.trapz(P11 * cos_t * sin_t, theta_rad)
       den = np.trapz(P11 * sin_t, theta_rad)
       return num / den if abs(den) > 1e-30 else float("nan")

   # Extract per-radius data
   radii = np.array([s.r_eff_nm for s in singles])
   theta_rad = singles[0].theta_rad  # same angles for all
   C_sca_per_r = np.array([s.C_sca[0] for s in singles])  # first wavelength

   # Compute g(r) for each radius
   g_per_r = []
   for s in singles:
       P11_theta = np.mean(s.P11[0, :, :], axis=1)  # azimuthal average
       g_per_r.append(g_from_phase_function(theta_rad, P11_theta))
   g_per_r = np.array(g_per_r)

   # Interpolate and integrate
   g_interp = LinearPCHIPInterpolator(radii, g_per_r)
   c_sca_interp = LinearPCHIPInterpolator(radii, C_sca_per_r)

   def integrand_num(r):
       return g_interp(r) * c_sca_interp(r)

   def integrand_den(r):
       return c_sca_interp(r)

   num = integrate_distribution_vectorized(
       integrand_num, bulk.size_distribution, n_quad=512
   )
   den = integrate_distribution_vectorized(
       integrand_den, bulk.size_distribution, n_quad=512
   )

   g_path_A = num / den if den > 0 else 0.0


Step 3: Compute g via Path B
----------------------------

.. code-block:: python

   # beta_1 is at index 1; divide by 3 for vSmartMOM convention
   g_path_B = bulk.beta[0, 1] / 3.0


Step 4: Compare and Validate
----------------------------

.. code-block:: python

   delta_g = abs(g_path_A - g_path_B)
   tol = 1e-5

   print(f"Path A (quadrature): g = {g_path_A:.8f}")
   print(f"Path B (Legendre):   g = {g_path_B:.8f}")
   print(f"|Δg| = {delta_g:.2e}")
   print("PASS" if delta_g < tol else "FAIL")

If :math:`|g_A / g_B| \approx 3` or :math:`\approx 1/3`, check whether
the :math:`\beta_1` convention is correct (with or without the
:math:`(2l+1)` factor).

Complete Script
---------------

The full working example with command-line interface is available at
``examples/bulk_cross_validation.py``. Run it with:

.. code-block:: bash

   python examples/bulk_cross_validation.py \
       --bulk-input output/bulk_aerosol.nc \
       --singles-input output/mie_r*.nc \
       --tol 1e-5
