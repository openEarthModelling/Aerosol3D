Bulk Cross-Validation
=====================

This tutorial demonstrates how to cross-validate the bulk asymmetry
parameter :math:`g` using two independent computational paths.

Prerequisites
-------------

- Completion of :doc:`bulk-optics-workflow` (bulk optics data ready)
- Single-particle optical data with phase functions (e.g. from Mie)

Overview
--------

The asymmetry parameter :math:`g` can be computed in two ways:

**Path A — Phase-function quadrature:**
For each radius, compute :math:`g(r)` from :math:`P_{11}(\theta)` via
angular quadrature, then energy-weighted integration over the size
distribution.

**Path B — Legendre moment:**
Extract :math:`\beta_1` from bulk data (computed by
``BulkOpticsBuilder``) and use :math:`g = \beta_1 / 3`.

If both paths agree within tolerance, the bulk computation is
internally consistent.


Step 1: Prepare Input Data
--------------------------

You need two types of NetCDF files:

1. **Bulk aerosol optics** — output from ``BulkOpticsBuilder``
   (e.g. ``bulk_aerosol.nc``).
2. **Single-size optics** — one file per radius from Mie or DDA
   (e.g. ``mie_r050.nc``, ``mie_r100.nc``, ...).

Generate single-size data with :func:`solve_optics`:

.. code-block:: python

   from Aerosol3D import solve_optics
   from Aerosol3D.optics.optics_export import AerosolOpticsData

   radii_nm = [50, 100, 200, 400, 800]
   for r in radii_nm:
       particle = ...  # define spherical particle of radius r
       result = solve_optics(particle, config, solver="MIE")
       data = AerosolOpticsData.from_optical_results([result], n_legendre=32)
       data.to_netcdf(f"mie_r{r:03d}.nc")


Step 2: Run the Cross-Validation Script
---------------------------------------

The example script ``examples/bulk_cross_validation.py`` performs the
dual-path check:

.. code-block:: bash

   python examples/bulk_cross_validation.py \
       --bulk-input output/bulk_aerosol.nc \
       --singles-input "output/mie_r*.nc" \
       --tol 1e-5

Expected output:

.. code-block:: text

   ========================================================================
       Wavelength (nm)     g_A (quad)   g_B (beta_1/3)         |diff|   Status
   ------------------------------------------------------------------------
              400.00     0.72345678       0.72345612     6.60e-07     PASS
              550.00     0.68123456       0.68123401     5.50e-07     PASS
              700.00     0.64567890       0.64567834     5.60e-07     PASS
   ========================================================================

   All wavelengths PASS (tol = 1e-5).


Step 3: Understand the Two Paths
--------------------------------

Path A: Phase-function quadrature
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each radius, :math:`g(r)` is computed from the azimuthally-averaged
phase function:

.. math::

   g(r) = \frac{\int_0^{\pi} P_{11}(\theta) \cos\theta \sin\theta \, d\theta}
                {\int_0^{\pi} P_{11}(\theta) \sin\theta \, d\theta}

Then energy-weighted integration over the size distribution:

.. math::

   g_A = \frac{\int g(r) \, C_{\text{sca}}(r) \, n(r) \, dr}
                {\int C_{\text{sca}}(r) \, n(r) \, dr}

Path B: Legendre moment
~~~~~~~~~~~~~~~~~~~~~~~

From the bulk ``beta`` array (vSmartMOM convention):

.. math::

   g_B = \frac{\beta_1}{3}

Because :math:`\beta_1 = 3g` in the vSmartMOM convention, this is a
direct extraction.


Troubleshooting
---------------

If :math:`g_A / g_B \approx 3.0`, check the Legendre coefficient
normalization. ``compute_legendre_moments()`` returns
:math:`k_l = (2l+1) \times \text{integral}`, but the bulk integration
expects :math:`\beta_l = k_l / (2l+1)`. Divide by :math:`(2l+1)` before
passing to ``BulkOpticsBuilder``.

If the difference is large at a specific wavelength, increase the
number of quadrature points (``n_quad``) or sample more radii.
