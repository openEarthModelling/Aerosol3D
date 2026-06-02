Bulk Aerosol Optics
===================

Overview
--------

Atmospheric aerosols are polydisperse — they consist of particles with a
continuous distribution of sizes. The ``Aerosol3D.bulk`` module aggregates
single-particle optical properties (computed via Mie theory or DDA) into
size-distribution-weighted **bulk** optical properties suitable for radiative
transfer models such as DISORT, libRadtran, and vSmartMOM.

The workflow is:

1. Define a particle size distribution (lognormal or gamma).
2. Compute optical properties for discrete radii (single-particle Mie or DDA).
3. Aggregate via ``BulkOpticsBuilder`` to obtain bulk properties.
4. Export to NetCDF (standard or vSmartMOM-compatible format).

For single-particle optical computation, see :doc:`optical-computation`.
For a hands-on tutorial, see :doc:`../tutorials/bulk-optics-workflow`.


Size Distributions
------------------

The module supports two analytic size distributions, both parameterized
by quantities with direct physical meaning.

Lognormal Distribution
^^^^^^^^^^^^^^^^^^^^^^

The lognormal probability density function is:

.. math::

   n(r) = \frac{1}{r \, \sigma_{\ln} \sqrt{2\pi}}
          \exp\!\left(-\frac{(\ln r - \ln r_g)^2}{2\sigma_{\ln}^2}\right)

where:

- :math:`r_g` — geometric mean radius (nm), equal to the median.
- :math:`\sigma_{\ln}` — log-space standard deviation (dimensionless).

The :math:`k`-th moment has a closed-form expression:

.. math::

   \langle r^k \rangle = r_g^{\,k} \exp\!\left(\frac{k^2 \sigma_{\ln}^2}{2}\right)

The bulk effective radius is:

.. math::

   r_{\text{eff}} = \frac{\langle r^3 \rangle}{\langle r^2 \rangle}
                  = r_g \exp\!\left(\frac{5}{2}\sigma_{\ln}^2\right)

Create with:

.. code-block:: python

   from Aerosol3D import SizeDistribution
   dist = SizeDistribution.lognormal(rg_nm=200.0, sigma_ln=0.5)


Gamma Distribution (Hansen \& Travis)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Hansen \& Travis (1974) gamma distribution uses the effective radius
:math:`r_{\text{eff}}` and effective variance :math:`v_{\text{eff}}` as
parameters. The PDF is proportional to:

.. math::

   n(r) \propto r^{\alpha} \exp(-r/\beta)

with :math:`\alpha = (1 - 3v_{\text{eff}})/v_{\text{eff}}` and
:math:`\beta = r_{\text{eff}} v_{\text{eff}}`. By construction, the
:math:`r_{\text{eff}}` parameter equals the bulk effective radius.

Create with:

.. code-block:: python

   dist = SizeDistribution.gamma(reff_nm=250.0, veff=0.1)


Method 1: Bin Weights
---------------------

Method 1 is a **local bin integration** approach. The discrete radii are
treated as representative points of bins, and the size distribution is
integrated within each bin to obtain fractional number counts.

Bin boundaries
^^^^^^^^^^^^^^

For sorted radii :math:`r_1 < r_2 < \dots < r_N`, the bin edges are:

.. math::

   e_0 &= 0 \\
   e_i &= \sqrt{r_{i-1} \, r_i} \quad (i = 1, \dots, N-1) \\
   e_N &= \infty

The geometric mean ensures scale-invariance on the log-radius axis.

Weight computation
^^^^^^^^^^^^^^^^^^

The fractional number count in bin :math:`i` is:

.. math::

   w_i = \int_{e_{i-1}}^{e_i} n(r) \, dr

with the normalization :math:`\sum_i w_i = 1` enforced explicitly.

Weighted averages
^^^^^^^^^^^^^^^^^

The bulk cross-sections are arithmetic means weighted by :math:`w_i`:

.. math::

   \bar{C}_{\text{ext}} &= \sum_{i=1}^{N} w_i \, C_{\text{ext}}(r_i) \\
   \bar{C}_{\text{sca}} &= \sum_{i=1}^{N} w_i \, C_{\text{sca}}(r_i)

For the Legendre coefficients, the **scattering-energy-weighted**
average is used to preserve the phase-function normalization:

.. math::

   \bar{M}_l = \sum_{i=1}^{N} w_i \, C_{\text{sca}}(r_i) \, \beta_l(r_i)

   \bar{\beta}_l = \frac{\bar{M}_l}{\bar{C}_{\text{sca}}}

The :math:`\beta_0 = 1` constraint is enforced exactly after computation.

When to use Method 1
^^^^^^^^^^^^^^^^^^^^

- Sparse sampling (few radii), where interpolation would be unreliable.
- Fallback mode when Mie ripple oscillations are undersampled.
- Validation baseline for Method 2.


Method 2: Continuous Integration
---------------------------------

Method 2 is the **primary** approach. It interpolates optical properties
as continuous functions of radius and integrates over the full size
distribution.

Interpolation strategy
^^^^^^^^^^^^^^^^^^^^^^

Different physical quantities require different interpolation spaces to
preserve their constraints:

- **Cross-sections** (:math:`C_{\text{ext}}`, :math:`C_{\text{sca}}`):
  Interpolated in **log-log space** using PCHIP. This preserves
  positivity and handles many orders of magnitude (e.g. :math:`C \sim r^2`).

- **Legendre coefficients** (:math:`\beta_l`):
  Interpolated in **linear space** using PCHIP. The values are bounded
  (:math:`|\beta_l| \leq 2l+1`) and linear interpolation avoids
  distorting the phase function shape.

**Important:** SSA and :math:`g` are **never** interpolated directly.
They are always derived from the interpolated :math:`C_{\text{ext}}`,
:math:`C_{\text{sca}}`, and :math:`\beta_l`.

Integration formulas
^^^^^^^^^^^^^^^^^^^^

For each wavelength :math:`\lambda`:

.. math::

   \bar{C}_{\text{ext}}(\lambda)
   &= \int_{r_{\min}}^{r_{\max}} C_{\text{ext}}(r,\lambda) \, n(r) \, dr

   \bar{C}_{\text{sca}}(\lambda)
   &= \int_{r_{\min}}^{r_{\max}} C_{\text{sca}}(r,\lambda) \, n(r) \, dr

   \bar{M}_l(\lambda)
   &= \int_{r_{\min}}^{r_{\max}}
      \beta_l(r,\lambda) \, C_{\text{sca}}(r,\lambda) \, n(r) \, dr

   \bar{\beta}_l(\lambda) &= \frac{\bar{M}_l(\lambda)}{\bar{C}_{\text{sca}}(\lambda)}

Numerical integration
^^^^^^^^^^^^^^^^^^^^^

Two quadrature methods are supported:

- **Adaptive quadrature** (``method="quad"``, default): SciPy's
  ``integrate.quad`` with Gauss-Kronrod rules. Best for smooth integrands.

- **Fixed Gauss-Legendre** (``method="fixed_quad"``): Pre-computed nodes
  and weights on a log-transformed interval. Faster for batch evaluation.

Why Method 2 is more accurate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Method 2 captures the **continuous variation** of optical properties
between sample points. For example, if :math:`C_{\text{ext}}(r)` has a
local maximum between two sample radii, Method 1 misses it entirely,
while Method 2 interpolates and integrates over the full structure.

In the limit of dense sampling, Method 1 converges to Method 2.


Derived Quantities
------------------

After computing :math:`\bar{C}_{\text{ext}}`, :math:`\bar{C}_{\text{sca}}`,
and :math:`\bar{\beta}_l`, the following derived quantities are computed:

Single-scattering albedo
^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

   \text{SSA} = \frac{\bar{C}_{\text{sca}}}{\bar{C}_{\text{ext}}}

Always in :math:`[0, 1]` by construction.

Asymmetry parameter
^^^^^^^^^^^^^^^^^^^

Using the vSmartMOM convention where :math:`\beta_l` includes the
:math:`(2l+1)` factor:

.. math::

   g = \frac{\beta_1}{3}

Normalization constraint
^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

   \beta_0 = 1

This is enforced exactly after both Method 1 and Method 2.


Physical Constraints and Validation
-----------------------------------

The module enforces and verifies the following physical constraints:

- **SSA bounds**: :math:`0 \leq \text{SSA} \leq 1` for all wavelengths.
- **Energy conservation**: :math:`\bar{C}_{\text{ext}} = \bar{C}_{\text{sca}} + \bar{C}_{\text{abs}}`
  (enforced at floating-point precision).
- **Beta normalization**: :math:`\beta_0 = 1` exactly.
- **Monodisperse limit**: With a single radius and narrow distribution,
  bulk properties converge to the single-particle result.

These are validated by the test suite in ``tests/test_bulk_physics.py``.


Mie Ripple Detection and Fallback
---------------------------------

Mie scattering for spherical particles produces rapid oscillations
(ripples) in the cross-sections as a function of radius. The oscillation
period is approximately:

.. math::

   \Delta r \approx \frac{\lambda}{2 \, |m - 1|}

where :math:`m` is the (real part of the) refractive index.

If the radius sampling is too sparse to resolve these ripples, Method 2
interpolation may introduce artifacts. The builder detects this condition
and automatically falls back to Method 1 for affected wavelengths.

Detection logic
^^^^^^^^^^^^^^^

For each wavelength, the builder checks:

.. math::

   \Delta r_i \leq \frac{\Delta r}{p_{\min}}

where :math:`\Delta r_i` are the spacing between adjacent sample radii
and :math:`p_{\min}` is the minimum required points per oscillation period
(default: 3).

User control
^^^^^^^^^^^^

Enable detection and provide the refractive index:

.. code-block:: python

   bulk = builder.compute(
       check_mie_ripples=True,
       refractive_index=1.55 + 0.02j,
       mie_ripple_min_points=3,
   )

If ``refractive_index`` is not provided, the builder attempts to
auto-extract it from the first added optics entry's
``refractive_index_real`` and ``refractive_index_imag`` fields. Only
the first wavelength's refractive index is used for the ripple period
estimate, so dispersion (wavelength-dependent refractive index) is not
accounted for. If these fields are not available, a ``ValueError`` is
raised.


NetCDF I/O Formats
------------------

Standard Format
^^^^^^^^^^^^^^^

The standard NetCDF format stores complete provenance:

- Bulk quantities: ``C_ext``, ``C_sca``, ``C_abs``, ``SSA``, ``g``, ``beta``
- Per-radius data: ``per_radius_C_ext``, ``per_radius_C_sca``, ``per_radius_beta``
- Metadata: size distribution parameters, interpolation method, integration
  method, effective radius, fallback flags

This format supports full round-trip serialization:

.. code-block:: python

   from Aerosol3D.bulk.io import bulk_to_netcdf, bulk_from_netcdf
   bulk_to_netcdf(bulk, "bulk_aerosol.nc")
   restored = bulk_from_netcdf("bulk_aerosol.nc")

vSmartMOM-Compatible Format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A minimal subset for direct ingestion by the vSmartMOM radiative transfer
model:

- ``bulk_SSA``, ``bulk_C_ext_nm2``, ``bulk_C_sca_nm2``, ``bulk_beta``
- ``wavelength_nm``, ``r_eff_nm``, ``tau_ref``

Export with:

.. code-block:: python

   from Aerosol3D.bulk.io import bulk_to_vsmartmom_netcdf
   bulk_to_vsmartmom_netcdf(bulk, "vsmartmom_input.nc", tau_ref=0.3)
