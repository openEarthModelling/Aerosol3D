Optical Computation
===================

Discrete Dipole Approximation (DDA)
-----------------------------------

DDA approximates a continuous particle as an array of interacting point
dipoles. The electromagnetic scattering problem is solved by finding the
induced dipole moments that satisfy the self-consistent field equations.

Convergence Criterion
---------------------

DDA accuracy depends on the product :math:`|m|kd`, where:

- :math:`|m|` is the magnitude of the refractive index
- :math:`k = 2\pi/\lambda` is the wavenumber
- :math:`d` is the dipole spacing

For accurate results, :math:`|m|kd \ll 1`. A common rule of thumb is
:math:`|m|kd < 0.5` for high accuracy.

LDR Polarizability
------------------

Aerosol3D uses the **Lattice Dispersion Relation (LDR)** polarizability
model (Draine & Goodman 1993) for DDA dipole polarizabilities. LDR replaces
the simpler radiative reaction correction and reduces DDA-Mie discrepancy
from 10-30% to ~1% at medium precision.

The LDR correction adds second-order terms to the Clausius-Mossotti
polarizability:

.. math::

    \alpha_{\text{LDR}} = \frac{\alpha_{\text{CM}}}{1 + \frac{k^2 d^2}{4\pi}
    \left[ b_1 + (b_2 + b_3 S) m_{\text{rel}}^2 \right]}

where :math:`S = \sum_\mu (e_\mu a_\mu)^2` depends on the propagation and
polarization directions, and :math:`b_1, b_2, b_3` are lattice geometry
constants.

Precision Levels
````````````````

The ``precision`` parameter on ``SimulationConfig`` controls the DDA dipole
spacing via the :math:`|m|kd` convergence criterion:

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Precision
     - :math:`|m|kd` target
     - Description
   * - ``"low"``
     - 0.63
     - Coarse grid, fastest. Suitable for quick estimates.
   * - ``"medium"`` (default)
     - 0.50
     - Good balance of speed and accuracy (~1% error vs Mie).
   * - ``"high"``
     - 0.30
     - Fine grid, higher accuracy. Recommended for publication.
   * - ``"ultra"``
     - 0.15
     - Very fine grid for maximum accuracy. Slowest.

Usage:

.. code-block:: python

    config = SimulationConfig(wavelength=550.0, precision="high")
    result = solve_optics(particle, config, solver="DDA")

Higher precision produces more dipoles (finer grid) and longer computation
time, but yields results closer to the exact Mie solution for spheres.

Mie Theory
----------

For spherical particles, Aerosol3D can use Mie theory (via PyMieScatt) as an
alternative to DDA. Mie theory is exact for spheres and much faster, making
it ideal for validation and spherical particle studies.

**EMA method selection.** When using the ``MIE`` solver with multi-component
particles, the ``ema_method`` parameter selects how the effective refractive
index is computed:

.. code-block:: python

    result = solve_optics(particle, config, solver="MIE", ema_method="maxwell_garnett")

Options: ``volume_weighted`` (default), ``maxwell_garnett``, ``bruggeman``.
See :doc:`ema-and-coreshell` for details.

**Core-shell Mie.** For coated spheres, use ``solver="MIE_CORESHELL"`` to solve
the exact layered-sphere Mie problem:

.. code-block:: python

    result = solve_optics(particle, config, solver="MIE_CORESHELL")

See :doc:`ema-and-coreshell` for a full comparison of EMA and core-shell approaches.

Orientational Averaging
-----------------------

Non-spherical particles have orientation-dependent optical properties.
Aerosol3D can average over multiple incident directions using Fibonacci
sphere sampling to obtain orientationally averaged cross-sections.

**Parallel execution.**  When ``orientational_average=True``, the solver
dispatches all (wavelength × orientation) tasks to a parallel worker pool
via ``joblib.Parallel`` with the ``loky`` backend (safe for PyJulia).  Set
``n_jobs`` to control the number of workers (default 32; set to 1 for
serial execution):

.. code-block:: python

    result = solve_optics(
        particle,
        config,
        solver="DDA",
        orientational_average=True,
        n_dirs=50,
        n_jobs=32,
        show_progress=True,
    )

**Key parameters:**

- ``n_dirs`` — Number of orientations sampled via Fibonacci sphere (default 50).
  A warning is issued when ``n_dirs < 30``.  For phase function convergence,
  ``n_dirs >= 100`` is recommended.
- ``n_jobs`` — Number of parallel workers.  Default 32.  Set to 1 for serial.
- ``show_progress`` — Display tqdm progress bars during orientation averaging.
  Default True.

**Error tolerance.**  If individual orientation solves fail, they are logged
and excluded from the average — the remaining successful orientations are used.
Near-field computation is disabled in the parallel path since those results
are not aggregated.

Phase Function
--------------

The phase function :math:`P_{11}(\theta, \phi)` describes the angular
distribution of scattered light. It is normalized such that::

    \int P_{11}(\theta, \phi) \, d\Omega = 1

Optical Data Export
-------------------

The ``AerosolOpticsData`` dataclass provides a unified container for
multi-wavelength optical properties. It is built from a list of
``OpticalResult`` objects via the ``from_optical_results()`` factory:

.. code-block:: python

    from Aerosol3D.optics import from_optical_results

    data = from_optical_results(results, n_legendre=32)

The container stores wavelength-dependent extinction, scattering,
absorption cross-sections, asymmetry parameter, phase functions, and
auto-computed Legendre moments.

**NetCDF I/O** — Persist and reload optical data:

.. code-block:: python

    data.to_netcdf("optics.nc")
    loaded = AerosolOpticsData.from_netcdf("optics.nc")

Legendre Moments
----------------

The ``compute_legendre_moments()`` function expands the scattering phase
function into Legendre polynomial coefficients:

.. math::

    k_l = (2l+1) \int_0^\pi P_{11}(\theta) P_l(\cos\theta) \sin\theta \, d\theta

.. note::
   ``compute_legendre_moments`` returns raw coefficients :math:`k_l`.
   For DISORT/libRadtran PMOM input, convert to :math:`\beta_l = k_l / (2l+1)`.

Optical Result Visualization
----------------------------

Aerosol3D provides plotting functions for analyzing and comparing optical
properties:

- ``plot_spectral_properties(data)`` — Extinction, scattering, and absorption
  spectra vs wavelength
- ``plot_phase_function(data)`` — Phase function P11 vs scattering angle
- ``plot_optical_comparison(data1, data2)`` — Side-by-side comparison of two
  datasets
- ``plot_phase_function_comparison(data1, data2)`` — Phase function comparison
  with relative difference
- ``plot_legendre_convergence(data)`` — Legendre moment convergence diagnostics
- ``plot_legendre_moments_spectrum(data)`` — Legendre moments as a function of
  wavelength
- ``generate_comparison_summary(data1, data2)`` — Full comparison summary with
  plots

.. code-block:: python

    from Aerosol3D.optics.visualization import plot_spectral_properties

    plot_spectral_properties(data)
