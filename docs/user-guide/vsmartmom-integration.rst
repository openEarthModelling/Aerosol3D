vSmartMOM Integration
=====================

Overview
--------

The ``Aerosol3D.vsmartmom`` subpackage provides a one-line Python interface
to the `vSmartMOM.jl <https://github.com/RemoteSensingTools/vSmartMOM.jl>`_
column radiative transfer (RT) solver. It enables end-to-end simulations
from :class:`Aerosol3D.bulk.BulkAerosolOpticsData` to top-of-atmosphere (TOA)
reflectance and bottom-of-atmosphere (BOA) transmittance.

Compared to the pyRadtran/DISORT pipeline (:doc:`optical-computation`),
vSmartMOM uses a method-of-moments (MOM) solver in Julia, operates in
**column mode** (plane-parallel layers), and natively supports polarization
(Stokes vector) output.

.. note::
   The ``vsmartmom`` subpackage is an **optional integration**. It requires
   Julia and vSmartMOM.jl to be installed separately.

Architecture & Data Flow
------------------------

The integration follows a subprocess pattern (similar to the DDA Julia
bridge) to avoid a runtime PyJulia dependency:

1. **Python** — :class:`VSmartMOMRunner.run_rt()` validates inputs and
   extracts bulk optical properties (``C_ext``, ``SSA``, ``beta``).
2. **Serialize** — ``serialize_input()`` writes a temporary NetCDF file
   containing the vertical profile, wavelengths, and Legendre coefficients.
3. **Julia subprocess** — ``scripts/run_rt.jl`` reads the NetCDF, builds a
   vSmartMOM model, overwrites per-band aerosol optical properties, and
   runs ``rt_run()``.
4. **Deserialize** — The Julia script writes an output NetCDF;
   :class:`VSmartMOMResult.from_netcdf()` loads it back into Python.

.. code-block:: text

   ┌─────────────────┐     ┌──────────────┐     ┌─────────────┐     ┌─────────────────┐
   │  Python         │────▶│  Temp NetCDF │────▶│  Julia      │────▶│  Temp NetCDF    │
   │  VSmartMOMRunner│     │  (input)     │     │  run_rt.jl  │     │  (output)       │
   └─────────────────┘     └──────────────┘     └─────────────┘     └─────────────────┘
                                                                              │
                                                                              ▼
                                                                    ┌─────────────────┐
                                                                    │  VSmartMOMResult│
                                                                    │  (Python)       │
                                                                    └─────────────────┘

Prerequisites
-------------

1. **Julia** must be installed and available on ``$PATH`` as ``julia``.
2. **vSmartMOM.jl** must be installed in a Julia project (environment).

   .. code-block:: bash

      # Activate/create a Julia project
      julia --project=/path/to/vSmartMOM-env -e 'using Pkg; Pkg.add("vSmartMOM")'

3. Pass the project path to :class:`VSmartMOMRunner`:

   .. code-block:: python

      from Aerosol3D.vsmartmom import VSmartMOMRunner

      runner = VSmartMOMRunner(julia_project="/path/to/vSmartMOM-env")

   If ``julia_project`` is ``None``, the system default Julia environment
   is used.

Python API
----------

VSmartMOMRunner
~~~~~~~~~~~~~~~

.. py:class:: VSmartMOMRunner(julia_project, julia_executable="julia", cleanup_temp=True)

   Orchestrate vSmartMOM radiative transfer simulations.

   :param julia_project: Path to the Julia project containing vSmartMOM
      and its dependencies. ``None`` to use the default environment.
   :param julia_executable: Name or path of the Julia executable.
   :param cleanup_temp: Whether to delete the temporary working directory
      after the run.

   .. py:method:: run_rt(bulk, heights, number_conc, sza, vza, vaz=None, tau_ref=None)

      Run the vSmartMOM radiative transfer model.

      :param bulk: Bulk aerosol optical properties.
      :type bulk: BulkAerosolOpticsData
      :param heights: Layer interface heights in metres.
         Length must equal ``len(number_conc) + 1``.
      :param number_conc: Number concentration per layer in cm\ :sup:`-3`.
      :param sza: Solar zenith angle in degrees.
      :param vza: Viewing zenith angles in degrees.
      :param vaz: Viewing azimuth angles in degrees. If ``None``, defaults
         to zeros.
      :param tau_ref: Optional total optical depth override. If given,
         concentrations are scaled so that the column optical depth at the
         first wavelength equals this value.
      :returns: The radiative transfer result.
      :rtype: VSmartMOMResult
      :raises ValueError: If input validation fails (dimension mismatch,
         negative concentration, non-monotonic heights).
      :raises RuntimeError: If Julia is not found or the subprocess fails.

VSmartMOMResult
~~~~~~~~~~~~~~~

.. py:class:: VSmartMOMResult

   Result container for vSmartMOM radiative transfer simulations.

   :ivar R: TOA reflectance, shape ``[n_stokes, n_vza, n_wl]``.
   :ivar T: BOA transmittance, shape ``[n_stokes, n_vza, n_wl]``.
   :ivar wavelengths: Wavelengths in nm, shape ``[n_wl]``.
   :ivar wavenumbers: Wavenumbers in cm\ :sup:`-1`, shape ``[n_wl]``.
   :ivar vza: Viewing zenith angles in degrees, shape ``[n_vza]``.
   :ivar vaz: Viewing azimuth angles in degrees, shape ``[n_vza]``.
   :ivar sza: Solar zenith angle in degrees.
   :ivar tau_per_layer: Optical depth per layer, shape ``[n_wl, n_layer]``.
   :ivar model_info: Metadata dictionary.

   .. py:method:: to_netcdf(path)

      Persist the full result to a NetCDF file.

      :param path: Output file path.

   .. py:classmethod:: from_netcdf(path)

      Load a result from a NetCDF file.

      :param path: Input file path.
      :returns: VSmartMOMResult instance.

Vertical Profile Input
----------------------

The vertical profile is defined by two arrays:

* ``heights`` — Layer interface heights in metres. Length ``n_layer + 1``.
* ``number_conc`` — Number concentration per layer in cm\ :sup:`-3`.
  Length ``n_layer``.

The per-layer optical depth is computed as:

.. math::

   \tau_{\text{layer}} = N \times C_{\text{ext}} \times \Delta z \times 10^{-6}

where:

* :math:`N` = ``number_conc`` [cm\ :sup:`-3`]
* :math:`C_{\text{ext}}` = ``bulk.C_ext`` [nm\ :sup:`2`]
* :math:`\Delta z` = layer thickness [m]
* :math:`10^{-6}` = unit conversion factor (nm\ :sup:`2` · m · cm\ :sup:`-3` → dimensionless)

SSA and Legendre coefficients (``beta``) are assumed **uniform across all
layers** — the microphysics is the same everywhere, only the concentration
varies with height.

Output Format
-------------

The :class:`VSmartMOMResult` contains:

* **R** — TOA reflectance. Dimensions ``[n_stokes, n_vza, n_wl]``.
* **T** — BOA transmittance. Dimensions ``[n_stokes, n_vza, n_wl]``.
* **tau_per_layer** — Optical depth per layer.
  Dimensions ``[n_wl, n_layer]``. Sum over the layer axis gives the total
  column optical depth at each wavelength.
* **wavelengths** / **wavenumbers** — Dual-unit spectral coordinates.

Error Handling
--------------

.. list-table::
   :header-rows: 1

   * - Scenario
     - Handling
   * - Julia not found on PATH
     - ``RuntimeError`` with installation instructions
   * - vSmartMOM project missing
     - ``RuntimeError`` with project setup guide
   * - ``heights`` length ≠ ``len(number_conc) + 1``
     - ``ValueError`` before serialization
   * - Negative concentration
     - ``ValueError``
   * - Heights not strictly increasing
     - ``ValueError``
   * - Wavelength mismatch (bulk vs vSmartMOM bands)
     - Log warning; use nearest-neighbor matching
   * - Julia subprocess error
     - ``RuntimeError`` with captured stderr
   * - Temporary file cleanup failure
     - Log warning; do not fail

Limitations
-----------

* **No heating rate output** — vSmartMOM's standard ``rt_run()`` returns
  only TOA/BOA radiances. Layer fluxes require solver extension.
* **Single aerosol type per run** — Multi-type mixing should be handled at
  the :class:`BulkOpticsBuilder` level beforehand.
* **No gas absorption** — ``absorption_params`` is not yet exposed through
  the runner.
* **Nearest-neighbor wavelength matching** — Off-grid wavelengths are matched
  to the closest vSmartMOM spectral band without interpolation.
* **Subprocess overhead** — Each ``run_rt()`` call spawns a fresh Julia
  process (~100–200 ms). For latency-sensitive workflows, a PyJulia path
  may be added in the future.
