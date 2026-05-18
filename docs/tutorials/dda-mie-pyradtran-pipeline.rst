DDA-Mie-pyRadtran Pipeline
===========================

This tutorial demonstrates a three-stage pipeline that compares DDA and Mie
optical properties and their impact on radiative transfer (RT) results using
pyRadtran (libRadtran's Python interface).

.. note::
   Stage 2 (radiative transfer) requires pyRadtran and libRadtran to be
   installed. Stages 1 and 3 work independently. See
   :doc:`/user-guide/installation` for setup instructions.

Stage 1: Compute Optical Properties
------------------------------------

Compute multi-wavelength optical properties using both DDA and Mie solvers,
then export to NetCDF via ``AerosolOpticsData``.

.. literalinclude:: ../../examples/dda_mie_pyradtran_pipeline/compute_optics.py
   :language: python
   :linenos:

Key steps:

1. **Create particle** — A 200 nm black carbon sphere, configured in ``config.py``
2. **Run solvers** — Loop over wavelengths (400–700 nm) with both DDA and Mie
3. **Export** — ``from_optical_results()`` builds an ``AerosolOpticsData`` container
   with auto-computed Legendre moments, saved via ``to_netcdf()``
4. **Compare** — ``plot_optical_comparison()`` and ``plot_phase_function_comparison()``
   generate side-by-side plots of the two methods

Stage 2: Run Radiative Transfer
--------------------------------

Load the precomputed optics and run DISORT radiative transfer through pyRadtran.

.. literalinclude:: ../../examples/dda_mie_pyradtran_pipeline/run_radiative_transfer.py
   :language: python
   :linenos:

Key steps:

1. **Load optics** — ``AerosolOpticsData.from_netcdf()`` loads the NetCDF from Stage 1
2. **Build aerosol** — Convert to pyRadtran ``CompositeAerosol`` with size distribution
   and vertical mass profile (exponential decay with scale height)
3. **Legendre moments** — ``k_l`` coefficients are converted to ``beta_l = k_l/(2l+1)``
   for DISORT PMOM compatibility
4. **Run DISORT** — Execute the RT solver and save irradiance/transmittance results

.. note::
   Set the ``PYRADTRAN_DATA_PATH`` environment variable to your libRadtran data
   directory before running this stage.

Stage 3: Compare Results
--------------------------

Compare optical properties and radiative transfer outputs between DDA and Mie.

.. literalinclude:: ../../examples/dda_mie_pyradtran_pipeline/compare_results.py
   :language: python
   :linenos:

Key steps:

1. **Load data** — Read both optics NetCDFs and RT result NetCDFs
2. **Optical comparison** — ``plot_optical_comparison()`` and
   ``plot_phase_function_comparison()`` compare spectral properties and phase functions
3. **RT comparison** — Direct/diffuse irradiance and transmittance spectra plotted
   with relative differences
4. **Summary** — ``generate_comparison_summary()`` produces a text summary of key metrics

Running the Pipeline
--------------------

From the ``examples/dda_mie_pyradtran_pipeline/`` directory::

    # Stage 1: Compute optics (Mie only, fast)
    python compute_optics.py --solver MIE

    # Stage 1: Compute optics (both solvers)
    python compute_optics.py

    # Stage 2: Run radiative transfer (requires pyRadtran)
    export PYRADTRAN_DATA_PATH=/path/to/libRadtran/data
    python run_radiative_transfer.py

    # Stage 3: Compare results
    python compare_results.py

Configuration
-------------

All parameters are centralized in ``config.py``: particle properties, DDA/Mie
settings, atmospheric scene, aerosol vertical profile, and size distribution.
