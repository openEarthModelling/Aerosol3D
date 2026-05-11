Installation
============

Requirements
------------

- Python >= 3.10
- Julia >= 1.9 (optional, required for DDA optical computation)
- CUDA-capable GPU (optional, for GPU-accelerated DDA)

Basic Installation
------------------

Install from PyPI::

    pip install Aerosol3D

For development (editable install)::

    git clone https://github.com/openEarthModelling/Aerosol3D.git
    cd Aerosol3D
    pip install -e ".[dev]"

Julia Backend (for DDA)
-----------------------

The DDA solver requires Julia and the ``CoupledElectricMagneticDipoles.jl``
package::

    pip install pyjulia
    python -c "import julia; julia.install()"

Then in a Python session::

    from julia import CoupledElectricMagneticDipoles as CEMD

GPU Acceleration
----------------

GPU-accelerated DDA requires CUDA and the Julia CUDA packages. See the
CEMD.jl documentation for setup instructions.
