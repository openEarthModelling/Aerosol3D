import numpy as np
import pyvista as pv
import pytest


@pytest.fixture
def soot_material():
    from Aerosol3D.core.material import Material
    return Material(name="soot", refractive_index=complex(1.8, 0.7), density=1.8)


@pytest.fixture
def sulfate_material():
    from Aerosol3D.core.material import Material
    return Material(name="sulfate", refractive_index=complex(1.4, 0.0), density=1.8)


@pytest.fixture
def sample_sphere_mesh():
    return pv.Sphere(radius=50.0, center=(0, 0, 0))


@pytest.fixture
def sample_fractal_coords():
    rng = np.random.default_rng(42)
    centers = rng.random((10, 3)) * 200
    radii = np.full(10, 25.0)
    return centers, radii