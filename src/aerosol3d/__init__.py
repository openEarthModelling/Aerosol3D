from importlib.metadata import version

__version__ = version("aerosol3d")

from .core import AerosolParticle, MixingState, Material, FractalAggregate
from .geometry import create_sphere, create_ellipsoid, create_cube
from .modeling import (
    apply_distance_coating,
    apply_potential_void_coating,
    apply_potential_edge_coating,
    apply_ccm_coating,
    apply_cam_coating,
)
from .factory import from_file, from_fractal
from .io import save_vtp, save_voxel
from .optics import solve_optics, SimulationConfig  # noqa: F401
from .utils.plot import (
    save_screenshot,
    save_rotation_video,
    plot_particle_as_voxels,
    save_particle_voxel_screenshot,
    save_particle_voxel_video,
)
from .materials import preset_material  # noqa: F401
