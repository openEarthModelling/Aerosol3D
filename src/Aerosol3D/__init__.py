from importlib.metadata import version

__version__ = version("Aerosol3D")

from .core import AerosolParticle, FractalAggregate, Material, MixingState
from .factory import from_file, from_fractal
from .geometry import create_cube, create_ellipsoid, create_sphere
from .io import save_voxel, save_vtp
from .materials import preset_material  # noqa: F401
from .modeling import (
    apply_cam_coating,
    apply_ccm_coating,
    apply_distance_coating,
    apply_potential_edge_coating,
    apply_potential_void_coating,
)
from .optics import SimulationConfig, solve_optics  # noqa: F401
from .utils.plot import (
    plot_particle_as_voxels,
    save_particle_voxel_screenshot,
    save_particle_voxel_video,
    save_rotation_video,
    save_screenshot,
)
