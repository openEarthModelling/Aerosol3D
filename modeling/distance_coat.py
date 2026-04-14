"""Distance coating using boolean operations (Kahnert 2017 onion model)."""

import logging

import numpy as np
import pyvista as pv

logger = logging.getLogger(__name__)

from aerosol3d.geometry.boolean import safe_difference
from aerosol3d.geometry.primitives import create_sphere


def _offset_surface(core_mesh: pv.PolyData, thickness: float) -> pv.PolyData:
    """Create an offset surface by expanding from centroid.

    For sphere-type geometry, uses analytical expansion.
    For general meshes, uses centroid-based scaling.
    """
    geom_type = core_mesh.field_data.get("geometry_type", ["unknown"])[0]
    if geom_type == "sphere":
        radius = core_mesh.field_data["analytic_radius"][0]
        center = core_mesh.field_data["analytic_center"]
        # analytic_center is a flat array [x, y, z]; pass as tuple
        if center.ndim == 0:
            center_tuple = (float(center),)
        else:
            center_tuple = tuple(float(c) for c in center)
        return create_sphere(center=center_tuple, radius=radius + thickness)
    else:
        # Centroid-based scaling for general meshes
        center = core_mesh.center
        bounds = core_mesh.bounds
        max_extent = max(bounds[1::2][i] - bounds[::2][i] for i in range(3))
        scale_factor = (max_extent + 2 * thickness) / max_extent
        scaled = core_mesh.copy()
        scaled.points = (scaled.points - center) * scale_factor + center
        return scaled


def _voxel_fallback_coating(core_mesh: pv.PolyData, thickness: float) -> pv.PolyData:
    """Fallback: voxel dilation + Marching Cubes for offset surface."""
    from scipy.ndimage import binary_dilation

    bounds = core_mesh.bounds
    voxel_size = thickness / 3.0
    origin = np.array(bounds[::2]) - thickness
    max_corner = np.array(bounds[1::2]) + thickness
    dims = np.ceil((max_corner - origin) / voxel_size).astype(int) + 1

    grid = pv.ImageData(dimensions=dims, spacing=voxel_size, origin=origin)
    enclosed = grid.select_enclosed_points(core_mesh, tolerance=1e-6)
    mask = (enclosed.point_data["Selected"] == 1).astype(np.int8)

    mask_3d = mask.reshape(dims[2], dims[1], dims[0])
    dilated = binary_dilation(mask_3d, iterations=int(thickness / voxel_size))
    dilated_flat = dilated.ravel()

    coating_field = np.zeros(len(mask), dtype=np.float64)
    coating_field[(dilated_flat == 1) & (mask == 0)] = 1.0

    grid.cell_data["coating"] = coating_field
    grid_point = grid.cell_data_to_point_data()
    coating = grid_point.contour(isosurfaces=[0.5], scalars="coating")
    if coating.n_cells > 0:
        coating = coating.smooth(n_iter=20)
    return coating


def apply_distance_coating(particle, thickness: float, material):
    """Apply onion-model coating via boolean difference (Kahnert 2017).

    Creates an offset surface around the particle's combined geometry,
    then computes the coating shell as outer - inner via boolean difference.
    Falls back to voxel dilation + Marching Cubes if boolean ops fail.

    Args:
        particle: AerosolParticle instance with at least one block.
        thickness: Coating thickness in the particle's unit (e.g. nm).
        material: Material instance for the coating.

    Returns:
        The same particle with coating added and mixing_state set to COATED.

    Raises:
        ValueError: If thickness is not positive.
    """
    if thickness <= 0:
        raise ValueError("Coating thickness must be positive.")

    from aerosol3d.core.particle import MixingState

    core_mesh = particle.combined
    try:
        outer_mesh = _offset_surface(core_mesh, thickness)
        coating = safe_difference(outer_mesh, core_mesh)
    except Exception:
        logger.warning("Boolean offset failed, falling back to voxel dilation", exc_info=True)
        coating = _voxel_fallback_coating(core_mesh, thickness)

    particle.add_mesh("coating", coating, material)
    particle.mixing_state = MixingState.COATED
    return particle
