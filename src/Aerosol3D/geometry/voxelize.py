import logging

import numpy as np
import pyvista as pv

logger = logging.getLogger(__name__)


def voxelize_with_materials(
    particle, voxel_size: float, bounds: list | None = None
) -> pv.ImageData:
    """Generate a regular 3D voxel grid with per-voxel material IDs.

    Uses the standard DDA discretisation: a voxel is assigned to a material
    if its centre point lies inside the material mesh. This ensures that
    touching monomers produce face-connected voxels, unlike the previous
    corner-voting approach which severely fragmented the discretised shape.

    Args:
        particle: AerosolParticle instance with .blocks and .combined
        voxel_size: Side length of each voxel in the particle's unit.
        bounds: Optional 6-element list [xmin, xmax, ymin, ymax, zmin, zmax]
            to override the particle's natural bounds. Useful when the
            grid must be larger than the particle (e.g. to accommodate
            outward-growing coatings).

    Returns:
        pv.ImageData with cell_data["material_id"] array.
        material_id == 0 means void (no material).
    """
    combined = particle.combined
    _bounds = bounds if bounds is not None else combined.bounds
    origin = np.array(_bounds[::2])
    max_corner = np.array(_bounds[1::2])
    extent = max_corner - origin

    dims = np.ceil(extent / voxel_size).astype(int) + 1

    grid = pv.ImageData(dimensions=dims, spacing=[voxel_size] * 3, origin=origin)

    material_grid = np.zeros(grid.n_cells, dtype=np.int32)

    # Cell-centre testing: standard DDA dipole-assignment rule.
    cell_centres = grid.cell_centers()

    for block_name in particle.blocks.keys():
        block = particle.blocks[block_name]
        if block is None:
            continue
        try:
            enclosed = cell_centres.select_interior_points(block, check_surface=False)
            inside = enclosed.point_data["selected_points"].astype(bool)
            mat_id = block.field_data.get("material_id", [0])[0]
            material_grid[inside] = mat_id
        except Exception:
            logger.warning("Failed to voxelize block %r, skipping", block_name)
            continue

    grid.cell_data["material_id"] = material_grid
    return grid
