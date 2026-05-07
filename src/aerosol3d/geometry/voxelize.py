import logging

import numpy as np
import pyvista as pv

logger = logging.getLogger(__name__)


def voxelize_with_materials(
    particle, voxel_size: float, bounds: list | None = None
) -> pv.ImageData:
    """Generate a regular 3D voxel grid with per-voxel material IDs.

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

    for block_name in particle.blocks.keys():
        block = particle.blocks[block_name]
        if block is None:
            continue
        try:
            enclosed = grid.select_interior_points(block, check_surface=False)
            # select_interior_points operates on grid points.
            # Convert point-level selection to cell-level by majority vote:
            # a cell is inside if its enclosing corner points are mostly inside.
            selected = enclosed.point_data["selected_points"]
            gdims = grid.dimensions
            # Reshape to (nx+1, ny+1, nz+1) for point grid
            point_sel = selected.reshape(gdims[0], gdims[1], gdims[2])
            # Average the 8 corner points of each cell to get cell-level membership
            cell_sel = (
                point_sel[:-1, :-1, :-1].astype(np.float32) +
                point_sel[1:, :-1, :-1].astype(np.float32) +
                point_sel[:-1, 1:, :-1].astype(np.float32) +
                point_sel[:-1, :-1, 1:].astype(np.float32) +
                point_sel[1:, 1:, :-1].astype(np.float32) +
                point_sel[1:, :-1, 1:].astype(np.float32) +
                point_sel[:-1, 1:, 1:].astype(np.float32) +
                point_sel[1:, 1:, 1:].astype(np.float32)
            ) / 8.0
            cell_mask = cell_sel >= 0.5
            mat_id = block.field_data.get("material_id", [0])[0]
            material_grid[cell_mask.ravel()] = mat_id
        except Exception:
            logger.warning("Failed to voxelize block %r, skipping", block_name)
            continue

    grid.cell_data["material_id"] = material_grid
    return grid
