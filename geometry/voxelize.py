import numpy as np
import pyvista as pv


def voxelize_with_materials(particle, voxel_size: float) -> pv.ImageData:
    """Generate a regular 3D voxel grid with per-voxel material IDs.

    Args:
        particle: AerosolParticle instance with .blocks and .combined
        voxel_size: Side length of each voxel in the particle's unit.

    Returns:
        pv.ImageData with cell_data["material_id"] array.
        material_id == 0 means void (no material).
    """
    combined = particle.combined
    bounds = combined.bounds
    origin = np.array(bounds[::2])
    max_corner = np.array(bounds[1::2])
    extent = max_corner - origin

    dims = np.ceil(extent / voxel_size).astype(int) + 1

    grid = pv.ImageData(dimensions=dims, spacing=[voxel_size] * 3, origin=origin)

    material_grid = np.zeros(grid.n_cells, dtype=np.int32)

    for block_name in particle.blocks.keys():
        block = particle.blocks[block_name]
        if block is None:
            continue
        try:
            enclosed = grid.select_interior_points(block)
            point_selected = enclosed.point_data["selected_points"]
            cell_mask = point_selected
            mat_id = block.field_data.get("material_id", [0])[0]
            material_grid[cell_mask] = mat_id
        except Exception:
            continue

    grid.cell_data["material_id"] = material_grid
    return grid