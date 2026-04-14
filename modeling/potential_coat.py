"""Potential field coating with Marching Cubes smooth output (Luo 2019).

Provides two coating algorithms:
- void_filling: Coated Way 2 from Luo 2019 - fills void space by
  potential relative to all BC monomer centers.
- edge_filling: Coated Way 1 from Luo 2019 - fills void space by
  potential relative to surface (edge) dipoles only.
"""

import numpy as np
from scipy.spatial.distance import cdist

from aerosol3d.geometry.voxelize import voxelize_with_materials


def apply_potential_coating(
    particle, target_f_bc: float, material,
    algorithm: str = "void_filling", k: float = 2.0,
    resolution: int = 64
) -> object:
    """Apply coating via potential field sorting (Luo 2019).

    Voxelize the particle, compute potential field on void voxels,
    select top-N by potential, extract smooth isosurface via Marching Cubes.

    Args:
        particle: AerosolParticle instance with at least one block.
        target_f_bc: Target volume fraction of BC (0.0 to 1.0, exclusive).
        material: Material instance for the coating.
        algorithm: "void_filling" or "edge_filling".
        k: Power law exponent for the potential field.
        resolution: Number of voxels along the largest axis.

    Returns:
        The same particle with coating added and mixing_state set to COATED.

    Raises:
        ValueError: If target_f_bc is not between 0 and 1, or algorithm
            is unknown.
        RuntimeError: If no BC voxels found or target f_bc too high.
    """
    if not 0 < target_f_bc < 1:
        raise ValueError("target_f_bc must be between 0 and 1 (exclusive).")

    from aerosol3d.core.particle import MixingState

    # 1. Compute voxel_size from resolution and particle bounds.
    combined = particle.combined
    bounds = combined.bounds
    extent = np.array(bounds[1::2]) - np.array(bounds[::2])
    max_extent = float(np.max(extent))
    voxel_size = max_extent / resolution

    # 2. Voxelize the particle.
    grid = voxelize_with_materials(particle, voxel_size=voxel_size)

    # 3. Identify BC voxels using the material_id from the first block.
    core_mat_id = particle.combined.field_data.get("material_id", [0])[0]
    material_ids = grid.cell_data["material_id"]
    bc_mask = material_ids == core_mat_id
    n_bc = int(np.sum(bc_mask))

    if n_bc == 0:
        raise RuntimeError("No BC voxels found. Increase resolution.")

    # 4. Compute target number of coating voxels.
    n_coating = int(n_bc / target_f_bc) - n_bc
    if n_coating <= 0:
        raise RuntimeError("Target f_bc too high -- no room for coating.")

    # 5. Compute potential field on non-BC voxels.
    cell_centers = grid.cell_centers().points
    bc_centers = cell_centers[bc_mask]
    candidate_mask = ~bc_mask
    candidate_centers = cell_centers[candidate_mask]

    if len(candidate_centers) == 0:
        raise RuntimeError("No candidate voxels available for coating.")

    if algorithm == "void_filling":
        # Luo 2019 Coated Way 2: potential relative to all BC monomer centers
        dists = cdist(candidate_centers, bc_centers)
        potential = np.sum(1.0 / (dists ** k + 1e-20), axis=1)
    elif algorithm == "edge_filling":
        # Luo 2019 Coated Way 1: potential relative to surface dipoles
        from scipy.ndimage import binary_erosion

        dims = grid.dimensions
        # ImageData dimensions are (nx, ny, nz); reshape in reverse for
        # (z, y, x) ordering consistent with ravel.
        bc_3d = bc_mask.reshape(dims[2], dims[1], dims[0])
        eroded = binary_erosion(bc_3d)
        edge_mask_3d = bc_3d & ~eroded
        edge_mask_flat = edge_mask_3d.ravel()

        if not np.any(edge_mask_flat):
            raise RuntimeError(
                "No surface voxels found for edge filling algorithm.")

        edge_centers = cell_centers[edge_mask_flat]
        dists = cdist(candidate_centers, edge_centers)
        potential = np.sum(1.0 / (dists ** k + 1e-20), axis=1)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # 6. Select top-N candidates by potential.
    sort_idx = np.argsort(potential)[::-1]
    n_select = min(n_coating, len(sort_idx))
    selected_idx = sort_idx[:n_select]

    # 7. Create scalar field and extract smooth isosurface via Marching Cubes.
    coating_field = np.zeros(grid.n_cells, dtype=np.float64)
    coating_field[selected_idx] = 1.0
    grid.cell_data["coating"] = coating_field

    # Convert cell data to point data for contour (marching cubes requires it).
    grid_point = grid.cell_data_to_point_data()
    coating = grid_point.contour(isosurfaces=[0.5], scalars="coating")
    if coating.n_cells > 0:
        coating = coating.smooth(n_iter=20)

    particle.add_mesh("coating", coating, material)
    particle.mixing_state = MixingState.COATED
    return particle
