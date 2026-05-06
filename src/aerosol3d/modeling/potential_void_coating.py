"""Potential-based void-filling coating with dual-parameter control.

Two-stage algorithm:
1. Surface contact layer: ensures coated_area_fraction of BC convex hull
   surface area is in contact with coating.
2. Bulk void filling: fills internal voids with remaining coating volume
   determined by dp_dc_ratio.
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
from scipy.ndimage import binary_erosion, binary_dilation

from aerosol3d.geometry.voxelize import voxelize_with_materials


def apply_potential_void_coating(
    particle,
    coated_area_fraction: float,
    dp_dc_ratio: float,
    material,
    k: float = 2.0,
    resolution: int = 64,
):
    """Apply potential-based void-filling coating.

    Stage 1 ensures surface coverage. Stage 2 fills internal voids.

    Args:
        particle: AerosolParticle with at least one core block.
        coated_area_fraction: Target fraction of BC convex hull surface
            area in contact with coating. Range (0, 1].
        dp_dc_ratio: Ratio of coated to core equivalent-volume-sphere
            diameter. Must be > 1.
        material: Material for the coating.
        k: Power-law exponent for potential field.
        resolution: Voxel grid resolution along longest axis.

    Returns:
        Modified AerosolParticle with coating added.

    Raises:
        ValueError: If parameters are out of range or constraints are
            infeasible.
        RuntimeError: If no BC or void voxels are found.
    """
    if not 0 < coated_area_fraction <= 1:
        raise ValueError("coated_area_fraction must be in (0, 1]")
    if dp_dc_ratio <= 1:
        raise ValueError("dp_dc_ratio must be > 1")

    from aerosol3d.core.particle import MixingState

    # 1. Voxelize particle with expanded grid for coating
    combined = particle.combined
    bounds = combined.bounds
    center = np.array(
        [
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2,
        ]
    )
    extent = np.array(bounds[1::2]) - np.array(bounds[::2])
    max_extent = float(np.max(extent))

    # Expand grid to accommodate dp_dc_ratio envelope
    coated_extent = max_extent * dp_dc_ratio
    voxel_size = coated_extent / resolution
    half = coated_extent / 2.0
    coated_bounds = [
        center[0] - half,
        center[0] + half,
        center[1] - half,
        center[1] + half,
        center[2] - half,
        center[2] + half,
    ]

    grid = voxelize_with_materials(
        particle, voxel_size=voxel_size, bounds=coated_bounds
    )

    # 2. Identify BC voxels
    core_mat_id = particle.combined.field_data.get("material_id", [0])[0]
    material_ids = grid.cell_data["material_id"]
    bc_mask = material_ids == core_mat_id
    n_bc = int(np.sum(bc_mask))

    if n_bc == 0:
        raise RuntimeError("No BC voxels found. Increase resolution.")

    cell_centers = grid.cell_centers().points
    bc_centers = cell_centers[bc_mask]

    # 3. Compute convex hull surface area of BC
    try:
        hull = ConvexHull(bc_centers)
        A_hull = hull.area
    except Exception:
        A_hull = 6.0 * (voxel_size ** 2) * (n_bc ** (2.0 / 3.0))

    # 4. Find BC surface voxels and surface-adjacent void voxels
    dims = grid.dimensions  # (nx, ny, nz)
    nx, ny, nz = dims
    bc_3d = bc_mask.reshape(nz - 1, ny - 1, nx - 1).astype(bool)

    eroded = binary_erosion(bc_3d).astype(bool)
    surface_3d = bc_3d & ~eroded
    void_3d = ~bc_3d

    surface_dilated = binary_dilation(surface_3d)
    surface_void_3d = surface_dilated & void_3d
    surface_void_flat = surface_void_3d.ravel()

    n_surface_void = int(np.sum(surface_void_flat))
    if n_surface_void == 0:
        raise RuntimeError("No surface-adjacent void voxels found.")

    # 5. Stage 1: Select surface void voxels for contact layer
    target_contact_area = coated_area_fraction * A_hull
    n_contact_target = min(
        int(np.ceil(target_contact_area / (voxel_size ** 2))),
        n_surface_void
    )

    surface_void_centers = cell_centers[surface_void_flat]
    dists = cdist(surface_void_centers, bc_centers)
    surface_potential = np.sum(1.0 / (dists ** k + 1e-20), axis=1)

    surface_sort_idx = np.argsort(surface_potential)[::-1]
    selected_surface = surface_sort_idx[:n_contact_target]

    surface_void_indices = np.where(surface_void_flat)[0]
    contact_indices = surface_void_indices[selected_surface]

    # 6. Volume feasibility check
    V_core = n_bc * (voxel_size ** 3)
    V_target = V_core * (dp_dc_ratio ** 3)
    V_contact = len(contact_indices) * (voxel_size ** 3)
    V_bulk = V_target - V_core - V_contact

    if V_bulk < 0:
        min_ratio = ((V_core + V_contact) / V_core) ** (1.0 / 3.0)
        raise ValueError(
            f"dp_dc_ratio={dp_dc_ratio:.3f} too small to achieve "
            f"coated_area_fraction={coated_area_fraction}. "
            f"Minimum required: dp_dc_ratio > {min_ratio:.3f}"
        )

    # 7. Stage 2: Fill remaining volume with void-filling potential
    void_mask_bulk = void_3d.ravel().copy()
    void_mask_bulk[contact_indices] = False
    n_void_bulk = int(np.sum(void_mask_bulk))

    if n_void_bulk == 0:
        raise RuntimeError("No bulk void voxels available for coating.")

    bulk_void_centers = cell_centers[void_mask_bulk]
    bulk_dists = cdist(bulk_void_centers, bc_centers)
    bulk_potential = np.sum(1.0 / (bulk_dists ** k + 1e-20), axis=1)

    n_bulk_target = min(
        int(np.ceil(V_bulk / (voxel_size ** 3))),
        n_void_bulk
    )
    bulk_sort_idx = np.argsort(bulk_potential)[::-1]
    selected_bulk = bulk_sort_idx[:n_bulk_target]

    bulk_void_indices = np.where(void_mask_bulk)[0]
    bulk_indices = bulk_void_indices[selected_bulk]

    # 8. Merge and extract smooth surface
    all_coating_indices = np.concatenate([contact_indices, bulk_indices])

    coating_field = np.zeros(grid.n_cells, dtype=np.float64)
    coating_field[all_coating_indices] = 1.0
    grid.cell_data["coating"] = coating_field

    grid_point = grid.cell_data_to_point_data()
    coating = grid_point.contour(isosurfaces=[0.5], scalars="coating")
    if coating.n_cells > 0:
        coating = coating.smooth(n_iter=20)

    particle.add_mesh("coating", coating, material)
    particle.mixing_state = MixingState.COATED
    return particle
