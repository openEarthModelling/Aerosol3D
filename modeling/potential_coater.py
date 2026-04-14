import numpy as np
from scipy.spatial.distance import cdist
from aerosol3d.core.particle import AerosolParticle, Component
from aerosol3d.physics.units import ureg
from .voxelizer import Voxelizer

def apply_potential_coating(
    particle: AerosolParticle, 
    target_f_bc: float,
    algorithm: str = "void_filling",
    k: float = 2.0,
    resolution: int = 64,
    material_name: str = "coating",
    refractive_index: complex = 1.4 + 0j
) -> AerosolParticle:
    """
    Apply a coating to the particle using potential field sorting (Luo 2019).
    
    Args:
        particle: The input AerosolParticle (usually containing a fractal core).
        target_f_bc: Target volume fraction of black carbon (0.0 to 1.0).
        algorithm: "void_filling" (Coated Way 2) or "edge_filling" (Coated Way 1).
        k: Power law exponent for the potential field.
        resolution: Voxel grid resolution.
        material_name: Name of the coating material.
        refractive_index: Refractive index of the coating.
        
    Returns:
        AerosolParticle: A new particle object with the added coating layer.
    """
    if not 0 < target_f_bc < 1:
        raise ValueError("target_f_bc must be between 0 and 1 (exclusive)")
        
    vox = Voxelizer(particle, resolution=resolution)
    
    # Target number of voxels for coating
    # V_total = V_bc / target_f_bc
    # V_coating = V_total - V_bc = V_bc * (1/target_f_bc - 1)
    
    # 1. Identify BC voxels
    # For now, assume component 0 is the BC core
    dist_field = vox.get_distance_field(0)
    bc_mask = dist_field <= 0
    n_bc_voxels = np.sum(bc_mask)
    
    if n_bc_voxels == 0:
        raise RuntimeError("No black carbon voxels found. Increase resolution or check monomer sizes.")
        
    n_total_target = int(n_bc_voxels / target_f_bc)
    n_coating_voxels = n_total_target - n_bc_voxels
    
    # 2. Calculate potential field for all non-BC voxels
    # Get centers of BC monomers
    bc_comp = particle.components[0]
    bc_centers = bc_comp.coordinates.to(ureg.meter).magnitude
    
    # Voxel centers that are NOT in the BC core
    candidate_centers = vox.voxel_centers[~bc_mask.ravel()]
    
    if algorithm == "void_filling":
        # p = sum(1 / l_i^k) where l_i is distance to monomer centers
        # Luo 2019 uses k=2 for void filling
        dists = cdist(candidate_centers, bc_centers)
        potential = np.sum(1.0 / (dists**k + 1e-20), axis=1) # Add epsilon to avoid div by zero
    elif algorithm == "edge_filling":
        # In Luo 2019, q = sum(1 / L_i^k) where L_i is distance to edge dipoles (surface voxels)
        surface_voxels = vox.get_surface_voxels([0])
        if len(surface_voxels) == 0:
            raise RuntimeError("No surface voxels found for edge filling algorithm.")
        dists = cdist(candidate_centers, surface_voxels)
        potential = np.sum(1.0 / (dists**k + 1e-20), axis=1)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
        
    # 3. Sort candidates by potential and select top N
    sort_idx = np.argsort(potential)[::-1] # Descending order
    selected_idx = sort_idx[:n_coating_voxels]
    
    # 4. Create new component for coating
    coating_coords = candidate_centers[selected_idx]
    # In a voxel representation, each 'point' has a radius half of voxel size
    # to approximate the volume correctly.
    v_radius = vox.voxel_size / 2.0
    
    coating_comp = Component(
        geometry_type="sphere",
        coordinates=coating_coords,
        radius=v_radius,
        material_name=material_name,
        refractive_index=refractive_index,
        role="coating",
        unit="m"
    )
    
    particle.add_component(coating_comp)
    particle.set_mixing_state(particle.mixing_state.COATED)
    
    return particle
