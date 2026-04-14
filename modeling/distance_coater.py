import numpy as np
from aerosol3d.core.particle import AerosolParticle, Component
from aerosol3d.physics.units import ureg
from .voxelizer import Voxelizer

def apply_distance_coating(
    particle: AerosolParticle,
    thickness: float,
    resolution: int = 64,
    material_name: str = "coating",
    refractive_index: complex = 1.4 + 0j
) -> AerosolParticle:
    """
    Apply a coating based on the distance field (Onion Coating, Kahnert 2017).
    
    Args:
        particle: The input AerosolParticle.
        thickness: The thickness of the coating layer in meters.
        resolution: Voxel grid resolution.
        material_name: Name of the coating material.
        refractive_index: Refractive index of the coating.
        
    Returns:
        AerosolParticle: A new particle object with the added coating layer.
    """
    if thickness <= 0:
        raise ValueError("Coating thickness must be positive.")
        
    vox = Voxelizer(particle, resolution=resolution)
    
    # 1. Get the union distance field for all existing components
    # Using all components because the coating should wrap around everything present
    union_dist = vox.get_union_distance_field()
    
    # 2. Identify coating voxels: 0 < distance <= thickness
    coating_mask = (union_dist > 0) & (union_dist <= thickness)
    
    coating_coords = vox.voxel_centers[coating_mask.ravel()]
    
    if len(coating_coords) == 0:
        raise RuntimeError("No voxels found for the specified coating thickness. Try increasing thickness or resolution.")
    
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
