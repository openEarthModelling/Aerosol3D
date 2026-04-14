import numpy as np
from typing import Union
from aerosol3d.core.particle import AerosolParticle, Component
from aerosol3d.physics.units import ureg

def apply_ccm_coating(
    particle: AerosolParticle,
    target_f_bc: float,
    material_name: str = "coating",
    refractive_index: complex = 1.4 + 0j
) -> AerosolParticle:
    """
    Apply a Closed-Cell Model (CCM) coating (Liu 2025).
    Each monomer is individually coated with a concentric shell.
    
    Args:
        particle: The input AerosolParticle.
        target_f_bc: Target volume fraction of black carbon (0.0 to 1.0).
        material_name: Name of the coating material.
        refractive_index: Refractive index of the coating.
        
    Returns:
        AerosolParticle: A new particle object with the CCM coating.
    """
    if not 0 < target_f_bc < 1:
        raise ValueError("target_f_bc must be between 0 and 1 (exclusive)")
        
    for comp in list(particle.components):
        if comp.role == "coating":
            continue
            
        coords = comp.coordinates.to(ureg.meter).magnitude
        
        if comp.geometry_type == 'sphere':
            r = comp.parameters['radius'].to(ureg.meter).magnitude
            # V_total = V_core / target_f_bc -> R_total = r / (target_f_bc**(1/3))
            R_total = r / (target_f_bc ** (1.0/3.0))
            
            coating_comp = Component(
                geometry_type="sphere",
                coordinates=coords,
                radius=R_total,
                material_name=material_name,
                refractive_index=refractive_index,
                role="coating",
                unit="m"
            )
            particle.add_component(coating_comp)
            
        elif comp.geometry_type == 'ellipsoid':
            axes = comp.parameters['axes'].to(ureg.meter).magnitude
            # a'*b'*c' = a*b*c / target_f_bc -> scale factor = (1/target_f_bc)**(1/3)
            scale = (1.0 / target_f_bc) ** (1.0/3.0)
            new_axes = axes * scale
            
            coating_comp = Component(
                geometry_type="ellipsoid",
                coordinates=coords,
                axes=new_axes,
                material_name=material_name,
                refractive_index=refractive_index,
                role="coating",
                unit="m"
            )
            particle.add_component(coating_comp)
            
        elif comp.geometry_type == 'cube':
            s = comp.parameters['side_lengths'].to(ureg.meter).magnitude
            scale = (1.0 / target_f_bc) ** (1.0/3.0)
            new_s = s * scale
            
            coating_comp = Component(
                geometry_type="cube",
                coordinates=coords,
                side_lengths=new_s,
                material_name=material_name,
                refractive_index=refractive_index,
                role="coating",
                unit="m"
            )
            particle.add_component(coating_comp)
        else:
            # For complex meshes, scaling might not be as straightforward without moving the center
            # Simple scaling by volume ratio
            scale = (1.0 / target_f_bc) ** (1.0/3.0)
            mesh = comp.parameters['trimesh'].copy()
            mesh.apply_scale(scale)
            
            coating_comp = Component(
                geometry_type="mesh",
                coordinates=coords,
                trimesh=mesh,
                material_name=material_name,
                refractive_index=refractive_index,
                role="coating",
                unit="m"
            )
            particle.add_component(coating_comp)
            
    particle.set_mixing_state(particle.mixing_state.COATED)
    return particle

def apply_cam_coating(
    particle: AerosolParticle,
    target_f_bc: float,
    material_name: str = "coating",
    refractive_index: complex = 1.4 + 0j
) -> AerosolParticle:
    """
    Apply a Coated-Aggregate Model (CAM) coating (Liu 2025).
    The entire aggregate is encapsulated by a single spherical (or bounding) envelope.
    
    Args:
        particle: The input AerosolParticle.
        target_f_bc: Target volume fraction of black carbon (0.0 to 1.0).
        material_name: Name of the coating material.
        refractive_index: Refractive index of the coating.
        
    Returns:
        AerosolParticle: A new particle object with the CAM coating.
    """
    if not 0 < target_f_bc < 1:
        raise ValueError("target_f_bc must be between 0 and 1 (exclusive)")
        
    total_core_volume = sum([comp.volume.to(ureg.meter**3).magnitude for comp in particle.components if comp.role != "coating"])
    
    if total_core_volume == 0:
        raise ValueError("Particle has no core volume to coat.")
        
    target_total_volume = total_core_volume / target_f_bc
    
    # Calculate aggregate center of mass
    all_coords = []
    for comp in particle.components:
        if comp.role != "coating":
            coords = comp.coordinates.to(ureg.meter).magnitude
            if coords.ndim == 1:
                all_coords.append(coords)
            else:
                all_coords.extend(coords)
                
    if not len(all_coords):
        raise ValueError("No valid coordinates found in the particle components.")
        
    all_coords = np.array(all_coords)
    center_of_mass = np.mean(all_coords, axis=0)
    
    # Generate a spherical envelope representing the coating
    # Volume = 4/3 * pi * R^3 -> R = (3 * Volume / (4 * pi))**(1/3)
    R_envelope = (3.0 * target_total_volume / (4.0 * np.pi)) ** (1.0/3.0)
    
    coating_comp = Component(
        geometry_type="sphere",
        coordinates=center_of_mass,
        radius=R_envelope,
        material_name=material_name,
        refractive_index=refractive_index,
        role="coating",
        unit="m"
    )
    
    particle.add_component(coating_comp)
    particle.set_mixing_state(particle.mixing_state.COATED)
    
    return particle
