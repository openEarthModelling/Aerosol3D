import pyvista as pv
import numpy as np
from aerosol3d.core.particle import AerosolParticle
from aerosol3d.physics.units import ureg, Q_

def save_vtp(particle: AerosolParticle, filename: str, unit: str = "nm"):
    """
    Export an AerosolParticle to a VTK PolyData (.vtp) file.
    
    Args:
        particle: The AerosolParticle to export.
        filename: Output filename (should end in .vtp).
        unit: The unit to scale coordinates and radii to (default 'nm').
    """
    target_unit = ureg(unit)
    
    # Material mapping to integer IDs
    mat_name_to_id = {}
    current_mat_id = 0
    
    blocks = []
    
    for i, component in enumerate(particle.components):
        mat_name = component.material_name
        if mat_name not in mat_name_to_id:
            mat_name_to_id[mat_name] = current_mat_id
            current_mat_id += 1
        mat_id = mat_name_to_id[mat_name]
        
        comp_poly = None
        
        if component.geometry_type == 'sphere':
            coords = component.coordinates.to(target_unit).magnitude
            radii = component.parameters['radius'].to(target_unit).magnitude
            if radii.ndim == 0:
                radii = np.full(len(coords) if coords.ndim > 1 else 1, radii)
            
            comp_poly = pv.PolyData(coords)
            comp_poly["radius"] = radii
            
        elif component.geometry_type == 'ellipsoid':
            # Create a sphere and scale it to ellipsoid
            axes = component.parameters['axes'].to(target_unit).magnitude
            center = component.coordinates.to(target_unit).magnitude
            comp_poly = pv.Sphere(radius=1.0, center=(0, 0, 0))
            comp_poly.points *= axes
            comp_poly.points += center
            # For ellipsoids, we don't have a single 'radius', 
            # but we can store mean radius for placeholder
            comp_poly["radius"] = np.full(comp_poly.n_points, np.mean(axes))
            
        elif component.geometry_type == 'cube':
            s = component.parameters['side_lengths'].to(target_unit).magnitude
            center = component.coordinates.to(target_unit).magnitude
            if s.size == 1:
                x, y, z = s, s, s
            else:
                x, y, z = s
            comp_poly = pv.Cube(center=center, x_length=x, y_length=y, z_length=z)
            comp_poly["radius"] = np.full(comp_poly.n_points, np.max(s)/2.0)
            
        elif component.geometry_type == 'mesh':
            mesh = component.parameters['trimesh']
            center = component.coordinates.to(target_unit).magnitude
            comp_poly = pv.wrap(mesh)
            # Scale and translate
            scale_factor = Q_(1.0, component.unit).to(target_unit).magnitude
            comp_poly.points = comp_poly.points * scale_factor + center
            comp_poly["radius"] = np.full(comp_poly.n_points, 0.0) # No radius for mesh points
            
        if comp_poly:
            # Add material and refractive index attributes
            comp_poly["material_id"] = np.full(comp_poly.n_points, mat_id)
            comp_poly["refractive_index_real"] = np.full(comp_poly.n_points, component.refractive_index.real)
            comp_poly["refractive_index_imag"] = np.full(comp_poly.n_points, component.refractive_index.imag)
            blocks.append(comp_poly)
            
    if not blocks:
        raise ValueError("Particle has no components to export.")
        
    # Merge all blocks into one PolyData
    # Note: Mixing different cell types (points, triangles) in PolyData is handled by merge
    combined = blocks[0]
    if len(blocks) > 1:
        for b in blocks[1:]:
            combined = combined.merge(b)
    
    # Add metadata
    combined.field_data["units"] = [unit]
    combined.field_data["mixing_state"] = [particle.mixing_state.name]
    
    # Store material names
    for name, m_id in mat_name_to_id.items():
        combined.field_data[f"material_name_{m_id}"] = [name]
        
    combined.save(filename)
