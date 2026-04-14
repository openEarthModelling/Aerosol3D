import trimesh
import numpy as np
from typing import Optional, Union, Any, Dict
from aerosol3d.core.component import Component
from aerosol3d.physics.units import ureg, Q_

def from_file(
    file_path: str,
    target_unit: str = "nm",
    source_unit: Optional[str] = None,
    role: str = "core",
    material_name: str = "mineral",
    refractive_index: complex = 1.5 + 0.0j,
    density: Optional[float] = None,
    coordinates: Optional[np.ndarray] = None,
    **extra_params
) -> Component:
    """
    Load a 3D mesh file (STL, OBJ, PLY, etc.) and return it as a Component.
    
    Args:
        file_path: Path to the mesh file.
        target_unit: The desired unit for the returned Component (default 'nm').
        source_unit: The unit used in the mesh file. If None, tries to detect or assumes target_unit.
        role: Functional role of the component (default 'core').
        material_name: Material name (default 'mineral').
        refractive_index: Complex refractive index (m = n + ik).
        density: Material density.
        coordinates: (3,) array for center position. Default (0,0,0).
        **extra_params: Additional parameters passed to the Component constructor.
        
    Returns:
        Component: A component with geometry_type='mesh'.
    """
    # Load the mesh using trimesh
    mesh = trimesh.load(file_path)
    
    # If the file contains a scene (multiple meshes), merge them into a single mesh
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            raise ValueError(f"No geometry found in file: {file_path}")
        mesh = mesh.dump(concatenate=True)
    
    if coordinates is None:
        coordinates = np.zeros(3)
        
    # Handle unit scaling
    # If trimesh has internal units and source_unit is None, use them.
    detected_source_unit = source_unit
    if detected_source_unit is None:
        if hasattr(mesh, 'units') and mesh.units:
            detected_source_unit = mesh.units
        else:
            # Fallback to target_unit if no unit is specified anywhere
            detected_source_unit = target_unit
            
    # Calculate scale factor to target_unit using pint
    try:
        q_source = Q_(1.0, detected_source_unit)
        q_target = Q_(1.0, target_unit)
        scale_factor = q_source.to(q_target).magnitude
    except Exception:
        # If units are not recognized by pint, default to no scaling
        scale_factor = 1.0
        
    # Scale the mesh data if necessary
    if not np.isclose(scale_factor, 1.0):
        mesh.apply_scale(scale_factor)
    
    # Return the Component
    return Component(
        geometry_type='mesh',
        coordinates=coordinates,
        material_name=material_name,
        refractive_index=refractive_index,
        density=density,
        role=role,
        unit=target_unit,
        trimesh=mesh,
        **extra_params
    )
