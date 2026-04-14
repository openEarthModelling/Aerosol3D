import numpy as np
from typing import Optional, Union
from aerosol3d.core.component import Component
from aerosol3d.physics.units import ureg

def from_pyfrac(
    aggregate_obj,
    role: str = 'core',
    material_name: str = 'soot',
    refractive_index: complex = complex(1.8, 0.7),
    density: Optional[float] = 1.8, # g/cm^3
    target_unit: str = 'nm'
) -> Component:
    """
    Adapter to create an Aerosol3D Component from a pyFracAggregate Aggregate object.
    
    Args:
        aggregate_obj: An instance of pyFracAggregate.core.aggregate.Aggregate.
        role: Functional role (e.g., 'core').
        material_name: Name of the material.
        refractive_index: Complex refractive index.
        density: Material density in g/cm^3 (default 1.8 for soot).
        target_unit: The unit to store the data in (default 'nm').
        
    Returns:
        Component: An Aerosol3D Component containing the fractal aggregate data.
    """
    # pyFracAggregate stores positions and radii
    # We assume the object has attributes 'positions', 'radii', and 'length_unit'
    coords = aggregate_obj.positions  # Expected (N, 3)
    radii = aggregate_obj.radii       # Expected (N,)
    source_unit = getattr(aggregate_obj, 'length_unit', 'nm')
    
    # Create the component
    comp = Component(
        geometry_type='sphere',
        coordinates=coords,
        radius=radii,
        material_name=material_name,
        refractive_index=refractive_index,
        density=density,
        role=role,
        unit=source_unit
    )
    
    # Convert to the target unit if different
    if source_unit != target_unit:
        comp.to_unit(target_unit)
        
    return comp
