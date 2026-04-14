import numpy as np
from typing import Optional, Union, Any, Dict, Tuple
from aerosol3d.physics.units import ureg, Q_

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

class Component:
    """
    Base class for a physical component of an aerosol particle.
    
    A component represents a geometric primitive (like a sphere, ellipsoid, or mesh)
    with specific material properties and spatial data.
    """
    
    def __init__(
        self,
        geometry_type: str,
        coordinates: np.ndarray,
        material_name: str = "unknown",
        refractive_index: complex = 1.5 + 0.0j,
        density: Optional[float] = None,
        role: str = "core",
        unit: str = "nm",
        **parameters
    ):
        """
        Initialize a Component.

        Args:
            geometry_type: The type of geometry ('sphere', 'ellipsoid', 'cube', 'mesh').
            coordinates: NumPy array of spatial coordinates (center or translation).
                         For 'sphere', can be (N, 3). For others, usually (3,).
            material_name: Name of the material.
            refractive_index: Complex refractive index (m = n + ik).
            density: Density of the material.
            role: Functional role (e.g., 'core', 'coating').
            unit: The unit for spatial data (default 'nm').
            **parameters: Geometry-specific parameters:
                - sphere: radius (scalar or array)
                - ellipsoid: axes (array of 3)
                - cube: side_lengths (scalar or array of 3)
                - mesh: trimesh (trimesh.Trimesh object)
        """
        self.geometry_type = geometry_type
        self.material_name = material_name
        self.refractive_index = refractive_index
        self.density = density
        self.role = role
        self.unit = ureg(unit)

        # Bind units to spatial data
        self.coordinates = Q_(np.asarray(coordinates), self.unit)
        
        self.parameters = {}
        for key, value in parameters.items():
            if key in ['radius', 'axes', 'side_lengths']:
                self.parameters[key] = Q_(np.asarray(value), self.unit)
            elif key == 'trimesh':
                if not TRIMESH_AVAILABLE:
                    raise ImportError("trimesh is required for 'mesh' geometry type but not installed.")
                self.parameters[key] = value
            else:
                self.parameters[key] = value

        # Backward compatibility / Convenience
        if 'radius' in self.parameters:
            self.radius = self.parameters['radius']

    def __repr__(self):
        return (f"Component(type={self.geometry_type}, material={self.material_name}, "
                f"role={self.role}, unit={self.unit})")

    @property
    def magnitude_coordinates(self) -> np.ndarray:
        """Return the coordinates as a raw NumPy array in the bound unit."""
        return self.coordinates.magnitude

    @property
    def magnitude_radius(self) -> np.ndarray:
        """Return the radius/radii as a raw NumPy array in the bound unit (sphere only)."""
        if 'radius' in self.parameters:
            return self.parameters['radius'].magnitude
        raise AttributeError("This component does not have a 'radius' attribute.")

    def to_unit(self, target_unit: str):
        """Convert spatial data to a different unit."""
        new_unit = ureg(target_unit)
        self.coordinates = self.coordinates.to(new_unit)
        for key in ['radius', 'axes', 'side_lengths']:
            if key in self.parameters:
                self.parameters[key] = self.parameters[key].to(new_unit)
        
        # Mesh vertices need conversion if stored inside trimesh
        if 'trimesh' in self.parameters:
            # trimesh objects don't natively handle pint units well, 
            # so we just scale the vertices.
            factor = Q_(1.0, self.unit).to(new_unit).magnitude
            self.parameters['trimesh'].vertices *= factor
            
        self.unit = new_unit
        if 'radius' in self.parameters:
            self.radius = self.parameters['radius']

    @property
    def volume(self) -> Q_:
        """Calculate the volume of the component."""
        if self.geometry_type == 'sphere':
            r = self.parameters['radius']
            # Sum volume if multiple spheres
            return np.sum(4/3 * np.pi * r**3)
        elif self.geometry_type == 'ellipsoid':
            a, b, c = self.parameters['axes']
            return 4/3 * np.pi * a * b * c
        elif self.geometry_type == 'cube':
            s = self.parameters['side_lengths']
            if s.size == 1:
                return s**3
            return s[0] * s[1] * s[2]
        elif self.geometry_type == 'mesh':
            # trimesh.volume returns value in cubic units of its vertices
            v = self.parameters['trimesh'].volume
            return Q_(v, self.unit**3)
        else:
            return Q_(0.0, self.unit**3)

    @property
    def surface_area(self) -> Q_:
        """Calculate the surface area of the component."""
        if self.geometry_type == 'sphere':
            r = self.parameters['radius']
            return np.sum(4 * np.pi * r**2)
        elif self.geometry_type == 'ellipsoid':
            # Knud Thomsen's formula approximation
            a, b, c = self.parameters['axes']
            p = 1.6075
            term = ( (a**p * b**p + a**p * c**p + b**p * c**p) / 3 )**(1/p)
            return 4 * np.pi * term
        elif self.geometry_type == 'cube':
            s = self.parameters['side_lengths']
            if s.size == 1:
                return 6 * s**2
            return 2 * (s[0]*s[1] + s[0]*s[2] + s[1]*s[2])
        elif self.geometry_type == 'mesh':
            a = self.parameters['trimesh'].area
            return Q_(a, self.unit**2)
        else:
            return Q_(0.0, self.unit**2)

    @property
    def bounds(self) -> Tuple[Q_, Q_]:
        """Return the bounding box (min, max) of the component."""
        if self.geometry_type == 'sphere':
            coords = self.coordinates
            r = self.parameters['radius']
            if coords.ndim == 1: # Single sphere
                return coords - r, coords + r
            else: # Multiple spheres
                # We need to broadcast r if it's a scalar but coords are multiple
                if r.size == 1 and coords.shape[0] > 1:
                    r_ext = np.full(coords.shape[0], r.magnitude) * r.units
                else:
                    r_ext = r
                
                # Check if r_ext needs reshaping for broadcasting
                if r_ext.ndim == 1:
                    r_ext = r_ext[:, np.newaxis]
                
                min_p = np.min(coords - r_ext, axis=0)
                max_p = np.max(coords + r_ext, axis=0)
                return min_p, max_p
                
        elif self.geometry_type == 'ellipsoid':
            axes = self.parameters['axes']
            return self.coordinates - axes, self.coordinates + axes
            
        elif self.geometry_type == 'cube':
            s = self.parameters['side_lengths']
            if s.size == 1:
                half_s = s / 2.0
            else:
                half_s = s / 2.0
            return self.coordinates - half_s, self.coordinates + half_s
            
        elif self.geometry_type == 'mesh':
            mesh = self.parameters['trimesh']
            b_min, b_max = mesh.bounds
            return self.coordinates + Q_(b_min, self.unit), self.coordinates + Q_(b_max, self.unit)
        else:
            return self.coordinates, self.coordinates
