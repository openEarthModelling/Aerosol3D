import numpy as np
from scipy.spatial.distance import cdist
from aerosol3d.core.particle import AerosolParticle
from aerosol3d.physics.units import ureg

class Voxelizer:
    """
    Utility class to convert discrete particle components into a voxel grid.
    """
    
    def __init__(self, particle: AerosolParticle, resolution: int = 64, padding: float = 0.2):
        """
        Initialize the voxelizer.
        
        Args:
            particle: The AerosolParticle to voxelize.
            resolution: The number of voxels along the largest dimension.
            padding: Extra space around the particle (as a fraction of total size).
        """
        self.particle = particle
        self.resolution = resolution
        self.padding = padding
        self._setup_grid()

    def _setup_grid(self):
        # Collect bounds from all components to find total bounding box
        all_min = []
        all_max = []
        
        if not self.particle.components:
            raise ValueError("Particle has no components to voxelize.")

        for comp in self.particle.components:
            b_min, b_max = comp.bounds
            all_min.append(b_min.to(ureg.meter).magnitude)
            all_max.append(b_max.to(ureg.meter).magnitude)
            
        min_p_raw = np.min(all_min, axis=0)
        max_p_raw = np.max(all_max, axis=0)
        
        # Calculate bounding box with padding
        extent_raw = max_p_raw - min_p_raw
        max_dim_extent = np.max(extent_raw)
        
        self.origin = min_p_raw - max_dim_extent * self.padding
        self.extent = max_p_raw + max_dim_extent * self.padding - self.origin
        self.voxel_size = np.max(self.extent) / self.resolution
        
        # Grid dimensions
        self.dims = np.ceil(self.extent / self.voxel_size).astype(int)
        
        # Create grid coordinates
        x = np.linspace(self.origin[0], self.origin[0] + self.dims[0] * self.voxel_size, self.dims[0])
        y = np.linspace(self.origin[1], self.origin[1] + self.dims[1] * self.voxel_size, self.dims[1])
        z = np.linspace(self.origin[2], self.origin[2] + self.dims[2] * self.voxel_size, self.dims[2])
        
        self.grid_z, self.grid_y, self.grid_x = np.meshgrid(z, y, x, indexing='ij')
        self.voxel_centers = np.stack([self.grid_x.ravel(), self.grid_y.ravel(), self.grid_z.ravel()], axis=1)

    def get_distance_field(self, component_index: int = 0) -> np.ndarray:
        """
        Calculate the distance from each voxel center to the nearest component surface.
        """
        comp = self.particle.components[component_index]
        coords = comp.coordinates.to(ureg.meter).magnitude
        
        if comp.geometry_type == 'sphere':
            rad = comp.parameters['radius'].to(ureg.meter).magnitude
            # Handle multiple spheres (fractal aggregate)
            if coords.ndim == 1: # Single center
                dist_to_centers = np.linalg.norm(self.voxel_centers - coords, axis=1)
            else: # Multiple centers
                dist_to_centers = cdist(self.voxel_centers, coords)
            
            # Subtract radii to get distance to surface
            dist_to_surface = dist_to_centers - rad
            if dist_to_surface.ndim > 1:
                min_dist = np.min(dist_to_surface, axis=1)
            else:
                min_dist = dist_to_surface
                
        elif comp.geometry_type == 'ellipsoid':
            axes = comp.parameters['axes'].to(ureg.meter).magnitude
            # Simplified ellipsoid distance field (approximate)
            # x^2/a^2 + y^2/b^2 + z^2/c^2 = 1
            rel_coords = self.voxel_centers - coords
            dist_norm = np.sqrt(np.sum((rel_coords / axes)**2, axis=1))
            # Distance approximation: d = (dist_norm - 1) * min(axes)
            min_dist = (dist_norm - 1.0) * np.min(axes)
            
        elif comp.geometry_type == 'cube':
            s = comp.parameters['side_lengths'].to(ureg.meter).magnitude
            if s.size == 1:
                half_s = np.full(3, s / 2.0)
            else:
                half_s = s / 2.0
            
            rel_coords = np.abs(self.voxel_centers - coords)
            q = rel_coords - half_s
            # Signed distance to box
            min_dist = np.linalg.norm(np.maximum(q, 0.0), axis=1) + \
                       np.minimum(np.max(q, axis=1), 0.0)
                       
        elif comp.geometry_type == 'mesh':
            mesh = comp.parameters['trimesh']
            # Convert voxel centers to mesh coordinate system
            # Since 'mesh' type uses 'coordinates' as translation
            query_points = self.voxel_centers - coords
            
            import trimesh.proximity
            prox = trimesh.proximity.ProximityQuery(mesh)
            min_dist = prox.signed_distance(query_points)
            # Signed distance in trimesh is positive inside, negative outside
            # We want positive outside, negative inside for consistency with spheres
            min_dist = -min_dist
            
        else:
            raise NotImplementedError(f"Distance field not implemented for {comp.geometry_type}")
        
        return min_dist.reshape(self.dims[::-1]) # Reshape to (Z, Y, X)

    def get_union_distance_field(self, component_indices=None) -> np.ndarray:
        """
        Calculate the combined (union) distance field for a set of components.
        If no indices are provided, uses all components.
        """
        if component_indices is None:
            component_indices = list(range(len(self.particle.components)))
            
        if not component_indices:
            raise ValueError("No components to calculate distance field.")
            
        union_dist = self.get_distance_field(component_indices[0])
        for idx in component_indices[1:]:
            dist = self.get_distance_field(idx)
            union_dist = np.minimum(union_dist, dist)
            
        return union_dist

    def get_surface_voxels(self, component_indices=None) -> np.ndarray:
        """
        Extract the surface voxels for the specified components.
        A voxel is considered a surface voxel if its distance field is near zero.
        """
        union_dist = self.get_union_distance_field(component_indices)
        # Find voxels on the surface (distance between -voxel_size and voxel_size)
        surface_mask = (union_dist <= self.voxel_size) & (union_dist > -self.voxel_size)
        return self.voxel_centers[surface_mask.ravel()]
