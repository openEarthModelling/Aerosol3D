import numpy as np
import pyvista as pv


class FractalAggregate:
    """Lightweight container for fractal aggregate data.

    Stores center coordinates and radii as NumPy arrays.
    Converts to mesh only when needed (visualization, boolean ops)
    via PyVista Glyph.
    """

    def __init__(self, centers: np.ndarray, radii: np.ndarray,
                 material, unit: str = "nm"):
        self.centers = np.asarray(centers, dtype=float)
        self.radii = np.asarray(radii, dtype=float)
        self.material = material
        self.unit = unit

    @property
    def n_monomers(self) -> int:
        return len(self.radii)

    @property
    def volume(self) -> float:
        """Total analytical volume from radii."""
        return float(np.sum(4/3 * np.pi * self.radii**3))

    @property
    def volume_weighted_center(self) -> np.ndarray:
        """Volume-weighted centroid: Σ(r³ × center) / Σ(r³)."""
        weights = self.radii**3
        return np.average(self.centers, axis=0, weights=weights)

    def to_mesh(self, theta_res: int = 20, phi_res: int = 20) -> pv.PolyData:
        """Expand to mesh via PyVista Glyph."""
        cloud = pv.PolyData(self.centers)
        cloud.point_data["radii"] = self.radii
        sphere = pv.Sphere(radius=1.0, theta_resolution=theta_res,
                           phi_resolution=phi_res)
        return cloud.glyph(geom=sphere, scale="radii", orient=False)

    def to_particle(self, name: str = "fractal_aggregate",
                    theta_res: int = 20, phi_res: int = 20):
        from .particle import AerosolParticle, MixingState
        particle = AerosolParticle(
            name=name, mixing_state=MixingState.AGGREGATED, unit=self.unit
        )
        mesh = self.to_mesh(theta_res, phi_res)
        particle.add_mesh("aggregate", mesh, self.material)
        return particle

    def __repr__(self) -> str:
        return (f"FractalAggregate(n={self.n_monomers}, "
                f"material={self.material.name!r})")