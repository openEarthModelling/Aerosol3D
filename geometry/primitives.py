import numpy as np
import pyvista as pv


def create_sphere(center, radius, theta_resolution=30, phi_resolution=30) -> pv.PolyData:
    """Create a sphere mesh with analytical parameters in field_data."""
    mesh = pv.Sphere(radius=radius, center=center,
                     theta_resolution=theta_resolution, phi_resolution=phi_resolution)
    mesh.field_data["geometry_type"] = np.array(["sphere"], dtype=object)
    mesh.field_data["analytic_radius"] = np.array([float(radius)], dtype=float)
    mesh.field_data["analytic_center"] = np.array([float(c) for c in center], dtype=float)
    return mesh


def create_ellipsoid(center, axes, theta_resolution=30, phi_resolution=30) -> pv.PolyData:
    """Create an ellipsoid mesh by scaling a unit sphere."""
    sphere = pv.Sphere(radius=1.0, center=(0, 0, 0),
                       theta_resolution=theta_resolution, phi_resolution=phi_resolution)
    sphere.points *= np.array(axes, dtype=float)
    sphere.points += np.array(center, dtype=float)
    sphere.field_data["geometry_type"] = np.array(["ellipsoid"], dtype=object)
    sphere.field_data["analytic_axes"] = np.array([float(a) for a in axes], dtype=float)
    sphere.field_data["analytic_center"] = np.array([float(c) for c in center], dtype=float)
    return sphere


def create_cube(center, side_lengths) -> pv.PolyData:
    """Create a cube/rectangular box mesh."""
    s = np.asarray(side_lengths, dtype=float)
    mesh = pv.Cube(center=center, x_length=s[0], y_length=s[1], z_length=s[2])
    mesh.field_data["geometry_type"] = np.array(["cube"], dtype=object)
    mesh.field_data["analytic_side_lengths"] = np.array(s.tolist(), dtype=float)
    mesh.field_data["analytic_center"] = np.array([float(c) for c in center], dtype=float)
    return mesh