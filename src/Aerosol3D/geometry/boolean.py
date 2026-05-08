import numpy as np
import pyvista as pv


def to_manifold(mesh: pv.PolyData):
    """Convert a pv.PolyData to a manifold3d Manifold."""
    import manifold3d as mn

    # Clean the mesh to ensure it's in good condition
    mesh = mesh.clean()

    # Get vertices and convert to float32 as expected by manifold3d
    verts = mesh.points.astype(np.float32)

    # Get faces - convert to triangle format if needed
    if mesh.is_all_triangles:
        # For triangle meshes, extract triangle vertices
        tris = mesh.faces.reshape(-1, 4)[:, 1:4]
    else:
        # Convert to triangles
        mesh_triangulated = mesh.triangulate()
        tris = mesh_triangulated.faces.reshape(-1, 4)[:, 1:4]

    # Create manifold3d Mesh and then Manifold
    mesh_mn = mn.Mesh(vert_properties=verts, tri_verts=tris.astype(np.uint32))
    return mn.Manifold(mesh_mn)


def from_manifold(m) -> pv.PolyData:
    """Convert a manifold3d Manifold to pv.PolyData."""
    # Get mesh from manifold
    mesh = m.to_mesh()

    # Extract vertices (first 3 columns are x,y,z coordinates)
    verts = mesh.vert_properties[:, :3]

    # Extract triangles
    tris = mesh.tri_verts.astype(np.int32)
    n = len(tris)

    # Create faces array for PyVista
    faces = np.column_stack([np.full(n, 3, dtype=np.int32), tris]).ravel()

    return pv.PolyData(verts, faces)


def safe_difference(a: pv.PolyData, b: pv.PolyData) -> pv.PolyData:
    """Boolean difference (a - b) using manifold3d, fallback to VTK."""
    try:
        ma = to_manifold(a.clean())
        mb = to_manifold(b.clean())
        return from_manifold(ma - mb)
    except Exception:
        # Fallback to VTK boolean operations
        return a.clean().triangulate().boolean_difference(b.clean().triangulate())


def safe_union(a: pv.PolyData, b: pv.PolyData) -> pv.PolyData:
    """Boolean union (a | b) using manifold3d, fallback to VTK."""
    try:
        ma = to_manifold(a.clean())
        mb = to_manifold(b.clean())
        return from_manifold(ma + mb)
    except Exception:
        # Fallback to VTK boolean operations
        return a.clean().triangulate().boolean_union(b.clean().triangulate())


def safe_intersection(a: pv.PolyData, b: pv.PolyData) -> pv.PolyData:
    """Boolean intersection (a & b) using manifold3d, fallback to VTK."""
    try:
        ma = to_manifold(a.clean())
        mb = to_manifold(b.clean())
        return from_manifold(ma ^ mb)
    except Exception:
        # Fallback to VTK boolean operations
        return a.clean().triangulate().boolean_intersection(b.clean().triangulate())
