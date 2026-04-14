from aerosol3d.geometry.voxelize import voxelize_with_materials


def save_vtp(particle, filename: str):
    """Save particle as .vtp (combined) or .vtmb (multi-block)."""
    particle.save(filename)


def save_voxel(particle, filename: str, voxel_size: float):
    """Save particle as voxelized ImageData (.vti)."""
    grid = voxelize_with_materials(particle, voxel_size)
    grid.save(filename)