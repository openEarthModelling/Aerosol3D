import numpy as np

from aerosol3d.geometry.boolean import safe_difference
from aerosol3d.geometry.primitives import create_sphere


def apply_cam_coating(particle, target_f_bc: float, material):
    """Apply Coated-Aggregate Model coating (Liu 2025).

    Encapsulates the entire aggregate in a spherical envelope.

    Args:
        particle: AerosolParticle with core blocks.
        target_f_bc: Target volume fraction of BC (0.0 < f_bc < 1.0).
        material: Material for the coating.

    Returns:
        The modified AerosolParticle.
    """
    if not 0 < target_f_bc < 1:
        raise ValueError("target_f_bc must be between 0 and 1 (exclusive).")

    from aerosol3d.core.particle import MixingState

    core_mesh = particle.combined
    total_volume = core_mesh.volume
    target_volume = total_volume / target_f_bc
    R = (3.0 * target_volume / (4.0 * np.pi)) ** (1.0 / 3.0)

    center = core_mesh.center
    envelope = create_sphere(center=center, radius=R)
    coating = safe_difference(envelope, core_mesh)

    particle.add_mesh("coating", coating, material)
    particle.mixing_state = MixingState.COATED
    return particle