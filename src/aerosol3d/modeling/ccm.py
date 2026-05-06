import numpy as np

from aerosol3d.geometry.boolean import safe_difference


def apply_ccm_coating(particle, f_bc: float, material):
    """Apply Closed-Cell Model coating (Liu 2025).

    Each core component is uniformly scaled by (1/f_bc)^(1/3),
    then boolean-differenced with the original to extract the coating shell.

    Args:
        particle: AerosolParticle with core blocks.
        f_bc: Target volume fraction of BC (0.0 < f_bc < 1.0).
        material: Material for the coating.

    Returns:
        The modified AerosolParticle.
    """
    if not 0 < f_bc < 1:
        raise ValueError("f_bc must be between 0 and 1 (exclusive).")

    from aerosol3d.core.particle import MixingState

    core_mesh = particle.combined
    scale = (1.0 / f_bc) ** (1.0 / 3.0)
    center = core_mesh.center

    expanded = core_mesh.copy()
    expanded.points = (expanded.points - center) * scale + center

    coating = safe_difference(expanded, core_mesh)
    particle.add_mesh("coating", coating, material)
    particle.mixing_state = MixingState.COATED
    return particle