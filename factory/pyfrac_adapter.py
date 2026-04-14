import numpy as np

from aerosol3d.core.aggregate import FractalAggregate
from aerosol3d.physics.units import Q_


def from_pyfrac(aggregate_obj, material, target_unit: str = "nm") -> FractalAggregate:
    """Convert a pyFracAggregate object to a FractalAggregate.

    Args:
        aggregate_obj: Object with .positions (N,3) and .radii (N,) attributes.
        material: Material instance.
        target_unit: Target unit for coordinates and radii.

    Returns:
        FractalAggregate with converted units.
    """
    centers = np.asarray(aggregate_obj.positions, dtype=float)
    radii = np.asarray(aggregate_obj.radii, dtype=float)
    source_unit = getattr(aggregate_obj, 'length_unit', 'nm')

    if source_unit != target_unit:
        try:
            factor = Q_(1.0, source_unit).to(Q_(1.0, target_unit)).magnitude
            centers = centers * factor
            radii = radii * factor
        except Exception:
            pass

    return FractalAggregate(
        centers=centers, radii=radii, material=material, unit=target_unit
    )
