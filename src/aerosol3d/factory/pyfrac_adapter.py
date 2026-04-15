import logging

import numpy as np

from aerosol3d.core.aggregate import FractalAggregate
from aerosol3d.physics.units import Q_

logger = logging.getLogger(__name__)


def from_fractal(aggregate_obj, material, target_unit: str = "nm") -> FractalAggregate:
    """Convert a pyFracAggregate Aggregate object to a FractalAggregate.

    The input must have .positions (N,3) and .radii (N,) attributes, as provided
    by the pyFracAggregate library.

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
            logger.warning("Unit conversion failed: %s -> %s", source_unit, target_unit)

    return FractalAggregate(
        centers=centers, radii=radii, material=material, unit=target_unit
    )
