import logging

import numpy as np
import pyvista as pv

from aerosol3d.physics.units import Q_

logger = logging.getLogger(__name__)


def from_file(file_path: str, unit: str = "nm",
              source_unit: str = None) -> pv.PolyData:
    """Load a 3D mesh file (STL, OBJ, PLY, etc.) as pv.PolyData.

    Args:
        file_path: Path to the mesh file.
        unit: Target unit for the returned mesh.
        source_unit: Unit used in the mesh file. Defaults to unit if None.

    Returns:
        pv.PolyData with scaled vertices.
    """
    mesh = pv.read(file_path)

    if isinstance(mesh, pv.MultiBlock):
        if len(mesh) == 0:
            raise ValueError(f"No geometry found in file: {file_path}")
        mesh = mesh.combine()

    if source_unit is None:
        if hasattr(mesh, 'units') and mesh.units:
            source_unit = mesh.units
        else:
            source_unit = unit

    if source_unit != unit:
        try:
            factor = Q_(1.0, source_unit).to(Q_(1.0, unit)).magnitude
            if not np.isclose(factor, 1.0):
                mesh.points *= factor
        except Exception:
            logger.warning("Unit conversion failed: %s -> %s", source_unit, unit)

    return mesh