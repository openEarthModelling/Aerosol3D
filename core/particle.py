from enum import Enum, auto
from typing import Dict, Optional

import numpy as np
import pyvista as pv


class MixingState(Enum):
    INTERNAL = auto()
    EXTERNAL = auto()
    COATED = auto()
    AGGREGATED = auto()


class AerosolParticle:
    """Aerosol particle container holding named geometry blocks.

    Each block represents a geometric component (core, coating, etc.)
    tagged with material properties via cell_data and field_data.
    """

    def __init__(self, name: str = "AerosolParticle",
                 mixing_state: MixingState = MixingState.INTERNAL,
                 unit: str = "nm"):
        self.name = name
        self.mixing_state = mixing_state
        self.unit = unit
        self._blocks: Dict[str, pv.PolyData] = {}

    @property
    def blocks(self) -> Dict[str, pv.PolyData]:
        return self._blocks

    def add_mesh(self, name: str, mesh: pv.PolyData, material) -> None:
        """Add a geometric component with material tagging."""
        mesh = mesh.copy()
        mesh.cell_data["material_id"] = np.full(
            mesh.n_cells, material.id, dtype=np.int32)
        mesh.cell_data["ri_n"] = np.full(
            mesh.n_cells, material.refractive_index.real)
        mesh.cell_data["ri_k"] = np.full(
            mesh.n_cells, material.refractive_index.imag)
        mesh.field_data["role"] = [name]
        mesh.field_data["material_name"] = [material.name]
        mesh.field_data["material_id"] = [material.id]
        self._blocks[name] = mesh

    @property
    def combined(self) -> pv.PolyData:
        """Merge all blocks into a single PolyData."""
        meshes = [b for b in self._blocks.values() if b is not None]
        if not meshes:
            raise ValueError("No blocks to combine.")
        result = meshes[0].copy()
        for m in meshes[1:]:
            result = result + m
        return result

    def save(self, filename: str) -> None:
        """Export to .vtmb or .vtp."""
        if filename.endswith(".vtmb"):
            mb = pv.MultiBlock(self._blocks)
            mb.save(filename)
        else:
            self.combined.save(filename)

    def __repr__(self) -> str:
        return (f"AerosolParticle(name={self.name!r}, "
                f"blocks={len(self._blocks)}, "
                f"mixing_state={self.mixing_state.name})")
