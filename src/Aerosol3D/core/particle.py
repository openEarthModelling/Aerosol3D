from enum import Enum, auto

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

    def __init__(
        self,
        name: str = "AerosolParticle",
        mixing_state: MixingState = MixingState.INTERNAL,
        unit: str = "nm",
    ):
        self.name = name
        self.mixing_state = mixing_state
        self.unit = unit
        self._blocks: dict[str, pv.PolyData] = {}

    @property
    def blocks(self) -> dict[str, pv.PolyData]:
        return self._blocks

    def add_mesh(self, name: str, mesh: pv.PolyData, material) -> None:
        """Add a geometric component with material tagging."""
        mesh = mesh.copy()
        mesh.cell_data["material_id"] = np.full(mesh.n_cells, material.id, dtype=np.int32)
        mesh.cell_data["ri_n"] = np.full(mesh.n_cells, material.refractive_index.real)
        mesh.cell_data["ri_k"] = np.full(mesh.n_cells, material.refractive_index.imag)
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

    @staticmethod
    def _block_volume(block: pv.PolyData) -> float:
        """Return the geometric volume of a block.

        For closed surface meshes (e.g. spheres) ``PolyData.volume`` gives the
        enclosed volume.  For volumetric meshes ``compute_cell_sizes()`` is used.
        """
        sized = block.compute_cell_sizes()
        vol = float(sized.cell_data["Volume"].sum())
        if vol > 0:
            return vol
        # Fallback for closed surface meshes
        volume = float(block.volume)
        if volume < 0:
            import logging

            logging.getLogger(__name__).warning(
                "Block '%s' has negative volume (%.3f). Check mesh orientation.",
                getattr(block, "field_data", {}).get("role", ["unknown"])[0]
                if hasattr(block, "field_data")
                else "unknown",
                volume,
            )
        return abs(volume)

    @property
    def equivalent_diameter(self) -> float:
        """Return the diameter of a sphere with the same total volume as all blocks.

        Note: This returns a raw geometric value. Unit conversion is the caller's responsibility.
        """
        total_volume = 0.0
        for block in self._blocks.values():
            if block is None:
                continue
            total_volume += self._block_volume(block)
        if total_volume <= 0:
            raise ValueError("Particle has no volume.")
        return (6.0 * total_volume / np.pi) ** (1.0 / 3.0)

    def effective_refractive_index(self, method: str = "volume_weighted") -> complex:
        """Compute effective refractive index using an EMA method.

        Args:
            method: One of 'volume_weighted', 'maxwell_garnett', 'bruggeman'.

        Returns:
            Effective complex refractive index.
        """
        from .ema import bruggeman, maxwell_garnett, volume_weighted

        volumes, ri_list = self._material_data()
        if method == "volume_weighted":
            return volume_weighted(volumes, ri_list)
        elif method == "maxwell_garnett":
            return maxwell_garnett(volumes, ri_list)
        elif method == "bruggeman":
            return bruggeman(volumes, ri_list)
        else:
            raise ValueError(
                f"Unknown EMA method '{method}'. "
                "Available: volume_weighted, maxwell_garnett, bruggeman"
            )

    def _material_data(self) -> tuple[list[float], list[complex]]:
        """Extract [(volume, ri), ...] from all blocks."""
        volumes = []
        ri_list = []
        for block in self._blocks.values():
            if block is None:
                continue
            ri = complex(
                float(np.mean(block.cell_data["ri_n"])),
                float(np.mean(block.cell_data["ri_k"])),
            )
            vol = self._block_volume(block)
            volumes.append(vol)
            ri_list.append(ri)
        return volumes, ri_list

    def __repr__(self) -> str:
        return (
            f"AerosolParticle(name={self.name!r}, "
            f"blocks={len(self._blocks)}, "
            f"mixing_state={self.mixing_state.name})"
        )
