# tests/test_extract_dipoles.py
import numpy as np
import pyvista as pv
import pytest


def _make_simple_grid(n_voxels_per_side=5, spacing=10.0):
    """Create a grid where center voxel has material_id=1."""
    dims = n_voxels_per_side + 1
    grid = pv.ImageData(
        dimensions=(dims, dims, dims),
        spacing=(spacing, spacing, spacing),
        origin=(-25, -25, -25),
    )
    mat_ids = np.zeros(grid.n_cells, dtype=np.int32)
    center_idx = (n_voxels_per_side // 2) * n_voxels_per_side * n_voxels_per_side \
               + (n_voxels_per_side // 2) * n_voxels_per_side \
               + (n_voxels_per_side // 2)
    mat_ids[center_idx] = 1
    grid.cell_data["material_id"] = mat_ids
    return grid


class TestApplyRadiativeCorrection:
    def test_known_value(self):
        from Aerosol3D.optics.dda_solver import _apply_radiative_correction
        alpha0 = 1.0 + 0.0j
        k = 0.01
        result = _apply_radiative_correction(alpha0, k)
        assert abs(result) > 0
        assert isinstance(result, complex)

    def test_absorbing_material(self):
        from Aerosol3D.optics.dda_solver import _apply_radiative_correction
        alpha0 = 1.0 + 0.5j
        k = 0.1
        result = _apply_radiative_correction(alpha0, k)
        assert result.imag != 0

    def test_array_input(self):
        from Aerosol3D.optics.dda_solver import _apply_radiative_correction
        alpha0 = np.array([1.0 + 0.1j, 2.0 + 0.3j])
        k = 0.1
        result = _apply_radiative_correction(alpha0, k)
        assert result.shape == (2,)
        assert result.dtype == np.complex128


class TestExtractDipoles:
    def test_single_voxel(self):
        from Aerosol3D.optics.dda_solver import _extract_dipoles
        from Aerosol3D.optics.datastructs import SimulationConfig
        grid = _make_simple_grid(n_voxels_per_side=5, spacing=10.0)

        config = SimulationConfig(wavelength=550.0, dipole_spacing=10.0)
        material_map = {
            1: type("Mat", (), {"refractive_index": complex(1.8, 0.7)})()
        }

        positions, alpha_e = _extract_dipoles(grid, config, material_map)
        assert positions.shape == (1, 3)
        assert alpha_e.shape == (1,)
        assert alpha_e.dtype == np.complex128
        assert positions.dtype == np.float64

    def test_multiple_voxels(self):
        from Aerosol3D.optics.dda_solver import _extract_dipoles
        from Aerosol3D.optics.datastructs import SimulationConfig
        grid = _make_simple_grid(n_voxels_per_side=5, spacing=10.0)
        mat_ids = grid.cell_data["material_id"]
        mat_ids[0] = 1
        mat_ids[1] = 1
        grid.cell_data["material_id"] = mat_ids

        config = SimulationConfig(wavelength=550.0, dipole_spacing=10.0)
        material_map = {
            1: type("Mat", (), {"refractive_index": complex(1.5, 0.0)})()
        }

        positions, alpha_e = _extract_dipoles(grid, config, material_map)
        assert positions.shape == (3, 3)  # 3 set above
        assert alpha_e.shape == (3,)

    def test_type_safety(self):
        from Aerosol3D.optics.dda_solver import _extract_dipoles
        from Aerosol3D.optics.datastructs import SimulationConfig
        grid = _make_simple_grid()
        config = SimulationConfig(wavelength=550.0, dipole_spacing=10.0)
        material_map = {
            1: type("Mat", (), {"refractive_index": complex(1.8, 0.7)})()
        }
        positions, alpha_e = _extract_dipoles(grid, config, material_map)
        assert positions.flags["C_CONTIGUOUS"]
        assert alpha_e.flags["C_CONTIGUOUS"]