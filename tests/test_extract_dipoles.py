# tests/test_extract_dipoles.py
import numpy as np
import pyvista as pv


def _make_simple_grid(n_voxels_per_side=5, spacing=10.0):
    """Create a grid where center voxel has material_id=1."""
    dims = n_voxels_per_side + 1
    grid = pv.ImageData(
        dimensions=(dims, dims, dims),
        spacing=(spacing, spacing, spacing),
        origin=(-25, -25, -25),
    )
    mat_ids = np.zeros(grid.n_cells, dtype=np.int32)
    center_idx = (
        (n_voxels_per_side // 2) * n_voxels_per_side * n_voxels_per_side
        + (n_voxels_per_side // 2) * n_voxels_per_side
        + (n_voxels_per_side // 2)
    )
    mat_ids[center_idx] = 1
    grid.cell_data["material_id"] = mat_ids
    return grid


class TestLDRPolarizabilityUnit:
    def test_ldr_s_parameter(self):
        from Aerosol3D.optics.dda_solver import _ldr_s

        # Propagation along z, x-polarization: S = 0
        assert _ldr_s((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)) == 0.0
        assert _ldr_s((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)) == 0.0
        # Propagation along z, z-polarization: S = 1
        assert _ldr_s((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)) == 1.0

    def test_compute_polarizability_single_material(self):
        """LDR alpha_e should be finite complex array for a simple grid."""
        from Aerosol3D.optics.datastructs import SimulationConfig
        from Aerosol3D.optics.dda_solver import _compute_polarizability

        grid = _make_simple_grid(n_voxels_per_side=5, spacing=10.0)
        material_map = {1: type("Mat", (), {"refractive_index": complex(1.8, 0.7)})()}

        config = SimulationConfig(wavelength=550.0, dipole_spacing=10.0)
        config.polarization = (1.0, 0.0, 0.0)

        alpha_e = _compute_polarizability(grid, config, material_map)
        assert alpha_e.ndim == 1
        assert len(alpha_e) == 1
        assert np.all(np.isfinite(alpha_e))
        assert alpha_e.dtype == np.complex128

    def test_compute_polarizability_multiple_voxels(self):
        """Multiple voxels with same material should get same alpha_e."""
        from Aerosol3D.optics.datastructs import SimulationConfig
        from Aerosol3D.optics.dda_solver import _compute_polarizability

        grid = _make_simple_grid(n_voxels_per_side=5, spacing=10.0)
        mat_ids = grid.cell_data["material_id"]
        mat_ids[0] = 1
        mat_ids[1] = 1
        grid.cell_data["material_id"] = mat_ids

        material_map = {1: type("Mat", (), {"refractive_index": complex(1.5, 0.0)})()}

        config = SimulationConfig(wavelength=550.0, dipole_spacing=10.0)
        config.polarization = (1.0, 0.0, 0.0)

        alpha_e = _compute_polarizability(grid, config, material_map)
        assert len(alpha_e) == 3
        # Same material, same alpha_e for all voxels
        np.testing.assert_allclose(alpha_e[0], alpha_e[1])

    def test_ldr_reduces_to_cm_for_small_kd(self):
        """As kd -> 0, LDR correction vanishes and alpha_LDR -> alpha_CM (unnormalized)."""
        from Aerosol3D.optics.datastructs import SimulationConfig
        from Aerosol3D.optics.dda_solver import _compute_polarizability

        grid = _make_simple_grid(n_voxels_per_side=5, spacing=10.0)
        material_map = {1: type("Mat", (), {"refractive_index": complex(1.5, 0.01)})()}

        # Very large wavelength -> kd very small -> LDR ~ CM
        config = SimulationConfig(wavelength=100000.0, dipole_spacing=10.0)
        config.polarization = (1.0, 0.0, 0.0)

        alpha_e = _compute_polarizability(grid, config, material_map)

        # Compute expected CM polarizability (unnormalized by k³/(4π))
        d = 10.0
        eps = complex(1.5, 0.01) ** 2
        a_cm = 3.0 * d**3 * (eps - 1.0) / (eps + 2.0)
        k = 2.0 * np.pi / 100000.0
        expected = (k**3 / (4.0 * np.pi)) * a_cm

        np.testing.assert_allclose(alpha_e[0], expected, rtol=1e-4)
