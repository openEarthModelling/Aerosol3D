# tests/test_integration.py
import numpy as np
import pyvista as pv
import pytest


class TestFullPipeline:
    def test_sphere_coating_export(self, tmp_path):
        """Create soot sphere → apply distance coating → save VTP."""
        from aerosol3d import (
            AerosolParticle, Material, MixingState,
            create_sphere, apply_distance_coating, save_vtp
        )

        soot = Material(name="soot", refractive_index=complex(1.8, 0.7), density=1.8)
        sulfate = Material(name="sulfate", refractive_index=complex(1.4, 0.0), density=1.8)

        p = AerosolParticle(name="coated_sphere", unit="nm")
        core = create_sphere(center=(0, 0, 0), radius=50.0)
        p.add_mesh("bc_core", core, soot)

        apply_distance_coating(p, thickness=10.0, material=sulfate)
        assert p.mixing_state == MixingState.COATED

        filepath = str(tmp_path / "coated.vtp")
        save_vtp(p, filepath)
        loaded = pv.read(filepath)
        assert loaded.n_points > 0

    def test_fractal_aggregate_ccm(self, tmp_path):
        """Create fractal aggregate → apply CCM → verify structure."""
        from aerosol3d import (
            AerosolParticle, Material, MixingState, FractalAggregate,
            apply_ccm_coating, save_vtp
        )

        soot = Material(name="soot", refractive_index=complex(1.8, 0.7), density=1.8)
        sulfate = Material(name="sulfate", refractive_index=complex(1.4, 0.0), density=1.8)

        rng = np.random.default_rng(42)
        agg = FractalAggregate(
            centers=rng.random((20, 3)) * 200,
            radii=np.full(20, 25.0),
            material=soot
        )

        p = AerosolParticle(name="ccm_aggregate", unit="nm")
        p.add_mesh("bc_core", agg.to_mesh(theta_res=10, phi_res=10), soot)

        apply_ccm_coating(p, f_bc=0.5, material=sulfate)
        assert p.mixing_state == MixingState.COATED
        assert "coating" in p.blocks

    def test_all_four_coatings(self, tmp_path):
        """Verify all 4 coating algorithms run without errors on a sphere."""
        from aerosol3d import (
            AerosolParticle, Material,
            create_sphere,
            apply_distance_coating, apply_potential_void_coating,
            apply_potential_edge_coating,
            apply_ccm_coating, apply_cam_coating,
        )

        soot = Material(name="soot", refractive_index=complex(1.8, 0.7), density=1.8)
        sulfate = Material(name="sulfate", refractive_index=complex(1.4, 0.0), density=1.8)

        for coat_fn in [apply_distance_coating, apply_ccm_coating, apply_cam_coating]:
            p = AerosolParticle(name="test", unit="nm")
            p.add_mesh("bc_core", create_sphere((0, 0, 0), 50.0), soot)
            if coat_fn == apply_distance_coating:
                coat_fn(p, thickness=10.0, material=sulfate)
            else:
                coat_fn(p, target_f_bc=0.5, material=sulfate)
            assert "coating" in p.blocks

        # Potential void coating with lower resolution for speed
        p = AerosolParticle(name="test", unit="nm")
        p.add_mesh("bc_core", create_sphere((0, 0, 0), 50.0), soot)
        apply_potential_void_coating(
            p, coated_area_fraction=0.5, dp_dc_ratio=1.5,
            material=sulfate, resolution=20)
        assert "coating" in p.blocks

        # Potential edge coating with lower resolution for speed
        p = AerosolParticle(name="test", unit="nm")
        p.add_mesh("bc_core", create_sphere((0, 0, 0), 50.0), soot)
        apply_potential_edge_coating(
            p, coated_area_fraction=0.5, dp_dc_ratio=1.5,
            material=sulfate, resolution=20)
        assert "coating" in p.blocks