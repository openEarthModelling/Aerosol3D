import numpy as np
import pyvista as pv
import pytest

from Aerosol3D.core.particle import AerosolParticle, MixingState


class TestFractalAggregate:
    def test_create(self, soot_material):
        from Aerosol3D.core.aggregate import FractalAggregate
        centers = np.zeros((5, 3))
        radii = np.full(5, 25.0)
        agg = FractalAggregate(centers=centers, radii=radii, material=soot_material)
        assert agg.centers.shape == (5, 3)
        assert agg.radii.shape == (5,)
        assert agg.material.name == "soot"

    def test_volume(self, soot_material):
        from Aerosol3D.core.aggregate import FractalAggregate
        centers = np.zeros((3, 3))
        radii = np.array([10.0, 20.0, 30.0])
        agg = FractalAggregate(centers=centers, radii=radii, material=soot_material)
        expected = 4/3 * np.pi * (10**3 + 20**3 + 30**3)
        assert abs(agg.volume - expected) < 1e-6

    def test_volume_weighted_center(self, soot_material):
        from Aerosol3D.core.aggregate import FractalAggregate
        centers = np.array([[0, 0, 0], [100, 0, 0]], dtype=float)
        radii = np.array([20.0, 20.0])  # equal weight → midpoint
        agg = FractalAggregate(centers=centers, radii=radii, material=soot_material)
        c = agg.volume_weighted_center
        assert abs(c[0] - 50.0) < 1e-6

    def test_to_mesh(self, soot_material):
        from Aerosol3D.core.aggregate import FractalAggregate
        centers = np.random.default_rng(42).random((5, 3)) * 100
        radii = np.full(5, 10.0)
        agg = FractalAggregate(centers=centers, radii=radii, material=soot_material)
        mesh = agg.to_mesh(theta_res=8, phi_res=8)
        assert isinstance(mesh, pv.PolyData)
        assert mesh.n_points > 0

    def test_to_particle(self, soot_material):
        from Aerosol3D.core.aggregate import FractalAggregate
        centers = np.random.default_rng(42).random((5, 3)) * 100
        radii = np.full(5, 10.0)
        agg = FractalAggregate(centers=centers, radii=radii, material=soot_material)
        particle = agg.to_particle()

        assert isinstance(particle, AerosolParticle)
        assert particle.name == "fractal_aggregate"
        assert particle.mixing_state == MixingState.AGGREGATED
        assert particle.unit == "nm"
        assert "aggregate" in particle.blocks
        assert particle.blocks["aggregate"].n_cells > 0

    def test_to_particle_custom_name(self, soot_material):
        from Aerosol3D.core.aggregate import FractalAggregate
        centers = np.zeros((3, 3))
        radii = np.full(3, 20.0)
        agg = FractalAggregate(centers=centers, radii=radii, material=soot_material)
        particle = agg.to_particle(name="my_custom_particle")

        assert isinstance(particle, AerosolParticle)
        assert particle.name == "my_custom_particle"
        assert particle.mixing_state == MixingState.AGGREGATED