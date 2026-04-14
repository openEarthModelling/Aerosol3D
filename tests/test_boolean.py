import numpy as np
import pyvista as pv
import pytest


class TestToFromManifold:
    def test_round_trip_preserves_topology(self):
        from aerosol3d.geometry.boolean import to_manifold, from_manifold
        sphere = pv.Sphere(radius=50.0)
        m = to_manifold(sphere)
        result = from_manifold(m)
        assert isinstance(result, pv.PolyData)
        assert result.n_points > 0
        assert result.n_cells > 0

    def test_round_trip_preserves_volume(self):
        from aerosol3d.geometry.boolean import to_manifold, from_manifold
        sphere = pv.Sphere(radius=50.0)
        original_vol = sphere.volume
        result = from_manifold(to_manifold(sphere))
        assert abs(result.volume - original_vol) / original_vol < 0.01


class TestSafeDifference:
    def test_sphere_minus_smaller_sphere(self):
        from aerosol3d.geometry.boolean import safe_difference
        outer = pv.Sphere(radius=50.0)
        inner = pv.Sphere(radius=30.0)
        shell = safe_difference(outer, inner)
        assert isinstance(shell, pv.PolyData)
        assert shell.volume > 0
        expected = 4/3 * np.pi * (50**3 - 30**3)
        assert abs(shell.volume - expected) / expected < 0.05

    def test_non_overlapping_returns_original(self):
        from aerosol3d.geometry.boolean import safe_difference
        sphere = pv.Sphere(radius=50.0, center=(0, 0, 0))
        far_away = pv.Sphere(radius=10.0, center=(200, 200, 200))
        result = safe_difference(sphere, far_away)
        assert result.n_cells > 0


class TestSafeUnion:
    def test_two_overlapping_spheres(self):
        from aerosol3d.geometry.boolean import safe_union
        s1 = pv.Sphere(radius=50.0, center=(-25, 0, 0))
        s2 = pv.Sphere(radius=50.0, center=(25, 0, 0))
        unioned = safe_union(s1, s2)
        assert unioned.n_cells > 0
        assert unioned.volume < s1.volume + s2.volume
        assert unioned.volume > max(s1.volume, s2.volume)