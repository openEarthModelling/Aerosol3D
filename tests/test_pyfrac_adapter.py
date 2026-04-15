import numpy as np
import pytest


class TestFractalAdapter:
    def test_returns_fractal_aggregate(self, soot_material):
        from aerosol3d.factory.pyfrac_adapter import from_fractal

        # Mock pyfrac aggregate object
        class MockAggregate:
            positions = np.random.default_rng(42).random((10, 3)) * 200
            radii = np.full(10, 25.0)
            length_unit = "nm"

        agg = from_fractal(MockAggregate(), material=soot_material)
        assert agg.n_monomers == 10
        assert agg.centers.shape == (10, 3)

    def test_unit_conversion(self, soot_material):
        from aerosol3d.factory.pyfrac_adapter import from_fractal

        class MockAggregate:
            positions = np.array([[100.0, 0.0, 0.0]])
            radii = np.array([50.0])
            length_unit = "um"

        agg = from_fractal(MockAggregate(), material=soot_material,
                          target_unit="nm")
        # Should have been converted from um to nm
        assert agg.unit == "nm"