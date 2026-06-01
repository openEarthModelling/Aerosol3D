"""Tests for Aerosol3D.bulk.interpolation physics-safe PCHIP kernels."""

from __future__ import annotations

import numpy as np
import pytest


class TestLogLogPCHIPInterpolator:
    def test_reconstructs_power_law_exactly(self):
        from Aerosol3D.bulk.interpolation import LogLogPCHIPInterpolator

        radii = np.array([10.0, 50.0, 100.0, 500.0, 1000.0])
        values = 2.5 * radii**2.0  # C ∝ r^2
        interp = LogLogPCHIPInterpolator(radii, values)

        r_test = np.array([20.0, 100.0, 750.0])
        expected = 2.5 * r_test**2.0
        result = interp(r_test)
        assert np.allclose(result, expected, rtol=1e-10)

    def test_never_produces_negative(self):
        from Aerosol3D.bulk.interpolation import LogLogPCHIPInterpolator

        radii = np.array([10.0, 50.0, 100.0, 500.0, 1000.0])
        values = np.array([1e-10, 1e-5, 1e-3, 1.0, 1e2])
        interp = LogLogPCHIPInterpolator(radii, values)

        r_test = np.logspace(0, 5, 1000)
        result = interp(r_test)
        assert np.all(result > 0.0)

    def test_scalar_input(self):
        from Aerosol3D.bulk.interpolation import LogLogPCHIPInterpolator

        radii = np.array([10.0, 50.0, 100.0])
        values = np.array([1.0, 4.0, 9.0])
        interp = LogLogPCHIPInterpolator(radii, values)

        result = interp(50.0)
        assert np.isscalar(result)
        assert result == pytest.approx(4.0, rel=1e-10)

    def test_extrapolation_below_returns_min(self):
        from Aerosol3D.bulk.interpolation import LogLogPCHIPInterpolator

        radii = np.array([10.0, 50.0, 100.0])
        values = np.array([1.0, 4.0, 9.0])
        interp = LogLogPCHIPInterpolator(radii, values)

        result = interp(1.0)
        assert result == pytest.approx(1.0, rel=1e-10)

    def test_extrapolation_above_returns_max(self):
        from Aerosol3D.bulk.interpolation import LogLogPCHIPInterpolator

        radii = np.array([10.0, 50.0, 100.0])
        values = np.array([1.0, 4.0, 9.0])
        interp = LogLogPCHIPInterpolator(radii, values)

        result = interp(1000.0)
        assert result == pytest.approx(9.0, rel=1e-10)

    def test_vectorized_input(self):
        from Aerosol3D.bulk.interpolation import LogLogPCHIPInterpolator

        radii = np.array([10.0, 50.0, 100.0])
        values = np.array([1.0, 4.0, 9.0])
        interp = LogLogPCHIPInterpolator(radii, values)

        r_test = np.array([20.0, 50.0, 80.0])
        result = interp(r_test)
        assert isinstance(result, np.ndarray)
        assert result.shape == r_test.shape

    def test_invalid_non_positive_radii_raises(self):
        from Aerosol3D.bulk.interpolation import LogLogPCHIPInterpolator

        with pytest.raises(ValueError, match="radii"):
            LogLogPCHIPInterpolator(np.array([-1.0, 10.0, 20.0]), np.array([1.0, 2.0, 3.0]))

    def test_invalid_non_positive_values_raises(self):
        from Aerosol3D.bulk.interpolation import LogLogPCHIPInterpolator

        with pytest.raises(ValueError, match="values"):
            LogLogPCHIPInterpolator(np.array([1.0, 10.0, 20.0]), np.array([0.0, 2.0, 3.0]))


class TestLinearPCHIPInterpolator:
    def test_interpolates_beta_correctly(self):
        from Aerosol3D.bulk.interpolation import LinearPCHIPInterpolator

        radii = np.array([10.0, 50.0, 100.0])
        values = np.array([0.1, 0.5, 1.0])
        interp = LinearPCHIPInterpolator(radii, values)

        result = interp(50.0)
        assert result == pytest.approx(0.5, rel=1e-10)

    def test_scalar_input(self):
        from Aerosol3D.bulk.interpolation import LinearPCHIPInterpolator

        radii = np.array([10.0, 50.0, 100.0])
        values = np.array([0.1, 0.5, 1.0])
        interp = LinearPCHIPInterpolator(radii, values)

        result = interp(50.0)
        assert np.isscalar(result)
        assert result == pytest.approx(0.5, rel=1e-10)

    def test_extrapolation_below_returns_min(self):
        from Aerosol3D.bulk.interpolation import LinearPCHIPInterpolator

        radii = np.array([10.0, 50.0, 100.0])
        values = np.array([0.1, 0.5, 1.0])
        interp = LinearPCHIPInterpolator(radii, values)

        result = interp(1.0)
        assert result == pytest.approx(0.1, rel=1e-10)

    def test_extrapolation_above_returns_max(self):
        from Aerosol3D.bulk.interpolation import LinearPCHIPInterpolator

        radii = np.array([10.0, 50.0, 100.0])
        values = np.array([0.1, 0.5, 1.0])
        interp = LinearPCHIPInterpolator(radii, values)

        result = interp(1000.0)
        assert result == pytest.approx(1.0, rel=1e-10)

    def test_vectorized_input(self):
        from Aerosol3D.bulk.interpolation import LinearPCHIPInterpolator

        radii = np.array([10.0, 50.0, 100.0])
        values = np.array([0.1, 0.5, 1.0])
        interp = LinearPCHIPInterpolator(radii, values)

        r_test = np.array([20.0, 50.0, 80.0])
        result = interp(r_test)
        assert isinstance(result, np.ndarray)
        assert result.shape == r_test.shape
