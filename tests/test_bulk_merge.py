"""Tests for Aerosol3D.bulk.merge — Method 1 (bin weights) and Method 2 (continuous)."""

from __future__ import annotations

import numpy as np
import pytest


class TestComputeBinWeights:
    def test_weights_sum_to_one(self):
        from Aerosol3D.bulk.datastructs import SizeDistribution
        from Aerosol3D.bulk.merge import compute_bin_weights

        sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.5)
        radii = np.array([50.0, 100.0, 200.0, 400.0])
        weights = compute_bin_weights(radii, sd)
        assert weights.sum() == pytest.approx(1.0, abs=1e-12)
        assert weights.shape == (4,)

    def test_weights_non_negative(self):
        from Aerosol3D.bulk.datastructs import SizeDistribution
        from Aerosol3D.bulk.merge import compute_bin_weights

        sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.5)
        radii = np.array([50.0, 100.0, 200.0, 400.0])
        weights = compute_bin_weights(radii, sd)
        assert np.all(weights >= 0.0)

    def test_unsorted_radii_sorted(self):
        from Aerosol3D.bulk.datastructs import SizeDistribution
        from Aerosol3D.bulk.merge import compute_bin_weights

        sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.5)
        radii = np.array([400.0, 50.0, 200.0, 100.0])
        weights = compute_bin_weights(radii, sd)
        assert weights.sum() == pytest.approx(1.0, abs=1e-12)

    def test_single_radius_raises(self):
        from Aerosol3D.bulk.datastructs import SizeDistribution
        from Aerosol3D.bulk.merge import compute_bin_weights

        sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.5)
        radii = np.array([100.0])
        with pytest.raises(ValueError, match="at least two radii"):
            compute_bin_weights(radii, sd)

    def test_weights_sum_verification_raises_on_bad_distribution(self):
        """If number_in_interval doesn't sum to ~1, raise RuntimeError."""
        from Aerosol3D.bulk.merge import compute_bin_weights

        # Create a mock size distribution whose number_in_interval
        # returns values that don't sum to 1
        class BadSD:
            def number_in_interval(self, left, right):
                # Always return 0.1 regardless of bounds
                return 0.1

        radii = np.array([50.0, 100.0, 200.0])
        with pytest.raises(RuntimeError, match="weights do not sum"):
            compute_bin_weights(radii, BadSD())


class TestMergeMethod1:
    def test_equal_weights_reproduce_arithmetic_mean(self):
        from Aerosol3D.bulk.merge import merge_method1

        n_radius, n_wavelength, n_legendre = 3, 2, 4
        C_ext = np.ones((n_radius, n_wavelength))
        C_sca = np.ones((n_radius, n_wavelength))
        beta = np.ones((n_radius, n_wavelength, n_legendre))
        weights = np.ones(n_radius) / n_radius

        bulk_C_ext, bulk_C_sca, bulk_beta = merge_method1(C_ext, C_sca, beta, weights)

        np.testing.assert_allclose(bulk_C_ext, np.ones(n_wavelength))
        np.testing.assert_allclose(bulk_C_sca, np.ones(n_wavelength))
        np.testing.assert_allclose(bulk_beta, np.ones((n_wavelength, n_legendre)))

    def test_beta0_always_one(self):
        from Aerosol3D.bulk.merge import merge_method1

        np.random.seed(42)
        n_radius, n_wavelength, n_legendre = 5, 3, 6
        C_ext = np.random.rand(n_radius, n_wavelength) + 0.1
        C_sca = np.random.rand(n_radius, n_wavelength) + 0.1
        beta = np.random.rand(n_radius, n_wavelength, n_legendre)
        weights = np.random.rand(n_radius)
        weights /= weights.sum()

        bulk_C_ext, bulk_C_sca, bulk_beta = merge_method1(C_ext, C_sca, beta, weights)

        np.testing.assert_allclose(bulk_beta[:, 0], 1.0, atol=1e-14)

    def test_weighted_sum_formula(self):
        from Aerosol3D.bulk.merge import merge_method1

        C_ext = np.array([[1.0], [3.0]])
        C_sca = np.array([[2.0], [4.0]])
        # beta shape: (n_radius, n_wavelength, n_legendre) = (2, 1, 2)
        beta = np.array(
            [
                [[1.0, 0.5]],  # radius 0
                [[1.0, 0.3]],  # radius 1
            ]
        )
        weights = np.array([0.25, 0.75])

        bulk_C_ext, bulk_C_sca, bulk_beta = merge_method1(C_ext, C_sca, beta, weights)

        # bulk_C_ext = 1*0.25 + 3*0.75 = 2.5
        assert bulk_C_ext[0] == pytest.approx(2.5)
        # bulk_C_sca = 2*0.25 + 4*0.75 = 3.5
        assert bulk_C_sca[0] == pytest.approx(3.5)
        # bulk_M_l[0,1] = 2*0.5*0.25 + 4*0.3*0.75 = 0.25 + 0.9 = 1.15
        # bulk_beta[0,1] = 1.15 / 3.5 = 0.32857...
        assert bulk_beta[0, 1] == pytest.approx(1.15 / 3.5, rel=1e-12)

    def test_zero_C_sca_handled(self):  # noqa: N802
        """If C_sca is all zero, beta should still have beta[:,0] = 1."""
        from Aerosol3D.bulk.merge import merge_method1

        n_radius, n_wavelength, n_legendre = 2, 1, 3
        C_ext = np.ones((n_radius, n_wavelength))
        C_sca = np.zeros((n_radius, n_wavelength))
        beta = np.ones((n_radius, n_wavelength, n_legendre))
        weights = np.array([0.5, 0.5])

        bulk_C_ext, bulk_C_sca, bulk_beta = merge_method1(C_ext, C_sca, beta, weights)

        np.testing.assert_allclose(bulk_C_ext, np.ones(n_wavelength))
        np.testing.assert_allclose(bulk_C_sca, np.zeros(n_wavelength))
        # beta[:, 0] should be forced to 1.0 even when C_sca is 0
        np.testing.assert_allclose(bulk_beta[:, 0], 1.0, atol=1e-14)

    def test_shape_mismatch_raises(self):
        from Aerosol3D.bulk.merge import merge_method1

        C_ext = np.ones((3, 2))
        C_sca = np.ones((3, 2))
        beta = np.ones((3, 2, 4))
        weights = np.ones(4) / 4  # mismatch: 4 vs 3

        with pytest.raises(ValueError, match="first dimension"):
            merge_method1(C_ext, C_sca, beta, weights)


class TestMergeMethod2:
    def test_dense_sampling_matches_method1(self):
        """Method 2 with many radii should converge to Method 1."""
        from Aerosol3D.bulk.datastructs import SizeDistribution
        from Aerosol3D.bulk.merge import compute_bin_weights, merge_method1, merge_method2

        np.random.seed(123)
        sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.3)
        n_wavelength, n_legendre = 2, 4

        # Use many radii spanning the distribution
        radii = np.logspace(1, 3, 64)  # 10 to 1000 nm

        # Synthetic optical properties: power-law-like
        C_ext = np.stack(
            [
                10.0 * (radii / 100.0) ** 1.5,
                5.0 * (radii / 100.0) ** 2.0,
            ],
            axis=1,
        )  # (64, 2)
        C_sca = 0.8 * C_ext

        # Synthetic beta: slightly varying with radius
        beta = np.zeros((len(radii), n_wavelength, n_legendre))
        for i, r in enumerate(radii):
            for j in range(n_wavelength):
                beta[i, j, 0] = 1.0
                beta[i, j, 1] = 0.8 + 0.1 * np.sin(np.log(r))
                beta[i, j, 2] = 0.3 + 0.05 * np.cos(np.log(r))
                beta[i, j, 3] = 0.05

        weights = compute_bin_weights(radii, sd)
        m1_C_ext, m1_C_sca, m1_beta = merge_method1(C_ext, C_sca, beta, weights)
        m2_C_ext, m2_C_sca, m2_beta = merge_method2(
            radii, C_ext, C_sca, beta, sd, n_quad=512, interpolation="pchip"
        )

        np.testing.assert_allclose(m2_C_ext, m1_C_ext, rtol=1e-2)
        np.testing.assert_allclose(m2_C_sca, m1_C_sca, rtol=1e-2)
        np.testing.assert_allclose(m2_beta, m1_beta, rtol=1e-2)

    def test_beta0_always_one_method2(self):
        from Aerosol3D.bulk.datastructs import SizeDistribution
        from Aerosol3D.bulk.merge import merge_method2

        np.random.seed(42)
        sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.5)
        n_radius, n_wavelength, n_legendre = 8, 2, 5
        radii = np.logspace(1, 3, n_radius)
        C_ext = np.random.rand(n_radius, n_wavelength) + 0.1
        C_sca = np.random.rand(n_radius, n_wavelength) + 0.1
        beta = np.random.rand(n_radius, n_wavelength, n_legendre)

        bulk_C_ext, bulk_C_sca, bulk_beta = merge_method2(
            radii, C_ext, C_sca, beta, sd, n_quad=256, interpolation="pchip"
        )

        np.testing.assert_allclose(bulk_beta[:, 0], 1.0, atol=1e-14)

    def test_method2_with_constant_properties(self):
        """If C_ext, C_sca, beta are constant with radius, Method 2 should
        reproduce those constants exactly (up to integration error)."""
        from Aerosol3D.bulk.datastructs import SizeDistribution
        from Aerosol3D.bulk.merge import merge_method2

        sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.5)
        n_radius, n_wavelength, n_legendre = 4, 2, 3
        radii = np.logspace(1, 3, n_radius)

        C_ext = np.ones((n_radius, n_wavelength))
        C_sca = np.ones((n_radius, n_wavelength))
        beta = np.zeros((n_radius, n_wavelength, n_legendre))
        beta[:, :, 0] = 1.0
        beta[:, :, 1] = 0.5
        beta[:, :, 2] = 0.1

        bulk_C_ext, bulk_C_sca, bulk_beta = merge_method2(
            radii, C_ext, C_sca, beta, sd, n_quad=512, interpolation="pchip"
        )

        np.testing.assert_allclose(bulk_C_ext, np.ones(n_wavelength), rtol=1e-6)
        np.testing.assert_allclose(bulk_C_sca, np.ones(n_wavelength), rtol=1e-6)
        np.testing.assert_allclose(bulk_beta[:, 0], 1.0, atol=1e-14)
        np.testing.assert_allclose(bulk_beta[:, 1], 0.5, rtol=1e-6)
        np.testing.assert_allclose(bulk_beta[:, 2], 0.1, rtol=1e-6)

    def test_method2_shape_mismatch_raises(self):
        from Aerosol3D.bulk.datastructs import SizeDistribution
        from Aerosol3D.bulk.merge import merge_method2

        sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.5)
        radii = np.array([50.0, 100.0, 200.0])
        C_ext = np.ones((3, 2))
        C_sca = np.ones((3, 2))
        beta = np.ones((4, 2, 3))  # mismatch: 4 vs 3 radii

        with pytest.raises(ValueError, match="first dimension"):
            merge_method2(radii, C_ext, C_sca, beta, sd)

    def test_method2_radii_length_mismatch_raises(self):
        from Aerosol3D.bulk.datastructs import SizeDistribution
        from Aerosol3D.bulk.merge import merge_method2

        sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.5)
        radii = np.array([50.0, 100.0, 200.0])
        C_ext = np.ones((4, 2))  # mismatch: 4 vs 3 radii
        C_sca = np.ones((4, 2))
        beta = np.ones((4, 2, 3))

        with pytest.raises(ValueError, match="first dimension"):
            merge_method2(radii, C_ext, C_sca, beta, sd)
