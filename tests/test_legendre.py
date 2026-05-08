import numpy as np
import pytest


class TestComputeLegendreMoments:
    def test_hg_phase_function_expansion(self):
        """Henyey-Greenstein phase function should expand to k_l = (2l+1) * g^l."""
        from Aerosol3D.optics.datastructs import PhaseFunction
        from Aerosol3D.optics.legendre import compute_legendre_moments

        g = 0.7
        n_theta = 181
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2 * np.pi, 180, endpoint=False)
        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

        # HG phase function: P11(cos_theta) proportional to (1-g^2) / (1+g^2-2g*cos_theta)^(3/2)
        mu = np.cos(theta_grid)
        P11 = (1 - g**2) / (1 + g**2 - 2 * g * mu) ** 1.5

        pf = PhaseFunction(theta=theta, phi=phi, P11=P11)
        moments = compute_legendre_moments(pf, n_legendre=8)

        assert moments.shape == (8,)
        assert moments[0] == pytest.approx(1.0, abs=1e-6)

        for l in range(1, 8):
            expected = (2 * l + 1) * (g ** l)
            assert moments[l] == pytest.approx(expected, rel=0.05)

    def test_isotropic_expansion(self):
        """Isotropic scattering (g=0) should give k_0=1, k_{l>0}=0."""
        from Aerosol3D.optics.datastructs import PhaseFunction
        from Aerosol3D.optics.legendre import compute_legendre_moments

        n_theta = 181
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2 * np.pi, 180, endpoint=False)
        P11 = np.ones((n_theta, len(phi)))

        pf = PhaseFunction(theta=theta, phi=phi, P11=P11)
        moments = compute_legendre_moments(pf, n_legendre=8)

        assert moments[0] == pytest.approx(1.0, abs=1e-6)
        for l in range(1, 8):
            assert abs(moments[l]) < 1e-6

    def test_reconstruction_check(self):
        """Reconstructed P11 from moments should approximate original."""
        from Aerosol3D.optics.datastructs import PhaseFunction
        from Aerosol3D.optics.legendre import compute_legendre_moments

        g = 0.7
        n_theta = 181
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2 * np.pi, 180, endpoint=False)
        mu = np.cos(theta[:, None])
        P11 = (1 - g**2) / (1 + g**2 - 2 * g * mu) ** 1.5

        pf = PhaseFunction(theta=theta, phi=phi, P11=P11)
        moments = compute_legendre_moments(pf, n_legendre=32)

        # Reconstruct P11(theta) = sum_l k_l * P_l(cos_theta)
        from numpy.polynomial.legendre import legval
        P11_reconstructed = legval(np.cos(theta), moments)

        P11_original = np.mean(P11, axis=1)
        assert np.allclose(P11_reconstructed, P11_original, rtol=0.05)
