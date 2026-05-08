"""Legendre moment expansion from DDA phase functions."""

import numpy as np
from numpy.polynomial.legendre import leggauss


def compute_legendre_moments(phase_function, n_legendre: int = 32) -> np.ndarray:
    """Expand azimuthally-averaged P11 into Legendre moments via Gauss-Legendre quadrature.

    The phase function is expanded as:
        P11(mu) = sum_{l=0}^{n-1}  k_l  *  P_l(mu)

    where k_l = (2l+1) * integral(P11(mu) * P_l(mu) dmu) / integral(P11(mu) dmu)

    Args:
        phase_function: PhaseFunction with P11(theta, phi).
        n_legendre: Number of Legendre moments to compute.

    Returns:
        (n_legendre,) array of expansion coefficients k_l, with k_0 = 1.0.
    """
    # Azimuthal average: P11 as function of theta only
    P11_theta = np.mean(phase_function.P11, axis=1)  # (n_theta,)
    theta_grid = phase_function.theta

    # Gauss-Legendre quadrature nodes and weights on [-1, 1]
    mu_nodes, weights = leggauss(max(n_legendre + 10, 64))

    # Map mu = cos(theta), theta in [0, pi]
    # Interpolate P11_theta from uniform grid to GL nodes
    mu_grid = np.cos(theta_grid)
    # Sort by mu for interpolation (theta=0 -> mu=1, theta=pi -> mu=-1)
    sort_idx = np.argsort(mu_grid)
    mu_sorted = mu_grid[sort_idx]
    P11_sorted = P11_theta[sort_idx]

    P11_at_nodes = np.interp(mu_nodes, mu_sorted, P11_sorted)

    # Compute Legendre polynomials at quadrature nodes
    from numpy.polynomial.legendre import legvander
    V = legvander(mu_nodes, n_legendre - 1)  # (n_nodes, n_legendre)

    # Normalize: integral of P11 over mu
    norm = np.sum(P11_at_nodes * weights)
    if norm <= 0:
        return np.zeros(n_legendre)

    # Compute expansion coefficients
    moments = np.zeros(n_legendre)
    for l in range(n_legendre):
        integrand = P11_at_nodes * V[:, l] * weights
        moments[l] = (2 * l + 1) * np.sum(integrand) / norm

    # Ensure k_0 = 1 (numerical noise may deviate slightly)
    if n_legendre > 0:
        moments[0] = 1.0

    return moments
