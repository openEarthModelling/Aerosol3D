"""Merge formulas for bulk aerosol optics.

Method 1: Local bin integration (weighted average over discrete radii).
Method 2: Continuous interpolation + integration over the size distribution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from Aerosol3D.bulk.interpolation import LinearPCHIPInterpolator, LogLogPCHIPInterpolator

if TYPE_CHECKING:
    from Aerosol3D.bulk.datastructs import SizeDistribution


def compute_bin_weights(radii_nm: np.ndarray, size_distribution: SizeDistribution) -> np.ndarray:
    """Compute fractional number count per bin.

    Sorts ``radii_nm`` and places bin edges at geometric means between
    adjacent radii.  The first bin extends from ``0.0`` and the last bin
    extends to ``inf``.

    Args:
        radii_nm: Radii in nanometers (n_radius,).  Will be sorted internally.
        size_distribution: The ``SizeDistribution`` instance.

    Returns:
        Normalized bin weights that sum to exactly ``1.0`` (n_radius,).

    Raises:
        ValueError: If fewer than two radii are provided.
        RuntimeError: If the raw weights do not sum to approximately ``1.0``.
    """
    radii = np.asarray(radii_nm, dtype=float)
    if radii.size < 2:
        raise ValueError("at least two radii are required for bin weight computation")

    radii = np.sort(radii)
    n = radii.size

    # Bin edges at geometric means between adjacent radii
    edges = np.empty(n + 1, dtype=float)
    edges[0] = 0.0
    edges[1:-1] = np.sqrt(radii[:-1] * radii[1:])
    edges[-1] = np.inf

    # Compute fractional number count in each bin
    weights = np.empty(n, dtype=float)
    for i in range(n):
        weights[i] = size_distribution.number_in_interval(edges[i], edges[i + 1])

    weight_sum = float(np.sum(weights))
    if not np.isclose(weight_sum, 1.0, atol=1e-5):
        raise RuntimeError(f"weights do not sum to approximately 1.0 (got {weight_sum:.6e})")

    # Renormalize to exactly 1.0
    weights = weights / weight_sum
    return weights


def merge_method1(
    C_ext: np.ndarray,  # noqa: N803
    C_sca: np.ndarray,  # noqa: N803
    beta: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Method 1: Local bin integration (weighted average).

    Args:
        C_ext: Extinction cross-section (n_radius, n_wavelength).
        C_sca: Scattering cross-section (n_radius, n_wavelength).
        beta: Legendre expansion coefficients (n_radius, n_wavelength, n_legendre).
        weights: Bin weights (n_radius,).

    Returns:
        ``(bulk_C_ext, bulk_C_sca, bulk_beta)`` where each array has shape
        ``(n_wavelength,)``, ``(n_wavelength,)``, and
        ``(n_wavelength, n_legendre)`` respectively.

    Raises:
        ValueError: If input shapes are inconsistent.
    """
    C_ext = np.asarray(C_ext)
    C_sca = np.asarray(C_sca)
    beta = np.asarray(beta)
    weights = np.asarray(weights)

    n_radius = weights.shape[0]
    if C_ext.shape[0] != n_radius or C_sca.shape[0] != n_radius or beta.shape[0] != n_radius:
        raise ValueError(
            f"first dimension of C_ext, C_sca, beta must match weights length "
            f"({n_radius}), got {C_ext.shape[0]}, {C_sca.shape[0]}, {beta.shape[0]}"
        )
    if C_ext.shape != C_sca.shape:
        raise ValueError(
            f"C_ext and C_sca must have the same shape, got {C_ext.shape} and {C_sca.shape}"
        )
    if beta.shape[:2] != C_ext.shape:
        raise ValueError(
            f"beta[..., 0] must match C_ext shape, got {beta.shape[:2]} vs {C_ext.shape}"
        )

    n_wavelength = C_ext.shape[1]
    n_legendre = beta.shape[2]

    # Weighted sums
    bulk_C_ext = np.dot(weights, C_ext)  # (n_wavelength,)
    bulk_C_sca = np.dot(weights, C_sca)  # (n_wavelength,)

    # M_l = sum_i C_sca[i,j] * beta[i,j,l] * w[i]
    bulk_M_l = np.zeros((n_wavelength, n_legendre), dtype=float)
    for i in range(n_radius):
        bulk_M_l += C_sca[i, :, None] * beta[i, :, :] * weights[i]

    # beta = M_l / C_sca
    bulk_beta = np.zeros_like(bulk_M_l)
    for j in range(n_wavelength):
        if bulk_C_sca[j] > 0.0:
            bulk_beta[j, :] = bulk_M_l[j, :] / bulk_C_sca[j]
        else:
            # If C_sca is zero, set beta[:,0] = 1 and leave rest as 0
            bulk_beta[j, 0] = 1.0

    # Ensure beta[:, 0] = 1.0 exactly
    bulk_beta[:, 0] = 1.0

    return bulk_C_ext, bulk_C_sca, bulk_beta


def merge_method2(
    radii_nm: np.ndarray,
    C_ext: np.ndarray,  # noqa: N803
    C_sca: np.ndarray,  # noqa: N803
    beta: np.ndarray,
    size_distribution: SizeDistribution,
    n_quad: int = 256,
    interpolation: str = "pchip",
    integration: str = "quad",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Method 2: Continuous interpolation + integration.

    For each wavelength, interpolates the per-radius optical properties
    and integrates them against the size-distribution PDF.

    Args:
        radii_nm: Radii in nanometers (n_radius,).
        C_ext: Extinction cross-section (n_radius, n_wavelength).
        C_sca: Scattering cross-section (n_radius, n_wavelength).
        beta: Legendre expansion coefficients (n_radius, n_wavelength, n_legendre).
        size_distribution: The ``SizeDistribution`` instance.
        n_quad: Number of quadrature points for integration.
        interpolation: Interpolation type (currently only ``"pchip"`` is supported).
        integration: Integration method — ``"quad"`` for adaptive Gauss-Kronrod
            (primary, spec-compliant) or ``"fixed_quad"`` for fixed Gauss-Legendre.

    Returns:
        ``(bulk_C_ext, bulk_C_sca, bulk_beta)``.

    Raises:
        ValueError: If input shapes are inconsistent or interpolation/integration
            type is unsupported.
    """
    import warnings

    if interpolation == "cspline":
        warnings.warn(
            "cspline interpolation is not recommended for optical properties; "
            "falling back to pchip (monotonic, no overshoot).",
            UserWarning,
            stacklevel=2,
        )
        interpolation = "pchip"
    elif interpolation != "pchip":
        raise ValueError(f"unsupported interpolation: {interpolation!r}")

    radii = np.asarray(radii_nm, dtype=float)
    C_ext = np.asarray(C_ext)
    C_sca = np.asarray(C_sca)
    beta = np.asarray(beta)

    n_radius = radii.shape[0]
    if C_ext.shape[0] != n_radius or C_sca.shape[0] != n_radius or beta.shape[0] != n_radius:
        raise ValueError(
            f"first dimension of C_ext, C_sca, beta must match radii length "
            f"({n_radius}), got {C_ext.shape[0]}, {C_sca.shape[0]}, {beta.shape[0]}"
        )
    if C_ext.shape != C_sca.shape:
        raise ValueError(
            f"C_ext and C_sca must have the same shape, got {C_ext.shape} and {C_sca.shape}"
        )
    if beta.shape[:2] != C_ext.shape:
        raise ValueError(
            f"beta[..., 0] must match C_ext shape, got {beta.shape[:2]} vs {C_ext.shape}"
        )

    n_wavelength = C_ext.shape[1]
    n_legendre = beta.shape[2]

    from Aerosol3D.bulk.integration import integrate_distribution, integrate_distribution_vectorized

    bulk_C_ext = np.empty(n_wavelength, dtype=float)
    bulk_C_sca = np.empty(n_wavelength, dtype=float)
    bulk_M_l = np.empty((n_wavelength, n_legendre), dtype=float)

    for j in range(n_wavelength):
        # Interpolators for this wavelength
        c_ext_interp = LogLogPCHIPInterpolator(radii, C_ext[:, j])
        c_sca_interp = LogLogPCHIPInterpolator(radii, C_sca[:, j])

        beta_interps = [LinearPCHIPInterpolator(radii, beta[:, j, l]) for l in range(n_legendre)]

        if integration == "quad":
            # Adaptive Gauss-Kronrod (primary, spec-compliant)
            # Integrate C_ext(r) * pdf(r) dr
            def _integrand_C_ext(r: np.ndarray) -> np.ndarray:  # noqa: N802
                return c_ext_interp(r)

            bulk_C_ext[j] = integrate_distribution(
                _integrand_C_ext, size_distribution, method="quad"
            )

            # Integrate C_sca(r) * pdf(r) dr
            def _integrand_C_sca(r: np.ndarray) -> np.ndarray:  # noqa: N802
                return c_sca_interp(r)

            bulk_C_sca[j] = integrate_distribution(
                _integrand_C_sca, size_distribution, method="quad"
            )

            # Integrate beta_l(r) * C_sca(r) * pdf(r) dr for each l
            for l in range(n_legendre):

                def _make_integrand(beta_interp, c_sca_interp):
                    def _integrand(r: np.ndarray) -> np.ndarray:
                        return beta_interp(r) * c_sca_interp(r)

                    return _integrand

                integrand = _make_integrand(beta_interps[l], c_sca_interp)
                bulk_M_l[j, l] = integrate_distribution(integrand, size_distribution, method="quad")
        elif integration == "fixed_quad":
            # Fixed Gauss-Legendre quadrature in log-space
            # Integrate C_ext(r) * pdf(r) dr
            def _integrand_C_ext(r: np.ndarray) -> np.ndarray:  # noqa: N802
                return c_ext_interp(r)

            bulk_C_ext[j] = integrate_distribution_vectorized(
                _integrand_C_ext, size_distribution, n_quad
            )

            # Integrate C_sca(r) * pdf(r) dr
            def _integrand_C_sca(r: np.ndarray) -> np.ndarray:  # noqa: N802
                return c_sca_interp(r)

            bulk_C_sca[j] = integrate_distribution_vectorized(
                _integrand_C_sca, size_distribution, n_quad
            )

            # Integrate beta_l(r) * C_sca(r) * pdf(r) dr for each l
            for l in range(n_legendre):

                def _make_integrand(beta_interp, c_sca_interp):
                    def _integrand(r: np.ndarray) -> np.ndarray:
                        return beta_interp(r) * c_sca_interp(r)

                    return _integrand

                integrand = _make_integrand(beta_interps[l], c_sca_interp)
                bulk_M_l[j, l] = integrate_distribution_vectorized(
                    integrand, size_distribution, n_quad
                )
        else:
            raise ValueError(f"unsupported integration: {integration!r}")

    # beta = M_l / C_sca
    bulk_beta = np.zeros_like(bulk_M_l)
    for j in range(n_wavelength):
        if bulk_C_sca[j] > 0.0:
            bulk_beta[j, :] = bulk_M_l[j, :] / bulk_C_sca[j]
        else:
            bulk_beta[j, 0] = 1.0

    # Ensure beta[:, 0] = 1.0 exactly
    bulk_beta[:, 0] = 1.0

    return bulk_C_ext, bulk_C_sca, bulk_beta
