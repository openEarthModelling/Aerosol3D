"""Quadrature integration engine for bulk aerosol optics.

Provides adaptive and fixed-quadrature methods for integrating a function
weighted by a size-distribution PDF over the radius domain.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad

if TYPE_CHECKING:
    from Aerosol3D.bulk.datastructs import SizeDistribution


def integrate_distribution(
    f: Callable[[np.ndarray], np.ndarray],
    size_distribution: SizeDistribution,
    n_quad: int = 256,
    method: str = "quad",
) -> float:
    """Integrate ``f(r) * pdf(r)`` over the size distribution domain.

    Computes:

    .. math::

        I = \\int_{r_{min}}^{r_{max}} f(r) \\cdot pdf(r) \\; dr

    where ``pdf(r)`` is the size-distribution probability density function.

    Args:
        f: Callable accepting an array of radii (nm) and returning an array
            of values. Must be vectorized for ``method="fixed_quad"``.
        size_distribution: The ``SizeDistribution`` instance providing
            ``pdf_values()`` and integration bounds.
        n_quad: Number of quadrature points. Used only for
            ``method="fixed_quad"``.
        method: ``"quad"`` for adaptive Gauss-Kronrod (``scipy.integrate.quad``)
            or ``"fixed_quad"`` for Gauss-Legendre quadrature in log-space.

    Returns:
        Scalar integral value.

    Raises:
        ValueError: If ``method`` is not supported.
    """
    if method == "quad":

        def _integrand(r: float) -> float:
            pdf_val = size_distribution.pdf_values(np.array([r]))[0]
            f_val = float(f(np.array([r]))[0])
            return f_val * pdf_val

        val, _ = quad(
            _integrand,
            size_distribution.r_min_nm,
            size_distribution.r_max_nm,
            limit=100,
        )
        return float(val)

    if method == "fixed_quad":
        return integrate_distribution_vectorized(f, size_distribution, n_quad)

    raise ValueError(f"Unsupported integration method: {method!r}")


def integrate_distribution_vectorized(
    f: Callable[[np.ndarray], np.ndarray],
    size_distribution: SizeDistribution,
    n_quad: int = 256,
) -> float:
    """Fully vectorized fixed-quadrature integration in log-space.

    Maps Gauss-Legendre nodes from ``[-1, 1]`` to
    ``[log(r_min), log(r_max)]`` and integrates with Jacobian
    ``dr = r * d(log r)``.

    Args:
        f: Vectorized callable ``f(r_nm) -> values``.
        size_distribution: The ``SizeDistribution`` instance.
        n_quad: Number of Gauss-Legendre quadrature points.

    Returns:
        Scalar integral value.
    """
    nodes, weights = leggauss(n_quad)

    log_min = np.log(size_distribution.r_min_nm)
    log_max = np.log(size_distribution.r_max_nm)

    # Affine map from [-1, 1] to [log_min, log_max]
    log_nodes = 0.5 * (log_max - log_min) * nodes + 0.5 * (log_max + log_min)
    r_nodes = np.exp(log_nodes)

    # Jacobian: dr = r * d(log r)
    jac = 0.5 * (log_max - log_min) * r_nodes

    f_vals = np.asarray(f(r_nodes))
    pdf_vals = size_distribution.pdf_values(r_nodes)

    integral = float(np.sum(f_vals * pdf_vals * weights * jac))
    return integral
