"""Effective Medium Approximation (EMA) methods for refractive index mixing."""

from __future__ import annotations

import numpy as np


def volume_weighted(volumes: list[float], refractive_indices: list[complex]) -> complex:
    """Volume-weighted linear mixing of refractive indices.

    m_eff = sum(f_i * m_i) where f_i = V_i / V_total.
    """
    if not volumes:
        raise ValueError("Material list is empty.")
    total = sum(volumes)
    return sum(v * m for v, m in zip(volumes, refractive_indices)) / total


def maxwell_garnett(
    volumes: list[float], refractive_indices: list[complex]
) -> complex:
    """Maxwell-Garnett effective medium approximation.

    Host is the component with the largest volume. All other components
    are treated as inclusions embedded in the host.

    (eps_eff - eps_h) / (eps_eff + 2*eps_h) = sum_i f_i * (eps_i - eps_h) / (eps_i + 2*eps_h)

    where eps = m^2 and f_i = V_i / V_total.
    """
    if not volumes:
        raise ValueError("Material list is empty.")
    if len(volumes) == 1:
        return refractive_indices[0]

    total = sum(volumes)
    # Sort by volume descending; largest is host
    pairs = sorted(zip(volumes, refractive_indices), key=lambda p: p[0], reverse=True)
    m_h = pairs[0][1]

    beta = 0.0 + 0j
    for v, m_i in pairs[1:]:
        f_i = v / total
        beta += f_i * (m_i**2 - m_h**2) / (m_i**2 + 2 * m_h**2)

    eps_eff = m_h**2 * (1 + 2 * beta) / (1 - beta)
    return np.sqrt(eps_eff)
