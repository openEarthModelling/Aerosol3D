"""Effective Medium Approximation (EMA) methods for refractive index mixing."""

from __future__ import annotations


def volume_weighted(volumes: list[float], refractive_indices: list[complex]) -> complex:
    """Volume-weighted linear mixing of refractive indices.

    m_eff = sum(f_i * m_i) where f_i = V_i / V_total.
    """
    if not volumes:
        raise ValueError("Material list is empty.")
    total = sum(volumes)
    return sum(v * m for v, m in zip(volumes, refractive_indices)) / total
