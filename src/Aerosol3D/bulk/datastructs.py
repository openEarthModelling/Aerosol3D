"""Data structures for bulk aerosol size distributions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.integrate import quad


@dataclass(frozen=True, slots=True)
class SizeDistribution:
    """A normalized aerosol size distribution n(r)/N_total.

    The ``pdf`` callable returns values in units of nm^-1 such that
    ``∫_{r_min}^{r_max} pdf(r) dr = 1``.

    Args:
        dist_type: One of "lognormal", "gamma", or "custom".
        pdf: Callable ``pdf(r_nm) -> density``.
        params: Serializable dictionary of distribution parameters.
        r_min_nm: Lower integration bound (nm). Defaults to 1.0.
        r_max_nm: Upper integration bound (nm). Defaults to 1e5.
    """

    dist_type: Literal["lognormal", "gamma", "custom"]
    pdf: Callable[[np.ndarray], np.ndarray]
    params: dict
    r_min_nm: float = 1.0
    r_max_nm: float = 1e5

    @classmethod
    def lognormal(cls, rg_nm: float, sigma_ln: float) -> SizeDistribution:
        """Create a log-normal size distribution.

        Uses the geometric-mean-radius parameterization (rg = median).

        PDF::

            n(r) = 1/(r * sigma_ln * sqrt(2*pi))
                   * exp(-(ln r - ln(r_g))^2 / (2 * sigma_ln^2))

        Args:
            rg_nm: Geometric mean radius (nm).
            sigma_ln: Geometric standard deviation (natural log).

        Returns:
            A frozen ``SizeDistribution`` instance.
        """

        def _pdf(r: np.ndarray) -> np.ndarray:
            r = np.asarray(r, dtype=float)
            out = np.zeros_like(r)
            mask = r > 0.0
            if np.any(mask):
                ln_r = np.log(r[mask])
                ln_rg = np.log(rg_nm)
                out[mask] = (
                    1.0
                    / (r[mask] * sigma_ln * np.sqrt(2.0 * np.pi))
                    * np.exp(-((ln_r - ln_rg) ** 2) / (2.0 * sigma_ln**2))
                )
            return out

        return cls(
            dist_type="lognormal",
            pdf=_pdf,
            params={"rg_nm": float(rg_nm), "sigma_ln": float(sigma_ln)},
        )

    @classmethod
    def gamma(cls, reff_nm: float, veff: float) -> SizeDistribution:
        """Create a gamma size distribution (Hansen & Travis 1974).

        The PDF is proportional to::

            r^((1 - 3*veff)/veff) * exp(-r / (reff * veff))

        The normalization constant is computed numerically via
        ``scipy.integrate.quad`` over ``[0, inf)``.

        Args:
            reff_nm: Effective radius (nm). By construction this equals
                ``moment(3) / moment(2)``.
            veff: Effective variance (dimensionless).

        Returns:
            A frozen ``SizeDistribution`` instance.
        """
        alpha = (1.0 - 3.0 * veff) / veff
        beta = 1.0 / (reff_nm * veff)

        # Compute normalization constant numerically over [0, inf)
        norm, _ = quad(lambda r: r**alpha * np.exp(-beta * r), 0.0, np.inf, limit=100)
        if norm <= 0.0 or not np.isfinite(norm):
            raise ValueError(
                f"Gamma normalization failed for reff={reff_nm}, veff={veff}"
            )

        def _pdf(r: np.ndarray) -> np.ndarray:
            r = np.asarray(r, dtype=float)
            out = np.zeros_like(r)
            mask = r > 0.0
            if np.any(mask):
                out[mask] = (r[mask] ** alpha) * np.exp(-beta * r[mask]) / norm
            return out

        return cls(
            dist_type="gamma",
            pdf=_pdf,
            params={"reff_nm": float(reff_nm), "veff": float(veff)},
        )

    @classmethod
    def from_scipy(
        cls, dist, params: dict, r_min_nm: float = 1.0, r_max_nm: float = 1e5
    ) -> SizeDistribution:
        """Wrap a SciPy continuous distribution.

        Args:
            dist: A ``scipy.stats.rv_continuous`` instance.
            params: Dictionary of SciPy distribution parameters.
            r_min_nm: Lower integration bound (nm).
            r_max_nm: Upper integration bound (nm).

        Returns:
            A frozen ``SizeDistribution`` instance.
        """

        def _pdf(r: np.ndarray) -> np.ndarray:
            r = np.asarray(r, dtype=float)
            out = np.zeros_like(r)
            mask = r > 0.0
            if np.any(mask):
                out[mask] = dist.pdf(r[mask], **params)
            return out

        return cls(
            dist_type="custom",
            pdf=_pdf,
            params=dict(params),
            r_min_nm=r_min_nm,
            r_max_nm=r_max_nm,
        )

    def pdf_values(self, r_nm: np.ndarray) -> np.ndarray:
        """Evaluate the PDF at the given radii.

        Args:
            r_nm: Radii in nanometers.

        Returns:
            PDF values in nm^-1.
        """
        return self.pdf(np.asarray(r_nm, dtype=float))

    def moment(self, order: float, method: str = "quad") -> float:
        """Compute the moment ``∫ r^order * pdf(r) dr``.

        For log-normal distributions the analytic formula is used::

            E[r^k] = rg^k * exp(k^2 * sigma_ln^2 / 2)

        For gamma and custom distributions numerical integration is used.

        Args:
            order: Moment order.
            method: Integration method (currently only "quad" is supported).

        Returns:
            The moment value.

        Raises:
            ValueError: If ``method`` is not supported.
        """
        if method != "quad":
            raise ValueError(f"Unsupported integration method: {method!r}")

        if self.dist_type == "lognormal":
            rg = self.params["rg_nm"]
            sigma = self.params["sigma_ln"]
            return rg**order * np.exp(order**2 * sigma**2 / 2.0)

        def _integrand(r: float) -> float:
            return r**order * self.pdf(np.array([r]))[0]

        val, _ = quad(_integrand, self.r_min_nm, self.r_max_nm, limit=100)
        return float(val)

    def effective_radius(self) -> float:
        """Return the effective radius ``moment(3) / moment(2)``.

        Returns:
            Effective radius in nm.
        """
        return self.moment(3.0) / self.moment(2.0)

    def number_in_interval(self, r_left: float, r_right: float) -> float:
        """Integrate the PDF over ``[r_left, r_right]``.

        Args:
            r_left: Lower bound (nm).
            r_right: Upper bound (nm).

        Returns:
            Fraction of particles in the interval.
        """
        val, _ = quad(
            lambda r: self.pdf(np.array([r]))[0], r_left, r_right, limit=100
        )
        return float(val)
