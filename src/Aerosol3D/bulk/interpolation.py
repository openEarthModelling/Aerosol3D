"""Physics-safe interpolation kernels for bulk aerosol optics."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator


class LogLogPCHIPInterpolator:
    """PCHIP interpolator in log-log space for positive-definite quantities.

    Suitable for cross-sections (C_ext, C_sca) which scale as power laws
    with radius.  Constant extrapolation outside the tabulated range.
    """

    def __init__(self, radii_nm: np.ndarray, values: np.ndarray) -> None:
        radii_nm = np.asarray(radii_nm)
        values = np.asarray(values)

        if radii_nm.ndim != 1 or values.ndim != 1:
            raise ValueError("radii_nm and values must be 1-D arrays")
        if radii_nm.size != values.size:
            raise ValueError("radii_nm and values must have the same length")
        if radii_nm.size < 2:
            raise ValueError("at least two points are required for interpolation")
        if np.any(radii_nm <= 0):
            raise ValueError("all radii_nm must be positive")
        if np.any(values <= 0):
            raise ValueError("all values must be positive")

        # Sort by radius
        order = np.argsort(radii_nm)
        self._radii = radii_nm[order]
        self._values = values[order]

        self._r_min = float(self._radii[0])
        self._r_max = float(self._radii[-1])
        self._v_min = float(self._values[0])
        self._v_max = float(self._values[-1])

        # Build PCHIP in log-log space
        self._interp = PchipInterpolator(
            np.log(self._radii),
            np.log(self._values),
            extrapolate=False,
        )

    def __call__(self, r_nm: float | np.ndarray) -> float | np.ndarray:
        scalar = np.isscalar(r_nm)
        r = np.atleast_1d(np.asarray(r_nm, dtype=float))

        # Evaluate in log-log space
        log_vals = self._interp(np.log(r))

        # Constant extrapolation: clip to edge values where interpolation is NaN
        log_v_min = np.log(self._v_min)
        log_v_max = np.log(self._v_max)
        log_vals = np.where(np.isnan(log_vals), log_v_min, log_vals)
        # For points above r_max, PchipInterpolator returns NaN; clip to max
        log_vals = np.where(r > self._r_max, log_v_max, log_vals)
        # For points below r_min, already handled by NaN -> v_min, but be explicit
        log_vals = np.where(r < self._r_min, log_v_min, log_vals)

        result = np.exp(log_vals)
        return float(result[0]) if scalar else result


class LinearPCHIPInterpolator:
    """PCHIP interpolator in linear space.

    Suitable for phase-function moments (beta_l) which may be zero or
    change sign.  Constant extrapolation outside the tabulated range.
    """

    def __init__(self, radii_nm: np.ndarray, values: np.ndarray) -> None:
        radii_nm = np.asarray(radii_nm)
        values = np.asarray(values)

        if radii_nm.ndim != 1 or values.ndim != 1:
            raise ValueError("radii_nm and values must be 1-D arrays")
        if radii_nm.size != values.size:
            raise ValueError("radii_nm and values must have the same length")
        if radii_nm.size < 2:
            raise ValueError("at least two points are required for interpolation")

        # Sort by radius
        order = np.argsort(radii_nm)
        self._radii = radii_nm[order]
        self._values = values[order]

        self._r_min = float(self._radii[0])
        self._r_max = float(self._radii[-1])
        self._v_min = float(self._values[0])
        self._v_max = float(self._values[-1])

        self._interp = PchipInterpolator(
            self._radii,
            self._values,
            extrapolate=False,
        )

    def __call__(self, r_nm: float | np.ndarray) -> float | np.ndarray:
        scalar = np.isscalar(r_nm)
        r = np.atleast_1d(np.asarray(r_nm, dtype=float))

        vals = self._interp(r)

        # Constant extrapolation
        vals = np.where(np.isnan(vals), self._v_min, vals)
        vals = np.where(r > self._r_max, self._v_max, vals)
        vals = np.where(r < self._r_min, self._v_min, vals)

        return float(vals[0]) if scalar else vals
