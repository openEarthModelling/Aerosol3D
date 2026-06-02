#!/usr/bin/env python3
"""Compare bulk optical properties computed by Method 1 (bin weights)
versus Method 2 (continuous interpolation).

This example uses synthetic data to simulate optical properties of
spherical particles at different sizes, computes bulk properties
with both methods, and compares the results.

Outputs:
- Console report: C_ext, C_sca, SSA, g comparison at each wavelength
- Figures: method1_vs_method2_comparison.png

Usage:
    python examples/bulk_method_comparison.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

# Ensure src is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from Aerosol3D.bulk.builder import BulkOpticsBuilder
from Aerosol3D.bulk.datastructs import SizeDistribution
from Aerosol3D.bulk.merge import compute_bin_weights, merge_method1, merge_method2
from Aerosol3D.optics.optics_export import AerosolOpticsData

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Wavelengths (nm)
WAVELENGTHS_NM = np.array([400.0, 550.0, 700.0])

# Size distribution: lognormal, rg=200nm, sigma=0.5
DIST = SizeDistribution.lognormal(rg_nm=200.0, sigma_ln=0.5)

# Sample radii: sparse sampling (to demonstrate Method 1 vs Method 2 differences)
SPARSE_RADII_NM = np.array([50.0, 100.0, 200.0, 400.0, 800.0])

# Sample radii: dense sampling (reference solution for Method 2)
DENSE_RADII_NM = np.logspace(np.log10(30), np.log10(1000), 64)

N_LEGENDRE = 8


def _make_synthetic_optics(radius_nm: float, wavelengths: np.ndarray) -> AerosolOpticsData:
    """Create synthetic optical data mimicking Mie scattering features.

    - C_ext ~ r^2 (geometric cross-section scaling)
    - C_sca = C_ext * SSA, SSA varies with wavelength and particle size
    - g (asymmetry parameter) increases with particle size
    """
    n_wl = len(wavelengths)

    # C_ext: ~ r^2 * frequency-dependent factor
    # Higher frequency (shorter wavelength) gives stronger scattering
    freq_factor = 1.0 + 0.3 * (400.0 / wavelengths)  # short-wavelength enhancement
    C_ext = np.full(n_wl, np.pi * radius_nm**2) * freq_factor

    # SSA: larger particles scatter more, smaller particles absorb more (simplified model)
    # Controlled by size parameter / wavelength ratio
    x = 2 * np.pi * radius_nm / wavelengths  # size parameter
    SSA = 0.4 + 0.5 / (1.0 + np.exp(-(x - 3.0)))  # sigmoid: small-x absorbing, large-x scattering
    SSA = np.clip(SSA, 0.3, 0.95)

    C_sca = C_ext * SSA
    C_abs = C_ext - C_sca

    # g: increases with x (forward scattering enhancement)
    g = np.clip(0.1 + 0.7 * (x / (x + 2.0)), 0.0, 0.85)

    # beta: vSmartMOM convention beta_l = (2l+1) * g_l
    # Using Henyey-Greenstein approximation: g_l = g^l
    beta = np.zeros((n_wl, N_LEGENDRE))
    for l in range(N_LEGENDRE):
        beta[:, l] = (2 * l + 1) * g**l

    return AerosolOpticsData(
        wavelength_nm=wavelengths.copy(),
        C_ext=C_ext,
        C_sca=C_sca,
        C_abs=C_abs,
        SSA=SSA,
        g=g,
        r_eff_nm=radius_nm,
        legendre_moments_beta=beta / np.arange(1, 2 * N_LEGENDRE + 1, 2),  # store g_l
        n_legendre=N_LEGENDRE,
        solver="SYNTHETIC",
    )


def compute_method1_direct(radii: np.ndarray, optics_list: list[AerosolOpticsData]) -> dict:
    """Compute directly using merge_method1 (without Builder)."""
    n_r = len(radii)
    n_wl = len(WAVELENGTHS_NM)

    C_ext = np.zeros((n_r, n_wl))
    C_sca = np.zeros((n_r, n_wl))
    beta = np.zeros((n_r, n_wl, N_LEGENDRE))

    for i, opt in enumerate(optics_list):
        C_ext[i, :] = opt.C_ext
        C_sca[i, :] = opt.C_sca
        if opt.legendre_moments_beta is not None:
            # Convert to vSmartMOM beta = (2l+1) * g_l
            l_vals = np.arange(N_LEGENDRE)
            beta[i, :, :] = opt.legendre_moments_beta * (2 * l_vals + 1)
        else:
            beta[i, :, 0] = 1.0
            if N_LEGENDRE > 1:
                beta[i, :, 1] = opt.g * 3.0

    weights = compute_bin_weights(radii, DIST)
    bulk_C_ext, bulk_C_sca, bulk_beta = merge_method1(C_ext, C_sca, beta, weights)

    return {
        "C_ext": bulk_C_ext,
        "C_sca": bulk_C_sca,
        "C_abs": bulk_C_ext - bulk_C_sca,
        "SSA": bulk_C_sca / bulk_C_ext,
        "g": bulk_beta[:, 1] / 3.0 if N_LEGENDRE > 1 else np.zeros(n_wl),
        "beta": bulk_beta,
        "method": "Method 1 (Bin Weights)",
    }


def compute_method2_direct(radii: np.ndarray, optics_list: list[AerosolOpticsData]) -> dict:
    """Compute directly using merge_method2 (without Builder)."""
    n_r = len(radii)
    n_wl = len(WAVELENGTHS_NM)

    C_ext = np.zeros((n_r, n_wl))
    C_sca = np.zeros((n_r, n_wl))
    beta = np.zeros((n_r, n_wl, N_LEGENDRE))

    for i, opt in enumerate(optics_list):
        C_ext[i, :] = opt.C_ext
        C_sca[i, :] = opt.C_sca
        if opt.legendre_moments_beta is not None:
            l_vals = np.arange(N_LEGENDRE)
            beta[i, :, :] = opt.legendre_moments_beta * (2 * l_vals + 1)
        else:
            beta[i, :, 0] = 1.0
            if N_LEGENDRE > 1:
                beta[i, :, 1] = opt.g * 3.0

    bulk_C_ext, bulk_C_sca, bulk_beta = merge_method2(
        radii, C_ext, C_sca, beta, DIST, n_quad=512, interpolation="pchip"
    )

    return {
        "C_ext": bulk_C_ext,
        "C_sca": bulk_C_sca,
        "C_abs": bulk_C_ext - bulk_C_sca,
        "SSA": bulk_C_sca / bulk_C_ext,
        "g": bulk_beta[:, 1] / 3.0 if N_LEGENDRE > 1 else np.zeros(n_wl),
        "beta": bulk_beta,
        "method": "Method 2 (Continuous Interp)",
    }


def compute_with_builder(radii: np.ndarray, optics_list: list[AerosolOpticsData]) -> dict:
    """Compute using BulkOpticsBuilder (wrapped API)."""
    builder = BulkOpticsBuilder(
        size_distribution=DIST,
        radii_nm=radii,
        n_legendre=N_LEGENDRE,
    )
    for r, opt in zip(radii, optics_list):
        builder.add(radius=float(r), optics=opt)

    bulk = builder.compute(n_quad=512)

    return {
        "C_ext": bulk.C_ext,
        "C_sca": bulk.C_sca,
        "C_abs": bulk.C_abs,
        "SSA": bulk.SSA,
        "g": bulk.g,
        "beta": bulk.beta,
        "method": "Builder (Method 2)",
        "fallback": bulk.fallback_used,
    }


def print_comparison(
    label: str,
    m1: dict,
    m2: dict,
    ref: dict | None = None,
):
    """Print comparison table."""
    print(f"\n{'=' * 80}")
    print(f"  {label}")
    print(f"{'=' * 80}")

    print(
        f"  {'nm':>10} | {'Method':>18} | {'C_ext':>12} | {'C_sca':>12} | {'SSA':>8} | {'g':>8} | {'beta_1':>8}"
    )
    print(f"  {'-' * 90}")

    for j, wl in enumerate(WAVELENGTHS_NM):
        # Method 1
        print(
            f"  {wl:>10.0f} | {'M1 (Bin)':>18} | "
            f"{m1['C_ext'][j]:>12.2f} | {m1['C_sca'][j]:>12.2f} | "
            f"{m1['SSA'][j]:>8.4f} | {m1['g'][j]:>8.4f} | {m1['beta'][j, 1]:>8.4f}"
        )
        # Method 2
        print(
            f"  {'':>10} | {'M2 (Interp)':>18} | "
            f"{m2['C_ext'][j]:>12.2f} | {m2['C_sca'][j]:>12.2f} | "
            f"{m2['SSA'][j]:>8.4f} | {m2['g'][j]:>8.4f} | {m2['beta'][j, 1]:>8.4f}"
        )

        # Difference
        dC_ext = (
            abs(m1["C_ext"][j] - m2["C_ext"][j]) / m1["C_ext"][j] * 100 if m1["C_ext"][j] > 0 else 0
        )
        dC_sca = (
            abs(m1["C_sca"][j] - m2["C_sca"][j]) / m1["C_sca"][j] * 100 if m1["C_sca"][j] > 0 else 0
        )
        dSSA = abs(m1["SSA"][j] - m2["SSA"][j]) * 100
        dg = abs(m1["g"][j] - m2["g"][j]) * 100
        print(
            f"  {'':>10} | {'Δ (%)':>18} | "
            f"{dC_ext:>11.2f}% | {dC_sca:>11.2f}% | {dSSA:>7.2f}% | {dg:>7.2f}% |"
        )
        print(f"  {'-' * 90}")

        if ref is not None:
            dref_C_ext = abs(m2["C_ext"][j] - ref["C_ext"][j]) / ref["C_ext"][j] * 100
            print(f"  {'':>10} | {'vs Ref (%)':>18} | {dref_C_ext:>11.2f}% | ...")


def _get_bin_edges(radii: np.ndarray) -> np.ndarray:
    """Compute Method 1 bin edges (geometric mean)."""
    radii = np.sort(radii)
    edges = np.zeros(len(radii) + 1)
    edges[0] = 0.0
    for i in range(1, len(radii)):
        edges[i] = np.sqrt(radii[i - 1] * radii[i])
    edges[-1] = np.inf
    return edges


def plot_all(
    sparse_radii: np.ndarray,
    sparse_optics: list[AerosolOpticsData],
    sparse_m1: dict,
    sparse_m2: dict,
    dense_m2: dict,
    output_dir: Path,
):
    """Plot the complete bulk computation process visualization (5 figures)."""
    import matplotlib.pyplot as plt
    from scipy.interpolate import PchipInterpolator

    wl = WAVELENGTHS_NM
    wl_colors = {400.0: "#8e44ad", 550.0: "#2980b9", 700.0: "#27ae60"}

    # ===================================================================
    # Figure 1: Size Distribution & Sampling
    # ===================================================================
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
    fig1.suptitle("Figure 1: Size Distribution & Sampling Grid", fontsize=14, fontweight="bold")

    # PDF curve
    r_fine = np.logspace(0.5, 3.5, 500)
    pdf_vals = DIST.pdf_values(r_fine)

    ax = axes1[0]
    ax.fill_between(r_fine, pdf_vals, alpha=0.3, color="#3498db", label="PDF")
    ax.plot(r_fine, pdf_vals, "-", color="#2980b9", lw=2)
    # Mark sample points
    pdf_at_samples = DIST.pdf_values(sparse_radii)
    ax.scatter(
        sparse_radii, pdf_at_samples, color="#e74c3c", s=100, zorder=5, label="Sparse samples (n=5)"
    )
    # Mark bin edges
    edges = _get_bin_edges(sparse_radii)
    finite_edges = edges[edges < np.inf]
    for e in finite_edges:
        ax.axvline(e, color="#95a5a6", linestyle="--", alpha=0.7, lw=1)
    ax.axvline(0, color="#95a5a6", linestyle="--", alpha=0.7, lw=1)
    ax.set_xlabel("Radius r (nm)")
    ax.set_ylabel("n(r) / N_total (nm⁻¹)")
    ax.set_title("Lognormal PDF with Sparse Samples & Bin Edges")
    ax.set_xlim(0, 1000)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes1[1]
    ax.semilogx(r_fine, pdf_vals, "-", color="#2980b9", lw=2)
    ax.fill_between(r_fine, pdf_vals, alpha=0.3, color="#3498db")
    ax.scatter(sparse_radii, pdf_at_samples, color="#e74c3c", s=100, zorder=5)
    for e in finite_edges:
        ax.axvline(e, color="#95a5a6", linestyle="--", alpha=0.7, lw=1)
    ax.set_xlabel("Radius r (nm) [log scale]")
    ax.set_ylabel("n(r) / N_total (nm⁻¹)")
    ax.set_title("PDF on Log Scale (showing distribution tails)")
    ax.legend(["PDF", "Sparse samples"], loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig1.savefig(output_dir / "fig1_size_distribution.png", dpi=150, bbox_inches="tight")
    logger.info("Figure 1 saved: size distribution")

    # ===================================================================
    # Figure 2: Single-Size Optical Properties (Raw Data)
    # ===================================================================
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
    fig2.suptitle(
        "Figure 2: Single-Size Optical Properties (Raw Data)", fontsize=14, fontweight="bold"
    )

    # Prepare data matrices
    C_ext_raw = np.array([opt.C_ext for opt in sparse_optics])  # (n_r, n_wl)
    C_sca_raw = np.array([opt.C_sca for opt in sparse_optics])
    SSA_raw = np.array([opt.SSA for opt in sparse_optics])
    g_raw = np.array([opt.g for opt in sparse_optics])

    for j, w in enumerate(wl):
        color = wl_colors[w]
        label = f"{w:.0f} nm"
        axes2[0, 0].semilogx(
            sparse_radii, C_ext_raw[:, j], "o-", color=color, label=label, markersize=8
        )
        axes2[0, 1].semilogx(
            sparse_radii, C_sca_raw[:, j], "o-", color=color, label=label, markersize=8
        )
        axes2[1, 0].semilogx(
            sparse_radii, SSA_raw[:, j], "o-", color=color, label=label, markersize=8
        )
        axes2[1, 1].semilogx(
            sparse_radii, g_raw[:, j], "o-", color=color, label=label, markersize=8
        )

    axes2[0, 0].set_ylabel(r"$C_{ext}$ (nm²)")
    axes2[0, 0].set_title("Extinction Cross-Section")
    axes2[0, 0].legend()
    axes2[0, 0].grid(True, alpha=0.3)

    axes2[0, 1].set_ylabel(r"$C_{sca}$ (nm²)")
    axes2[0, 1].set_title("Scattering Cross-Section")
    axes2[0, 1].legend()
    axes2[0, 1].grid(True, alpha=0.3)

    axes2[1, 0].set_ylabel("SSA")
    axes2[1, 0].set_xlabel("Radius r (nm)")
    axes2[1, 0].set_title("Single Scattering Albedo")
    axes2[1, 0].legend()
    axes2[1, 0].grid(True, alpha=0.3)

    axes2[1, 1].set_ylabel("g")
    axes2[1, 1].set_xlabel("Radius r (nm)")
    axes2[1, 1].set_title("Asymmetry Parameter")
    axes2[1, 1].legend()
    axes2[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig2.savefig(output_dir / "fig2_raw_optical_properties.png", dpi=150, bbox_inches="tight")
    logger.info("Figure 2 saved: raw optical properties")

    # ===================================================================
    # Figure 3: Method 2 — Continuous Interpolation Process
    # ===================================================================
    fig3, axes3 = plt.subplots(2, 2, figsize=(12, 10))
    fig3.suptitle(
        "Figure 3: Method 2 — Continuous Interpolation + Integration\n"
        "(Log-Log PCHIP for C, Linear PCHIP for β)",
        fontsize=13,
        fontweight="bold",
    )

    r_interp = np.logspace(np.log10(30), np.log10(1000), 200)
    pdf_interp = DIST.pdf_values(r_interp)

    for j, w in enumerate(wl):
        color = wl_colors[w]
        label = f"{w:.0f} nm"

        # C_ext: log-log PCHIP
        log_C_ext = np.log(C_ext_raw[:, j])
        pchip_C_ext = PchipInterpolator(
            np.log(np.sort(sparse_radii)), log_C_ext[np.argsort(sparse_radii)]
        )
        C_ext_smooth = np.exp(pchip_C_ext(np.log(r_interp)))

        ax = axes3[0, 0]
        ax.loglog(sparse_radii, C_ext_raw[:, j], "o", color=color, markersize=10, zorder=5)
        ax.loglog(r_interp, C_ext_smooth, "-", color=color, alpha=0.7, label=label)
        ax.set_ylabel(r"$C_{ext}(r)$ (nm²)")
        ax.set_title(r"Step 1: Interpolate $C_{ext}(r)$ in Log-Log Space")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # C_sca: log-log PCHIP
        log_C_sca = np.log(C_sca_raw[:, j])
        pchip_C_sca = PchipInterpolator(
            np.log(np.sort(sparse_radii)), log_C_sca[np.argsort(sparse_radii)]
        )
        C_sca_smooth = np.exp(pchip_C_sca(np.log(r_interp)))

        ax = axes3[0, 1]
        ax.loglog(sparse_radii, C_sca_raw[:, j], "o", color=color, markersize=10, zorder=5)
        ax.loglog(r_interp, C_sca_smooth, "-", color=color, alpha=0.7, label=label)
        ax.set_ylabel(r"$C_{sca}(r)$ (nm²)")
        ax.set_title(r"Step 2: Interpolate $C_{sca}(r)$ in Log-Log Space")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # beta_1: linear PCHIP
        beta_1_raw = np.array(
            [
                opt.legendre_moments_beta[j, 1] * 3.0
                if opt.legendre_moments_beta is not None
                else opt.g[j] * 3.0
                for opt in sparse_optics
            ]
        )
        pchip_beta = PchipInterpolator(np.sort(sparse_radii), beta_1_raw[np.argsort(sparse_radii)])
        beta_1_smooth = pchip_beta(r_interp)

        ax = axes3[1, 0]
        ax.semilogx(sparse_radii, beta_1_raw, "o", color=color, markersize=10, zorder=5)
        ax.semilogx(r_interp, beta_1_smooth, "-", color=color, alpha=0.7, label=label)
        ax.set_xlabel("Radius r (nm)")
        ax.set_ylabel(r"$\beta_1(r)$")
        ax.set_title(r"Step 3: Interpolate $\beta_1(r)$ in Linear Space")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Integrand: C_sca(r) * pdf(r)
        integrand = C_sca_smooth * pdf_interp
        ax = axes3[1, 1]
        ax.semilogx(r_interp, integrand, "-", color=color, alpha=0.7, label=label)
        ax.fill_between(r_interp, integrand, alpha=0.2, color=color)
        ax.set_xlabel("Radius r (nm)")
        ax.set_ylabel(r"$C_{sca}(r) \cdot n(r)$")
        ax.set_title(r"Step 4: Integrand = $C_{sca}(r) \cdot n(r)$")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig3.savefig(output_dir / "fig3_method2_interpolation.png", dpi=150, bbox_inches="tight")
    logger.info("Figure 3 saved: Method 2 interpolation process")

    # ===================================================================
    # Figure 4: Method 1 — Local Bin Integration
    # ===================================================================
    fig4, axes4 = plt.subplots(1, 3, figsize=(15, 5))
    fig4.suptitle("Figure 4: Method 1 — Local Bin Integration", fontsize=14, fontweight="bold")

    # Bin edges
    edges = _get_bin_edges(sparse_radii)
    weights = compute_bin_weights(sparse_radii, DIST)

    # Panel 4a: PDF + Bins
    ax = axes4[0]
    ax.fill_between(r_fine, DIST.pdf_values(r_fine), alpha=0.3, color="#3498db")
    ax.plot(r_fine, DIST.pdf_values(r_fine), "-", color="#2980b9", lw=2, label="PDF")
    for i, e in enumerate(finite_edges):
        ax.axvline(e, color="#e67e22", linestyle="--", alpha=0.8, lw=1.5)
    ax.scatter(
        sparse_radii,
        DIST.pdf_values(sparse_radii),
        color="#e74c3c",
        s=80,
        zorder=5,
        label="Sample points",
    )
    # Annotate bin numbers
    for i in range(len(sparse_radii)):
        mid = (edges[i] + edges[i + 1]) / 2 if edges[i + 1] < np.inf else sparse_radii[i] * 1.3
        ax.text(
            mid,
            DIST.pdf_values(mid) * 1.15,
            f"Bin {i + 1}",
            ha="center",
            fontsize=9,
            color="#e67e22",
            fontweight="bold",
        )
    ax.set_xlim(0, 1000)
    ax.set_xlabel("Radius r (nm)")
    ax.set_ylabel("n(r) / N_total (nm⁻¹)")
    ax.set_title("Step 1: Divide into Bins (geometric mean edges)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel 4b: Bin Weights
    ax = axes4[1]
    bin_labels = [f"Bin {i + 1}\n{r:.0f}nm" for i, r in enumerate(np.sort(sparse_radii))]
    bars = ax.bar(range(len(weights)), weights, color="#3498db", edgecolor="#2980b9", alpha=0.8)
    ax.set_xticks(range(len(weights)))
    ax.set_xticklabels(bin_labels, fontsize=8)
    ax.set_ylabel(r"Fractional Number Count $N_i$")
    ax.set_title(
        r"Step 2: Compute Weights $N_i = \int_{bin_i} n(r)dr$"
        + "\n"
        + rf"($\sum N_i$ = {np.sum(weights):.4f})"
    )
    ax.grid(True, alpha=0.3, axis="y")
    for bar, w in zip(bars, weights):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{w:.3f}",
            ha="center",
            fontsize=9,
        )

    # Panel 4c: Weighted Contributions
    ax = axes4[2]
    # Use 550 nm as example
    j_mid = 1
    contributions = C_ext_raw[:, j_mid] * weights
    bars = ax.bar(
        range(len(contributions)), contributions, color="#e74c3c", edgecolor="#c0392b", alpha=0.8
    )
    ax.set_xticks(range(len(contributions)))
    ax.set_xticklabels(bin_labels, fontsize=8)
    ax.set_ylabel(r"$C_{ext,i} \cdot N_i$ (nm²)")
    ax.set_title(
        rf"Step 3: Weighted Contribution (λ={wl[j_mid]:.0f}nm)"
        + "\n"
        + rf"$\bar{{C}}_{{ext}}$ = {sparse_m1['C_ext'][j_mid]:.1f} nm²"
    )
    ax.grid(True, alpha=0.3, axis="y")
    for bar, c in zip(bars, contributions):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(contributions) * 0.01,
            f"{c:.0f}",
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()
    fig4.savefig(output_dir / "fig4_method1_bin_weights.png", dpi=150, bbox_inches="tight")
    logger.info("Figure 4 saved: Method 1 bin weight process")

    # ===================================================================
    # Figure 5: Final Results Comparison
    # ===================================================================
    fig5, axes5 = plt.subplots(2, 2, figsize=(12, 10))
    fig5.suptitle(
        "Figure 5: Final Bulk Results — Method 1 vs Method 2", fontsize=14, fontweight="bold"
    )

    colors = {"M1": "#e74c3c", "M2_sparse": "#3498db", "M2_dense": "#2ecc71"}
    labels = {
        "M1": "Method 1 (Bin)",
        "M2_sparse": "Method 2 (Sparse)",
        "M2_dense": "Method 2 (Dense, Ref)",
    }

    # C_ext
    ax = axes5[0, 0]
    ax.plot(
        wl, sparse_m1["C_ext"], "o-", color=colors["M1"], label=labels["M1"], markersize=10, lw=2
    )
    ax.plot(
        wl,
        sparse_m2["C_ext"],
        "s-",
        color=colors["M2_sparse"],
        label=labels["M2_sparse"],
        markersize=10,
        lw=2,
    )
    ax.plot(
        wl,
        dense_m2["C_ext"],
        "^-",
        color=colors["M2_dense"],
        label=labels["M2_dense"],
        markersize=8,
        lw=2,
    )
    # Add error annotations
    for j, w in enumerate(wl):
        err_m1 = abs(sparse_m1["C_ext"][j] - dense_m2["C_ext"][j]) / dense_m2["C_ext"][j] * 100
        err_m2 = abs(sparse_m2["C_ext"][j] - dense_m2["C_ext"][j]) / dense_m2["C_ext"][j] * 100
        ax.annotate(
            f"M1 err: {err_m1:.1f}%",
            xy=(w, sparse_m1["C_ext"][j]),
            xytext=(10, 15),
            textcoords="offset points",
            fontsize=8,
            color=colors["M1"],
        )
        ax.annotate(
            f"M2 err: {err_m2:.1f}%",
            xy=(w, sparse_m2["C_ext"][j]),
            xytext=(10, -20),
            textcoords="offset points",
            fontsize=8,
            color=colors["M2_sparse"],
        )
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(r"$\bar{C}_{ext}$ (nm²)")
    ax.set_title("Bulk Extinction Cross-Section")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # SSA
    ax = axes5[0, 1]
    ax.plot(wl, sparse_m1["SSA"], "o-", color=colors["M1"], label=labels["M1"], markersize=10, lw=2)
    ax.plot(
        wl,
        sparse_m2["SSA"],
        "s-",
        color=colors["M2_sparse"],
        label=labels["M2_sparse"],
        markersize=10,
        lw=2,
    )
    ax.plot(
        wl,
        dense_m2["SSA"],
        "^-",
        color=colors["M2_dense"],
        label=labels["M2_dense"],
        markersize=8,
        lw=2,
    )
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("SSA")
    ax.set_title("Bulk Single Scattering Albedo")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # g
    ax = axes5[1, 0]
    ax.plot(wl, sparse_m1["g"], "o-", color=colors["M1"], label=labels["M1"], markersize=10, lw=2)
    ax.plot(
        wl,
        sparse_m2["g"],
        "s-",
        color=colors["M2_sparse"],
        label=labels["M2_sparse"],
        markersize=10,
        lw=2,
    )
    ax.plot(
        wl,
        dense_m2["g"],
        "^-",
        color=colors["M2_dense"],
        label=labels["M2_dense"],
        markersize=8,
        lw=2,
    )
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("g")
    ax.set_title("Bulk Asymmetry Parameter")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # beta_1
    ax = axes5[1, 1]
    ax.plot(
        wl,
        sparse_m1["beta"][:, 1],
        "o-",
        color=colors["M1"],
        label=labels["M1"],
        markersize=10,
        lw=2,
    )
    ax.plot(
        wl,
        sparse_m2["beta"][:, 1],
        "s-",
        color=colors["M2_sparse"],
        label=labels["M2_sparse"],
        markersize=10,
        lw=2,
    )
    ax.plot(
        wl,
        dense_m2["beta"][:, 1],
        "^-",
        color=colors["M2_dense"],
        label=labels["M2_dense"],
        markersize=8,
        lw=2,
    )
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(r"$\beta_1$")
    ax.set_title("Bulk First Legendre Coefficient")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig5.savefig(output_dir / "fig5_final_comparison.png", dpi=150, bbox_inches="tight")
    logger.info("Figure 5 saved: final comparison")

    # Close all figures to free memory
    plt.close("all")


def main():
    logger.info("=" * 70)
    logger.info("Bulk Aerosol Optics: Method 1 vs Method 2 Comparison")
    logger.info("=" * 70)
    logger.info(
        f"Size Distribution: Lognormal(rg={DIST.params['rg_nm']:.0f}nm, σ={DIST.params['sigma_ln']})"
    )
    logger.info(f"Sparse Radii: {SPARSE_RADII_NM}")
    logger.info(
        f"Dense Radii: {len(DENSE_RADII_NM)} points from {DENSE_RADII_NM.min():.0f} to {DENSE_RADII_NM.max():.0f} nm"
    )
    logger.info(f"Wavelengths: {WAVELENGTHS_NM} nm")
    logger.info(f"Legendre Order: {N_LEGENDRE}")

    # -----------------------------------------------------------------------
    # Generate synthetic data (sparse)
    # -----------------------------------------------------------------------
    logger.info("\n--- Generating synthetic data (sparse sampling) ---")
    sparse_optics = [_make_synthetic_optics(r, WAVELENGTHS_NM) for r in SPARSE_RADII_NM]

    # -----------------------------------------------------------------------
    # Generate synthetic data (dense)
    # -----------------------------------------------------------------------
    logger.info("--- Generating synthetic data (dense sampling) ---")
    dense_optics = [_make_synthetic_optics(r, WAVELENGTHS_NM) for r in DENSE_RADII_NM]

    # -----------------------------------------------------------------------
    # Compute
    # -----------------------------------------------------------------------
    logger.info("\n--- Computing bulk properties ---")

    # Method 1 (sparse)
    sparse_m1 = compute_method1_direct(SPARSE_RADII_NM, sparse_optics)
    logger.info("Method 1 (sparse) done")

    # Method 2 (sparse)
    sparse_m2 = compute_method2_direct(SPARSE_RADII_NM, sparse_optics)
    logger.info("Method 2 (sparse) done")

    # Method 2 (dense) — reference solution
    dense_m2 = compute_method2_direct(DENSE_RADII_NM, dense_optics)
    logger.info("Method 2 (dense, reference) done")

    # Builder API (validation)
    builder_result = compute_with_builder(SPARSE_RADII_NM, sparse_optics)
    logger.info(f"Builder API done (fallback={builder_result['fallback']})")

    # -----------------------------------------------------------------------
    # Print comparison
    # -----------------------------------------------------------------------
    print_comparison("Sparse Sampling: Method 1 vs Method 2", sparse_m1, sparse_m2, dense_m2)

    # Verify Builder result matches Method 2
    print(f"\n{'=' * 80}")
    print("  Builder API Consistency Check")
    print(f"{'=' * 80}")
    for j, wl in enumerate(WAVELENGTHS_NM):
        dC = abs(builder_result["C_ext"][j] - sparse_m2["C_ext"][j]) / sparse_m2["C_ext"][j] * 100
        dg = abs(builder_result["g"][j] - sparse_m2["g"][j]) * 100
        print(
            f"  {wl:.0f}nm: ΔC_ext={dC:.4f}%, Δg={dg:.4f}% → {'PASS' if dC < 0.01 and dg < 0.01 else 'FAIL'}"
        )

    # -----------------------------------------------------------------------
    # Compare with reference
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("  Convergence to Dense Reference (Method 2)")
    print(f"{'=' * 80}")
    print(f"  {'Wavelength':>12} | {'M1 vs Ref (%)':>15} | {'M2 vs Ref (%)':>15} | {'Winner':>10}")
    print(f"  {'-' * 60}")
    for j, wl in enumerate(WAVELENGTHS_NM):
        err_m1 = abs(sparse_m1["C_ext"][j] - dense_m2["C_ext"][j]) / dense_m2["C_ext"][j] * 100
        err_m2 = abs(sparse_m2["C_ext"][j] - dense_m2["C_ext"][j]) / dense_m2["C_ext"][j] * 100
        winner = "M2" if err_m2 < err_m1 else "M1"
        print(f"  {wl:>12.0f} | {err_m1:>14.2f}% | {err_m2:>14.2f}% | {winner:>10}")

    # -----------------------------------------------------------------------
    # Plotting (full process visualization)
    # -----------------------------------------------------------------------
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    plot_all(
        SPARSE_RADII_NM,
        sparse_optics,
        sparse_m1,
        sparse_m2,
        dense_m2,
        output_dir,
    )

    logger.info("\n" + "=" * 70)
    logger.info("Done!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
