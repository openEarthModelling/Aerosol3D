"""Example: Black carbon fractal aggregate — full pipeline.

Demonstrates the complete aerosol3d workflow with pyFracAggregate:
1. Generate a BC fractal aggregate using pyFracAggregate
2. Convert to AerosolParticle via from_fractal()
3. 3D screenshot and rotation video
4. DDA optical computation

Usage:
    # Visualization only:
    python black_carbon_fractal.py --no-optics

    # Full pipeline:
    python black_carbon_fractal.py

    # Save all outputs:
    python black_carbon_fractal.py --save
"""

import argparse
import sys
from pathlib import Path

EXAMPLE_DIR = Path(__file__).parent
OUTPUT_DIR = EXAMPLE_DIR / "output"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-optics", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    # --- Step 1: Generate fractal aggregate ---
    from pyFracAggregate import generate as gen_fractal, Monodisperse

    print("Generating BC fractal aggregate (50 monomers, PCA, Df=1.8)...")
    agg = gen_fractal(
        n_particles=50,
        df=1.8,
        kf=1.2,
        method="pca",
        particle_dist=Monodisperse(25.0),
    )
    print(f"  Generated: {agg.current_size} monomers, unit={agg.length_unit}")

    # --- Step 2: Convert to AerosolParticle ---
    from aerosol3d import Material, from_fractal, save_screenshot, save_rotation_video

    soot = Material("soot", refractive_index=complex(1.95, 0.79), density=1.8)
    fractal = from_fractal(agg, soot)
    particle = fractal.to_particle()
    print(f"  Particle: {particle}")

    # --- Step 3: 3D visualization ---
    save_screenshot(
        particle,
        str(OUTPUT_DIR / "bc_fractal_3d.png"),
        colors={"aggregate": "black"},
        opacity={"aggregate": 0.9},
    )
    print(f"  Screenshot saved: {OUTPUT_DIR / 'bc_fractal_3d.png'}")

    save_rotation_video(
        particle,
        str(OUTPUT_DIR / "bc_fractal_rotation.mp4"),
        colors={"aggregate": "black"},
        opacity={"aggregate": 0.9},
        n_frames=72,
        fps=24,
    )
    print(f"  Rotation video saved: {OUTPUT_DIR / 'bc_fractal_rotation.mp4'}")

    # --- Step 4: DDA optical computation ---
    if args.no_optics:
        print("Optical computation skipped (--no-optics).")
        return

    from aerosol3d import solve_optics, SimulationConfig
    from aerosol3d.optics.visualization import print_macroscopic, plot_phase_function_2d

    # Dipole spacing ~5 nm for good DDA convergence with soot at 550 nm
    config = SimulationConfig(
        wavelength=550.0,
        polarization=(1.0, 0.0, 0.0),
        propagation=(0.0, 0.0, 1.0),
        n_host=1.0,
        dipole_spacing=5.0,
    )

    print(f"\nRunning DDA solve (dipole d = 5 nm)...")
    result = solve_optics(
        particle,
        config,
        voxel_size=5.0,
        compute_near_field=True,
        compute_phase_func=True,
    )
    print_macroscopic(result)

    if args.save and result.phase_function is not None:
        phase_path = str(OUTPUT_DIR / "bc_fractal_phase.png")
        plot_phase_function_2d(result, save_path=phase_path, plane="xz", log_scale=True)
        print(f"Phase function saved: {phase_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
