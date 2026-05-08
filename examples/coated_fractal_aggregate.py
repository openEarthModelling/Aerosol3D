"""Example: Coated black carbon fractal aggregate with dual-parameter coating.

Demonstrates the new two-parameter coating algorithms:
1. Generate a BC fractal aggregate
2. Apply potential void-filling coating (fills internal voids)
3. Apply potential edge-growing coating (grows outward)
4. 3D screenshots and rotation videos for comparison (mesh + voxel)

Usage:
    python coated_fractal_aggregate.py
    python coated_fractal_aggregate.py --no-video
"""

import argparse
from pathlib import Path

EXAMPLE_DIR = Path(__file__).parent
OUTPUT_DIR = EXAMPLE_DIR / "output"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-video", action="store_true", help="Skip video generation")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Imports
    from pyFracAggregate import generate as gen_fractal, Monodisperse
    from Aerosol3D import (
        from_fractal,
        save_screenshot,
        save_rotation_video,
        save_particle_voxel_screenshot,
        save_particle_voxel_video,
        preset_material,
        apply_potential_void_coating,
        apply_potential_edge_coating,
    )

    # Materials
    soot = preset_material("black_carbon")
    sulfate = preset_material("sulfate")

    # --- Step 1: Generate BC fractal aggregate ---
    print("Generating BC fractal aggregate (30 monomers, PCA, Df=1.8)...")
    agg = gen_fractal(
        n_particles=30,
        df=1.8,
        kf=1.2,
        method="pca",
        particle_dist=Monodisperse(25.0),
    )
    print(f"  Generated: {agg.current_size} monomers")

    fractal = from_fractal(agg, soot)
    particle = fractal.to_particle()
    print(f"  Particle: {particle}")

    # --- Step 2: Screenshot of bare aggregate ---
    save_screenshot(
        particle,
        str(OUTPUT_DIR / "coated_bare_aggregate.png"),
        colors={"aggregate": "black"},
        opacity={"aggregate": 0.9},
    )
    print(f"  Screenshot saved: {OUTPUT_DIR / 'coated_bare_aggregate.png'}")

    save_particle_voxel_screenshot(
        particle,
        voxel_size=5.0,
        path=str(OUTPUT_DIR / "coated_bare_aggregate_voxels.png"),
        colors={soot.id: "black"},
    )
    print(f"  Voxel screenshot saved: {OUTPUT_DIR / 'coated_bare_aggregate_voxels.png'}")

    # --- Step 3: Apply void-filling coating ---
    print("\nApplying potential void-filling coating...")
    void_particle = particle  # work in-place
    apply_potential_void_coating(
        void_particle,
        coated_area_fraction=0.8,
        dp_dc_ratio=2.0,
        material=sulfate,
        resolution=48,
    )
    print(f"  Coated particle: {void_particle}")

    save_screenshot(
        void_particle,
        str(OUTPUT_DIR / "coated_void_filling.png"),
        colors={"aggregate": "black", "coating": "dodgerblue"},
        opacity={"aggregate": 0.9, "coating": 0.5},
    )
    print(f"  Screenshot saved: {OUTPUT_DIR / 'coated_void_filling.png'}")

    save_particle_voxel_screenshot(
        void_particle,
        voxel_size=5.0,
        path=str(OUTPUT_DIR / "coated_void_filling_voxels.png"),
        colors={soot.id: "black", sulfate.id: "dodgerblue"},
        opacity={soot.id: 0.9, sulfate.id: 0.3},
    )
    print(f"  Voxel screenshot saved: {OUTPUT_DIR / 'coated_void_filling_voxels.png'}")

    # --- Step 4: Apply edge-growing coating ---
    print("\nApplying potential edge-growing coating...")
    # Start fresh from the bare aggregate
    edge_particle = fractal.to_particle()
    apply_potential_edge_coating(
        edge_particle,
        coated_area_fraction=0.8,
        dp_dc_ratio=2.0,
        material=sulfate,
        resolution=48,
    )
    print(f"  Coated particle: {edge_particle}")

    save_screenshot(
        edge_particle,
        str(OUTPUT_DIR / "coated_edge_growing.png"),
        colors={"aggregate": "black", "coating": "crimson"},
        opacity={"aggregate": 0.9, "coating": 0.5},
    )
    print(f"  Screenshot saved: {OUTPUT_DIR / 'coated_edge_growing.png'}")

    save_particle_voxel_screenshot(
        edge_particle,
        voxel_size=5.0,
        path=str(OUTPUT_DIR / "coated_edge_growing_voxels.png"),
        colors={soot.id: "black", sulfate.id: "crimson"},
        opacity={soot.id: 0.9, sulfate.id: 0.3},
    )
    print(f"  Voxel screenshot saved: {OUTPUT_DIR / 'coated_edge_growing_voxels.png'}")

    # --- Step 5: Rotation videos ---
    if args.no_video:
        print("\nVideo generation skipped (--no-video).")
        return

    print("\nGenerating rotation videos (this may take a minute)...")

    save_rotation_video(
        void_particle,
        str(OUTPUT_DIR / "coated_void_rotation.mp4"),
        colors={"aggregate": "black", "coating": "dodgerblue"},
        opacity={"aggregate": 0.9, "coating": 0.5},
        n_frames=72,
        fps=24,
    )
    print(f"  Void-filling rotation video: {OUTPUT_DIR / 'coated_void_rotation.mp4'}")

    save_particle_voxel_video(
        void_particle,
        voxel_size=5.0,
        path=str(OUTPUT_DIR / "coated_void_rotation_voxels.mp4"),
        colors={soot.id: "black", sulfate.id: "dodgerblue"},
        opacity={soot.id: 0.9, sulfate.id: 0.3},
        n_frames=72,
        fps=24,
    )
    print(f"  Void-filling voxel rotation video: {OUTPUT_DIR / 'coated_void_rotation_voxels.mp4'}")

    save_rotation_video(
        edge_particle,
        str(OUTPUT_DIR / "coated_edge_rotation.mp4"),
        colors={"aggregate": "black", "coating": "crimson"},
        opacity={"aggregate": 0.9, "coating": 0.5},
        n_frames=72,
        fps=24,
    )
    print(f"  Edge-growing rotation video: {OUTPUT_DIR / 'coated_edge_rotation.mp4'}")

    save_particle_voxel_video(
        edge_particle,
        voxel_size=5.0,
        path=str(OUTPUT_DIR / "coated_edge_rotation_voxels.mp4"),
        colors={soot.id: "black", sulfate.id: "crimson"},
        opacity={soot.id: 0.9, sulfate.id: 0.3},
        n_frames=72,
        fps=24,
    )
    print(f"  Edge-growing voxel rotation video: {OUTPUT_DIR / 'coated_edge_rotation_voxels.mp4'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
