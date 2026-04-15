import pyvista as pv


def plot_particle(particle, colors=None, opacity=None, off_screen=False):
    """Visualize an aerosol particle with per-block coloring.

    Args:
        particle: AerosolParticle instance.
        colors: Dict mapping block name to color string. Defaults to auto.
        opacity: Dict mapping block name to opacity float. Defaults to 1.0.
        off_screen: If True, render without opening a window.

    Returns:
        pv.Plotter instance (call .show() to display).
    """
    plotter = pv.Plotter(off_screen=off_screen)

    if colors is None:
        colors = {}
    if opacity is None:
        opacity = {}

    for name, block in particle.blocks.items():
        if block is None:
            continue
        color = colors.get(name, None)
        opac = opacity.get(name, 1.0)
        plotter.add_mesh(block, color=color, opacity=opac, name=name)

    return plotter


def save_screenshot(
    particle,
    path: str,
    colors=None,
    opacity=None,
    camera_position="iso",
    window_size=(1024, 768),
    background="white",
):
    """Render AerosolParticle and save as PNG screenshot.

    Args:
        particle: AerosolParticle instance.
        path: Output PNG file path.
        colors: Dict mapping block name to color string.
        opacity: Dict mapping block name to opacity float.
        camera_position: Camera preset ("iso", "xy", "xz", "yz", "front")
            or a manual tuple.
        window_size: (width, height) in pixels.
        background: Background color string.
    """
    if colors is None:
        colors = {}
    if opacity is None:
        opacity = {}

    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background(background)

    for name, block in particle.blocks.items():
        if block is None:
            continue
        plotter.add_mesh(
            block,
            color=colors.get(name, None),
            opacity=opacity.get(name, 1.0),
            name=name,
        )

    plotter.camera_position = camera_position
    plotter.screenshot(path)
    plotter.close()


def save_rotation_video(
    particle,
    path: str,
    colors=None,
    opacity=None,
    n_frames: int = 72,
    fps: int = 24,
    elevation: float = 30.0,
    window_size=(1024, 768),
    background="white",
):
    """Render AerosolParticle 360-degree rotation and save as MP4.

    Args:
        particle: AerosolParticle instance.
        path: Output MP4 file path (must end with .mp4).
        colors: Dict mapping block name to color string.
        opacity: Dict mapping block name to opacity float.
        n_frames: Total frames (72 = one frame per 5 degrees).
        fps: Frame rate.
        elevation: Camera elevation angle in degrees.
        window_size: (width, height) in pixels.
        background: Background color string.

    Raises:
        ValueError: If path does not end with .mp4.
    """
    import numpy as np
    import imageio.v2 as imageio

    if not path.endswith(".mp4"):
        raise ValueError(f"Output path must end with .mp4, got: {path}")

    if colors is None:
        colors = {}
    if opacity is None:
        opacity = {}

    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background(background)

    for name, block in particle.blocks.items():
        if block is None:
            continue
        plotter.add_mesh(
            block,
            color=colors.get(name, None),
            opacity=opacity.get(name, 1.0),
            name=name,
        )

    combined = particle.combined
    focal_point = combined.center
    radius = max(combined.length, 1.0) * 1.5

    writer = imageio.get_writer(path, fps=fps)

    for i in range(n_frames):
        angle = 360.0 * i / n_frames
        azimuth_rad = np.radians(angle)
        elev_rad = np.radians(elevation)
        pos = [
            radius * np.cos(azimuth_rad) * np.cos(elev_rad) + focal_point[0],
            radius * np.sin(azimuth_rad) * np.cos(elev_rad) + focal_point[1],
            radius * np.sin(elev_rad) + focal_point[2],
        ]
        plotter.camera_position = (pos, focal_point, (0, 0, 1))
        frame = plotter.screenshot()
        writer.append_data(frame)

    writer.close()
    plotter.close()