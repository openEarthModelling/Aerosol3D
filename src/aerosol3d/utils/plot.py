import numpy as np
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

    elev_rad = np.radians(elevation)
    writer = imageio.get_writer(path, fps=fps, format="FFMPEG")

    for i in range(n_frames):
        azimuth = 360.0 * i / n_frames
        az_rad = np.radians(azimuth)
        pos = [
            radius * np.cos(elev_rad) * np.cos(az_rad) + focal_point[0],
            radius * np.cos(elev_rad) * np.sin(az_rad) + focal_point[1],
            radius * np.sin(elev_rad) + focal_point[2],
        ]
        # camera_position tuple + render() required in off_screen mode;
        # individual camera attribute assignments are ignored by pyvista
        plotter.camera_position = [pos, list(focal_point), [0, 0, 1]]
        plotter.render()
        frame = plotter.screenshot()
        writer.append_data(frame)

    writer.close()
    plotter.close()


def _build_voxel_glyph_mesh(grid):
    """Convert a voxel grid into a renderable glyph mesh of cubes.

    Args:
        grid: pv.ImageData with cell_data['material_id'].

    Returns:
        pv.PolyData where each occupied voxel is represented as a cube.

    Raises:
        ValueError: If grid lacks 'material_id' or has no occupied cells.
    """
    if "material_id" not in grid.cell_data:
        raise ValueError("Grid must contain cell_data['material_id']")

    thresholded = grid.threshold(0.5, scalars="material_id")
    if thresholded.n_cells == 0:
        raise ValueError("No occupied voxels to visualize")

    centers = thresholded.cell_centers()
    voxel_size = float(grid.spacing[0])
    cube = pv.Cube().scale((voxel_size, voxel_size, voxel_size))
    glyphs = centers.glyph(geom=cube, orient=False, scale=False)

    # glyph() does not preserve cell data; replicate material_id per glyph
    n_cells_per_glyph = cube.n_cells
    material_ids = thresholded.cell_data["material_id"]
    replicated = np.repeat(material_ids, n_cells_per_glyph)
    glyphs.cell_data["material_id"] = replicated

    return glyphs


_DEFAULT_PALETTE = [
    "red", "blue", "green", "orange", "purple",
    "cyan", "magenta", "brown", "pink", "olive",
]


def _resolve_voxel_colors(grid, colors):
    """Return a dict mapping material_id to color string.

    Args:
        grid: pv.ImageData with cell_data['material_id'].
        colors: Dict mapping material_id to color string, or None for auto.

    Returns:
        Dict[int, str] with a color for every occupied material_id.
    """
    unique_ids = np.unique(grid.cell_data["material_id"])
    occupied_ids = unique_ids[unique_ids > 0]

    if colors is None:
        colors = {}

    resolved = {}
    for i, mat_id in enumerate(occupied_ids):
        resolved[int(mat_id)] = colors.get(int(mat_id), _DEFAULT_PALETTE[i % len(_DEFAULT_PALETTE)])
    return resolved


def plot_voxel_grid(grid, colors=None, opacity=None, off_screen=False):
    """Return a PyVista Plotter rendering each occupied voxel as a cube.

    Args:
        grid: pv.ImageData with cell_data['material_id'].
        colors: Dict mapping material_id (int) to color string.
            e.g. {1: "black", 2: "dodgerblue"}. Auto-assigned if None.
        opacity: Dict mapping material_id to float in [0, 1].
        off_screen: If True, render without opening a window.

    Returns:
        pv.Plotter instance.
    """
    if opacity is None:
        opacity = {}

    glyphs = _build_voxel_glyph_mesh(grid)
    resolved_colors = _resolve_voxel_colors(grid, colors)

    plotter = pv.Plotter(off_screen=off_screen)
    for mat_id, color in resolved_colors.items():
        mask = glyphs.cell_data["material_id"] == mat_id
        if not np.any(mask):
            continue
        sub_mesh = glyphs.extract_cells(mask)
        plotter.add_mesh(
            sub_mesh,
            color=color,
            opacity=opacity.get(mat_id, 1.0),
            show_edges=False,
        )
    return plotter


def save_voxel_grid_screenshot(
    grid,
    path: str,
    colors=None,
    opacity=None,
    camera_position="iso",
    window_size=(1024, 768),
    background="white",
):
    """Render voxel grid and save as PNG screenshot.

    Args:
        grid: pv.ImageData with cell_data['material_id'].
        path: Output PNG file path.
        colors: Dict mapping material_id to color string.
        opacity: Dict mapping material_id to opacity float.
        camera_position: Camera preset ("iso", "xy", "xz", "yz", "front")
            or a manual tuple.
        window_size: (width, height) in pixels.
        background: Background color string.
    """
    if not path.endswith(".png"):
        raise ValueError(f"Output path must end with .png, got: {path}")

    plotter = plot_voxel_grid(grid, colors=colors, opacity=opacity, off_screen=True)
    plotter.set_background(background)
    plotter.window_size = window_size
    plotter.camera_position = camera_position
    plotter.screenshot(path)
    plotter.close()


def save_voxel_grid_video(
    grid,
    path: str,
    colors=None,
    opacity=None,
    n_frames: int = 72,
    fps: int = 24,
    elevation: float = 30.0,
    window_size=(1024, 768),
    background="white",
):
    """Render 360-degree rotation of voxel grid and save as MP4.

    Args:
        grid: pv.ImageData with cell_data['material_id'].
        path: Output MP4 file path (must end with .mp4).
        colors: Dict mapping material_id to color string.
        opacity: Dict mapping material_id to opacity float.
        n_frames: Total frames (72 = one frame per 5 degrees).
        fps: Frame rate.
        elevation: Camera elevation angle in degrees.
        window_size: (width, height) in pixels.
        background: Background color string.
    """
    import numpy as np
    import imageio.v2 as imageio

    if not path.endswith(".mp4"):
        raise ValueError(f"Output path must end with .mp4, got: {path}")

    plotter = plot_voxel_grid(grid, colors=colors, opacity=opacity, off_screen=True)
    plotter.set_background(background)
    plotter.window_size = window_size

    glyphs = _build_voxel_glyph_mesh(grid)
    bounds = glyphs.bounds
    focal_point = [
        (bounds[0] + bounds[1]) / 2,
        (bounds[2] + bounds[3]) / 2,
        (bounds[4] + bounds[5]) / 2,
    ]
    radius = max(
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
        bounds[5] - bounds[4],
    ) * 1.5

    elev_rad = np.radians(elevation)
    writer = imageio.get_writer(path, fps=fps, format="FFMPEG")

    for i in range(n_frames):
        azimuth = 360.0 * i / n_frames
        az_rad = np.radians(azimuth)
        pos = [
            radius * np.cos(elev_rad) * np.cos(az_rad) + focal_point[0],
            radius * np.cos(elev_rad) * np.sin(az_rad) + focal_point[1],
            radius * np.sin(elev_rad) + focal_point[2],
        ]
        plotter.camera_position = [pos, list(focal_point), [0, 0, 1]]
        plotter.render()
        frame = plotter.screenshot()
        writer.append_data(frame)

    writer.close()
    plotter.close()