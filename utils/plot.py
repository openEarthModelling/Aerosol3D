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