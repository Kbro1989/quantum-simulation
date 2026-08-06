import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from quantumlab.visualization.style import set_style


def plot_density_2d(wf, potential=None, title: str = '2D Probability Density $|\\Psi(x,y)|^2$',
                    save_path: str = None, show: bool = True, theme: str = 'light',
                    log_scale: bool = False):
    """
    Heatmap of the 2D probability density |ψ(x,y)|² with optional potential overlay.

    Parameters
    ----------
    wf : WaveFunction2D
        The wavefunction to visualise.
    potential : Potential, optional
        If provided, draws a contour overlay of V(x, y).
    title : str
        Plot title.
    save_path : str, optional
        File path to save the figure (PNG, PDF, etc.).
    show : bool
        Whether to call plt.show().
    theme : str
        'light' or 'dark'.
    log_scale : bool
        Use logarithmic colour scale (useful for tunnelling tails).
    """
    set_style(theme)
    cmap = 'viridis' if theme == 'light' else 'inferno'
    bg_color = 'white' if theme == 'light' else '#111111'

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    x = wf.grid.x
    y = wf.grid.y
    density = wf.probability_density.T   # transpose so x→col, y→row for imshow

    extent = [x.min(), x.max(), y.min(), y.max()]
    norm = LogNorm(vmin=max(density.max() * 1e-6, density[density > 0].min()),
                   vmax=density.max()) if log_scale else None

    im = ax.imshow(density, origin='lower', extent=extent, aspect='equal',
                   cmap=cmap, interpolation='bilinear', norm=norm)

    if potential is not None:
        V = potential.evaluate(wf.grid).T
        V_norm = (V - V.min()) / (V.max() - V.min() + 1e-30)
        contour_color = '#ff4444' if theme == 'light' else '#ffaa00'
        ax.contour(x, y, V_norm, levels=5, colors=contour_color, linewidths=0.8, alpha=0.7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Probability Density $|\\Psi|^2$', fontsize=10)
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=14)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=bg_color)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_density_snapshots_2d(wf_list, times, potential=None, ncols: int = 4,
                               title: str = '2D Wave Packet Evolution',
                               save_path: str = None, show: bool = True,
                               theme: str = 'light'):
    """
    Grid of |ψ|² heatmap snapshots at different times.

    Parameters
    ----------
    wf_list : list of WaveFunction2D
        Ordered list of wavefunctions (one per snapshot).
    times : array-like
        Corresponding time values for axis labels.
    ncols : int
        Number of columns in the snapshot grid.
    """
    set_style(theme)
    cmap = 'viridis' if theme == 'light' else 'inferno'
    bg_color = 'white' if theme == 'light' else '#111111'
    text_color = 'black' if theme == 'light' else 'white'

    n = len(wf_list)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.2))
    fig.patch.set_facecolor(bg_color)
    axes = np.array(axes).reshape(-1)

    # Shared colour scale across all frames
    global_max = max(wf.probability_density.max() for wf in wf_list)

    x = wf_list[0].grid.x
    y = wf_list[0].grid.y
    extent = [x.min(), x.max(), y.min(), y.max()]

    for i, (wf, t) in enumerate(zip(wf_list, times)):
        ax = axes[i]
        ax.set_facecolor(bg_color)
        density = wf.probability_density.T
        ax.imshow(density, origin='lower', extent=extent, aspect='equal',
                  cmap=cmap, interpolation='bilinear',
                  vmin=0.0, vmax=global_max)
        if potential is not None:
            V = potential.evaluate(wf.grid).T
            V_norm = (V - V.min()) / (V.max() - V.min() + 1e-30)
            contour_color = '#ff4444' if theme == 'light' else '#ffaa00'
            ax.contour(x, y, V_norm, levels=3, colors=contour_color,
                       linewidths=0.6, alpha=0.6)
        ax.set_title(f't = {t:.2f}', fontsize=9, color=text_color)
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight='bold', color=text_color, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=bg_color)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_orthogonal_slices_3d(wf3d, potential=None,
                               title: str = '3D Wave Packet — Orthogonal Slices',
                               save_path: str = None, show: bool = True,
                               theme: str = 'light'):
    """
    Three-panel plot showing |ψ|² slices through the centroid of a 3D wavefunction:
    XY (z=centre), XZ (y=centre), YZ (x=centre).

    Parameters
    ----------
    wf3d : WaveFunction3D
        The 3D wavefunction to slice.
    potential : Potential, optional
        Draws a contour overlay on each panel.
    """
    set_style(theme)
    cmap = 'viridis' if theme == 'light' else 'inferno'
    bg_color = 'white' if theme == 'light' else '#111111'
    text_color = 'black' if theme == 'light' else 'white'
    contour_color = '#ff4444' if theme == 'light' else '#ffaa00'

    grid = wf3d.grid
    cx = grid.N_x // 2
    cy = grid.N_y // 2
    cz = grid.N_z // 2

    # Find centroid of probability mass for more informative slicing
    prob = wf3d.probability_density
    total = prob.sum()
    if total > 0:
        cx = int(np.round(np.sum(np.arange(grid.N_x)[:, None, None] * prob) / total))
        cy = int(np.round(np.sum(np.arange(grid.N_y)[None, :, None] * prob) / total))
        cz = int(np.round(np.sum(np.arange(grid.N_z)[None, None, :] * prob) / total))
        cx = np.clip(cx, 0, grid.N_x - 1)
        cy = np.clip(cy, 0, grid.N_y - 1)
        cz = np.clip(cz, 0, grid.N_z - 1)

    slice_xy = prob[:, :, cz].T
    slice_xz = prob[:, cy, :].T
    slice_yz = prob[cx, :, :].T

    global_max = max(slice_xy.max(), slice_xz.max(), slice_yz.max())

    fig = plt.figure(figsize=(15, 5))
    fig.patch.set_facecolor(bg_color)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    panels = [
        (gs[0], slice_xy, grid.x, grid.y, 'x', 'y', f'XY  (z = {grid.z[cz]:.1f})'),
        (gs[1], slice_xz, grid.x, grid.z, 'x', 'z', f'XZ  (y = {grid.y[cy]:.1f})'),
        (gs[2], slice_yz, grid.y, grid.z, 'y', 'z', f'YZ  (x = {grid.x[cx]:.1f})'),
    ]

    for spec, data, ax_arr, ay_arr, xlabel, ylabel, sub_title in panels:
        ax = fig.add_subplot(spec)
        ax.set_facecolor(bg_color)
        extent = [ax_arr.min(), ax_arr.max(), ay_arr.min(), ay_arr.max()]
        im = ax.imshow(data, origin='lower', extent=extent, aspect='equal',
                       cmap=cmap, interpolation='bilinear',
                       vmin=0.0, vmax=global_max)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel(xlabel, fontsize=10, color=text_color)
        ax.set_ylabel(ylabel, fontsize=10, color=text_color)
        ax.set_title(sub_title, fontsize=11, fontweight='bold', color=text_color)
        ax.tick_params(colors=text_color)
        for spine in ax.spines.values():
            spine.set_edgecolor(text_color)

    fig.suptitle(title, fontsize=14, fontweight='bold', color=text_color, y=1.02)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=bg_color)
    if show:
        plt.show()
    else:
        plt.close(fig)
