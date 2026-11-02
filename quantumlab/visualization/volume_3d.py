"""
3D volumetric rendering of wavefunctions: nested |ψ|² isosurfaces extracted
with marching cubes, coloured by the local quantum phase arg ψ and lit with
a simple Lambert model. Supports orbiting-camera animations (MP4/GIF) and
interactive HTML export (Plotly).

Requires scikit-image (marching cubes); Plotly is only needed for HTML export.
"""
from pathlib import Path
from typing import Optional, Sequence
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize, to_rgb
from matplotlib.animation import FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import map_coordinates
from quantumlab.core.grid import Grid3D
from quantumlab.visualization.style import set_style

try:
    from skimage.measure import marching_cubes
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

PHASE_CMAP = 'hsv'


def _require_skimage():
    if not SKIMAGE_AVAILABLE:
        raise ImportError('3D isosurface rendering requires scikit-image: pip install scikit-image')


def extract_isosurface(field: np.ndarray, level: float, grid: Grid3D, step_size: int = 1):
    """
    Marching-cubes isosurface of a scalar field sampled on ``grid``.

    Returns (vertices in physical coordinates, faces, unit face normals),
    or None if the level is outside the field range.
    """
    _require_skimage()
    if not (field.min() < level < field.max()):
        return None
    spacing = (grid.dx, grid.dy, grid.dz)
    verts, faces, _, _ = marching_cubes(field, level=level, spacing=spacing, step_size=step_size)
    verts = verts + np.array([grid.x[0], grid.y[0], grid.z[0]])
    tri = verts[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-30
    return verts, faces, normals


def _sample_phase(psi: np.ndarray, points: np.ndarray, grid: Grid3D) -> np.ndarray:
    """Phase arg ψ at arbitrary physical points (trilinear interpolation of Re/Im)."""
    idx = np.stack([(points[:, 0] - grid.x[0]) / grid.dx,
                    (points[:, 1] - grid.y[0]) / grid.dy,
                    (points[:, 2] - grid.z[0]) / grid.dz])
    re = map_coordinates(psi.real, idx, order=1, mode='nearest')
    im = map_coordinates(psi.imag, idx, order=1, mode='nearest')
    return np.arctan2(im, re)


class IsosurfaceRenderer:
    """
    Render |ψ(x, y, z)|² as nested translucent isosurfaces.

    Parameters
    ----------
    grid : Grid3D
    levels : sequence of float
        Isosurface levels as fractions of the reference peak density.
    alphas : sequence of float, optional
        Opacity per level (defaults fade from opaque inner to faint outer).
    color_by : {'phase', 'level'}
        'phase' colours each face by arg ψ on a cyclic map; 'level' uses one
        colour per isosurface.
    reference_peak : float, optional
        Density that ``levels`` refer to. If None, each frame uses its own peak.
        Pass the global peak of a series for a fixed, comparable scale.
    potential : Potential, optional
        Drawn as a grey translucent isosurface at ``potential_level`` · max V.
    step_size : int
        Marching-cubes step (2 roughly quarters the triangle count).
    """

    def __init__(self, grid: Grid3D, levels: Sequence[float] = (0.5, 0.15, 0.03),
                 alphas: Optional[Sequence[float]] = None, color_by: str = 'phase',
                 reference_peak: Optional[float] = None, potential=None,
                 potential_level: float = 0.5, theme: str = 'dark', step_size: int = 1,
                 figsize=(9, 8), light_direction=(0.4, -0.5, 0.75), show_axes: bool = True,
                 zoom: float = 1.15, phase_cmap: str = PHASE_CMAP):
        if not isinstance(grid, Grid3D):
            raise TypeError('IsosurfaceRenderer requires a Grid3D')
        if color_by not in ('phase', 'level'):
            raise ValueError("color_by must be 'phase' or 'level'")
        self.grid = grid
        self.levels = tuple(sorted(levels, reverse=True))
        n = len(self.levels)
        self.alphas = tuple(alphas) if alphas is not None else tuple(np.linspace(0.9, 0.18, n)) if n > 1 else (0.85,)
        self.color_by = color_by
        self.reference_peak = reference_peak
        self.theme = theme
        self.step_size = step_size
        self.figsize = figsize
        self.light = np.asarray(light_direction, dtype=float) / np.linalg.norm(light_direction)
        self.show_axes = show_axes
        self.zoom = zoom
        self.phase_cmap = phase_cmap
        self._potential_mesh = None
        if potential is not None:
            V = potential.evaluate(grid)
            if V.max() > V.min():
                self._potential_mesh = extract_isosurface(V, V.min() + potential_level * (V.max() - V.min()),
                                                          grid, step_size=1)
        self.fig = None
        self.ax = None

    # ------------------------------------------------------------------ drawing
    def _shade(self, rgb: np.ndarray, normals: np.ndarray) -> np.ndarray:
        lambert = np.abs(normals @ self.light)
        intensity = 0.35 + 0.65 * lambert
        return np.clip(rgb * intensity[:, None], 0.0, 1.0)

    def _setup_axes(self):
        set_style(self.theme)
        bg = '#0b0b10' if self.theme == 'dark' else 'white'
        self.fig = plt.figure(figsize=self.figsize)
        self.fig.patch.set_facecolor(bg)
        self.ax = self.fig.add_axes([-0.06, -0.02, 0.87, 0.94], projection='3d')
        self.ax.set_facecolor(bg)
        self._bg = bg
        self._title = self.fig.text(0.5, 0.965, '', ha='center', va='top', fontsize=14, fontweight='bold')
        self._time = self.fig.text(0.03, 0.965, '', ha='left', va='top', fontsize=12,
                                   family='monospace', fontweight='bold')
        self._caption = self.fig.text(0.03, 0.02, '', ha='left', va='bottom', fontsize=8, alpha=0.75)
        cax = self.fig.add_axes([0.9, 0.28, 0.018, 0.42])
        if self.color_by == 'phase':
            sm = cm.ScalarMappable(norm=Normalize(-np.pi, np.pi), cmap=self.phase_cmap)
            cbar = self.fig.colorbar(sm, cax=cax, ticks=[-np.pi, 0, np.pi])
            cbar.ax.set_yticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
            cbar.set_label(r'phase  arg $\psi$', fontsize=10)
        else:
            cax.set_visible(False)

    def _style_axes(self, elev, azim):
        ax, g = self.ax, self.grid
        ax.set_xlim(g.x[0], g.x[-1])
        ax.set_ylim(g.y[0], g.y[-1])
        ax.set_zlim(g.z[0], g.z[-1])
        ax.set_box_aspect((g.L_x, g.L_y, g.L_z), zoom=self.zoom)
        ax.view_init(elev=elev, azim=azim)
        edge = '#3a3a48' if self.theme == 'dark' else '#cccccc'
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis.pane.set_edgecolor(edge)
            axis._axinfo['grid']['color'] = edge
            axis._axinfo['grid']['linewidth'] = 0.4
        if self.show_axes:
            ax.set_xlabel('x', labelpad=2)
            ax.set_ylabel('y', labelpad=2)
            ax.set_zlabel('z', labelpad=2)
            ax.tick_params(labelsize=7, pad=0)
        else:
            ax.set_axis_off()

    def draw(self, psi: np.ndarray, t: Optional[float] = None, elev: float = 22.0,
             azim: float = -58.0, title: Optional[str] = None, note: Optional[str] = None,
             overlays: Optional[Sequence[dict]] = None):
        """
        Draw one frame (re-uses the figure across calls) and return it.

        ``overlays`` is a list of extra opaque surfaces drawn underneath the
        translucent density shells, each a dict with keys ``field`` (array on
        the grid), ``level``, and optionally ``color`` and ``alpha``. Example:
        vortex cores as an isosurface of the normalised density.
        """
        if self.fig is None:
            self._setup_axes()
        ax = self.ax
        ax.cla()
        if overlays:
            # Keep the drawing order: opaque overlays first, translucent shells on top
            ax.computed_zorder = False
            for ov in overlays:
                mesh = extract_isosurface(np.asarray(ov['field']), ov['level'], self.grid, self.step_size)
                if mesh is None:
                    continue
                verts, faces, normals = mesh
                rgb = np.tile(to_rgb(ov.get('color', '#4fd8ff')), (len(faces), 1))
                ax.add_collection3d(Poly3DCollection(verts[faces], facecolors=self._shade(rgb, normals),
                                                     edgecolor='none', alpha=ov.get('alpha', 1.0)))
        psi = np.asarray(psi)
        density = np.abs(psi) ** 2
        peak = self.reference_peak if self.reference_peak is not None else density.max()

        if self._potential_mesh is not None:
            verts, faces, normals = self._potential_mesh
            grey = np.tile(to_rgb('#9aa0b5' if self.theme == 'dark' else '#7a7f90'), (len(faces), 1))
            coll = Poly3DCollection(verts[faces], facecolors=self._shade(grey, normals),
                                    edgecolor='none', alpha=0.22)
            ax.add_collection3d(coll)

        level_colors = cm.plasma(np.linspace(0.95, 0.2, len(self.levels)))[:, :3]
        phase_norm = Normalize(-np.pi, np.pi)
        phase_cmap = plt.get_cmap(self.phase_cmap)
        # Draw outer (faint) shells first so inner shells are composited on top
        for level, alpha, lc in reversed(list(zip(self.levels, self.alphas, level_colors))):
            mesh = extract_isosurface(density, level * peak, self.grid, self.step_size)
            if mesh is None:
                continue
            verts, faces, normals = mesh
            if self.color_by == 'phase':
                centroids = verts[faces].mean(axis=1)
                rgb = phase_cmap(phase_norm(_sample_phase(psi, centroids, self.grid)))[:, :3]
            else:
                rgb = np.tile(lc, (len(faces), 1))
            coll = Poly3DCollection(verts[faces], facecolors=self._shade(rgb, normals),
                                    edgecolor='none', alpha=alpha)
            ax.add_collection3d(coll)

        self._style_axes(elev, azim)
        levels_txt = ', '.join(f'{100 * lv:g}%' for lv in self.levels)
        ref_txt = 'global peak' if self.reference_peak is not None else 'frame peak'
        caption = f'isosurfaces of $|\\psi|^2$ at {levels_txt} of {ref_txt}'
        if note:
            caption = f'{caption}   |   {note}'
        self._title.set_text(title or '')
        self._caption.set_text(caption)
        self._time.set_text('' if t is None else f't = {t:6.2f}')
        return self.fig

    def save(self, psi, path, t=None, elev=22.0, azim=-58.0, title=None, note=None, dpi=150,
             overlays=None):
        fig = self.draw(psi, t=t, elev=elev, azim=azim, title=title, note=note, overlays=overlays)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, facecolor=self._bg)
        return path

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None


class IsosurfaceAnimation:
    """
    Streaming MP4/GIF writer: frames are rendered as they are added, so a long
    high-resolution simulation never has to keep its snapshots in memory.

    Usage::

        with IsosurfaceAnimation('out.mp4', grid, n_frames=120, reference_peak=p0) as anim:
            for ...:
                anim.add_frame(wf.psi, t)

    The camera orbits from ``azim_start`` by ``orbit_degrees`` over ``n_frames``;
    if ``elev_end`` is given the elevation moves linearly from ``elev`` to it.
    """

    def __init__(self, output_path, grid: Grid3D, n_frames: int, fps: int = 20, dpi: int = 100,
                 elev: float = 22.0, azim_start: float = -60.0, orbit_degrees: float = 90.0,
                 title: Optional[str] = None, note: Optional[str] = None,
                 elev_end: Optional[float] = None, **renderer_kwargs):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.n_frames = n_frames
        self.dpi = dpi
        self.elev = elev
        self.elev_end = elev if elev_end is None else elev_end
        self.azim_start = azim_start
        self.orbit_degrees = orbit_degrees
        self.title = title
        self.note = note
        self.renderer = IsosurfaceRenderer(grid, **renderer_kwargs)
        self.renderer._setup_axes()
        if self.output_path.suffix.lower() == '.gif':
            self.writer = PillowWriter(fps=fps)
        else:
            self.writer = FFMpegWriter(fps=fps, codec='libx264',
                                       extra_args=['-pix_fmt', 'yuv420p', '-crf', '20'],
                                       metadata={'title': title or 'QuantumLab 3D evolution',
                                                 'artist': 'QuantumLab'})
        self.frames_written = 0

    def __enter__(self):
        self.writer.setup(self.renderer.fig, str(self.output_path), self.dpi)
        return self

    def add_frame(self, psi: np.ndarray, t: Optional[float] = None, overlays=None):
        frac = self.frames_written / max(self.n_frames - 1, 1)
        azim = self.azim_start + self.orbit_degrees * frac
        elev = self.elev + (self.elev_end - self.elev) * frac
        self.renderer.draw(psi, t=t, elev=elev, azim=azim, title=self.title, note=self.note,
                           overlays=overlays)
        self.writer.grab_frame(facecolor=self.renderer._bg)
        self.frames_written += 1

    def __exit__(self, exc_type, exc, tb):
        self.writer.finish()
        self.renderer.close()
        return False


def animate_isosurfaces(snapshots: Sequence[np.ndarray], times: Sequence[float], grid: Grid3D,
                        output_path, fps: int = 20, dpi: int = 100, elev: float = 22.0,
                        azim_start: float = -60.0, orbit_degrees: float = 90.0,
                        title: Optional[str] = None, note: Optional[str] = None,
                        fixed_scale: bool = True, progress: bool = True, **renderer_kwargs):
    """
    Render a series of 3D wavefunctions (complex arrays on ``grid``) to MP4 or
    GIF while the camera orbits by ``orbit_degrees`` over the clip.

    With ``fixed_scale=True`` the isosurface levels refer to the peak density
    of the whole series, so shrinking/spreading is shown faithfully.
    """
    if fixed_scale and 'reference_peak' not in renderer_kwargs:
        renderer_kwargs['reference_peak'] = max(float(np.max(np.abs(s) ** 2)) for s in snapshots)
    n = len(snapshots)
    with IsosurfaceAnimation(output_path, grid, n, fps=fps, dpi=dpi, elev=elev, azim_start=azim_start,
                             orbit_degrees=orbit_degrees, title=title, note=note,
                             **renderer_kwargs) as anim:
        for i, (psi, t) in enumerate(zip(snapshots, times)):
            anim.add_frame(psi, t)
            if progress and (i % 10 == 0 or i == n - 1):
                print(f'  rendered frame {i + 1}/{n}')
    return Path(output_path)


def export_interactive_html(snapshots: Sequence[np.ndarray], times: Sequence[float], grid: Grid3D,
                            output_path, level: float = 0.1, stride: int = 2,
                            title: str = 'QuantumLab — 3D wavefunction'):
    """
    Interactive, rotatable HTML (Plotly Mesh3d) of one |ψ|² isosurface per
    snapshot, coloured by phase, with a time slider. ``stride`` subsamples the
    grid to keep the file small. Plotly.js is loaded from its CDN.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as err:
        raise ImportError('Interactive export requires plotly: pip install plotly') from err
    _require_skimage()
    sub = Grid3D(len(grid.x[::stride]), grid.x[0], grid.x[0] + grid.L_x,
                 len(grid.y[::stride]), grid.y[0], grid.y[0] + grid.L_y,
                 len(grid.z[::stride]), grid.z[0], grid.z[0] + grid.L_z)
    peak = max(float(np.max(np.abs(s) ** 2)) for s in snapshots)

    def mesh_for(psi):
        psi = np.asarray(psi)[::stride, ::stride, ::stride]
        mesh = extract_isosurface(np.abs(psi) ** 2, level * peak, sub)
        if mesh is None:
            return go.Mesh3d(x=[], y=[], z=[])
        verts, faces, _ = mesh
        phase = _sample_phase(psi, verts, sub)
        return go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                         i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                         intensity=phase, cmin=-np.pi, cmax=np.pi, colorscale='Twilight',
                         colorbar=dict(title='arg ψ'), flatshading=False,
                         lighting=dict(ambient=0.45, diffuse=0.8, specular=0.3), opacity=0.9)

    frames = [go.Frame(data=[mesh_for(s)], name=f'{t:.2f}') for s, t in zip(snapshots, times)]
    fig = go.Figure(data=frames[0].data, frames=frames)
    steps = [dict(method='animate', label=f.name,
                  args=[[f.name], dict(mode='immediate', frame=dict(duration=0, redraw=True))]) for f in frames]
    fig.update_layout(
        title=f'{title}  (|ψ|² = {100 * level:g}% of peak)', template='plotly_dark',
        scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z', aspectmode='data'),
        sliders=[dict(steps=steps, currentvalue=dict(prefix='t = '))],
        updatemenus=[dict(type='buttons', showactive=False, buttons=[
            dict(label='▶ Play', method='animate',
                 args=[None, dict(frame=dict(duration=120, redraw=True), fromcurrent=True)]),
            dict(label='⏸ Pause', method='animate',
                 args=[[None], dict(mode='immediate', frame=dict(duration=0, redraw=False))])])])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs='cdn')
    return output_path
