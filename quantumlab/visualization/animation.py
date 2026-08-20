"""
Time-evolution animation exporters for 2D and 3D simulations.
Requires ffmpeg for MP4 output; falls back to GIF via Pillow.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.colors import LogNorm
from pathlib import Path
from quantumlab.visualization.style import set_style


class WavePacketAnimator:
    """
    Deterministic animation pipeline. No randomness in frame generation.
    Global colour scale is pre-computed across all frames for consistency.
    """
    def __init__(self, wf_series, times, grid, potential=None,
                 theme='dark', log_scale=False, fps=30):
        self.wf_series = wf_series
        self.times = times
        self.grid = grid
        self.potential = potential
        self.theme = theme
        self.log_scale = log_scale
        self.fps = fps
        self.fig = None
        self.ax = None
        self.im = None
        self.title_obj = None

    def _init_frame(self):
        set_style(self.theme)
        cmap = 'viridis' if self.theme == 'light' else 'inferno'
        bg = 'white' if self.theme == 'light' else '#111111'

        self.fig, self.ax = plt.subplots(figsize=(8, 7))
        self.fig.patch.set_facecolor(bg)
        self.ax.set_facecolor(bg)

        # Pre-compute global colour scale for deterministic consistency
        self.global_max = max(wf.probability_density.max() for wf in self.wf_series)
        self.global_min = max(
            self.global_max * 1e-6,
            min(wf.probability_density[wf.probability_density > 0].min()
                for wf in self.wf_series)
        )

        norm = LogNorm(vmin=self.global_min, vmax=self.global_max) if self.log_scale else None

        density = self.wf_series[0].probability_density.T
        extent = [self.grid.x.min(), self.grid.x.max(),
                  self.grid.y.min(), self.grid.y.max()]

        self.im = self.ax.imshow(density, origin='lower', extent=extent,
                                 aspect='equal', cmap=cmap,
                                 interpolation='bilinear', norm=norm,
                                 vmin=0 if not self.log_scale else None,
                                 vmax=self.global_max if not self.log_scale else None)

        if self.potential is not None:
            V = self.potential.evaluate(self.grid).T
            V_norm = (V - V.min()) / (V.max() - V.min() + 1e-30)
            contour_color = '#ff4444' if self.theme == 'light' else '#ffaa00'
            self.ax.contour(self.grid.x, self.grid.y, V_norm, levels=5,
                           colors=contour_color, linewidths=0.8, alpha=0.7)

        self.fig.colorbar(self.im, ax=self.ax, fraction=0.046, pad=0.04)
        self.ax.set_xlabel('x', fontsize=11)
        self.ax.set_ylabel('y', fontsize=11)
        self.title_obj = self.ax.set_title(
            f't = {self.times[0]:.2f}', fontsize=13, fontweight='bold', pad=14
        )
        plt.tight_layout()
        return [self.im]

    def _update_frame(self, frame):
        density = self.wf_series[frame].probability_density.T
        self.im.set_data(density)
        self.title_obj.set_text(f't = {self.times[frame]:.2f}')
        return [self.im]

    def render(self, output_path: str, fmt: str = 'mp4'):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        anim = FuncAnimation(
            self.fig, self._update_frame,
            init_func=self._init_frame,
            frames=len(self.wf_series),
            blit=True, interval=1000 / self.fps
        )

        if fmt == 'mp4':
            writer = FFMpegWriter(fps=self.fps, metadata={
                'title': 'QuantumLab Wave Packet Evolution',
                'artist': 'QuantumLab',
                'comment': 'Deterministic SSFM simulation'
            })
        else:
            writer = PillowWriter(fps=self.fps)

        anim.save(str(output_path), writer=writer)
        plt.close(self.fig)
        print(f"Animation saved: {output_path}")
