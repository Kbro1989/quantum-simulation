from quantumlab.visualization.style import set_style
from quantumlab.visualization.plots_1d import plot_wavefunction_1d, plot_dual_space_1d
from quantumlab.visualization.plots_3d import plot_space_time_3d
from quantumlab.visualization.plots_2d import (
    plot_density_2d,
    plot_density_snapshots_2d,
    plot_orthogonal_slices_3d,
)

try:
    from quantumlab.visualization.volume_3d import (
        IsosurfaceRenderer, IsosurfaceAnimation, animate_isosurfaces, export_interactive_html,
    )
except ImportError:  # matplotlib 3D toolkit / scipy missing
    pass

__all__ = [
    'set_style',
    'plot_wavefunction_1d', 'plot_dual_space_1d',
    'plot_space_time_3d',
    'plot_density_2d', 'plot_density_snapshots_2d', 'plot_orthogonal_slices_3d',
    'IsosurfaceRenderer', 'IsosurfaceAnimation', 'animate_isosurfaces', 'export_interactive_html',
]
