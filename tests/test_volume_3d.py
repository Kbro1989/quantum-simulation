import numpy as np
import pytest
from quantumlab.core.grid import Grid3D
from quantumlab.potentials import ApertureScreen

skimage = pytest.importorskip('skimage')
import matplotlib
matplotlib.use('Agg')
from quantumlab.visualization.volume_3d import (
    extract_isosurface, IsosurfaceRenderer, IsosurfaceAnimation, animate_isosurfaces,
    export_interactive_html, _sample_phase,
)


@pytest.fixture
def grid():
    return Grid3D(24, -4.0, 4.0, 24, -4.0, 4.0, 24, -4.0, 4.0)


def test_isosurface_of_sphere_has_correct_radius(grid):
    r = np.sqrt(grid.X ** 2 + grid.Y ** 2 + grid.Z ** 2)
    verts, faces, normals = extract_isosurface(r, 2.5, grid)
    radii = np.linalg.norm(verts, axis=1)
    assert np.abs(radii - 2.5).max() < 0.05
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-6)


def test_isosurface_outside_range_returns_none(grid):
    assert extract_isosurface(np.zeros(grid.shape), 1.0, grid) is None


def test_phase_sampling_follows_plane_wave(grid):
    psi = np.exp(1j * 0.7 * grid.X)
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, -0.5], [-2.0, 1.0, 1.0]])
    np.testing.assert_allclose(_sample_phase(psi, pts, grid), np.angle(np.exp(1j * 0.7 * pts[:, 0])), atol=0.02)


def _vortex(grid):
    return (grid.X + 1j * grid.Y) * np.exp(-(grid.X ** 2 + grid.Y ** 2 + grid.Z ** 2) / 2)


@pytest.mark.parametrize('color_by', ['phase', 'level'])
def test_renderer_writes_png(tmp_path, grid, color_by):
    screen = ApertureScreen(V0=5.0, wall_position=2.0, separation=2.0, aperture='pinhole')
    r = IsosurfaceRenderer(grid, levels=(0.5, 0.1), color_by=color_by, potential=screen)
    path = r.save(_vortex(grid), tmp_path / 'frame.png', t=1.0, title='test', note='note', dpi=40)
    r.close()
    assert path.exists() and path.stat().st_size > 1000


def test_streaming_gif_animation(tmp_path, grid):
    psi = _vortex(grid)
    out = tmp_path / 'anim.gif'
    with IsosurfaceAnimation(out, grid, n_frames=3, fps=5, dpi=30, elev_end=40.0, levels=(0.4,)) as anim:
        for k in range(3):
            anim.add_frame(psi * np.exp(-1j * k), t=float(k))
    assert anim.frames_written == 3
    assert out.stat().st_size > 1000


def test_animate_isosurfaces_helper(tmp_path, grid):
    frames = [_vortex(grid) * np.exp(-1j * k) for k in range(2)]
    out = animate_isosurfaces(frames, [0.0, 1.0], grid, tmp_path / 'a.gif', fps=5, dpi=30, progress=False,
                              levels=(0.3,))
    assert out.exists()


def test_interactive_html_export(tmp_path, grid):
    pytest.importorskip('plotly')
    out = export_interactive_html([_vortex(grid)] * 2, [0.0, 0.5], grid, tmp_path / 'v.html', stride=1)
    text = out.read_text()
    assert 'plotly' in text.lower() and 'Mesh3d'.lower() in text.lower()


def test_renderer_rejects_2d_grid():
    from quantumlab.core.grid import Grid2D
    with pytest.raises(TypeError):
        IsosurfaceRenderer(Grid2D(8, -1, 1, 8, -1, 1))


def test_renderer_draws_overlay_surfaces(tmp_path, grid):
    r = np.sqrt(grid.X ** 2 + grid.Y ** 2)
    overlay = {'field': r, 'level': 0.8, 'color': '#00ffff', 'alpha': 1.0}
    renderer = IsosurfaceRenderer(grid, levels=(0.2,), color_by='level')
    fig = renderer.draw(_vortex(grid), t=0.0, overlays=[overlay])
    # one collection for the overlay tube plus one for the density shell
    assert len(fig.axes[0].collections) == 2
    renderer.save(_vortex(grid), tmp_path / 'ov.png', overlays=[overlay], dpi=30)
    renderer.close()
