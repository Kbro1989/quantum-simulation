"""
Example 10: Quantum chaos — eigenstates of the Bunimovich stadium billiard.

The classical stadium billiard is fully chaotic. Its quantum eigenstates
are computed with the matrix-free FFT Hamiltonian and the k-space
preconditioned LOBPCG eigensolver. Many states look like random waves,
while some concentrate along unstable periodic orbits ("scars").

Checks:
  * the lowest levels agree between LOBPCG and imaginary-time propagation;
  * the level staircase N(E) follows Weyl's law with the Dirichlet
    perimeter correction,  N(E) ≈ A·E/(2π) − P·√(2E)/(4π)  (ℏ = m = 1).
    The walls here are soft (finite V0, smoothed over a short length), so
    the effective boundary is only approximately sharp.

Outputs (in output/):
    10_stadium_eigenstates.png   gallery of |ψ_n|²
    10_stadium_weyl.png          level staircase vs Weyl's law
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from quantumlab.core.grid import Grid2D
from quantumlab.potentials import StadiumBilliard
from quantumlab.stationary import eigenstates, excited_states
from quantumlab.visualization.style import set_style


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    radius, length = 2.0, 4.0
    grid = Grid2D(176, -5.5, 5.5, 96, -3.0, 3.0)
    stadium = StadiumBilliard(V0=300.0, radius=radius, length=length, smoothing=0.08)
    n_states = 60

    os.makedirs('output', exist_ok=True)
    print('--- Example 10: eigenstates of the chaotic stadium billiard ---')
    print(f'Grid {grid.N_x}×{grid.N_y}, stadium R = {radius}, L = {length}, {n_states} states (LOBPCG)')
    energies, states = eigenstates(grid, stadium, n_states=n_states, method='lobpcg', tol=1e-6)
    print(f'  E_0 = {energies[0]:.6f}   E_{n_states - 1} = {energies[-1]:.6f}')

    e_it, _, infos = excited_states(grid, stadium, n_states=3, dtau=0.002)
    print('\n  Cross-check of the lowest levels:')
    for n in range(3):
        print(f'    n = {n}:  LOBPCG {energies[n]:.6f}   imaginary time {e_it[n]:.6f}   '
              f'|Δ| = {abs(energies[n] - e_it[n]):.1e}')

    area = np.pi * radius ** 2 + 2 * radius * length
    perimeter = 2 * np.pi * radius + 2 * length
    E = np.linspace(0, energies[-1], 400)
    weyl = area * E / (2 * np.pi) - perimeter * np.sqrt(2 * E) / (4 * np.pi)
    weyl_at_levels = area * energies / (2 * np.pi) - perimeter * np.sqrt(2 * energies) / (4 * np.pi)
    staircase = np.arange(1, n_states + 1)
    print(f'\n  Weyl check: N({energies[-1]:.2f}) = {n_states}, Weyl estimate = '
          f'{weyl_at_levels[-1]:.1f}; mean |N − N_Weyl| = {np.mean(np.abs(staircase - 0.5 - weyl_at_levels)):.2f}')

    picks = [0, 1, 2, 3, 5, 8, 12, 17, 23, 30, 38, 44, 49, 53, 56, 59]
    set_style('dark')
    fig, axes = plt.subplots(4, 4, figsize=(14, 8.2))
    for ax, n in zip(axes.flat, picks):
        ax.imshow(states[n].probability_density.T, origin='lower', cmap='magma',
                  extent=[grid.x[0], grid.x[-1] + grid.dx, grid.y[0], grid.y[-1] + grid.dy])
        ax.set_xlim(-4.3, 4.3)
        ax.set_ylim(-2.3, 2.3)
        ax.set_title(f'n = {n},  E = {energies[n]:.2f}', fontsize=9)
        ax.axis('off')
    fig.suptitle('Eigenstates $|\\psi_n|^2$ of the Bunimovich stadium (classically chaotic)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig('output/10_stadium_eigenstates.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(np.concatenate([[0], energies]), np.concatenate([[0], staircase]), where='post',
            color='#ffb000', lw=1.8, label='numerical $N(E)$')
    ax.plot(E, weyl, color='#7fd4ff', ls='--', lw=1.5, label="Weyl: $AE/2\\pi - P\\sqrt{2E}/4\\pi$")
    ax.set_xlabel('E')
    ax.set_ylabel('number of levels below E')
    ax.set_title('Level staircase vs Weyl\'s law', fontweight='bold')
    ax.legend()
    fig.tight_layout()
    fig.savefig('output/10_stadium_weyl.png', dpi=150)
    plt.close(fig)
    print('\nPlots saved: output/10_stadium_eigenstates.png, output/10_stadium_weyl.png')


if __name__ == '__main__':
    main()
