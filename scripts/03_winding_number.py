#!/usr/bin/env python3
"""
TASK 3: WINDING NUMBER

Purpose:
    Compute the topological invariant D from the d-vector of H(k).
    Visualize the winding trajectory and produce a phase diagram.

This is the "topology" step - it answers: what phase are we in?
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.hamiltonian import SSHHamiltonian
from src.topology import SSHTopology


def compute_D_for_phases():
    """Compute D for representative points."""
    print("\n[1/3] Winding numbers for representative phases")
    print("-" * 50)

    cases = [
        (1.0, 0.5, 'Trivial'),
        (0.5, 1.0, 'Topological'),
        (1.0, 1.0, 'Critical'),
    ]
    for v, w, label in cases:
        D = SSHTopology.winding_number(v, w, Nk=2000)
        D_ana = SSHTopology.winding_number_analytic(v, w)
        print(f"  {label:12s}  v={v:.1f}, w={w:.1f}  →  D = {D:+.4f}  (analytic: {D_ana:.1f})")


def plot_winding_trajectories():
    """Show the d-vector path for trivial vs topological."""
    print("\n[2/3] Winding trajectories in d-space")
    print("-" * 50)

    N = 10
    k = np.linspace(-np.pi, np.pi, 400)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax, (v, w, title) in zip(axes, [
        (1.0, 0.5, 'Trivial (v > w): circle misses origin'),
        (0.5, 1.0, 'Topological (v < w): circle encloses origin'),
    ]):
        h = SSHHamiltonian(N, v=v, w=w)
        dx, dy = h.get_d_vector(k)
        D = SSHTopology.winding_number(v, w)

        ax.plot(dx, dy, 'b-', lw=2)
        ax.plot(dx[0], dy[0], 'go', ms=10, label='k = -π')
        ax.plot(dx[-1], dy[-1], 'ro', ms=10, label='k = +π')
        ax.plot(0, 0, 'kx', ms=15, mew=3, label='origin')

        # Arrows to show direction
        for i in range(0, len(k) - 20, 40):
            ax.annotate('', xy=(dx[i+20], dy[i+20]), xytext=(dx[i], dy[i]),
                        arrowprops=dict(arrowstyle='->', color='blue', alpha=0.5))

        ax.set_aspect('equal')
        ax.set_xlabel('$d_x(k)$')
        ax.set_ylabel('$d_y(k)$')
        ax.set_title(f"{title}\nD = {D:.3f}", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('figures/03_winding_trajectories.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  → Saved figures/03_winding_trajectories.png")


def plot_phase_diagram():
    """Compute D over a grid of (v, w)."""
    print("\n[3/3] Topological phase diagram")
    print("-" * 50)

    D_map, v_vals, w_vals = SSHTopology.phase_diagram(
        v_range=(0.2, 2.0), w_range=(0.2, 2.0), N_points=60
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(D_map, extent=[v_vals[0], v_vals[-1], w_vals[0], w_vals[-1]],
                   origin='lower', cmap='RdBu', aspect='auto', vmin=-0.1, vmax=1.1)
    ax.plot([0.2, 2.0], [0.2, 2.0], 'k--', lw=2, label='v = w')
    ax.set_xlabel('v')
    ax.set_ylabel('w')
    ax.set_title('Winding number D(v, w)')
    ax.legend()
    plt.colorbar(im, ax=ax, label='D')

    plt.tight_layout()
    plt.savefig('figures/03_phase_diagram.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  → Saved figures/03_phase_diagram.png")


if __name__ == "__main__":
    os.makedirs('figures', exist_ok=True)
    print("=" * 55)
    print("TASK 3: WINDING NUMBER")
    print("=" * 55)
    compute_D_for_phases()
    plot_winding_trajectories()
    plot_phase_diagram()
    print("\n Topological invariant computed")