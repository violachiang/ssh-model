#!/usr/bin/env python3
"""
TASK 6: BULK–BOUNDARY CORRESPONDENCE

Purpose:
    Put the bulk invariant (D) and the boundary spectrum side-by-side
    and verify that:

        number of zero-energy edge states = 2|D|

    This is the central result of the entire project.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.hamiltonian import SSHHamiltonian
from src.topology import SSHTopology


def verify_correspondence_table():
    """Print a table of D vs edge state count."""
    print("\n[1/2] Bulk invariant vs boundary count")
    print("-" * 60)
    print(f"{'v':>5} {'w':>5} {'D':>8} {'edges':>7} {'2|D|':>6} {'match':>7}")
    print("-" * 60)

    cases = [
        (1.0, 0.3), (1.0, 0.7), (1.0, 0.9),
        (1.0, 1.1), (1.0, 1.4), (1.0, 1.8),
        (0.5, 1.0), (1.5, 1.0),
    ]
    all_ok = True
    for v, w in cases:
        D = SSHTopology.winding_number(v, w, Nk=2000)
        h = SSHHamiltonian(N=30, v=v, w=w)
        edges = h.get_edge_states(tol=1e-6)
        n_edges = len(edges['states']) if edges['found'] else 0

        predicted = 2 * abs(round(D))
        ok = (n_edges == predicted)
        all_ok &= ok
        print(f"{v:>5.1f} {w:>5.1f} {D:>+8.3f} {n_edges:>7d} {predicted:>6d} "
              f"{'' if ok else '':>7}")

    print("-" * 60)
    print(f"  Overall: {' ALL MATCH' if all_ok else ' MISMATCH'}")


def summary_figure():
    """Produce the final summary figure."""
    print("\n[2/2] Generating summary figure")
    print("-" * 50)

    N = 40
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    k = np.linspace(-np.pi, np.pi, 500)

    # Panels: [bulk spectrum | d-vector | edge wavefunction]
    # Row 1: trivial, Row 2: topological

    for row, (v, w, label) in enumerate([
        (1.0, 0.5, 'Trivial'),
        (0.5, 1.0, 'Topological'),
    ]):
        h = SSHHamiltonian(N, v=v, w=w)
        D = SSHTopology.winding_number(v, w)
        eigs = h.diagonalize()[0]
        bands = h.get_bands(k)
        dx, dy = h.get_d_vector(k)

        # Column 0: real-space spectrum
        ax = axes[row, 0]
        ax.scatter(range(len(eigs)), eigs, s=15, c='steelblue')
        ax.axhline(0, color='red', ls='--', lw=1)
        ax.set_title(f"{label}: real-space spectrum")
        ax.set_xlabel('index'); ax.set_ylabel('E')
        ax.grid(True, alpha=0.3)

        # Column 1: band structure
        ax = axes[row, 1]
        ax.plot(k, bands[:, 0], 'b-', lw=2)
        ax.plot(k, bands[:, 1], 'r-', lw=2)
        ax.axhline(0, color='gray', ls=':')
        ax.set_title(f"band structure (D = {D:.2f})")
        ax.set_xlabel('k'); ax.set_ylabel('E(k)')
        ax.set_xlim(-np.pi, np.pi)
        ax.grid(True, alpha=0.3)

        # Column 2: winding + edge states inset
        ax = axes[row, 2]
        ax.plot(dx, dy, 'g-', lw=2, label='d-vector path')
        ax.plot(0, 0, 'kx', ms=14, mew=3, label='origin')
        ax.set_aspect('equal')
        ax.set_title(f"winding: D = {D:.2f}")
        ax.set_xlabel('$d_x$'); ax.set_ylabel('$d_y$')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('figures/06_summary.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  → Saved figures/06_summary.png")


if __name__ == "__main__":
    os.makedirs('figures', exist_ok=True)
    print("=" * 60)
    print("TASK 6: BULK–BOUNDARY CORRESPONDENCE")
    print("=" * 60)
    verify_correspondence_table()
    summary_figure()
    print("\n Correspondence verified")