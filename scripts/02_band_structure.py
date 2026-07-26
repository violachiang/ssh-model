#!/usr/bin/env python3
"""
TASK 2: BAND STRUCTURE

Purpose:
    Compute and visualize E(k) for the infinite (bulk) system.
    Show the gap and identify the topological phase transition
    at v = w.

This is the first "physics result" step - it tells us what the
bulk looks like.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.hamiltonian import SSHHamiltonian


def plot_three_phases():
    """Show band structures for trivial, critical, topological phases."""
    print("\n[1/2] Band structures across the phase transition")
    print("-" * 50)

    N = 10
    w = 1.0
    cases = [
        (2.0, 'Trivial  (v=2.0 > w)'),
        (1.0, 'Critical (v=1.0 = w)'),
        (0.5, 'Topological (v=0.5 < w)'),
    ]

    k = np.linspace(-np.pi, np.pi, 500)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (v, label) in zip(axes, cases):
        h = SSHHamiltonian(N, v=v, w=w)
        bands = h.get_bands(k)
        bands_ana = h.get_analytical_bands(k)

        ax.plot(k, bands[:, 0], 'b-', lw=2, label='Lower band')
        ax.plot(k, bands[:, 1], 'r-', lw=2, label='Upper band')
        ax.plot(k, bands_ana[:, 0], 'k--', lw=0.8, alpha=0.5)
        ax.plot(k, bands_ana[:, 1], 'k--', lw=0.8, alpha=0.5)

        gap_at_pi = abs(bands[-1, 1] - bands[-1, 0])
        ax.set_title(f"{label}\nGap at k=π: {gap_at_pi:.3f}", fontsize=11)
        ax.set_xlabel('k')
        ax.set_ylabel('E(k)')
        ax.set_xlim(-np.pi, np.pi)
        ax.set_xticks([-np.pi, 0, np.pi])
        ax.set_xticklabels(['-π', '0', 'π'])
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('figures/02_band_structure.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  → Saved figures/02_band_structure.png")


def plot_gap_closure():
    """Sweep v and track the gap to find the transition point."""
    print("\n[2/2] Gap closure across the transition")
    print("-" * 50)

    N, w = 10, 1.0
    v_values = np.linspace(0.2, 2.0, 200)

    gaps_pi = []
    gaps_0 = []
    k = np.array([0.0, np.pi])

    for v in v_values:
        h = SSHHamiltonian(N, v=v, w=w)
        bands = h.get_bands(k)
        gaps_0.append(abs(bands[0, 1] - bands[0, 0]))
        gaps_pi.append(abs(bands[1, 1] - bands[1, 0]))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(v_values, gaps_0, 'b-', lw=2, label='Gap at k=0')
    ax.plot(v_values, gaps_pi, 'r-', lw=2, label='Gap at k=π')
    ax.axvline(1.0, color='k', ls='--', alpha=0.6, label='v = w (transition)')
    ax.set_xlabel('v (with w = 1.0)')
    ax.set_ylabel('Gap size |E_+ - E_-|')
    ax.set_title('Gap closure: the topological phase transition')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/02_gap_closure.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Find actual minimum
    min_idx = np.argmin(gaps_pi)
    print(f"  Gap at k=π closes near v = {v_values[min_idx]:.3f}")
    print(f"  (theory: v = w = {w})")


if __name__ == "__main__":
    os.makedirs('figures', exist_ok=True)
    print("=" * 55)
    print("TASK 2: BAND STRUCTURE")
    print("=" * 55)
    plot_three_phases()
    plot_gap_closure()
    print("\n Band structure computed")