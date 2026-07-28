#!/usr/bin/env python3
"""
TASK 4: ENERGY SPECTRUM (Open Boundary Conditions)

Purpose:
    Diagonalize the finite real-space Hamiltonian and inspect the
    eigenvalues. Show how the spectrum changes across the phase
    transition and where mid-gap states appear.

This is the first OBC (boundary) analysis step.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.hamiltonian import SSHHamiltonian


def plot_spectrum_two_phases():
    """Compare spectra in trivial and topological phases."""
    print("\n[1/2] Spectrum in each phase")
    print("-" * 50)

    N, v = 20, 1.0

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, w, title in zip(axes, [0.5, 1.5],
                             ['Trivial (w=0.5 < v)', 'Topological (w=1.5 > v)']):
        h = SSHHamiltonian(N, v=v, w=w)
        eigs = h.diagonalize()[0]
        info = h.get_spectrum_info()

        ax.scatter(range(len(eigs)), eigs, s=30, c='steelblue',
                   edgecolors='k', linewidths=0.5, zorder=3)
        ax.axhline(0, color='red', ls='--', lw=1, alpha=0.6)
        ax.set_xlabel('Eigenstate index')
        ax.set_ylabel('Energy E')
        ax.set_title(f"{title}\ngap = {info['gap']:.3f}, "
                     f"zero-modes = {info['num_zero_states']}", fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/04_spectrum_two_phases.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  → Saved figures/04_spectrum_two_phases.png")


def plot_spectrum_vs_w():
    """Sweep w and track every eigenvalue."""
    print("\n[2/2] Spectrum as w varies")
    print("-" * 50)

    N, v = 20, 1.0
    w_values = np.linspace(0, 2.5, 150)
    all_eigs = np.array([SSHHamiltonian(N, v, w).diagonalize()[0] for w in w_values])

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(2 * N):
        ax.plot(w_values, all_eigs[:, i], 'b-', lw=0.4, alpha=0.4)

    ax.axvline(v, color='red', ls='--', lw=2, label='w = v (transition)')
    ax.axhline(0, color='black', lw=0.5, alpha=0.4)
    ax.set_xlabel('w (with v = 1.0)')
    ax.set_ylabel('Energy E')
    ax.set_title(f'Energy spectrum vs w (N={N})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/04_spectrum_vs_w.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Detect edge states (states near E=0 for w > v)
    zero_mask = np.abs(all_eigs) < 1e-6
    w_with_zero_modes = w_values[np.any(zero_mask, axis=1)]
    if len(w_with_zero_modes):
        print(f"  Zero modes appear for w ≥ {w_with_zero_modes.min():.3f}")
        print(f"  Predicted transition: w = v = {v}")


if __name__ == "__main__":
    os.makedirs('figures', exist_ok=True)
    print("=" * 55)
    print("TASK 4: ENERGY SPECTRUM")
    print("=" * 55)
    plot_spectrum_two_phases()
    plot_spectrum_vs_w()
    print("\n Spectrum computed")