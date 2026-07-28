#!/usr/bin/env python3
"""
TASK 5: EDGE STATE CHARACTERIZATION

Purpose:
    Take the zero-energy states found in the OBC spectrum and
    characterize them: where are they? Are they localized?
    How fast do they decay?

This produces the direct "edge state" evidence.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.hamiltonian import SSHHamiltonian
from src.analysis import SSHAnalysis


def plot_edge_wavefunctions():
    """Plot the two zero-energy wavefunctions."""
    print("\n[1/2] Edge state wavefunctions")
    print("-" * 50)

    N, v, w = 40, 0.4, 1.0
    h = SSHHamiltonian(N, v=v, w=w)
    edge = h.get_edge_states(tol=1e-6)

    if not edge['found']:
        print("   No edge states found")
        return

    print(f"  Found {len(edge['states'])} zero-energy states")
    sites = np.arange(2 * N)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, state in zip(axes, edge['states']):
        prob = state['probability']
        com = SSHAnalysis.center_of_mass(state['wavefunction'])
        xi = SSHAnalysis.localization_length(state['wavefunction'])

        ax.semilogy(sites, prob + 1e-14, 'b-', lw=2)
        ax.set_xlabel('Site index')
        ax.set_ylabel(r'$|\psi(x)|^2$ (log scale)')
        ax.set_title(f"E = {state['energy']:+.2e}\n"
                     f"COM = {com:.1f}, spread = {xi:.2f}", fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Edge states: exponential decay (N={N}, v={v}, w={w})', y=1.02)
    plt.tight_layout()
    plt.savefig('figures/05_edge_wavefunctions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  → Saved figures/05_edge_wavefunctions.png")


def fit_decay_lengths():
    """Fit the exponential decay length for each edge state."""
    print("\n[2/2] Exponential decay fits")
    print("-" * 50)

    N, v = 60, 0.3
    w_values = [0.5, 0.7, 0.9, 1.0, 1.2, 1.5]

    fig, ax = plt.subplots(figsize=(9, 5))

    for w in w_values:
        h = SSHHamiltonian(N, v=v, w=w)
        edge = h.get_edge_states(tol=1e-6)
        if not edge['found']:
            continue
        for state in edge['states']:
            fit = SSHAnalysis.fit_exponential_decay(state['probability'], N)
            if fit['success'] and np.isfinite(fit['decay_length']):
                # One marker per (w, edge state)
                ax.scatter(w, fit['decay_length'], s=80, alpha=0.7,
                           edgecolors='k', zorder=3)

    ax.set_xlabel('w (with v = 0.3)')
    ax.set_ylabel(r'Decay length $\xi$ (sites)')
    ax.set_title('Edge state penetration depth vs w')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/05_decay_length.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  → Saved figures/05_decay_length.png")
    print("  Physics: ξ decreases as w/v grows — edge states are more tightly bound.")


if __name__ == "__main__":
    os.makedirs('figures', exist_ok=True)
    print("=" * 55)
    print("TASK 5: EDGE STATE CHARACTERIZATION")
    print("=" * 55)
    plot_edge_wavefunctions()
    fit_decay_lengths()
    print("\n Edge states characterized")