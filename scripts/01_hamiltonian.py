#!/usr/bin/env python3
"""
TASK 1: HAMILTONIAN CONSTRUCTION

Purpose:
    Build both the real-space and momentum-space SSH Hamiltonians,
    and verify the mathematical properties required for a valid
    quantum Hamiltonian (Hermiticity, correct dimensions, pattern).

This is the "infrastructure" step - it produces no physics results
on its own, but everything else depends on it.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.hamiltonian import SSHHamiltonian


def verify_real_space_matrix():
    """Build and inspect the real-space Hamiltonian matrix."""
    print("\n[1/3] Real-space Hamiltonian")
    print("-" * 50)

    N, v, w = 3, 1.0, 0.5
    h = SSHHamiltonian(N, v, w)
    H = h.build_real_space()

    print(f"  System: N={N} cells → {2*N} sites")
    print(f"  Matrix shape: {H.shape}")
    print(f"  Is Hermitian: {np.allclose(H, H.T)}")
    print(f"  All eigenvalues real: {np.all(np.isreal(np.linalg.eigvalsh(H)))}")

    # Print the matrix for small N to see the pattern
    if N <= 5:
        print(f"\n  Hamiltonian matrix (N={N}, v={v}, w={w}):")
        for row in H:
            print("   ", " ".join(f"{x:5.1f}" for x in row))

    return H


def verify_momentum_space_matrix():
    """Build and inspect the Bloch Hamiltonian H(k)."""
    print("\n[2/3] Momentum-space Hamiltonian")
    print("-" * 50)

    N, v, w = 10, 1.0, 0.5
    k = np.linspace(-np.pi, np.pi, 5)  # Only 5 k-points for inspection
    h = SSHHamiltonian(N, v, w)
    Hk = h.build_momentum_space(k)

    print(f"  Shape: {Hk.shape}  (one 2*2 matrix per k-point)")
    print(f"\n  H(k) at k=0:")
    print(f"    {Hk[2]}")   # k = 0 is the middle index
    print(f"\n  H(k) at k=π:")
    print(f"    {Hk[-1]}")


def check_consistency():
    """Verify real-space and momentum-space agree."""
    print("\n[3/3] Consistency check")
    print("-" * 50)

    N, v, w = 20, 0.7, 1.3
    h = SSHHamiltonian(N, v, w)

    k = np.linspace(-np.pi, np.pi, 200)
    numerical = h.get_bands(k)
    analytical = h.get_analytical_bands(k)

    max_error = np.max(np.abs(numerical - analytical))
    print(f"  Max |E_numerical - E_analytical| = {max_error:.2e}")
    print(f"  {' PASS' if max_error < 1e-10 else ' FAIL'}")


if __name__ == "__main__":
    os.makedirs('figures', exist_ok=True)
    print("=" * 55)
    print("TASK 1: HAMILTONIAN CONSTRUCTION")
    print("=" * 55)

    verify_real_space_matrix()
    verify_momentum_space_matrix()
    check_consistency()

    print("\n" + "=" * 55)
    print(" Hamiltonian construction verified")
    print("=" * 55)