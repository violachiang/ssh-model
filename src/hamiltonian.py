"""
hamiltonian.py — SSH Hamiltonian construction and diagonalization.

Public API (what the scripts rely on):

    SSHHamiltonian(N, v, w)                      — constructor
    .build_real_space()          -> (2N,2N) ndarray
    .diagonalize()               -> (eigenvalues, eigenvectors)
    .get_spectrum_info()         -> dict with keys:
                                      'gap', 'num_zero_states',
                                      'phase', 'min_energy', 'max_energy'
    .get_edge_states(tol)        -> dict with keys:
                                      'found' (bool),
                                      'states' (list of dicts each with
                                          'energy', 'wavefunction',
                                          'probability', 'center', 'spread')
    .build_momentum_space(k)     -> (len(k), 2, 2) complex ndarray
    .get_bands(k)                -> (len(k), 2) ndarray
    .get_analytical_bands(k)     -> (len(k), 2) ndarray
    .get_d_vector(k)             -> (dx, dy) each shape (len(k),)
    SSHHamiltonian.winding_number(v, w, Nk)  -> float  (static)
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.linalg import eigh


class SSHHamiltonian:
    """
    Su-Schrieffer-Heeger Hamiltonian on a 1D dimerised chain.

    Parameters
    ----------
    N : int
        Number of unit cells. The chain has 2N sites.
    v : float
        Intra-cell hopping amplitude (the "strong" bond inside a cell).
    w : float
        Inter-cell hopping amplitude (the bond between adjacent cells).
    """

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #
    def __init__(self, N: int, v: float = 1.0, w: float = 1.0) -> None:
        if N <= 0:
            raise ValueError("N must be a positive integer")

        self.N = int(N)
        self.v = float(v)
        self.w = float(w)

        # Caches — filled lazily
        self._H_real: np.ndarray | None = None
        self._eigenvalues: np.ndarray | None = None
        self._eigenvectors: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    #  Real-space Hamiltonian
    # ------------------------------------------------------------------ #
    def build_real_space(self) -> np.ndarray:
        """
        Build the 2N x 2N real-space Hamiltonian (Open Boundary Conditions).

        Site ordering:  A0, B0, A1, B1, A2, B2, ...
        with  A_n <-> B_n  hopping = v
              B_n <-> A_{n+1}  hopping = w
        """
        N = self.N
        H = np.zeros((2 * N, 2 * N), dtype=float)

        # Intra-cell hopping v: (A_n, B_n) = (2n, 2n+1)
        for n in range(N):
            H[2 * n, 2 * n + 1] = self.v
            H[2 * n + 1, 2 * n] = self.v

        # Inter-cell hopping w: (B_n, A_{n+1}) = (2n+1, 2(n+1))
        for n in range(N - 1):
            H[2 * n + 1, 2 * (n + 1)] = self.w
            H[2 * (n + 1), 2 * n + 1] = self.w

        self._H_real = H
        return H

    def diagonalize(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diagonalise the real-space Hamiltonian.

        Returns
        -------
        eigenvalues : ndarray, shape (2N,)   ascending order
        eigenvectors : ndarray, shape (2N, 2N)
                       column i is the eigenvector for eigenvalues[i]
        """
        if self._H_real is None:
            self.build_real_space()

        # eigh() is the right routine: Hermitian/symmetric, real, sorted.
        self._eigenvalues, self._eigenvectors = eigh(self._H_real)
        return self._eigenvalues, self._eigenvectors

    # ------------------------------------------------------------------ #
    #  Spectral analysis
    # ------------------------------------------------------------------ #
    def get_spectrum_info(self) -> Dict:
        """
        Summarise the real-space spectrum.

        Returns
        -------
        dict with keys:
            'gap'             : (float)  E[N] - E[N-1]  (zero if gap closes)
            'num_zero_states' : (int)    number of |E| < 1e-6
            'phase'           : (str)    'topological' or 'trivial'
            'min_energy'      : (float)
            'max_energy'      : (float)
        """
        if self._eigenvalues is None:
            self.diagonalize()

        eigs = self._eigenvalues
        N = self.N

        gap = float(eigs[N] - eigs[N - 1])
        num_zero = int(np.sum(np.abs(eigs) < 1e-6))
        phase = "topological" if self.w > self.v else "trivial"

        return {
            "gap": gap,
            "num_zero_states": num_zero,
            "phase": phase,
            "min_energy": float(np.min(eigs)),
            "max_energy": float(np.max(eigs)),
        }

    def get_edge_states(self, tol: float = 1e-6) -> Dict:
        """
        Identify and characterise zero-energy states.

        Parameters
        ----------
        tol : float
            States with |E| < tol are considered "zero-energy".

        Returns
        -------
        dict with keys:
            'found'  : bool
            'states' : list of dicts, each containing:
                          'energy'       (float)
                          'wavefunction' (ndarray, shape (2N,), complex)
                          'probability'  (ndarray, shape (2N,), real)
                          'center'       (float)  center of mass
                          'spread'       (float)  sqrt variance
        """
        if self._eigenvectors is None:
            self.diagonalize()

        zero_indices = np.where(np.abs(self._eigenvalues) < tol)[0]

        if zero_indices.size == 0:
            return {"found": False, "states": []}

        sites = np.arange(2 * self.N)
        states = []

        for idx in zero_indices:
            psi = self._eigenvectors[:, idx]
            prob = np.abs(psi) ** 2

            center = float(np.sum(sites * prob))
            variance = float(np.sum((sites - center) ** 2 * prob))
            spread = float(np.sqrt(variance))

            states.append(
                {
                    "energy": float(self._eigenvalues[idx]),
                    "wavefunction": psi,
                    "probability": prob,
                    "center": center,
                    "spread": spread,
                }
            )

        return {"found": True, "states": states}

    # ------------------------------------------------------------------ #
    #  Momentum-space Hamiltonian (PBC)
    # ------------------------------------------------------------------ #
    def build_momentum_space(self, k: np.ndarray) -> np.ndarray:
        """
        Bloch Hamiltonian H(k) = [[0, v + w e^{-ik}],
                                  [v + w e^{+ik}, 0]].

        Parameters
        ----------
        k : ndarray, shape (Nk,)

        Returns
        -------
        Hk : ndarray, shape (Nk, 2, 2), complex
        """
        k = np.atleast_1d(k).astype(float)
        Hk = np.zeros((k.size, 2, 2), dtype=complex)

        for i, ki in enumerate(k):
            Hk[i, 0, 0] = 0.0
            Hk[i, 0, 1] = self.v + self.w * np.exp(-1j * ki)
            Hk[i, 1, 0] = self.v + self.w * np.exp(+1j * ki)
            Hk[i, 1, 1] = 0.0

        return Hk

    def get_bands(self, k: np.ndarray) -> np.ndarray:
        """
        Numerical energy bands E(k).

        Returns
        -------
        bands : ndarray, shape (len(k), 2)
                bands[:, 0] = lower band, bands[:, 1] = upper band
        """
        Hk = self.build_momentum_space(k)
        bands = np.array([np.linalg.eigvalsh(Hk[i]) for i in range(Hk.shape[0])])
        return bands

    def get_analytical_bands(self, k: np.ndarray) -> np.ndarray:
        """
        Analytical energy bands: E(k) = ± sqrt(v² + w² + 2vw cos k).

        Returns
        -------
        bands : ndarray, shape (len(k), 2)
                bands[:, 0] = -sqrt(...), bands[:, 1] = +sqrt(...)
        """
        k = np.atleast_1d(k).astype(float)
        radicand = self.v ** 2 + self.w ** 2 + 2 * self.v * self.w * np.cos(k)
        # Clamp tiny negatives from round-off
        radicand = np.maximum(radicand, 0.0)
        d = np.sqrt(radicand)
        return np.stack([-d, +d], axis=1)

    # ------------------------------------------------------------------ #
    #  d-vector for winding-number calculations
    # ------------------------------------------------------------------ #
    def get_d_vector(self, k: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the two components of the d-vector:

            d_x(k) = v + w cos(k)
            d_y(k) = w sin(k)
        """
        k = np.atleast_1d(k).astype(float)
        dx = self.v + self.w * np.cos(k)
        dy = self.w * np.sin(k)
        return dx, dy

    # ------------------------------------------------------------------ #
    #  Winding number (static utility — kept for convenience)
    # ------------------------------------------------------------------ #
    @staticmethod
    def winding_number(v: float, w: float, Nk: int = 2000) -> float:
        """
        Numerically compute the topological winding number

            D = (1/2π) ∮ (d̂_x d d̂_y/dk − d̂_y d d̂_x/dk) dk

        over k ∈ [-π, π].
        """
        k = np.linspace(-np.pi, np.pi, Nk)
        dx = v + w * np.cos(k)
        dy = w * np.sin(k)

        norm = np.sqrt(dx ** 2 + dy ** 2)
        # If the loop passes through the origin the winding is undefined.
        if np.any(norm < 1e-12):
            return float("nan")

        dx_hat = dx / norm
        dy_hat = dy / norm

        ddx = np.gradient(dx_hat, k)
        ddy = np.gradient(dy_hat, k)

        integrand = dx_hat * ddy - dy_hat * ddx
        return float(np.trapz(integrand, k) / (2 * np.pi))