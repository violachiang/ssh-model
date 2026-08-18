# `README.md` for the SSH Model Project

Save this as `README.md` in the root of your repository.

---

```markdown
# SSH Model: Topological Phases and Bulk-Boundary Correspondence

A numerical implementation of the **Su-Schrieffer-Heeger (SSH) model** — the
canonical 1D toy model of topological matter. The project demonstrates, from
first principles, how a bulk topological invariant (the winding number) predicts
the existence of zero-energy edge states on a finite chain.

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/numpy-%3E%3D1.26,%3C2.0-blue)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/scipy-%3E%3D1.11,%3C1.13-blue)](https://scipy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Table of Contents

- [Overview](#overview)
- [Physics Background](#physics-background)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Testing](#testing)
- [Key Concepts](#key-concepts)
- [Common Issues](#common-issues)
- [References](#references)
- [License](#license)

---

## Overview

The SSH model describes a 1D chain of dimers with alternating hopping
amplitudes **v** (intra-cell) and **w** (inter-cell). Despite its simplicity,
it exhibits a **topological phase transition** at `v = w` that separates two
topologically distinct phases:

| Phase | Condition | Winding Number `D` | Edge States |
|-------|-----------|--------------------|-------------|
| Trivial     | `v > w` | `D = 0` | None |
| Critical    | `v = w` | Undefined (gap closes) | — |
| Topological | `v < w` | `D = 1` | Two, at `E = 0` |

The project verifies the **bulk-boundary correspondence** numerically:

```
   bulk invariant D          boundary physics
   ─────────────────   ⟹    ─────────────────────
   computed from H(k)        observed in H_real
   (infinite chain, PBC)     (finite chain, OBC)
```

---

## Physics Background

### Real-space Hamiltonian (Open Boundary Conditions)

For a chain of `N` unit cells (2`N` sites), the tight-binding Hamiltonian is

$$
H = \sum_{n=1}^{N} v \left( c^\dagger_{n,A} c_{n,B} + \text{h.c.} \right)
  + \sum_{n=1}^{N-1} w \left( c^\dagger_{n+1,A} c_{n,B} + \text{h.c.} \right)
$$

which becomes a `2N × 2N` real symmetric matrix with `v` and `w` alternating on
the first off-diagonals.

### Momentum-space Hamiltonian (Periodic Boundary Conditions)

Fourier transforming gives a `2 × 2` Bloch Hamiltonian

$$
H(k) = \begin{pmatrix}
0 & v + w e^{-ik} \\[2pt]
v + w e^{+ik} & 0
\end{pmatrix}
= d_x(k)\,\sigma_x + d_y(k)\,\sigma_y
$$

with

$$
d_x(k) = v + w\cos k, \qquad d_y(k) = w\sin k.
$$

The energy bands are

$$
E(k) = \pm\sqrt{v^2 + w^2 + 2vw\cos k}.
$$

### Topological Invariant

As `k` sweeps `[-π, π]`, the vector `(d_x, d_y)` traces a circle of radius `w`
centered at `(v, 0)`. The **winding number** counts how many times this loop
winds around the origin:

$$
D = \frac{1}{2\pi} \int_{-\pi}^{\pi}
    \left( \hat{d}_x \, \partial_k \hat{d}_y - \hat{d}_y \, \partial_k \hat{d}_x \right) dk
$$

- `v > w` → circle misses the origin → `D = 0`
- `v < w` → circle encloses the origin → `D = 1`

### Edge States

In the topological phase, a finite open chain supports **two zero-energy states**
localized exponentially at the boundaries:

$$
|\psi(x)|^2 \sim e^{-x/\xi}, \qquad \xi \sim \frac{1}{\ln(w/v)}
$$

These states are protected by **chiral (sublattice) symmetry** and cannot be
removed without closing the bulk gap.

---

## Project Structure

```
ssh-model/
├── src/                                    # Core library
│   ├── __init__.py
│   ├── hamiltonian.py                      # Hamiltonian construction & diagonalization
│   ├── topology.py                         # Winding number, phase diagram, Zak phase
│   ├── analysis.py                         # Wavefunction analysis (localization, decay)
│   └── visualization.py                    # Shared plotting utilities
│
├── scripts/                                # Analysis pipeline (run in order)
│   ├── 01_hamiltonian.py                   # Build & verify H_real and H(k)
│   ├── 02_band_structure.py                # Bulk dispersion E(k), gap closure
│   ├── 03_winding_number.py                # Winding trajectories & phase diagram
│   ├── 04_energy_spectrum.py               # OBC spectrum vs w
│   ├── 05_edge_states.py                   # Edge-state wavefunctions & decay length
│   ├── 06_bulk_boundary_correspondence.py  # Synthesize bulk D with edge states
│   └── run_all.py                          # Run the full pipeline
│
├── tests/                                  # Unit tests
│   ├── test_hamiltonian.py
│   └── test_topology.py
│
├── figures/                                # Generated output (gitignored)
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

### Design Philosophy

Scripts are organized by **logical task flow**, not by implementation order:

```
     PHASE 1 — CONSTRUCTION
        01_hamiltonian.py
                │
     ┌──────────┴──────────┐
     ▼                     ▼
  PHASE 2 — BULK        PHASE 3 — BOUNDARY
  02_band_structure     04_energy_spectrum
  03_winding_number     05_edge_states
     └──────────┬──────────┘
                ▼
     PHASE 4 — SYNTHESIS
        06_bulk_boundary_correspondence
```

Bulk and boundary calculations are independent; the final script verifies that
they agree — this **is** the bulk-boundary correspondence.

---

## Installation

### Prerequisites

- **Python ≥ 3.9** (3.10 or 3.11 recommended)
- **NumPy 1.26.x** (⚠️ NumPy ≥ 2.0 is **not** compatible with SciPy < 1.12)

### Option A — Using `uv` (recommended)

```bash
git clone https://github.com/<your-username>/ssh-model.git
cd ssh-model

uv venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

uv pip install -r requirements.txt
```

### Option B — Using `pip`

```bash
git clone https://github.com/<your-username>/ssh-model.git
cd ssh-model

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Option C — Using `conda`

```bash
conda create -n ssh-model python=3.10
conda activate ssh-model
conda install numpy=1.26.4 scipy=1.11.4 matplotlib=3.8
pip install pytest
```

### `requirements.txt`

```
numpy>=1.26,<2.0
scipy>=1.11,<1.13
matplotlib>=3.8
pytest>=7.0
```

---

## Usage

### Run the full pipeline

```bash
MPLBACKEND=Agg python scripts/run_all.py
```

The `MPLBACKEND=Agg` environment variable tells Matplotlib to render figures
without opening blocking windows — essential for headless or automated runs.
All figures are still written to `figures/`.

### Run individual tasks

```bash
# Interactive (shows figures one at a time)
python scripts/02_band_structure.py

# Headless (saves figures only)
MPLBACKEND=Agg python scripts/03_winding_number.py
```

### Run as a library

```python
import sys
sys.path.append('.')
from src.hamiltonian import SSHHamiltonian
from src.topology import SSHTopology

# Build and diagonalize
h = SSHHamiltonian(N=20, v=0.5, w=1.0)
eigs, vecs = h.diagonalize()

# Spectrum diagnostics
info = h.get_spectrum_info()
print(info['phase'])           # 'topological'
print(info['num_zero_states']) # 2

# Edge-state wavefunctions
edge = h.get_edge_states(tol=1e-6)
print(len(edge['states']))     # 2

# Winding number
D = SSHTopology.winding_number(v=0.5, w=1.0)
print(D)                       # 1.0
```

### Expected pipeline output

```
STEP 1/6  Hamiltonian construction       ✅ verified
STEP 2/6  Band structure                 ✅ saved figures/02_*.png
STEP 3/6  Winding number                 ✅ D=0 (trivial), D=1 (topological)
STEP 4/6  Energy spectrum                ✅ spectrum vs w
STEP 5/6  Edge states                    ✅ 2 zero-modes found
STEP 6/6  Bulk-boundary correspondence   ✅ ALL MATCH
```

---

## Results

The pipeline produces the following figures (in `figures/`):

| File | Content |
|------|---------|
| `02_band_structure.png`     | `E(k)` for trivial, critical, topological phases |
| `02_gap_closure.png`        | Gap size vs `v` — shows gap closing at `v = w` |
| `03_winding_trajectories.png` | `(d_x, d_y)` loops with and without origin enclosed |
| `03_phase_diagram.png`      | `D(v, w)` heatmap with the `v = w` phase boundary |
| `04_spectrum_two_phases.png` | Real-space spectrum: trivial vs topological |
| `04_spectrum_vs_w.png`      | All eigenvalues as `w` sweeps across transition |
| `05_edge_wavefunctions.png` | Exponential decay of edge states (log scale) |
| `05_decay_length.png`       | Edge-state penetration depth vs `w` |
| `06_summary.png`            | Combined figure: spectrum, bands, winding, edge states |

### Representative Result: Bulk-Boundary Correspondence

```
    v     w        D   edges   2|D|   match
------------------------------------------------
  1.0   0.3   +0.000       0      0       ✓
  1.0   0.7   +0.000       0      0       ✓
  1.0   1.1   +1.000       2      2       ✓
  1.0   1.4   +1.000       2      2       ✓
  1.0   1.8   +1.000       2      2       ✓
  0.5   1.0   +1.000       2      2       ✓
  1.5   1.0   +0.000       0      0       ✓
------------------------------------------------
  Overall: ✓ ALL MATCH
```

### A Note on Finite-Size Effects

For `w` very close to `v`, the two edge states **hybridize across the bulk**
and split symmetrically to `±E_edge`, with

$$
E_\text{edge} \sim v \left( \frac{v}{w} \right)^{N}
$$

At `N = 30` and `v = 1.0`, the splitting is:

| `w` | `(v/w)³⁰` | Edge energy `|E|` |
|-----|-----------|-------------------|
| 1.1 | `5.7 × 10⁻²` | `~2.9 × 10⁻²` |
| 1.4 | `3.1 × 10⁻⁵` | `~1.5 × 10⁻⁵` |
| 1.8 | `2.6 × 10⁻⁸` | `~1.3 × 10⁻⁸` |

An **energy-tolerance** detector (`|E| < 1e-6`) will miss the split states for
`w ≲ 1.4`. The pipeline therefore identifies edge states by **spatial
localization** instead:

```python
def count_edge_states(h, boundary_threshold=0.8):
    eigs, vecs = h.diagonalize()
    N = h.N
    closest = np.argsort(np.abs(eigs))[:6]
    count = 0
    for idx in closest:
        prob = np.abs(vecs[:, idx])**2
        if prob[:N//2].sum() + prob[3*N//2:].sum() > boundary_threshold:
            count += 1
    return count
```

This is physically correct: **"edge state" means "localized at the boundary"**,
not "exactly at E = 0". At any finite `N`, exact zero energy holds only in the
thermodynamic limit.

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run only Hamiltonian tests
pytest tests/test_hamiltonian.py -v

# Run only topology tests
pytest tests/test_topology.py -v

# Run a specific test
pytest tests/test_hamiltonian.py::TestHamiltonianConstruction::test_trivial_gap -v
```

The tests verify:

- **`test_hamiltonian.py`** — matrix dimensions, Hermiticity, hopping placement,
  chiral symmetry of the spectrum, correct phases, edge-state localization,
  agreement between numerical and analytical bands.
- **`test_topology.py`** — winding numbers are `0` / `1` / `0.5` in the expected
  regimes, convergence with `Nk`, discontinuity at the phase transition, correct
  Zak phase quantization, and bulk-boundary correspondence.

---

## Key Concepts

| Concept | Where it lives |
|---------|----------------|
| Real-space Hamiltonian | `src/hamiltonian.py::build_real_space` |
| Bloch Hamiltonian `H(k)` | `src/hamiltonian.py::build_momentum_space` |
| Exact diagonalization | `src/hamiltonian.py::diagonalize` |
| Band structure `E(k)` | `src/hamiltonian.py::get_bands` |
| `d`-vector `(d_x, d_y)` | `src/hamiltonian.py::get_d_vector` |
| Winding number `D` | `src/topology.py::winding_number` |
| Phase diagram | `src/topology.py::phase_diagram` |
| Zak phase | `src/topology.py::zak_phase` |
| Edge-state wavefunction | `src/hamiltonian.py::get_edge_states` |
| Localization length | `src/analysis.py::localization_length` |
| Exponential decay fit | `src/analysis.py::fit_exponential_decay` |

---

## References

1. **Asbóth, Oroszlány, Pályi** (2016).
   *A Short Course on Topological Insulators*.
   Springer. [arXiv:1509.02295](https://arxiv.org/abs/1509.02295) —
   *Chapters 1–2 cover the SSH model step by step.*

2. **Su, Schrieffer, Heeger** (1979).
   "Solitons in Polyacetylene."
   *Phys. Rev. Lett.* **42**, 1698.

3. **Vanderbilt, D.** (2018).
   *Berry Phases in Electronic Structure Theory*.
   Cambridge University Press.

4. **Shen, S.-Q.** (2017).
   *Topological Insulators: Dirac Equation in Condensed Matter*.
   Springer.

5. **Ryu, Schnyder, Furusaki, Ludwig** (2010).
   "Topological insulators and superconductors: tenfold way and dimensional
   hierarchy."
   *New J. Phys.* **12**, 065010.

---

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

Model originally introduced by Su, Schrieffer, and Heeger (1979). Numerical
framework inspired by the pedagogical treatments in Asbóth *et al.* and
Vanderbilt. Built with NumPy, SciPy, and Matplotlib.
```

---

## Additional Files to Include

### `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/

# Generated outputs
figures/*.png
*.npy
*.pkl

# Editor
.vscode/
.idea/
.DS_Store
```

### `LICENSE` (MIT)

```text
MIT License

Copyright (c) 2026 <Your Name>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```