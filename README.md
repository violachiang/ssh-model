# SSH Model: Topological Phases and Bulk-Boundary Correspondence

A numerical implementation of the **Su-Schrieffer-Heeger (SSH) model** — the
canonical one-dimensional topological insulator. The project computes the
topological winding number `D` from the bulk Bloch Hamiltonian `H(k)` and,
independently, exactly diagonalizes the finite real-space Hamiltonian to show
that `D` predicts the number of zero-energy edge states.

The two calculations are performed separately and then checked against each
other. That check **is** the bulk-boundary correspondence.

---

## Contents

- [What it does](#what-it-does)
- [Physics background](#physics-background)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Testing](#testing)
- [Notes on finite-size effects](#notes-on-finite-size-effects)
- [References](#references)
- [License](#license)

---

## What it does

The SSH model describes a chain of dimers with alternating hopping amplitudes
`v` (intra-cell) and `w` (inter-cell). It has two phases separated by a
transition at `v = w`:

| Phase | Condition | Winding number `D` | Edge states |
|---|---|---|---|
| Trivial | `v > w` | 0 | none |
| Critical | `v = w` | undefined (gap closes) | — |
| Topological | `v < w` | 1 | two, at `E = 0` |

The project computes each column of this table numerically and verifies that
the last one follows from the third.

```text
     bulk (PBC, H(k))                  boundary (OBC, H_real)
     ────────────────                  ──────────────────────
     02_band_structure.py              04_energy_spectrum.py
     03_winding_number.py              05_edge_states.py
              │                                  │
              └────────────────┬─────────────────┘
                               ▼
               06_bulk_boundary_correspondence.py
                    "D predicts the edge count"
```

---

## Physics background

### Real-space Hamiltonian (open boundary conditions)

For `N` unit cells (2`N` sites):

$$
H = \sum_{n=1}^{N} v \left( c^\dagger_{n,A} c_{n,B} + \text{h.c.} \right)
  + \sum_{n=1}^{N-1} w \left( c^\dagger_{n+1,A} c_{n,B} + \text{h.c.} \right)
$$

which becomes a `2N × 2N` real symmetric matrix with `v` and `w` alternating on
the first off-diagonals.

### Momentum-space Hamiltonian (periodic boundary conditions)

Fourier transforming gives a `2 × 2` Bloch Hamiltonian:

$$
H(k) = \begin{pmatrix}
0 & v + w e^{-ik} \\
v + w e^{+ik} & 0
\end{pmatrix}
= d_x(k)\,\sigma_x + d_y(k)\,\sigma_y,
\qquad
\begin{aligned}
d_x(k) &= v + w\cos k,\\
d_y(k) &= w\sin k.
\end{aligned}
$$

Energy bands:

$$
E(k) = \pm\sqrt{v^2 + w^2 + 2vw\cos k}.
$$

### Winding number

As `k` sweeps `[−π, π]`, the vector `(d_x, d_y)` traces a circle of radius `w`
centered at `(v, 0)`. The winding number counts how many times that loop
encircles the origin:

$$
D = \frac{1}{2\pi} \int_{-\pi}^{\pi}
    \left( \hat{d}_x \, \partial_k \hat{d}_y
         - \hat{d}_y \, \partial_k \hat{d}_x \right) dk
$$

`v > w` → circle misses the origin → `D = 0`.
`v < w` → circle encloses the origin → `D = 1`.

### Edge states

In the topological phase, a finite open chain supports two zero-energy states
localized at the boundaries:

$$
|\psi(x)|^2 \sim e^{-x/\xi}, \qquad \xi \sim \frac{1}{\ln(w/v)}
$$

They are protected by chiral (sublattice) symmetry — they cannot be removed
without closing the bulk gap.

---

## Project structure

```text
ssh-model/
├── src/
│   ├── __init__.py
│   ├── hamiltonian.py          # H_real, H(k), diagonalization, edge states
│   ├── topology.py             # winding number, phase diagram, Zak phase
│   ├── analysis.py             # localization, exponential fit, DOS, entropy
│   └── visualization.py        # shared plotting utilities
│
├── scripts/
│   ├── 01_hamiltonian.py                   # build & verify H_real and H(k)
│   ├── 02_band_structure.py                # E(k), gap closure
│   ├── 03_winding_number.py                # d-vector loops, phase diagram
│   ├── 04_energy_spectrum.py               # OBC spectrum vs w
│   ├── 05_edge_states.py                   # wavefunctions, decay length
│   ├── 06_bulk_boundary_correspondence.py  # D vs edge count
│   └── run_all.py                          # run the pipeline in order
│
├── tests/
│   ├── test_hamiltonian.py
│   └── test_topology.py
│
├── figures/                    # generated PNGs (gitignored)
├── requirements.txt
├── pyproject.toml
├── README.md
└── LICENSE
```

**Scripts are organized by logical task, not by implementation order.** Bulk and
boundary analyses are independent; the final script verifies that they agree.

---

## Installation

Requires **Python ≥ 3.9**. NumPy is pinned to `1.26.x` — NumPy 2.x breaks the
SciPy version this project depends on.

### With `uv` (recommended)

```bash
git clone https://github.com/<your-username>/ssh-model.git
cd ssh-model

uv venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

uv pip install -r requirements.txt
```

### With `pip`

```bash
git clone https://github.com/<your-username>/ssh-model.git
cd ssh-model

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### With `conda`

```bash
conda create -n ssh-model python=3.10
conda activate ssh-model
conda install numpy=1.26.4 scipy=1.11.4 matplotlib=3.8
pip install pytest
```

### `requirements.txt`

```text
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

`MPLBACKEND=Agg` renders figures without opening blocking windows — required
for headless or automated runs. All figures are still written to `figures/`.

### Run a single task

```bash
python scripts/02_band_structure.py             # interactive, shows figures
MPLBACKEND=Agg python scripts/03_winding_number.py   # saves figures only
```

### Use it as a library

```python
import sys
sys.path.append('.')
from src.hamiltonian import SSHHamiltonian
from src.topology import SSHTopology

h = SSHHamiltonian(N=20, v=0.5, w=1.0)
eigs, vecs = h.diagonalize()

info = h.get_spectrum_info()
print(info['phase'])            # 'topological'
print(info['num_zero_states'])  # 2

edge = h.get_edge_states(tol=1e-6)
print(len(edge['states']))      # 2

D = SSHTopology.winding_number(v=0.5, w=1.0)
print(D)                        # 1.0
```

### Expected pipeline output

```text
STEP 1/6  Hamiltonian construction       verified
STEP 2/6  Band structure                 saved figures/02_*.png
STEP 3/6  Winding number                 D = 0 (trivial), D = 1 (topological)
STEP 4/6  Energy spectrum                saved figures/04_*.png
STEP 5/6  Edge states                    2 zero-modes found
STEP 6/6  Bulk-boundary correspondence   ALL MATCH
```

---

## Results

The pipeline produces these figures in `figures/`:

| File | Content |
|---|---|
| `02_band_structure.png` | `E(k)` for trivial, critical, topological phases |
| `02_gap_closure.png` | Gap size vs `v` — closes at `v = w` |
| `03_winding_trajectories.png` | `(d_x, d_y)` loops with and without origin enclosed |
| `03_phase_diagram.png` | `D(v, w)` heatmap with the `v = w` boundary |
| `04_spectrum_two_phases.png` | OBC spectrum: trivial vs topological |
| `04_spectrum_vs_w.png` | All eigenvalues as `w` sweeps across transition |
| `05_edge_wavefunctions.png` | Exponential decay of edge states (log scale) |
| `05_decay_length.png` | Edge-state penetration depth vs `w` |
| `06_summary.png` | Combined spectrum / bands / winding / edge states |

### Bulk-boundary correspondence

```text
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
  Overall: ALL MATCH
```

---

## Testing

```bash
pytest tests/ -v

pytest tests/test_hamiltonian.py -v
pytest tests/test_topology.py -v
```

The tests cover:

- **`test_hamiltonian.py`** — matrix shape, Hermiticity, hopping placement,
  chiral symmetry of the spectrum, phase classification, edge-state
  localization, numerical vs analytical bands.
- **`test_topology.py`** — winding number values in each phase, convergence
  with `Nk`, discontinuity at the transition, phase-diagram structure, Zak
  phase quantization, Berry-curvature integral, bulk-boundary correspondence.

---

## Notes on finite-size effects

For `w` very close to `v`, the two edge states hybridize across the bulk and
split symmetrically to `±E_edge`, with

$$
E_\text{edge} \sim v \left( \frac{v}{w} \right)^{N}
$$

At `N = 30`, `v = 1.0`:

| `w` | `(v/w)³⁰` | Edge energy `|E|` |
|---|---|---|
| 1.1 | `5.7 × 10⁻²` | `~2.9 × 10⁻²` |
| 1.4 | `3.1 × 10⁻⁵` | `~1.5 × 10⁻⁵` |
| 1.8 | `2.6 × 10⁻⁸` | `~1.3 × 10⁻⁸` |

An energy-tolerance detector (`|E| < 1e-6`) misses the split states for
`w ≲ 1.4`. The pipeline therefore identifies edge states by **spatial
localization**:

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

This is the physically correct definition: an edge state is one **localized at
the boundary**, not one exactly at `E = 0`. Exact zero energy holds only in the
thermodynamic limit.

---

## References

1. **Asbóth, Oroszlány, Pályi** (2016).
   *A Short Course on Topological Insulators*.
   Springer. [arXiv:1509.02295](https://arxiv.org/abs/1509.02295).
   Chapters 1–2 cover the SSH model in detail.

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