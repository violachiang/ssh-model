#!/usr/bin/env python3
"""
Run the full SSH model analysis in logical order.

Pipeline:
    1. Hamiltonian construction   (infrastructure)
    2. Band structure             (bulk, PBC)
    3. Winding number             (bulk topology)
    4. Energy spectrum            (boundary, OBC)
    5. Edge states                (boundary characterization)
    6. Bulk-boundary correspondence (synthesis)
"""

import subprocess
import sys

SCRIPTS = [
    ("01_hamiltonian.py",                    "Hamiltonian construction"),
    ("02_band_structure.py",                 "Band structure (bulk)"),
    ("03_winding_number.py",                 "Winding number (topology)"),
    ("04_energy_spectrum.py",                "Real-space spectrum (OBC)"),
    ("05_edge_states.py",                    "Edge state characterization"),
    ("06_bulk_boundary_correspondence.py",   "Bulk-boundary verification"),
]

def main():
    for i, (script, desc) in enumerate(SCRIPTS, 1):
        print("\n" + "=" * 60)
        print(f"  STEP {i}/{len(SCRIPTS)}: {desc}")
        print(f"  → {script}")
        print("=" * 60)
        result = subprocess.run([sys.executable, f"scripts/{script}"])
        if result.returncode != 0:
            print(f" {script} failed — aborting.")
            sys.exit(1)
    print("\n" + "=" * 60)
    print("   FULL PIPELINE COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()