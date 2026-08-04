"""
test_topology.py - Unit tests for topological invariants.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.topology import SSHTopology
from src.hamiltonian import SSHHamiltonian


class TestWindingNumber:
    """
    Tests for winding number calculations.
    """
    
    def test_winding_number_trivial(self):
        """
        Test winding number in trivial phase.
        
        Why: For v > w, D should be exactly 0.
        """
        v, w = 1.0, 0.5
        D = SSHTopology.winding_number(v, w)
        
        # Should be very close to 0
        assert abs(D) < 0.01, f"Winding number should be 0, got {D}"
        
        # Also test analytic result
        D_analytic = SSHTopology.winding_number_analytic(v, w)
        assert abs(D_analytic - 0.0) < 0.01
        
        print(f" Trivial phase: D={D:.6f} (numeric), D={D_analytic:.1f} (analytic)")
    
    def test_winding_number_topological(self):
        """
        Test winding number in topological phase.
        
        Why: For w > v, D should be exactly 1.
        """
        v, w = 0.5, 1.0
        D = SSHTopology.winding_number(v, w)
        
        # Should be very close to 1
        assert abs(D - 1.0) < 0.01, f"Winding number should be 1, got {D}"
        
        # Also test analytic result
        D_analytic = SSHTopology.winding_number_analytic(v, w)
        assert abs(D_analytic - 1.0) < 0.01
        
        print(f" Topological phase: D={D:.6f} (numeric), D={D_analytic:.1f} (analytic)")
    
    def test_winding_number_critical(self):
        """
        Test winding number at critical point.
        
        Why: At v = w, the gap closes and D is undefined.
        """
        v, w = 1.0, 1.0
        D = SSHTopology.winding_number(v, w, Nk=1000)
        
        # Should be between 0 and 1 (typically 0.5)
        assert 0.4 < D < 0.6, f"At critical point, D should be ~0.5, got {D}"
        
        print(f" Critical point: D={D:.6f}")
    
    def test_winding_number_convergence(self):
        """
        Test winding number converges with Nk.
        
        Why: Numerical integration should converge as we increase points.
        """
        v, w = 0.5, 1.0  # Topological phase
        
        D_100 = SSHTopology.winding_number(v, w, Nk=100)
        D_1000 = SSHTopology.winding_number(v, w, Nk=1000)
        D_10000 = SSHTopology.winding_number(v, w, Nk=10000)
        
        # Should converge
        diff_100_1000 = abs(D_100 - D_1000)
        diff_1000_10000 = abs(D_1000 - D_10000)
        
        # Higher Nk should be closer
        assert diff_1000_10000 < diff_100_1000, \
               f"Not converging: {diff_100_1000:.6f} vs {diff_1000_10000:.6f}"
        
        print(f" Convergence test: D_100={D_100:.6f}, D_1000={D_1000:.6f}, D_10000={D_10000:.6f}")
    
    def test_winding_number_discontinuity(self):
        """
        Test winding number discontinuity at phase transition.
        
        Why: D jumps from 0 to 1 at v = w.
        """
        v_range = np.linspace(0.2, 2.0, 50)
        w = 1.0
        
        D_values = []
        for v in v_range:
            D = SSHTopology.winding_number(v, w)
            D_values.append(D)
        
        # Find where D changes
        D_changes = np.where(np.abs(np.diff(D_values)) > 0.5)[0]
        
        # Should have one jump
        assert len(D_changes) == 1, f"Expected 1 jump, got {len(D_changes)}"
        
        # Jump should be near v = w = 1.0
        jump_index = D_changes[0]
        jump_v = v_range[jump_index]
        assert abs(jump_v - 1.0) < 0.1, f"Jump at v={jump_v:.3f}, expected v=1.0"
        
        print(f" Winding number jumps at v={jump_v:.3f} (expected 1.0)")
    
    def test_winding_number_phase_diagram(self):
        """
        Test phase diagram structure.
        
        Why: Phase diagram should show D=0 for v>w, D=1 for w>v.
        """
        D_map, v_vals, w_vals = SSHTopology.phase_diagram()
        
        # Test a few points
        v_idx = len(v_vals) // 3
        w_idx = len(w_vals) // 3
        
        # Trivial region (v > w)
        v_triv = v_vals[v_idx + 10]
        w_triv = w_vals[w_idx]
        D_triv = D_map[np.where(v_vals == v_triv)[0][0], 
                      np.where(w_vals == w_triv)[0][0]]
        assert abs(D_triv) < 0.1, f"In trivial region, D should be 0, got {D_triv}"
        
        # Topological region (w > v)
        v_top = v_vals[v_idx]
        w_top = w_vals[w_idx + 10]
        D_top = D_map[np.where(v_vals == v_top)[0][0],
                     np.where(w_vals == w_top)[0][0]]
        assert abs(D_top - 1.0) < 0.1, f"In topological region, D should be 1, got {D_top}"
        
        print(" Phase diagram has correct structure")


class TestZakPhase:
    """
    Tests for Zak phase calculations.
    """
    
    def test_zak_phase_trivial(self):
        """
        Test Zak phase in trivial phase.
        
        Why: Zak phase should be 0 for trivial phase.
        """
        v, w = 1.0, 0.5
        phase = SSHTopology.zak_phase(v, w)
        
        assert abs(phase) < 0.01, f"Zak phase should be 0, got {phase}"
        print(f" Trivial phase Zak phase: {phase:.6f}")
    
    def test_zak_phase_topological(self):
        """
        Test Zak phase in topological phase.
        
        Why: Zak phase should be π for topological phase.
        """
        v, w = 0.5, 1.0
        phase = SSHTopology.zak_phase(v, w)
        
        assert abs(phase - np.pi) < 0.01, f"Zak phase should be π, got {phase}"
        print(f" Topological phase Zak phase: {phase:.6f}")
    
    def test_zak_phase_quantization(self):
        """
        Test Zak phase is quantized.
        
        Why: Zak phase should be 0 or π due to symmetry.
        """
        test_params = [(1.0, 0.5), (0.5, 1.0), (1.2, 0.8), (0.3, 1.5)]
        
        for v, w in test_params:
            phase = SSHTopology.zak_phase(v, w)
            # Should be close to 0 or π
            assert abs(phase) < 0.01 or abs(phase - np.pi) < 0.01, \
                   f"Zak phase not quantized: {phase:.6f}"
        
        print(" Zak phase is quantized (0 or π)")


class TestBerryCurvature:
    """
    Tests for Berry curvature calculations.
    """
    
    def test_berry_curvature_integral(self):
        """
        Test integral of Berry curvature equals winding number.
        
        Why: ∫ Berry curvature dk = 2πD.
        """
        v, w = 0.5, 1.0  # Topological phase
        k = np.linspace(-np.pi, np.pi, 1000)
        
        berry = SSHTopology.berry_curvature(v, w, k)
        
        # Integrate Berry curvature
        integral = np.trapz(berry, k)
        
        # Should equal 2πD
        D = SSHTopology.winding_number(v, w, Nk=1000)
        expected = 2 * np.pi * D
        
        assert abs(integral - expected) < 0.01, \
               f"Berry curvature integral {integral:.6f} != {expected:.6f}"
        
        print(f" Berry curvature integral: {integral:.6f} ≈ 2πD = {expected:.6f}")


class TestBulkBoundaryCorrespondence:
    """
    Tests for bulk-boundary correspondence.
    """
    
    def test_bulk_boundary_correspondence(self):
        """
        Test that D predicts edge states.
        
        Why: This is the central principle of topological physics.
        """
        test_params = [
            (1.0, 0.5, 0),   # Trivial: D=0, 0 edge states
            (0.8, 1.0, 2),   # Topological: D=1, 2 edge states
            (0.5, 1.5, 2),   # Strongly topological: D=1, 2 edge states
            (1.5, 0.5, 0),   # Strongly trivial: D=0, 0 edge states
        ]
        
        for v, w, expected_edges in test_params:
            # Compute winding number
            D = SSHTopology.winding_number(v, w, Nk=1000)
            D_rounded = round(D)
            
            # Find edge states
            h = SSHHamiltonian(N=20, v=v, w=w)
            edge_states = h.get_edge_states()
            num_edges = len(edge_states['states']) if edge_states['found'] else 0
            
            # Check correspondence
            expected_D = 1 if expected_edges > 0 else 0
            
            assert abs(D_rounded - expected_D) < 0.1, \
                   f"D={D:.3f}, expected {expected_D}"
            assert num_edges == expected_edges, \
                   f"Got {num_edges} edges, expected {expected_edges}"
        
        print(" Bulk-boundary correspondence verified for multiple parameters")


def run_all_tests():
    """
    Run all topology tests.
    """
    test_classes = [
        TestWindingNumber,
        TestZakPhase,
        TestBerryCurvature,
        TestBulkBoundaryCorrespondence
    ]
    
    print("="*70)
    print("RUNNING SSH TOPOLOGY TESTS")
    print("="*70)
    print()
    
    passed = 0
    failed = 0
    
    for test_class in test_classes:
        instance = test_class()
        test_methods = [method for method in dir(instance) 
                       if method.startswith('test_') and callable(getattr(instance, method))]
        
        print(f"\nTesting {test_class.__name__}:")
        print("-"*50)
        
        for method_name in test_methods:
            try:
                getattr(instance, method_name)()
                passed += 1
                print(f"   {method_name}")
            except AssertionError as e:
                failed += 1
                print(f"   {method_name}: {str(e)}")
            except Exception as e:
                failed += 1
                print(f"   {method_name}: Unexpected error: {str(e)}")
    
    print("\n" + "="*70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*70)
    
    return passed, failed


if __name__ == "__main__":
    # Run tests
    passed, failed = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)