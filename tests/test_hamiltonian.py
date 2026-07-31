"""
test_hamiltonian.py - Unit tests for SSH Hamiltonian construction.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.hamiltonian import SSHHamiltonian


class TestHamiltonianConstruction:
    """
    Test suite for Hamiltonian construction.
    """
    
    def test_hamiltonian_dimensions(self):
        """
        Test that Hamiltonian has correct dimensions.
        
        Why: The SSH Hamiltonian should be (2N) × (2N).
        This is a fundamental property we must verify.
        """
        N = 10
        v, w = 1.0, 0.5
        
        h = SSHHamiltonian(N, v, w)
        H = h.build_real_space()
        
        # Check dimensions
        assert H.shape == (2*N, 2*N), f"Expected {(2*N, 2*N)}, got {H.shape}"
        print(f" Hamiltonian dimensions: {H.shape} (correct)")
    
    def test_hopping_placement(self):
        """
        Test that hopping amplitudes are placed correctly.
        
        Why: The pattern of v and w alternates. Wrong placement
        would break the SSH model physics.
        """
        N = 5
        v, w = 1.0, 0.5
        
        h = SSHHamiltonian(N, v, w)
        H = h.build_real_space()
        
        # Check intra-cell hopping (v) at correct positions
        for n in range(N):
            assert H[2*n, 2*n+1] == v, f"v not at position ({2*n}, {2*n+1})"
            assert H[2*n+1, 2*n] == v, f"v not at position ({2*n+1}, {2*n})"
        
        # Check inter-cell hopping (w) at correct positions
        for n in range(N-1):
            assert H[2*n+1, 2*(n+1)] == w, f"w not at position ({2*n+1}, {2*(n+1)})"
            assert H[2*(n+1), 2*n+1] == w, f"w not at position ({2*(n+1)}, {2*n+1})"
        
        print(" Hopping amplitudes correctly placed")
    
    def test_hermiticity(self):
        """
        Test that Hamiltonian is Hermitian (symmetric for real parameters).
        
        Why: Quantum Hamiltonians must be Hermitian to have real eigenvalues.
        This is a fundamental requirement of quantum mechanics.
        """
        N = 10
        v, w = 1.0, 0.5
        
        h = SSHHamiltonian(N, v, w)
        H = h.build_real_space()
        
        # Check symmetry: H[i,j] == H[j,i]
        assert np.allclose(H, H.T), "Hamiltonian is not symmetric"
        print(" Hamiltonian is Hermitian (symmetric)")
    
    def test_eigenvalues_real(self):
        """
        Test that eigenvalues are real.
        
        Why: Hermitian matrices have real eigenvalues.
        This validates our diagonalization is correct.
        """
        N = 20
        v, w = 1.0, 0.5
        
        h = SSHHamiltonian(N, v, w)
        eigenvalues = h.diagonalize()[0]
        
        # Check all eigenvalues are real (no imaginary parts)
        assert np.all(np.isreal(eigenvalues)), "Eigenvalues have imaginary parts"
        print(" All eigenvalues are real")
    
    def test_spectrum_symmetry(self):
        """
        Test that spectrum is symmetric around E=0.
        
        Why: The SSH model has chiral symmetry, which implies
        the spectrum is particle-hole symmetric: E and -E appear together.
        """
        N = 20
        v, w = 1.0, 0.5
        
        h = SSHHamiltonian(N, v, w)
        eigenvalues = h.diagonalize()[0]
        
        # Check eigenvalues come in ± pairs
        # Sort absolute values and compare pairs
        abs_eigenvals = np.sort(np.abs(eigenvalues))
        # Pair up: abs_eigenvals[0] with abs_eigenvals[1], etc.
        for i in range(0, len(abs_eigenvals)-1, 2):
            assert abs(abs_eigenvals[i] - abs_eigenvals[i+1]) < 1e-10, \
                   f"Eigenvalue pair {i}: {abs_eigenvals[i]:.10f}, {abs_eigenvals[i+1]:.10f} not equal"
        
        print(" Spectrum is symmetric around E=0 (chiral symmetry preserved)")
    
    def test_trivial_gap(self):
        """
        Test that trivial phase has a gap with no edge states.
        
        Why: For v > w, the system is trivial with no zero-energy states.
        """
        N = 20
        v, w = 1.0, 0.5  # Trivial phase: v > w
        
        h = SSHHamiltonian(N, v, w)
        info = h.get_spectrum_info()
        
        # Gap should be positive
        assert info['gap'] > 0, f"Gap should be positive, got {info['gap']}"
        
        # No edge states
        assert info['num_zero_states'] == 0, \
               f"Found {info['num_zero_states']} edge states in trivial phase"
        
        # Phase should be trivial
        assert info['phase'] == 'trivial', f"Phase should be trivial, got {info['phase']}"
        
        print(f" Trivial phase: gap={info['gap']:.4f}, no edge states")
    
    def test_topological_gap(self):
        """
        Test that topological phase has mid-gap edge states.
        
        Why: For w > v, two zero-energy edge states should appear.
        """
        N = 20
        v, w = 0.5, 1.0  # Topological phase: w > v
        
        h = SSHHamiltonian(N, v, w)
        info = h.get_spectrum_info()
        
        # Gap should exist
        assert info['gap'] > 0, f"Gap should be positive, got {info['gap']}"
        
        # Should have exactly 2 edge states
        assert info['num_zero_states'] == 2, \
               f"Found {info['num_zero_states']} edge states, expected 2"
        
        # Phase should be topological
        assert info['phase'] == 'topological', f"Phase should be topological, got {info['phase']}"
        
        print(f" Topological phase: gap={info['gap']:.4f}, 2 edge states")
    
    def test_edge_state_localization(self):
        """
        Test that edge states are localized at boundaries.
        
        Why: Edge states should be exponentially localized near the chain ends.
        """
        N = 40
        v, w = 0.4, 1.0  # Topological phase
        
        h = SSHHamiltonian(N, v, w)
        edge_states = h.get_edge_states()
        
        # Should find edge states
        assert edge_states['found'], "No edge states found"
        
        for state in edge_states['states']:
            # Center of mass should be near boundaries (site 0 or site 2N-1)
            com = state['center']
            assert com < N or com > 3*N, \
                   f"Edge state centered at {com:.2f}, expected near boundary"
            
            # Localization length should be finite (< N)
            assert state['spread'] < N/2, \
                   f"Edge state spread {state['spread']:.2f} too large"
        
        print(f" Edge states localized at boundaries (N={N})")
    
    def test_momentum_space_dimensions(self):
        """
        Test momentum-space Hamiltonian dimensions.
        
        Why: H(k) should be 2×2 for each k-point.
        """
        N = 10
        v, w = 1.0, 0.5
        k = np.linspace(-np.pi, np.pi, 100)
        
        h = SSHHamiltonian(N, v, w)
        Hk = h.build_momentum_space(k)
        
        assert Hk.shape == (len(k), 2, 2), \
               f"Expected {(len(k), 2, 2)}, got {Hk.shape}"
        print(" Momentum-space Hamiltonian dimensions correct")
    
    def test_band_structure_analytical_match(self):
        """
        Test numerical bands match analytical bands.
        
        Why: The SSH model has analytical solution E(k) = ±√(v²+w²+2vw cos(k)).
        Numerical results should match this exactly.
        """
        N = 10
        v, w = 1.0, 0.5
        k = np.linspace(-np.pi, np.pi, 100)
        
        h = SSHHamiltonian(N, v, w)
        numerical_bands = h.get_bands(k)
        analytical_bands = h.get_analytical_bands(k)
        
        # Check both bands match
        assert np.allclose(numerical_bands, analytical_bands, atol=1e-10), \
               "Numerical bands don't match analytical bands"
        
        print(" Numerical bands match analytical solution")
    
    def test_d_vector(self):
        """
        Test d-vector components.
        
        Why: d_x(k) = v + w cos(k), d_y(k) = w sin(k).
        These are needed for the winding number.
        """
        N = 10
        v, w = 1.0, 0.5
        k = np.array([0, np.pi/2, np.pi])
        
        h = SSHHamiltonian(N, v, w)
        d_x, d_y = h.get_d_vector(k)
        
        # Check at specific k values
        expected_dx = v + w * np.cos(k)
        expected_dy = w * np.sin(k)
        
        assert np.allclose(d_x, expected_dx), f"d_x mismatch: {d_x} vs {expected_dx}"
        assert np.allclose(d_y, expected_dy), f"d_y mismatch: {d_y} vs {expected_dy}"
        
        print(" d-vector components correct")
    
    def test_winding_number_values(self):
        """
        Test winding number gives correct values.
        
        Why: D = 0 for trivial, D = 1 for topological.
        """
        # Test trivial phase
        D_trivial = SSHHamiltonian.winding_number(1.0, 0.5)
        assert abs(D_trivial - 0.0) < 0.01, f"Expected 0, got {D_trivial}"
        
        # Test topological phase
        D_topological = SSHHamiltonian.winding_number(0.5, 1.0)
        assert abs(D_topological - 1.0) < 0.01, f"Expected 1, got {D_topological}"
        
        print(f" Winding numbers: trivial={D_trivial:.3f}, topological={D_topological:.3f}")


class TestEdgeStateProperties:
    """
    Tests specific to edge state properties.
    """
    
    def test_edge_state_energy(self):
        """
        Test that edge states are exactly at E=0.
        
        Why: Chiral symmetry protects zero-energy edge states.
        """
        N = 40
        v, w = 0.4, 1.0
        
        h = SSHHamiltonian(N, v, w)
        edge_states = h.get_edge_states(tol=1e-6)
        
        assert edge_states['found'], "No edge states found"
        
        for state in edge_states['states']:
            assert abs(state['energy']) < 1e-6, \
                   f"Edge state energy {state['energy']:.10f} not near zero"
        
        print(" Edge states at E=0 (chiral symmetry protected)")
    
    def test_edge_state_number(self):
        """
        Test that number of edge states equals 2|D|.
        
        Why: Bulk-boundary correspondence says 2 edge states per winding number.
        """
        # Topological phase (D=1) -> 2 edge states
        h_top = SSHHamiltonian(20, 0.5, 1.0)
        top_states = h_top.get_edge_states()
        assert len(top_states['states']) == 2, \
               f"Expected 2 edge states, got {len(top_states['states'])}"
        
        # Trivial phase (D=0) -> 0 edge states
        h_triv = SSHHamiltonian(20, 1.0, 0.5)
        triv_states = h_triv.get_edge_states()
        assert len(triv_states['states']) == 0, \
               f"Expected 0 edge states, got {len(triv_states['states'])}"
        
        print(" Edge state number follows bulk-boundary correspondence")
    
    def test_edge_state_exponential_decay(self):
        """
        Test that edge states decay exponentially.
        
        Why: Topological edge states have exponential localization.
        """
        N = 60
        v, w = 0.3, 1.0
        
        h = SSHHamiltonian(N, v, w)
        edge_states = h.get_edge_states()
        
        assert edge_states['found'], "No edge states found"
        
        for state in edge_states['states']:
            prob = state['probability']
            
            # Check probability decays from edges
            # Left edge: should decay from left to right
            left_half = prob[:N]
            right_half = prob[N:]
            
            # On log scale, should be linear (exponential decay)
            # Check left edge decays
            left_log = np.log(left_half + 1e-12)
            # Should be decreasing
            assert left_log[0] > left_log[-1], "Left edge not decaying"
            
            # Check right edge decays
            right_log = np.log(right_half + 1e-12)
            # Should be decreasing from right to left (so increasing left to right)
            assert right_log[0] < right_log[-1], "Right edge not decaying"
        
        print(" Edge states show exponential decay")


def run_all_tests():
    """
    Run all tests and display results.
    """
    test_classes = [TestHamiltonianConstruction, TestEdgeStateProperties]
    
    print("="*70)
    print("RUNNING SSH HAMILTONIAN TESTS")
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