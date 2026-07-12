"""
analysis.py - Data analysis functions for SSH model.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from scipy.optimize import curve_fit

class SSHAnalysis:
    """
    Class for analyzing SSH model results.
    """
    
    @staticmethod
    def participation_ratio(psi: np.ndarray) -> float:
        """
        Compute the inverse participation ratio.
        
        Step 1: Quantify wavefunction delocalization.
        
        Why: The participation ratio tells us if a state is localized or extended.
        - Small PR (< 1): Localized state
        - Large PR (~N): Extended state
        
        Implementation:
        PR = 1 / Σ |ψ_i|⁴
        
        For edge states, PR should be small (localized at boundaries).
        """
        prob = np.abs(psi)**2
        ipr = np.sum(prob**2)
        return 1 / ipr if ipr > 0 else 0
    
    @staticmethod
    def localization_length(psi: np.ndarray) -> float:
        """
        Compute the localization length.
        
        Step 2: Measure spatial extent of wavefunction.
        
        Why: The localization length ξ quantifies how far the wavefunction
        penetrates into the bulk. For edge states, ξ should be finite
        and proportional to 1/log(w/v).
        
        Implementation:
        ξ = √(Σ (i - i)² |ψ_i|²)
        
        where i is the center of mass.
        """
        prob = np.abs(psi)**2
        N = len(prob)
        sites = np.arange(N)
        
        # Step 2a: Center of mass
        center = np.sum(sites * prob)
        
        # Step 2b: Variance
        variance = np.sum((sites - center)**2 * prob)
        
        return np.sqrt(variance) if variance > 0 else 0
    
    @staticmethod
    def center_of_mass(psi: np.ndarray) -> float:
        """
        Compute the center of mass of wavefunction.
        
        Step 3: Locate the wavefunction position.
        
        Why: For edge states, the center of mass should be near the boundaries
        (site 0 or site 2N-1). For bulk states, it should be in the middle.
        """
        prob = np.abs(psi)**2
        N = len(prob)
        sites = np.arange(N)
        return np.sum(sites * prob)
    
    @staticmethod
    def fit_exponential_decay(prob: np.ndarray, N: int) -> Dict:
        """
        Fit exponential decay of edge states.
        
        Step 4: Verify exponential localization.
        
        Why: Edge states should decay exponentially from the boundary:
        |ψ(x)|² ∝ exp(-x/ξ)
        
        This exponential decay is the signature of a topological edge state.
        
        Implementation:
        1. Take the tail of the probability distribution
        2. Fit to A*exp(-x/ξ)
        3. Extract the decay length ξ
        """
        # Step 4a: Use the tail of the distribution
        # Why: The exponential decay is most visible away from the boundary
        tail_start = N // 2
        tail = prob[tail_start:2*N]
        x_data = np.arange(len(tail))
        
        # Step 4b: Avoid numerical issues with zeros
        tail_safe = np.maximum(tail, 1e-12)
        
        try:
            # Step 4c: Fit in log space
            # Why: Linearizing the exponential makes fitting easier
            log_tail = np.log(tail_safe)
            coeffs = np.polyfit(x_data, log_tail, 1)
            
            # Step 4d: Extract decay length
            # Why: ξ = -1/coeff[0] where coeff[0] is the decay rate
            # FIXED: Use coeffs[0] instead of coeff[0]
            decay_length = -1/coeffs[0] if coeffs[0] < 0 else np.inf
            
            return {
                'decay_length': decay_length,
                'amplitude': np.exp(coeffs[1]),
                'fit_coeffs': coeffs,
                'success': True
            }
        except Exception as e:
            return {
                'decay_length': np.inf,
                'amplitude': None,
                'fit_coeffs': None,
                'success': False,
                'error': str(e)
            }
    
    @staticmethod
    def fit_exponential_decay_scipy(prob: np.ndarray, N: int) -> Dict:
        """
        Fit exponential decay using scipy.optimize.curve_fit.
        
        Step 4 (Alternative): Use scipy for more robust fitting.
        
        Why: curve_fit provides error estimates and is more robust
        than polyfit for exponential fits.
        """
        # Define exponential function
        def exp_decay(x, A, xi):
            return A * np.exp(-x / xi)
        
        # Step 4a: Use the tail of the distribution
        tail_start = N // 2
        tail = prob[tail_start:2*N]
        x_data = np.arange(len(tail))
        
        # Step 4b: Initial guess for parameters
        # A ≈ max(tail), xi ≈ 1 (reasonable guess)
        A_guess = np.max(tail)
        xi_guess = 1.0
        
        try:
            # Step 4c: Fit using curve_fit
            popt, pcov = curve_fit(exp_decay, x_data, tail, 
                                  p0=[A_guess, xi_guess],
                                  bounds=([0, 0], [np.inf, np.inf]))
            
            A_fit, xi_fit = popt
            
            # Step 4d: Calculate R² to assess fit quality
            residuals = tail - exp_decay(x_data, A_fit, xi_fit)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((tail - np.mean(tail))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return {
                'decay_length': xi_fit,
                'amplitude': A_fit,
                'r_squared': r_squared,
                'success': True,
                'fit_params': popt,
                'fit_covariance': pcov
            }
        except Exception as e:
            return {
                'decay_length': np.inf,
                'amplitude': None,
                'r_squared': 0,
                'success': False,
                'error': str(e)
            }
    
    @staticmethod
    def gap_analysis(eigenvalues: np.ndarray, N: int) -> Dict:
        """
        Analyze the energy gap.
        
        Step 5: Quantify the band gap.
        
        Why: The gap size and its behavior determine many physical properties:
        1. Topological phase: gap closes at v = w
        2. Trivial phase: gap remains open
        3. Edge states: appear in the gap
        """
        # Step 5a: Identify bulk states
        # Why: Bulk states are the N highest occupied and N lowest unoccupied
        gap = eigenvalues[N] - eigenvalues[N-1]
        
        # Step 5b: Look for mid-gap states
        mid_gap_states = []
        for i, e in enumerate(eigenvalues):
            if abs(e) < gap/2 and (i < N-1 or i > N):
                mid_gap_states.append({'index': i, 'energy': e})
        
        return {
            'gap': gap,
            'mid_gap_states': mid_gap_states,
            'num_mid_gap_states': len(mid_gap_states)
        }
    
    @staticmethod
    def density_of_states(eigenvalues: np.ndarray, 
                         energy_range: Tuple[float, float],
                         n_bins: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the density of states.
        
        Step 6: Calculate the DOS.
        
        Why: The density of states shows how many states exist at each energy.
        It's a useful diagnostic for:
        1. Identifying gaps
        2. Seeing edge states (peaks at E=0)
        3. Understanding the bulk spectrum
        """
        # Step 6a: Create energy bins
        energy_min, energy_max = energy_range
        bins = np.linspace(energy_min, energy_max, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Step 6b: Histogram eigenvalues
        hist, _ = np.histogram(eigenvalues, bins=bins)
        
        return bin_centers, hist
    
    @staticmethod
    def entanglement_entropy(psi: np.ndarray, partition_size: int) -> float:
        """
        Compute entanglement entropy.
        
        Step 7: Measure quantum entanglement.
        
        Why: Topological states often have non-trivial entanglement properties.
        The entanglement entropy of the ground state is a useful diagnostic.
        
        Implementation:
        1. Partition the system into A and B
        2. Compute the reduced density matrix
        3. Calculate von Neumann entropy S = -Tr(ρ_A log ρ_A)
        """
        # Step 7a: Partition wavefunction
        psi_A = psi[:partition_size]
        psi_B = psi[partition_size:]
        
        # Step 7b: Compute reduced density matrix
        # ρ_A = Tr_B |ψψ|
        rho_A = np.outer(psi_A, psi_A.conjugate())
        
        # Step 7c: Compute eigenvalues of ρ_A
        eigvals = np.linalg.eigvalsh(rho_A)
        eigvals = eigvals[eigvals > 1e-10]  # Remove zeros
        
        # Step 7d: Compute entropy
        S = -np.sum(eigvals * np.log(eigvals))
        
        return S
    
    @staticmethod
    def analyze_edge_state(psi: np.ndarray, N: int) -> Dict:
        """
        Comprehensive analysis of an edge state.
        
        Step 8: Combine all edge state analyses.
        
        Why: This provides a complete characterization of an edge state
        in one function call.
        """
        prob = np.abs(psi)**2
        
        # Compute all relevant quantities
        pr = SSHAnalysis.participation_ratio(psi)
        loc = SSHAnalysis.localization_length(psi)
        com = SSHAnalysis.center_of_mass(psi)
        
        # Fit exponential decay
        fit_result = SSHAnalysis.fit_exponential_decay(prob, N)
        
        # Determine if it's truly an edge state
        is_edge = (com < N/4 or com > 3*N/4) and loc < N/2
        
        return {
            'participation_ratio': pr,
            'localization_length': loc,
            'center_of_mass': com,
            'is_edge_state': is_edge,
            'decay_length': fit_result.get('decay_length', np.inf),
            'fit_success': fit_result.get('success', False),
            'fit_details': fit_result
        }