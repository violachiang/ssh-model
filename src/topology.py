"""
topology.py - Topological invariant calculations for SSH model.
"""

import numpy as np
from typing import Tuple, Optional

class SSHTopology:
    """
    Class for computing topological invariants.
    """
    
    @staticmethod
    def winding_number(v: float, w: float, Nk: int = 1000) -> float:
        """
        Calculate the winding number.
        
        Step 1: Compute D for given v and w.
        
        Why: The winding number is the topological invariant that
        classifies the SSH model. It determines whether edge states exist.
        
        Implementation Details:
        1. Discretize the Brillouin zone (k ∈ [-π, π])
        2. Compute d_x(k) and d_y(k)
        3. Normalize to get (d̂_x, d̂_y)
        4. Calculate the winding using numerical integration
        """
        k = np.linspace(-np.pi, np.pi, Nk)
        dk = k[1] - k[0]
        
        # Step 1a: Compute vector components
        d_x = v + w * np.cos(k)
        d_y = w * np.sin(k)
        
        # Step 1b: Normalize
        # Why: We need the direction, not magnitude
        mag = np.sqrt(d_x**2 + d_y**2)
        dx_hat = d_x / mag
        dy_hat = d_y / mag
        
        # Step 1c: Derivatives
        # Why: The winding number involves derivatives with respect to k
        ddx = np.gradient(dx_hat, k)
        ddy = np.gradient(dy_hat, k)
        
        # Step 1d: Integrand
        # Why: This is the Berry connection in 1D
        integrand = dx_hat * ddy - dy_hat * ddx
        
        # Step 1e: Integrate
        D = (1/(2*np.pi)) * np.trapz(integrand, k)
        
        return D
    
    @staticmethod
    def winding_number_analytic(v: float, w: float) -> float:
        """
        Compute winding number analytically.
        
        Step 2: Analytic verification.
        
        Why: The SSH model has a known analytic winding number:
        D = 0 if v > w, D = 1 if w > v
        
        This validates our numerical implementation.
        """
        if abs(v - w) < 1e-10:
            return 0.5  # Undefined at phase transition
        return 1.0 if w > v else 0.0
    
    @staticmethod
    def phase_diagram(v_range: Tuple[float, float] = (0.2, 2.0),
                     w_range: Tuple[float, float] = (0.2, 2.0),
                     N_points: int = 40) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the topological phase diagram.
        
        Step 3: Map D over parameter space.
        
        Why: The phase diagram shows the topological phase transition.
        It clearly demonstrates that the phase boundary is v = w.
        
        Implementation:
        1. Create a grid of (v, w) points
        2. Compute D at each point
        3. Return the D-map for visualization
        """
        v_vals = np.linspace(v_range[0], v_range[1], N_points)
        w_vals = np.linspace(w_range[0], w_range[1], N_points)
        
        D_map = np.zeros((N_points, N_points))
        
        # Step 3a: Loop over parameter space
        # Why: We need D for each combination of v and w
        for i, v in enumerate(v_vals):
            for j, w in enumerate(w_vals):
                D_map[i, j] = SSHTopology.winding_number(v, w, Nk=200)
        
        return D_map, v_vals, w_vals
    
    @staticmethod
    def zak_phase(v: float, w: float) -> float:
        """
        Calculate the Zak phase.
        
        Step 4: Compute the Berry phase.
        
        Why: The Zak phase is the Berry phase across the Brillouin zone.
        For the SSH model, it's quantized: 0 (trivial) or π (topological).
        """
        # Step 4a: Compute the Zak phase
        # Why: For SSH, the Zak phase = π * (number of winding)
        return np.pi if w > v else 0.0
    
    @staticmethod
    def berry_curvature(v: float, w: float, k: np.ndarray) -> np.ndarray:
        """
        Compute the Berry curvature.
        
        Step 5: Calculate local topological quantity.
        
        Why: The Berry curvature is the local manifestation of topology.
        Its integral gives the winding number.
        """
        # Step 5a: Compute d-vector components
        d_x = v + w * np.cos(k)
        d_y = w * np.sin(k)
        
        # Step 5b: Berry curvature in 1D
        # Why: In 1D, the Berry curvature is simply dθ/dk
        # where θ = atan2(d_y, d_x)
        dphi = (d_x * np.gradient(d_y, k) - d_y * np.gradient(d_x, k)) / (d_x**2 + d_y**2)
        
        return dphi
    
    @staticmethod
    def gap_closing_condition(v: float, w: float) -> bool:
        """
        Check if gap is closing.
        
        Step 6: Identify phase transition.
        
        Why: The gap closes at v = w, marking the topological phase transition.
        This is where the winding number changes discontinuously.
        """
        return abs(v - w) < 1e-10
    
    @staticmethod
    def critical_exponent(v: float, w: float) -> float:
        """
        Compute critical exponent near phase transition.
        
        Step 7: Analyze critical behavior.
        
        Why: Near the phase transition, the gap closes as |v-w|^ν.
        This is a universal feature of topological phase transitions.
        """
        # Step 7a: Gap size near transition
        # Why: The gap scales as |v-w| for the SSH model
        return np.abs(v - w)