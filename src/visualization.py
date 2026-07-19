"""
visualization.py - Plotting functions for SSH model.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Optional, Tuple, List, Dict
import warnings

class SSHVisualizer:
    """
    Class for visualizing SSH model results.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize visualizer with style settings.
        
        Step 1: Set up plotting style.
        
        Why: Consistent styling makes figures publication-ready.
        Good style choices improve readability and aesthetics.
        """
        try:
            plt.style.use(style)
        except:
            # Fallback to default if style not available
            warnings.warn(f"Style {style} not found. Using default.")
            plt.style.use('default')
        
        self.colors = {
            'trivial': '#2E86AB',      # Blue
            'topological': '#E63946',   # Red
            'critical': '#2D6A4F',      # Green
            'edge': '#F4A261',           # Orange
            'bulk': '#6C757D'            # Gray
        }
    
    def plot_spectrum(self, eigenvalues: np.ndarray, 
                     title: str = "Energy Spectrum",
                     highlight_gap: bool = True,
                     show_zero: bool = True,
                     ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot energy spectrum.
        
        Step 2: Visualize energy levels.
        
        Why: The spectrum plot shows:
        1. The band structure (bulk states)
        2. The gap (if present)
        3. Mid-gap states (edge states)
        
        This is the most fundamental plot in the SSH model.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        N = len(eigenvalues) // 2
        
        # Step 2a: Plot eigenvalues as scatter
        # Why: Scatter plot clearly shows individual energy levels
        ax.scatter(range(len(eigenvalues)), eigenvalues, 
                  s=30, c='blue', alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Step 2b: Highlight gap if present
        # Why: The gap is the key feature of the spectrum
        if highlight_gap and N > 0 and N < len(eigenvalues):
            gap = eigenvalues[N] - eigenvalues[N-1]
            if gap > 0.1:
                ax.axhspan(eigenvalues[N-1], eigenvalues[N], 
                          alpha=0.2, color='green')
                ax.text(0.02, 0.98, f'Gap = {gap:.4f}', 
                       transform=ax.transAxes,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Step 2c: Show zero energy line
        # Why: Zero energy is special (edge states appear here)
        if show_zero:
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        ax.set_xlabel('Eigenvalue Index')
        ax.set_ylabel('Energy (E)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_edge_spectrum(self, w_values: np.ndarray, 
                          all_eigenvalues: np.ndarray,
                          v: float = 1.0,
                          ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot spectrum vs w to show edge state emergence.
        
        Step 3: Visualize topological transition.
        
        Why: This plot shows:
        1. How the gap closes at v = w
        2. How edge states emerge for w > v
        3. The bulk-boundary correspondence in action
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        N = all_eigenvalues.shape[1] // 2
        
        # Step 3a: Plot all eigenvalues as function of w
        for i in range(all_eigenvalues.shape[1]):
            ax.plot(w_values, all_eigenvalues[:, i], 
                   'b-', alpha=0.15, linewidth=0.5)
        
        # Step 3b: Highlight phase transition
        ax.axvline(x=v, color='red', linestyle='--', linewidth=2, 
                  alpha=0.7, label=f'Phase Transition (w=v={v})')
        
        # Step 3c: Mark gap closing
        gap_indices = np.where(np.abs(w_values - v) < 0.01)[0]
        if len(gap_indices) > 0:
            idx = gap_indices[0]
            gap = all_eigenvalues[idx, N] - all_eigenvalues[idx, N-1]
            if gap < 0.1:
                ax.scatter([v], [0], color='red', s=200, zorder=5,
                          label='Gap Closing Point', marker='x')
        
        ax.set_xlabel('w (inter-cell hopping)')
        ax.set_ylabel('Energy (E)')
        ax.set_title(f'Energy Spectrum vs w (v={v})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_edge_wavefunction(self, sites: np.ndarray, prob: np.ndarray,
                              title: str = "Edge State Wavefunction",
                              log_scale: bool = False,
                              ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot edge state wavefunction.
        
        Step 4: Visualize edge state localization.
        
        Why: This plot demonstrates:
        1. Exponential localization at boundaries
        2. The spatial extent of edge states
        3. That edge states are physical, not just mathematical constructs
        """
        if ax is None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        else:
            ax1, ax2 = ax, None
        
        # Step 4a: Linear plot
        ax1.plot(sites, prob, 'b-', linewidth=2, alpha=0.8)
        ax1.set_xlabel('Site Index')
        ax1.set_ylabel('$|\psi(x)|^2$')
        ax1.set_title('Linear Scale')
        ax1.grid(True, alpha=0.3)
        
        # Step 4b: Log plot to show exponential decay
        ax2.semilogy(sites, prob + 1e-12, 'r-', linewidth=2, alpha=0.8)
        ax2.set_xlabel('Site Index')
        ax2.set_ylabel('$|\psi(x)|^2$ (log scale)')
        ax2.set_title('Log Scale (Exponential Decay)')
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        return fig
    
    def plot_band_structure(self, k: np.ndarray, bands: np.ndarray,
                           title: str = "Band Structure",
                           analytical: Optional[np.ndarray] = None,
                           ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot band structure.
        
        Step 5: Visualize bulk bands.
        
        Why: The band structure shows:
        1. The dispersion relation E(k)
        2. The band gap
        3. The topological phase (through gap behavior)
        
        This is the "smoking gun" of the SSH model.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Step 5a: Plot numerical bands
        ax.plot(k, bands[:, 0], 'b-', linewidth=2, label='Lower band')
        ax.plot(k, bands[:, 1], 'r-', linewidth=2, label='Upper band')
        
        # Step 5b: Overlay analytical result if provided
        if analytical is not None:
            ax.plot(k, analytical[:, 0], 'b--', linewidth=1, alpha=0.5)
            ax.plot(k, analytical[:, 1], 'r--', linewidth=1, alpha=0.5)
            ax.text(0.02, 0.98, 'Solid: Numerical\nDashed: Analytical',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.set_xlabel('k')
        ax.set_ylabel('E(k)')
        ax.set_title(title)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_winding_trajectory(self, dx: np.ndarray, dy: np.ndarray,
                               winding_number: float,
                               title: str = "Winding Trajectory",
                               ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot winding trajectory in d-space.
        
        Step 6: Visualize the topological invariant.
        
        Why: This plot shows:
        1. The path traced by (d_x, d_y) in the Brillouin zone
        2. Whether the origin is enclosed (winding)
        3. The geometric meaning of the winding number
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        # Step 6a: Plot trajectory
        ax.plot(dx, dy, 'b-', linewidth=2, alpha=0.8)
        
        # Step 6b: Mark key points
        ax.plot(dx[0], dy[0], 'go', markersize=10, label='k=0')
        ax.plot(dx[-1], dy[-1], 'ro', markersize=10, label='k=π')
        ax.plot(0, 0, 'kx', markersize=15, markeredgewidth=2, label='Origin')
        
        # Step 6c: Add arrows to show direction
        N = len(dx)
        arrow_indices = np.linspace(0, N-1, 15, dtype=int)
        for i in arrow_indices:
            if i < N - 10:
                ax.arrow(dx[i], dy[i], 
                        dx[i+10]-dx[i], dy[i+10]-dy[i],
                        head_width=0.05, head_length=0.05,
                        fc='blue', ec='blue', alpha=0.5)
        
        # Step 6d: Add color-coded phase indicator
        phase = 'Topological' if winding_number > 0.5 else 'Trivial'
        color = 'red' if winding_number > 0.5 else 'blue'
        ax.text(0.5, 0.95, f'Phase: {phase} (D={winding_number:.3f})',
               transform=ax.transAxes, fontsize=14,
               color=color, ha='center',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        ax.set_xlabel('$d_x$')
        ax.set_ylabel('$d_y$')
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_phase_diagram(self, v_vals: np.ndarray, w_vals: np.ndarray, 
                          D_map: np.ndarray,
                          ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot phase diagram.
        
        Step 7: Visualize the topological phase diagram.
        
        Why: The phase diagram shows:
        1. The boundary between topological and trivial phases
        2. The universality of the phase transition
        3. The parameter space of the SSH model
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 7))
        
        # Step 7a: Plot winding number as colormap
        im = ax.imshow(D_map, 
                      extent=[v_vals[0], v_vals[-1], w_vals[0], w_vals[-1]],
                      origin='lower', cmap='RdBu', aspect='auto',
                      vmin=-0.1, vmax=1.1)
        
        # Step 7b: Mark phase boundary
        ax.plot([v_vals[0], v_vals[-1]], [v_vals[0], v_vals[-1]], 
               'k--', linewidth=2, alpha=0.7, label='v = w')
        
        ax.set_xlabel('v')
        ax.set_ylabel('w')
        ax.set_title('Topological Phase Diagram')
        ax.legend()
        plt.colorbar(im, ax=ax, label='Winding Number D')
        
        return ax
    
    def plot_summary_figure(self, results: Dict) -> None:
        """
        Create comprehensive summary figure.
        
        Step 8: Compile all results into one figure.
        
        Why: A summary figure shows the complete picture:
        1. Spectrum (trivial vs topological)
        2. Edge states
        3. Band structure
        4. Winding number
        5. Phase diagram
        
        This is the ultimate product of the project.
        """
        fig = plt.figure(figsize=(18, 14))
        
        # Step 8a: Trivial spectrum
        ax1 = plt.subplot(2, 3, 1)
        self.plot_spectrum(results['eigenvals_trivial'], 
                          title='Trivial Phase: v > w', ax=ax1)
        
        # Step 8b: Topological spectrum
        ax2 = plt.subplot(2, 3, 2)
        self.plot_spectrum(results['eigenvals_topological'], 
                          title='Topological Phase: w > v', ax=ax2)
        
        # Step 8c: Edge state
        ax3 = plt.subplot(2, 3, 3)
        if 'edge_state' in results:
            self.plot_edge_wavefunction(
                results['edge_state']['sites'],
                results['edge_state']['probability'],
                title='Edge State Localization', ax=ax3
            )
        
        # Step 8d: Band structure
        ax4 = plt.subplot(2, 3, 4)
        if 'bands' in results:
            self.plot_band_structure(
                results['bands']['k'],
                results['bands']['energy_trivial'],
                title='Band Structure (Trivial)', ax=ax4
            )
        
        # Step 8e: Winding trajectory
        ax5 = plt.subplot(2, 3, 5)
        if 'winding' in results:
            self.plot_winding_trajectory(
                results['winding']['dx'],
                results['winding']['dy'],
                results['winding']['number'],
                title='Winding Number', ax=ax5
            )
        
        # Step 8f: Phase diagram
        ax6 = plt.subplot(2, 3, 6)
        if 'phase_diagram' in results:
            self.plot_phase_diagram(
                results['phase_diagram']['v'],
                results['phase_diagram']['w'],
                results['phase_diagram']['D'],
                ax=ax6
            )
        
        plt.suptitle('SSH Model: Complete Analysis', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.show()