import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import sys
import os
import time
import argparse

from LLG_solver import compare_phases

# Utility class to suppress the verbose output from LLG_solver during the long sweep
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def generate_phase_diagram(n_H=26, n_A=33, L=32):
    """
    Generates a phase diagram by scanning H and A.
    Saves the data and automatically calls the plotter.
    """
    # H ranges from 0 to 2.5
    # A ranges from -1.5 to 1.7
    H_vals = np.linspace(0, 2.5, n_H)
    A_vals = np.linspace(-1.5, 1.7, n_A)
    
    # Matrix to store the result phase IDs
    phase_grid = np.zeros((n_A, n_H))
    
    # Mapping strings to integer IDs for the heatmap
    phase_map = {"SkX": 0, "SC": 1, "SP": 2, "FM": 3}
    
    print(f"=== Starting Phase Diagram Generation ===")
    print(f"Total points: {n_H * n_A} (H resolution: {n_H}, A resolution: {n_A})")
    print(f"Using lattice size L={L} for the sweep.")
    print("-----------------------------------------")
    
    start_time = time.time()
    
    for i, a in enumerate(A_vals):
        for j, h in enumerate(H_vals):
            # Print without newline to track progress on a single line
            sys.stdout.write(f"\rComputing Point {i*n_H + j + 1}/{n_H * n_A} | H = {h:.2f}, A = {a:.2f} ... ")
            sys.stdout.flush()
            
            try:
                # Suppress output from the inner LLG solver to keep the console clean
                with HiddenPrints():
                    winner, _ = compare_phases(H_scaled=h, A_scaled=a, L=L, 
                                               plot_ansatz=False, live_plot=False)
                
                # Retrieve the integer ID for the winning phase, or -1 if something went wrong
                phase_grid[i, j] = phase_map.get(winner, -1)
                
            except Exception as e:
                print(f"\nFailed to converge at H={h}, A={a}: {e}")
                phase_grid[i, j] = -1 # Mark as invalid
                
    elapsed = time.time() - start_time
    print(f"\n\nSweep finished in {elapsed:.2f} seconds!")
    
    # Save the exact simulation data so you don't have to recompute just to adjust the plot design!
    np.save("phase_diagram_data.npy", phase_grid)
    np.save("H_vals.npy", H_vals)
    np.save("A_vals.npy", A_vals)
    
    print("Data saved to 'phase_diagram_data.npy'. Generating plot...")
    plot_phase_diagram(phase_grid, H_vals, A_vals)


def plot_phase_diagram(phase_grid, H_vals, A_vals):
    """
    Renders the integer grid as a clean, colored phase diagram.
    """
    # Use standard light background for academic figures
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Muted, professional pastel/academic palette
    # SkX (Muted Blue), SC (Muted Green), SP (Muted Orange/Brown), FM (Light Gray/White)
    cmap = ListedColormap(['#4C72B0', '#55A868', '#DD8452', '#EAEAF2'])
    cmap.set_under('#808080') # Failed params as gray
    
    # Setup boundaries to lock integer categories 0, 1, 2, 3 into distinct colors
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(bounds, cmap.N)
    
    # Pcolormesh requires the coordinates of the grid *edges*, not centers
    dH = H_vals[1] - H_vals[0] if len(H_vals) > 1 else 1.0
    dA = A_vals[1] - A_vals[0] if len(A_vals) > 1 else 1.0
    
    H_edges = np.append(H_vals - dH/2, H_vals[-1] + dH/2)
    A_edges = np.append(A_vals - dA/2, A_vals[-1] + dA/2)
    
    A_mesh, H_mesh = np.meshgrid(A_edges, H_edges)
    
    # Plot the matrix
    c = ax.pcolormesh(A_mesh, H_mesh, phase_grid.T, cmap=cmap, norm=norm, edgecolors='none')
    
    # Setup categorical colorbar
    cbar = plt.colorbar(c, ax=ax, ticks=[0, 1, 2, 3], pad=0.03)
    cbar.ax.set_yticklabels(['Skyrmion Lattice (SkX)', 'Square Cell (SC)', 'Spiral Phase (SP)', 'Ferromagnetic (FM)'])
    cbar.set_label('Ground State Phase', rotation=270, labelpad=25, fontsize=13)
    
    # Aesthetics
    ax.set_xlabel('Scaled Anisotropy ($A_s$)', fontsize=14, labelpad=10)
    ax.set_ylabel('Scaled Magnetic Field ($H$)', fontsize=14, labelpad=10)
    ax.set_title('Topological Magnetic Phase Diagram', fontsize=16, pad=20)
    
    # Subtle dashed grid over the heatmap
    ax.grid(color='white', alpha=0.5, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig("phase_diagram.png", dpi=300)
    print("Saved high-res plot to 'phase_diagram.png'")
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Skyrmion Phase Diagram")
    parser.add_argument("--nH", type=int, default=26, help="Number of points along the H axis")
    parser.add_argument("--nA", type=int, default=33, help="Number of points along the A axis")
    parser.add_argument("--L", type=int, default=32, help="Lattice size for relaxation")
    parser.add_argument("--recompute", action="store_true", help="Force recomputation even if data exists")
    
    args = parser.parse_args()
    
    # Check if data was previously generated so we don't accidentally waste time recomputing
    if os.path.exists("phase_diagram_data.npy") and not args.recompute:
        ans = input("Found existing 'phase_diagram_data.npy'. Load it instead of recomputing? (y/n): ")
        if ans.lower() == 'y':
            print("Loading existing data...")
            grid = np.load("phase_diagram_data.npy")
            H = np.load("H_vals.npy")
            A = np.load("A_vals.npy")
            plot_phase_diagram(grid, H, A)
        else:
            generate_phase_diagram(n_H=args.nH, n_A=args.nA, L=args.L)
    else:
        generate_phase_diagram(n_H=args.nH, n_A=args.nA, L=args.L)
