import os
import sys
# Allow imports from the parent directory (project root) when running this script directly or as a module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import argparse
import time

# Import the classification functions from the new state_analysis module
from state_analysis import analyze_state_mc, analyze_state_real_space, analyze_state_llg

def process_mc_data(input_npz_path, output_npz_path, classifier_type="mc"):
    """
    Loads MC spin configurations from input_npz_path, runs analyze_state on each point,
    and returns sorted H_vals, A_vals, and the classified phase grid.
    """
    print(f"Loading Monte Carlo simulation data from: {input_npz_path}")
    data = np.load(input_npz_path)
    
    # Extract coordinate ranges and metadata
    h_range = data['h_range']
    a_range = data['a_range']
    spins_all = data['spins']
    L = int(data['L'])
    T = float(data['temperature'])
    
    n_H = len(h_range)
    n_A = len(a_range)
    print(f"Dataset summary: L={L}, T={T}, H resolution={n_H}, A resolution={n_A}")
    
    # We construct sorted ascending coordinate axes for standard plotting layout
    sorted_h_indices = np.argsort(h_range)
    sorted_a_indices = np.argsort(a_range)
    
    H_vals = h_range[sorted_h_indices]
    A_vals = a_range[sorted_a_indices]
    
    # Initialize the phase ID matrix (A_res x H_res)
    # Categorization mapping matching phase_diagram.py:
    # SkX: 0, SC: 1, SP: 2, FM: 3, Unknown / Error: -1
    phase_map = {"SkX": 0, "SC": 1, "SP": 2, "FM": 3}
    phase_grid = np.zeros((n_A, n_H))
    
    # Dictionary to keep track of statistics
    stats_counts = {"SkX": 0, "SC": 0, "SP": 0, "FM": 0, "Unknown Phase": 0}
    
    print("\n--- Starting Phase Identification Sweep ---")
    start_time = time.time()
    
    # Loop over sorted coordinate positions
    total_pts = n_A * n_H
    completed = 0
    
    for i, a_idx in enumerate(sorted_a_indices):
        a_val = a_range[a_idx]
        for j, h_idx in enumerate(sorted_h_indices):
            h_val = h_range[h_idx]
            
            # Extract spin configuration: spins shape is (n_H, n_A, L, L, 3)
            # In L31_Res50_T0.01_cooled_data.npz, the axes are:
            # Axis 0 corresponds to h_range, Axis 1 corresponds to a_range
            spins = spins_all[h_idx, a_idx]
            
            # Select the requested classification function
            if classifier_type == "mc":
                classifier_func = analyze_state_mc
            elif classifier_type == "real-space":
                classifier_func = analyze_state_real_space
            elif classifier_type == "llg":
                classifier_func = analyze_state_llg
            else:
                raise ValueError(f"Unknown classifier type: {classifier_type}")
                
            # Analyze phase using the selected classification function
            stats = classifier_func(spins, ax=1.0, ay=1.0, phase_name=f"H{h_val:.2f}_As{a_val:.2f}", plot_fft=False)
            
            state = stats["classified_state"]
            phase_id = phase_map.get(state, -1)
            phase_grid[i, j] = phase_id
            
            stats_counts[state] = stats_counts.get(state, 0) + 1
            
            completed += 1
            if completed % 100 == 0 or completed == total_pts:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                eta = (total_pts - completed) / rate if rate > 0 else 0.0
                print(f"Processed Point {completed}/{total_pts} | H={h_val:.2f}, A={a_val:.2f} | Phase: {state:<13} | Rate: {rate:.1f} pts/s | ETA: {eta:.1f}s")
                
    elapsed = time.time() - start_time
    print(f"\nPhase classification completed in {elapsed:.2f} seconds!")
    print(f"Phase distribution: {dict(stats_counts)}")
    
    # Save the processed data for instant future plotting
    os.makedirs(os.path.dirname(output_npz_path), exist_ok=True)
    np.savez(output_npz_path, grid=phase_grid, H_vals=H_vals, A_vals=A_vals, L=L, T=T)
    print(f"Classification cache saved to '{output_npz_path}'")
    
    return phase_grid, H_vals, A_vals, L, T

def plot_mc_phase_diagram(phase_grid, H_vals, A_vals, L, T, out_png_path, title=None):
    """
    Draws the phase diagram matching the exact publication aesthetics of phase_diagram.py.
    """
    if title is None:
        title = f"Monte Carlo Phase Diagram (L={L}x{L}, T={T})"
        
    print(f"Generating plot: {out_png_path}")
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Muted, academic color scheme matching phase_diagram.py:
    # SkX (Muted Blue), SC (Muted Green), SP (Muted Orange/Brown), FM (Light Gray/White)
    cmap = ListedColormap(['#4C72B0', '#55A868', '#DD8452', '#EAEAF2'])
    cmap.set_under('#808080')  # Unknown / failed points colored gray
    
    # Lock integers 0, 1, 2, 3 into categorical bounds
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(bounds, cmap.N)
    
    # Pcolormesh coordinates of the grid edges
    dH = H_vals[1] - H_vals[0] if len(H_vals) > 1 else 1.0
    dA = A_vals[1] - A_vals[0] if len(A_vals) > 1 else 1.0
    
    H_edges = np.append(H_vals - dH/2, H_vals[-1] + dH/2)
    A_edges = np.append(A_vals - dA/2, A_vals[-1] + dA/2)
    
    A_mesh, H_mesh = np.meshgrid(A_edges, H_edges)
    
    # Plot the matrix (transposed to align axes correctly: Anisotropy x, Field y)
    c = ax.pcolormesh(A_mesh, H_mesh, phase_grid.T, cmap=cmap, norm=norm, edgecolors='none')
    
    # Setup categorical colorbar
    cbar = plt.colorbar(c, ax=ax, ticks=[0, 1, 2, 3], pad=0.03)
    cbar.ax.set_yticklabels(['Skyrmion Lattice (SkX)', 'Square Cell (SC)', 'Spiral Phase (SP)', 'Ferromagnetic (FM)'])
    cbar.set_label('Ground State Phase', rotation=270, labelpad=25, fontsize=13)
    
    # Aesthetics
    ax.set_xlabel('Scaled Uniaxial Anisotropy ($A_s$)', fontsize=14, labelpad=10)
    ax.set_ylabel('Scaled Magnetic Field ($H$)', fontsize=14, labelpad=10)
    ax.set_title(title, fontsize=16, pad=20)
    
    # White dashed grid lines separating grid bins
    ax.grid(color='white', alpha=0.5, linestyle='--', linewidth=0.5)
    
    # Ensure axes limits bound the data nicely
    ax.set_xlim(A_edges[0], A_edges[-1])
    ax.set_ylim(H_edges[0], H_edges[-1])
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    plt.savefig(out_png_path, dpi=300)
    print(f"Saved publication-grade plot to '{out_png_path}'")
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo Skyrmion Phase Diagram Generator")
    parser.add_argument("--input", type=str, default="output/phase_diagrams/mc/L31_Res50_T0.01_cooled_data.npz",
                        help="Path to the Monte Carlo simulation npz file")
    parser.add_argument("--classifier", type=str, choices=["mc", "real-space", "llg"], default="mc",
                        help="Which classifier to use for phase identification: 'mc' (reciprocal FFT), 'real-space' (structure tensor & bond order), or 'llg' (zero-temp FFT)")
    parser.add_argument("--cache", type=str, default=None,
                        help="Path to save/load classified phase grid cache (defaults to classifier-specific path)")
    parser.add_argument("--output-png", type=str, default=None,
                        help="Path to save the generated phase diagram plot (defaults to classifier-specific path)")
    parser.add_argument("--recompute", action="store_true", help="Force recalculation even if cache exists")
    parser.add_argument("--title", type=str, default=None, help="Custom title for the phase diagram plot")
    
    args = parser.parse_args()
    
    # Resolve default paths based on chosen classifier to avoid overwriting caches/images
    if args.cache is None:
        if args.input == "output/phase_diagrams/mc/L31_Res50_T0.01_cooled_data.npz":
            if args.classifier == "mc":
                args.cache = "output/phase_diagrams/mc/L31_Res50_T0.01_classified_grid.npz"
            elif args.classifier == "real-space":
                args.cache = "output/phase_diagrams/mc/L31_Res50_T0.01_realspace_grid.npz"
            else:  # llg
                args.cache = "output/phase_diagrams/mc/L31_Res50_T0.01_llg_grid.npz"
        else:
            # Extract directory and base name from the input file
            input_dir = os.path.dirname(args.input)
            input_base = os.path.splitext(os.path.basename(args.input))[0]
            if input_base.endswith("_data"):
                clean_base = input_base[:-5]
            else:
                clean_base = input_base
            args.cache = os.path.join(input_dir, f"{clean_base}_{args.classifier}_grid.npz")
            
    if args.output_png is None:
        if args.input == "output/phase_diagrams/mc/L31_Res50_T0.01_cooled_data.npz":
            if args.classifier == "mc":
                args.output_png = "output/phase_diagrams/mc/mc_phase_diagram.png"
            elif args.classifier == "real-space":
                args.output_png = "output/phase_diagrams/mc/mc_phase_diagram_realspace.png"
            else:  # llg
                args.output_png = "output/phase_diagrams/mc/mc_phase_diagram_llg.png"
        else:
            input_dir = os.path.dirname(args.input)
            input_base = os.path.splitext(os.path.basename(args.input))[0]
            if input_base.endswith("_data"):
                clean_base = input_base[:-5]
            else:
                clean_base = input_base
            args.output_png = os.path.join(input_dir, f"{clean_base}_{args.classifier}_diagram.png")
            
    # Check if cache exists to save computation time
    if os.path.exists(args.cache) and not args.recompute:
        print(f"Found existing classified phase grid cache at: {args.cache}")
        data = np.load(args.cache)
        phase_grid = data['grid']
        H_vals = data['H_vals']
        A_vals = data['A_vals']
        L = int(data['L']) if 'L' in data else 31
        T = float(data['T']) if 'T' in data else 0.01
    else:
        # Check that the input file exists
        if not os.path.exists(args.input):
            print(f"Error: Input file '{args.input}' not found.", file=sys.stderr)
            if os.path.basename(os.getcwd()) == "plotting":
                print("\n[TIP] It looks like you ran this script from inside the 'plotting' folder.")
                print("Please navigate to the project root directory first, then run:")
                print("  cd ..")
                print("  python .\\plotting\\plot_mc_phase_diagram.py")
            sys.exit(1)
        phase_grid, H_vals, A_vals, L, T = process_mc_data(args.input, args.cache, classifier_type=args.classifier)
        
    # Format a descriptive title if not custom-specified
    if args.title is None:
        classifier_names = {
            "mc": "Reciprocal FFT (MC-Optimized)",
            "real-space": "Real-Space Classifier",
            "llg": "Standard Zero-Temp FFT"
        }
        args.title = f"Monte Carlo Phase Diagram (L={L}x{L}, T={T})\nClassifier: {classifier_names[args.classifier]}"
        
    # Plot the phase diagram
    plot_mc_phase_diagram(phase_grid, H_vals, A_vals, L, T, args.output_png, title=args.title)
