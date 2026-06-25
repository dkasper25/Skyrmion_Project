import os
# CRITICAL: Prevent JAX from spanning infinite threads within multiprocessing workers!
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["JAX_CPU_DEFAULT_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import sys
import os
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

from LLG_solver import compare_phases

# Utility class to suppress the verbose output from LLG_solver during the long sweep
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def _evaluate_phase_point(task):
    """
    Worker function for one phase-diagram point.
    Kept at module scope for Windows multiprocessing compatibility.
    """
    i, j, a, h, L, iso_scale = task
    phase_map = {"SkX": 0, "SC": 1, "SP": 2, "FM": 3}

    try:
        with HiddenPrints():
            winner, energies = compare_phases(
                H_scaled=h,
                A_scaled=a,
                L=L,
                plot_ansatz=False,
                live_plot=False,
                save_outputs=False,
                iso_scale=iso_scale
            )
        return i, j, phase_map.get(winner, -1), energies, None
    except Exception as exc:
        return i, j, -1, {"SkX": np.nan, "SC": np.nan, "SP": np.nan, "FM": np.nan}, str(exc)

def generate_phase_diagram(n_H=26, n_A=33, L=32, workers=None, H_min=0.0, H_max=2.5, A_min=-1.5, A_max=1.7, iso_scale=False):
    """
    Generates a phase diagram by scanning H and A.
    Saves the data and automatically calls the plotter.
    """
    H_vals = np.linspace(H_min, H_max, n_H)
    A_vals = np.linspace(A_min, A_max, n_A)
    
    # Matrix to store the result phase IDs
    phase_grid = np.zeros((n_A, n_H))
    
    # Matrices to store the energies for each phase
    energy_SkX = np.full((n_A, n_H), np.nan)
    energy_SC = np.full((n_A, n_H), np.nan)
    energy_SP = np.full((n_A, n_H), np.nan)
    energy_FM = np.full((n_A, n_H), np.nan)
    
    if workers is None:
        workers = min(os.cpu_count() or 1, n_H * n_A)
    if workers < 1:
        workers = 1
    
    print(f"=== Starting Phase Diagram Generation ===")
    print(f"Total points: {n_H * n_A} (H resolution: {n_H}, A resolution: {n_A})")
    print(f"Using lattice size L={L} for the sweep.")
    print(f"Using {workers} worker process(es).")
    print("-----------------------------------------")
    
    start_time = time.time()
    
    tasks = [(i, j, a, h, L, iso_scale) for i, a in enumerate(A_vals) for j, h in enumerate(H_vals)]
    
    if workers == 1:
        for idx, task in enumerate(tasks, start=1):
            i, j, a, h, _, _ = task
            sys.stdout.write(f"\rComputing Point {idx}/{n_H * n_A} | H = {h:.2f}, A = {a:.2f} ... ")
            sys.stdout.flush()
            
            ii, jj, phase_id, energies, err = _evaluate_phase_point(task)
            phase_grid[ii, jj] = phase_id
            energy_SkX[ii, jj] = energies.get("SkX", np.nan)
            energy_SC[ii, jj] = energies.get("SC", np.nan)
            energy_SP[ii, jj] = energies.get("SP", np.nan)
            energy_FM[ii, jj] = energies.get("FM", np.nan)
            if err is not None:
                print(f"\nFailed to converge at H={h}, A={a}: {err}")
    else:
        completed = 0
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_task = {executor.submit(_evaluate_phase_point, task): task for task in tasks}
            
            for future in as_completed(future_to_task):
                completed += 1
                _, _, a, h, _, _ = future_to_task[future]
                sys.stdout.write(f"\rComputing Point {completed}/{n_H * n_A} | H = {h:.2f}, A = {a:.2f} ... ")
                sys.stdout.flush()
                
                try:
                    ii, jj, phase_id, energies, err = future.result()
                    phase_grid[ii, jj] = phase_id
                    energy_SkX[ii, jj] = energies.get("SkX", np.nan)
                    energy_SC[ii, jj] = energies.get("SC", np.nan)
                    energy_SP[ii, jj] = energies.get("SP", np.nan)
                    energy_FM[ii, jj] = energies.get("FM", np.nan)
                    if err is not None:
                        print(f"\nFailed to converge at H={h}, A={a}: {err}")
                except Exception as exc:
                    i, j, _, _, _, _ = future_to_task[future]
                    phase_grid[i, j] = -1
                    print(f"\nWorker crashed at H={h}, A={a}: {exc}")
                
    elapsed = time.time() - start_time
    print(f"\n\nSweep finished in {elapsed:.2f} seconds!")
    
    # Save the exact simulation data so you don't have to recompute just to adjust the plot design!
    total_pts = n_H * n_A
    os.makedirs("output/phase_diagrams/llg", exist_ok=True)
    out_path = f"output/phase_diagrams/llg/phase_diagram_L{L}_{total_pts}.npz"
    np.savez(out_path, grid=phase_grid, H_vals=H_vals, A_vals=A_vals,
             energy_SkX=energy_SkX, energy_SC=energy_SC, energy_SP=energy_SP, energy_FM=energy_FM)
    
    print(f"Data bundled and saved to '{out_path}'. Generating plot...")
    plot_phase_diagram(phase_grid, H_vals, A_vals, out_name=f"phase_diagram_L{L}_{total_pts}.png")
    
    energies_dict = {
        'SkX': energy_SkX,
        'SC': energy_SC,
        'SP': energy_SP,
        'FM': energy_FM
    }
    plot_energy_difference(energies_dict, H_vals, A_vals, out_name=f"phase_diagram_energy_diff_L{L}_{total_pts}.png")
    plot_fm_stabilization_energy(energies_dict, H_vals, A_vals, out_name=f"phase_diagram_fm_stab_L{L}_{total_pts}.png")


def plot_phase_diagram(phase_grid, H_vals, A_vals, out_name="phase_diagram.png", title="Topological Magnetic Phase Diagram"):
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
    ax.set_title(title, fontsize=16, pad=20)
    
    # Subtle dashed grid over the heatmap
    ax.grid(color='white', alpha=0.5, linestyle='--', linewidth=0.5)
    plt.style.use('default')
    os.makedirs("output/phase_diagrams/llg", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"output/phase_diagrams/llg/{out_name}", dpi=300)
    print(f"Saved high-res plot to 'output/phase_diagrams/llg/{out_name}'")
    
    plt.show()

def plot_energy_difference(energies_dict, H_vals, A_vals, out_name="energy_diff.png", title="Energy Gap to First Excited Phase"):
    """
    Plots a heatmap of the energy difference between the lowest and 
    second lowest energy phases at each point.
    """
    n_A, n_H = len(A_vals), len(H_vals)
    energy_diff_grid = np.zeros((n_A, n_H))
    
    # Stack the arrays: shape will be (4, n_A, n_H)
    energy_stack = np.stack([
        energies_dict['SkX'],
        energies_dict['SC'],
        energies_dict['SP'],
        energies_dict['FM']
    ])
    
    for i in range(n_A):
        for j in range(n_H):
            # Extract energies at point (i, j)
            point_energies = energy_stack[:, i, j]
            # Remove NaNs
            valid_energies = point_energies[~np.isnan(point_energies)]
            
            if len(valid_energies) >= 2:
                # Sort to find the two lowest
                sorted_energies = np.sort(valid_energies)
                energy_diff_grid[i, j] = sorted_energies[1] - sorted_energies[0]
            else:
                # If less than 2 valid phases, difference is np.nan
                energy_diff_grid[i, j] = np.nan
                
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 8))
    
    dH = H_vals[1] - H_vals[0] if len(H_vals) > 1 else 1.0
    dA = A_vals[1] - A_vals[0] if len(A_vals) > 1 else 1.0
    
    H_edges = np.append(H_vals - dH/2, H_vals[-1] + dH/2)
    A_edges = np.append(A_vals - dA/2, A_vals[-1] + dA/2)
    A_mesh, H_mesh = np.meshgrid(A_edges, H_edges)
    
    # Plot using a sequential colormap
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color='#808080') # set color for NaN
    
    c = ax.pcolormesh(A_mesh, H_mesh, energy_diff_grid.T, cmap=cmap, edgecolors='none')
    
    cbar = plt.colorbar(c, ax=ax, pad=0.03)
    cbar.set_label('Energy Difference $\\Delta E$ (2nd Lowest - Lowest)', rotation=270, labelpad=25, fontsize=13)
    
    ax.set_xlabel('Scaled Anisotropy ($A_s$)', fontsize=14, labelpad=10)
    ax.set_ylabel('Scaled Magnetic Field ($H$)', fontsize=14, labelpad=10)
    ax.set_title(title, fontsize=16, pad=20)
    
    ax.grid(color='white', alpha=0.3, linestyle='--', linewidth=0.5)
    plt.style.use('default')
    os.makedirs("output/phase_diagrams/llg", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"output/phase_diagrams/llg/{out_name}", dpi=300)
    print(f"Saved high-res energy diff plot to 'output/phase_diagrams/llg/{out_name}'")
    
    plt.show()

def plot_fm_stabilization_energy(energies_dict, H_vals, A_vals, out_name="fm_stabilization.png", title="FM Stabilization Energy"):
    """
    Plots a heatmap of the energy difference between the Ferromagnetic (FM) state 
    and the lowest energy state at each point. Positive values mean the structured phase
    is more stable than the FM state.
    """
    import matplotlib.colors as mcolors
    
    n_A, n_H = len(A_vals), len(H_vals)
    energy_diff_grid = np.zeros((n_A, n_H))
    
    # Stack the arrays
    energy_stack = np.stack([
        energies_dict['SkX'],
        energies_dict['SC'],
        energies_dict['SP'],
        energies_dict['FM']
    ])
    
    for i in range(n_A):
        for j in range(n_H):
            point_energies = energy_stack[:, i, j]
            fm_energy = energies_dict['FM'][i, j]
            
            # Remove NaNs
            valid_energies = point_energies[~np.isnan(point_energies)]
            
            if len(valid_energies) > 0 and not np.isnan(fm_energy):
                lowest_energy = np.min(valid_energies)
                diff = fm_energy - lowest_energy
                if diff <= 1e-10:
                    # FM is the ground state (or effectively degenerate)
                    energy_diff_grid[i, j] = 0.0
                else:
                    energy_diff_grid[i, j] = diff
            else:
                energy_diff_grid[i, j] = np.nan
                
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 8))
    
    dH = H_vals[1] - H_vals[0] if len(H_vals) > 1 else 1.0
    dA = A_vals[1] - A_vals[0] if len(A_vals) > 1 else 1.0
    
    H_edges = np.append(H_vals - dH/2, H_vals[-1] + dH/2)
    A_edges = np.append(A_vals - dA/2, A_vals[-1] + dA/2)
    A_mesh, H_mesh = np.meshgrid(A_edges, H_edges)
    
    # Truncate magma so the lowest value isn't black
    base_cmap = plt.cm.magma
    colors = base_cmap(np.linspace(0.15, 1.0, 256))
    cmap = mcolors.LinearSegmentedColormap.from_list('trunc_magma', colors)
    
    cmap.set_bad(color='#808080') # gray for NaN/Failed
    cmap.set_under(color='black') # black for exactly 0 (FM ground state)
    
    # Use vmin=1e-8 so that exactly 0.0 falls "under" the colormap range
    valid_diffs = energy_diff_grid[energy_diff_grid > 1e-10]
    vmax = np.max(valid_diffs) if len(valid_diffs) > 0 else 1.0
    
    c = ax.pcolormesh(A_mesh, H_mesh, energy_diff_grid.T, cmap=cmap, vmin=1e-8, vmax=vmax, edgecolors='none')
    
    cbar = plt.colorbar(c, ax=ax, pad=0.03)
    cbar.set_label('Stabilization Energy $\\Delta E$ (FM - Lowest Phase)', rotation=270, labelpad=25, fontsize=13)
    
    ax.set_xlabel('Scaled Anisotropy ($A_s$)', fontsize=14, labelpad=10)
    ax.set_ylabel('Scaled Magnetic Field ($H$)', fontsize=14, labelpad=10)
    ax.set_title(title, fontsize=16, pad=20)
    
    ax.grid(color='white', alpha=0.3, linestyle='--', linewidth=0.5)
    plt.style.use('default')
    os.makedirs("output/phase_diagrams/llg", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"output/phase_diagrams/llg/{out_name}", dpi=300)
    print(f"Saved high-res FM stabilization plot to 'output/phase_diagrams/llg/{out_name}'")
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Skyrmion Phase Diagram")
    parser.add_argument("--nH", type=int, default=26, help="Number of points along the H axis")
    parser.add_argument("--nA", type=int, default=33, help="Number of points along the A axis")
    parser.add_argument("--L", type=int, default=32, help="Lattice size for relaxation")
    parser.add_argument("--recompute", action="store_true", help="Force recomputation even if data exists")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (defaults to CPU count)")
    parser.add_argument("--iso-scale", action="store_true", help="Enforce isotropic scaling (preserve initial aspect ratio) in phase relaxation")
    parser.add_argument("--H_min", type=float, default=0.0, help="Minimum H value")
    parser.add_argument("--H_max", type=float, default=2.5, help="Maximum H value")
    parser.add_argument("--A_min", type=float, default=-1.5, help="Minimum A value")
    parser.add_argument("--A_max", type=float, default=1.7, help="Maximum A value")
    
    args = parser.parse_args()
    
    import glob
    existing_files = glob.glob("output/phase_diagrams/llg/*.npz")
    
    if len(existing_files) > 0 and not args.recompute:
        print("\nFound existing phase diagram data files:")
        print(" [0] : Generate NEW phase diagram")
        for idx, f in enumerate(existing_files, 1):
            print(f" [{idx}] : Load {os.path.basename(f)}")
        
        choice = input(f"\nWhich one would you like to load? [0-{len(existing_files)}] (default 0): ").strip()
        if choice and choice.isdigit() and 0 < int(choice) <= len(existing_files):
            sel_file = existing_files[int(choice)-1]
            print(f"Loading existing data from {os.path.basename(sel_file)}...")
            data = np.load(sel_file)
            grid = data['grid']
            H = data['H_vals']
            A = data['A_vals']
            out_png = os.path.basename(sel_file).replace('.npz', '.png')
            plot_phase_diagram(grid, H, A, out_name=out_png)
            
            if 'energy_SkX' in data:
                energies_dict = {
                    'SkX': data['energy_SkX'],
                    'SC': data['energy_SC'],
                    'SP': data['energy_SP'],
                    'FM': data['energy_FM']
                }
                out_diff_png = os.path.basename(sel_file).replace('.npz', '_energy_diff.png')
                plot_energy_difference(energies_dict, H, A, out_name=out_diff_png)
                out_fm_png = os.path.basename(sel_file).replace('.npz', '_fm_stabilization.png')
                plot_fm_stabilization_energy(energies_dict, H, A, out_name=out_fm_png)
        else:
            if args.workers is not None and args.workers > 1:
                print("WARNING: Multiprocessing on Windows requires __main__ block isolation.")
                generate_phase_diagram(n_H=args.nH, n_A=args.nA, L=args.L, workers=args.workers, H_min=args.H_min, H_max=args.H_max, A_min=args.A_min, A_max=args.A_max, iso_scale=args.iso_scale)
            else:
                generate_phase_diagram(n_H=args.nH, n_A=args.nA, L=args.L, workers=args.workers, H_min=args.H_min, H_max=args.H_max, A_min=args.A_min, A_max=args.A_max, iso_scale=args.iso_scale)
    else:
        generate_phase_diagram(n_H=args.nH, n_A=args.nA, L=args.L, workers=args.workers, H_min=args.H_min, H_max=args.H_max, A_min=args.A_min, A_max=args.A_max, iso_scale=args.iso_scale)
