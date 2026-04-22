import os
# CRITICAL: Prevent JAX from spanning infinite threads within multiprocessing workers!
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["JAX_CPU_DEFAULT_THREADS"] = "1"

import numpy as np
import sys
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

from fintemp_LLG import compare_fintemp_phases
from phase_diagram import plot_phase_diagram, plot_energy_difference, HiddenPrints

class FintempArgs:
    """Mock namespace to pass parameters smoothly into compare_fintemp_phases"""
    def __init__(self, L=32, L_super=None, H=0.0, A=0.0, T=0.1, dt=0.005, steps=1000, block=100, seed=42, no_plot=True, live_mode="quiver"):
        self.L = L
        self.L_super = L_super
        self.H = H
        self.A = A
        self.T = T
        self.dt = dt
        self.steps = steps
        self.block = block
        self.seed = seed
        self.no_plot = no_plot
        self.live_mode = live_mode

def _evaluate_fintemp_point(task):
    """
    Worker function for one fintemp phase-diagram point.
    Kept at module scope for Windows multiprocessing compatibility.
    """
    i, j, a, h, T_sel, L, steps, block = task
    phase_map = {"SkX": 0, "SC": 1, "SP": 2, "FM": 3}
    
    sim_args = FintempArgs(L=L, T=T_sel, steps=steps, block=block, no_plot=True)
    sim_args.H = h
    sim_args.A = a
    
    try:
        with HiddenPrints():
            winner, energies = compare_fintemp_phases(sim_args, save_outputs=False)
        return i, j, phase_map.get(winner, -1), energies, None
    except Exception as exc:
        return i, j, -1, {"SkX": np.nan, "SC": np.nan, "SP": np.nan, "FM": np.nan}, str(exc)

def generate_fintemp_phase_diagram(T_sel=0.1, n_H=26, n_A=33, L=32, steps=1000, block=100, workers=None):
    """
    Generates a finite-temperature phase diagram by sweeping H and A and integrating SDEs.
    """
    H_vals = np.linspace(0, 2.5, n_H)
    A_vals = np.linspace(-1.5, 1.7, n_A)
    
    phase_grid = np.zeros((n_A, n_H))
    
    energy_SkX = np.full((n_A, n_H), np.nan)
    energy_SC = np.full((n_A, n_H), np.nan)
    energy_SP = np.full((n_A, n_H), np.nan)
    energy_FM = np.full((n_A, n_H), np.nan)
    
    if workers is None:
        workers = min(os.cpu_count() or 1, n_H * n_A)
    if workers < 1:
        workers = 1
    
    print(f"=== Starting Finite-Temperature Phase Diagram Gen ===")
    print(f"Temperature: T={T_sel}")
    print(f"Resolution: {n_H * n_A} total points (H:{n_H}, A:{n_A})")
    print(f"SDE Config: {steps} steps (Block:{block}) measured at L={L}")
    print(f"Using {workers} worker process(es).")
    print("-----------------------------------------------------")
    
    start_time = time.time()
    
    # Create task list for all points
    tasks = [(i, j, a, h, T_sel, L, steps, block) for i, a in enumerate(A_vals) for j, h in enumerate(H_vals)]
    
    if workers == 1:
        # Sequential execution for debugging
        for idx, task in enumerate(tasks, start=1):
            i, j, a, h, _, _, _, _ = task
            sys.stdout.write(f"\rComputing Point {idx}/{n_H * n_A} | H = {h:.2f}, A = {a:.2f} ... ")
            sys.stdout.flush()
            
            ii, jj, phase_id, energies, err = _evaluate_fintemp_point(task)
            phase_grid[ii, jj] = phase_id
            energy_SkX[ii, jj] = energies.get("SkX", np.nan)
            energy_SC[ii, jj] = energies.get("SC", np.nan)
            energy_SP[ii, jj] = energies.get("SP", np.nan)
            energy_FM[ii, jj] = energies.get("FM", np.nan)
            if err is not None:
                print(f"\nFailed to converge at H={h}, A={a}: {err}")
    else:
        # Parallel execution with ProcessPoolExecutor
        completed = 0
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_task = {executor.submit(_evaluate_fintemp_point, task): task for task in tasks}
            
            for future in as_completed(future_to_task):
                completed += 1
                _, _, a, h, _, _, _, _ = future_to_task[future]
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
                    i, j, _, _, _, _, _, _ = future_to_task[future]
                    phase_grid[i, j] = -1
                    print(f"\nWorker crashed at H={h}, A={a}: {exc}")
                
    elapsed = time.time() - start_time
    print(f"\n\nSDE Sweep finished in {elapsed:.2f} seconds!")
    
    # Save Paths
    out_dir = "output/Fintemp/Phase Diagram Data"
    os.makedirs(out_dir, exist_ok=True)
    total_pts = n_H * n_A
    
    out_path = f"{out_dir}/fintemp_pd_T{T_sel}_L{L}_{total_pts}.npz"
    np.savez(out_path, grid=phase_grid, H_vals=H_vals, A_vals=A_vals,
             energy_SkX=energy_SkX, energy_SC=energy_SC, energy_SP=energy_SP, energy_FM=energy_FM, T=T_sel)
    
    print(f"Data bundled and saved to '{out_path}'. Generating plots...")
    
    # Leverage existing plot functions by utilizing geometric path substitution 
    # to redirect the hardcoded 'output/LLG/Graphs/' save route towards 'output/Fintemp/Graphs/' natively.
    os.makedirs("output/Fintemp/Graphs", exist_ok=True)
    redirect_dir = "../../Fintemp/Graphs" 
    
    pd_title = f"Topological Magnetic Phase Diagram (T = {T_sel})"
    plot_phase_diagram(phase_grid, H_vals, A_vals, out_name=f"{redirect_dir}/fintemp_pd_T{T_sel}_L{L}_{total_pts}.png", title=pd_title)
    
    energies_dict = {
        'SkX': energy_SkX,
        'SC': energy_SC,
        'SP': energy_SP,
        'FM': energy_FM
    }
    ed_title = f"Energy Gap to First Excited Phase (T = {T_sel})"
    plot_energy_difference(energies_dict, H_vals, A_vals, out_name=f"{redirect_dir}/fintemp_energy_diff_T{T_sel}_L{L}_{total_pts}.png", title=ed_title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Finite-Temperature Phase Diagram")
    parser.add_argument("--T", type=float, default=0.1, help="Temperature base for SDE thermalization")
    parser.add_argument("--nH", type=int, default=26, help="Number of points along the H axis")
    parser.add_argument("--nA", type=int, default=33, help="Number of points along the A axis")
    parser.add_argument("--L", type=int, default=32, help="Lattice size")
    parser.add_argument("--steps", type=int, default=1000, help="SDE thermal equilibration steps per point")
    parser.add_argument("--block", type=int, default=100, help="SDE energy averaging block size")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel multiprocessing workers")
    
    args = parser.parse_args()
    generate_fintemp_phase_diagram(T_sel=args.T, n_H=args.nH, n_A=args.nA, L=args.L, steps=args.steps, block=args.block, workers=args.workers)
