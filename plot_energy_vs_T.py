import numpy as np
import matplotlib.pyplot as plt
import os
from fintemp_LLG import compare_fintemp_phases

# Create a clean mock arguments class to pass to the function
class Args:
    def __init__(self, T):
        self.L = 32           # Base unit cell size
        self.L_super = 64   # Supercell size for thermodynamics
        self.H = 1
        self.A = 0.5
        self.T = T
        self.dt = 0.005
        self.steps = 2000     # Reduced steps for a faster sweep
        self.block = 50
        self.seed = 42
        self.no_plot = True   # Disable live plotting so it sweeps instantly
        self.live_mode = "quiver"

def plot_energy_scaling():
    print("Starting Setup: Energy vs Temperature Sweep...")
    # Temperatures from almost 0 to 0.8
    temperatures = np.linspace(0.01, 10, 30) 
    
    phase_names = ["SkX", "SC", "SP", "FM"]
    energies_all = {p: [] for p in phase_names}

    import contextlib
    for T in temperatures:
        print(f"Sweeping at T = {T:.3f}... ", end="", flush=True)
        
        args = Args(T)
        # We set save_outputs=False so it doesn't flood your drive with .npz files during the sweep
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            winner, results = compare_fintemp_phases(args, save_outputs=False)
        
        print("Energies: ", end="")
        for p in phase_names:
            val = results.get(p, np.nan)
            energies_all[p].append(val)
            if not np.isnan(val):
                print(f"{p}: {val:.4f} | ", end="")
        print(f"Winner: {winner}")

    # Plot the results
    plt.figure(figsize=(9, 6))
    
    colors = {'SkX': 'red', 'SC': 'blue', 'SP': 'green', 'FM': 'purple'}
    markers = {'SkX': 'o', 'SC': 's', 'SP': '^', 'FM': 'D'}
    
    for p in phase_names:
        plt.plot(temperatures, energies_all[p], marker=markers[p], color=colors[p], 
                 label=p, markersize=8, linewidth=2, alpha=0.8, fillstyle='none', markeredgewidth=2)

    plt.xlabel("Scaled Temperature (T)", fontsize=12)
    plt.ylabel("Thermal Energy Density (Equilibrium)", fontsize=12)
    plt.title(f"Phase Energy Comparison (H={args.H}, A={args.A})", fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(title="Magnetic Phase")
    
    # Save and show
    os.makedirs("output/Fintemp", exist_ok=True)
    outpath = f"output/Fintemp/energy_vs_temperature_H{args.H}_A{args.A}_Tmax{args.T}.png"
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"\nSweep Complete! Plot saved to '{outpath}'")
    
    plt.show()

if __name__ == "__main__":
    plot_energy_scaling()
